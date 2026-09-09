"""Build sampled or exhaustive PAIN paths for DBLP link prediction.

The three physical variants attach research-area information to papers (v1),
conferences (v2), or authors (v3).  The universal baseline is their edge union.
Paper-conference target edges are train-only in every message-passing graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from pain_nc.experiment import atomic_torch_save, atomic_write_json


SEED = 1566911444
PATH_LENGTH = 3
VARIANTS = ("v1", "v2", "v3", "universal")
NODE_TYPE_NAMES = ("author", "paper", "term", "conference", "area")
EDGE_TYPE_NAMES = (
    "author-paper",
    "paper-term",
    "paper-conference",
    "paper-area",
    "conference-area",
    "author-area",
)
EDGE_TYPE = {name: index for index, name in enumerate(EDGE_TYPE_NAMES)}


@dataclass
class Contract:
    author_label: pd.DataFrame
    paper_author: pd.DataFrame
    paper_conf: pd.DataFrame
    paper_term: pd.DataFrame
    paper_area: pd.DataFrame
    maps: dict[str, dict[int, int]]
    offsets: dict[str, int]
    counts: dict[str, int]
    node_type: torch.Tensor
    splits: dict[str, torch.Tensor]
    raw_split_papers: dict[str, np.ndarray]


def read_tables(raw_dir: Path) -> tuple[pd.DataFrame, ...]:
    author_label = pd.read_csv(
        raw_dir / "author_label.txt",
        sep="\t",
        names=["author_id", "area_id", "author_name"],
        header=None,
        encoding="utf-8",
    )
    paper_author = pd.read_csv(
        raw_dir / "paper_author.txt",
        sep="\t",
        names=["paper_id", "author_id"],
        header=None,
        encoding="utf-8",
    )
    paper_conf = pd.read_csv(
        raw_dir / "paper_conf.txt",
        sep="\t",
        names=["paper_id", "conf_id"],
        header=None,
        encoding="utf-8",
    )
    paper_term = pd.read_csv(
        raw_dir / "paper_term.txt",
        sep="\t",
        names=["paper_id", "term_id"],
        header=None,
        encoding="utf-8",
    )
    return author_label, paper_author, paper_conf, paper_term


def filter_tables(
    author_label: pd.DataFrame,
    paper_author: pd.DataFrame,
    paper_conf: pd.DataFrame,
    paper_term: pd.DataFrame,
    min_conf: int,
) -> tuple[pd.DataFrame, ...]:
    valid_authors = set(author_label["author_id"])
    paper_author = paper_author[
        paper_author["author_id"].isin(valid_authors)
    ].drop_duplicates().reset_index(drop=True)
    valid_papers = set(paper_author["paper_id"])
    paper_conf = paper_conf[
        paper_conf["paper_id"].isin(valid_papers)
    ].drop_duplicates().reset_index(drop=True)
    paper_term = paper_term[
        paper_term["paper_id"].isin(valid_papers)
    ].drop_duplicates().reset_index(drop=True)
    if min_conf > 0:
        counts = paper_conf["conf_id"].value_counts()
        conferences = counts[counts >= min_conf].index
        paper_conf = paper_conf[
            paper_conf["conf_id"].isin(conferences)
        ].reset_index(drop=True)
    valid_papers = set(paper_conf["paper_id"])
    paper_author = paper_author[
        paper_author["paper_id"].isin(valid_papers)
    ].reset_index(drop=True)
    paper_term = paper_term[
        paper_term["paper_id"].isin(valid_papers)
    ].reset_index(drop=True)
    return author_label, paper_author, paper_conf, paper_term


def split_papers(
    paper_ids: Iterable[int], seed: int
) -> dict[str, np.ndarray]:
    papers = np.asarray(sorted(set(paper_ids)), dtype=np.int64)
    train, remainder = train_test_split(
        papers, test_size=0.30, random_state=seed, shuffle=True
    )
    relative_test_ratio = 2.0 / 3.0
    val, test = train_test_split(
        remainder, test_size=relative_test_ratio, random_state=seed, shuffle=True
    )
    return {
        "train": np.sort(train).astype(np.int64),
        "val": np.sort(val).astype(np.int64),
        "test": np.sort(test).astype(np.int64),
    }


def make_maps(
    author_label: pd.DataFrame,
    paper_author: pd.DataFrame,
    paper_conf: pd.DataFrame,
    paper_term: pd.DataFrame,
    paper_area: pd.DataFrame,
) -> tuple[dict, dict, dict, torch.Tensor]:
    values = {
        "author": sorted(paper_author["author_id"].unique()),
        "paper": sorted(paper_conf["paper_id"].unique()),
        "term": sorted(paper_term["term_id"].unique()),
        "conference": sorted(paper_conf["conf_id"].unique()),
        "area": sorted(
            set(paper_area["area_id"].unique())
            | set(author_label["area_id"].unique())
        ),
    }
    maps: dict[str, dict[int, int]] = {}
    offsets: dict[str, int] = {}
    counts: dict[str, int] = {}
    node_types = []
    offset = 0
    for type_id, name in enumerate(NODE_TYPE_NAMES):
        offsets[name] = offset
        counts[name] = len(values[name])
        maps[name] = {
            int(raw): offset + local for local, raw in enumerate(values[name])
        }
        node_types.append(
            torch.full((len(values[name]),), type_id, dtype=torch.int8)
        )
        offset += len(values[name])
    return maps, offsets, counts, torch.cat(node_types)


def positive_pairs(
    paper_conf: pd.DataFrame,
    papers: np.ndarray,
    maps: dict[str, dict[int, int]],
) -> torch.Tensor:
    selected = paper_conf[
        paper_conf["paper_id"].isin(set(papers.tolist()))
    ][["paper_id", "conf_id"]].drop_duplicates()
    rows = [
        (maps["paper"][int(paper)], maps["conference"][int(conf)])
        for paper, conf in selected.itertuples(index=False)
    ]
    return torch.tensor(rows, dtype=torch.long).reshape(-1, 2)


def all_negative_tails(
    positives: torch.Tensor,
    conference_ids: torch.Tensor,
) -> torch.Tensor:
    """All false conference ids for each positive query, in canonical order."""
    known: dict[int, set[int]] = {}
    for paper, conf in positives.tolist():
        known.setdefault(int(paper), set()).add(int(conf))
    rows = [
        [int(conf) for conf in conference_ids.tolist() if int(conf) not in known[int(paper)]]
        for paper, _ in positives.tolist()
    ]
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(
            "DBLP training negatives are ragged because papers have differing "
            "numbers of true conferences; this pipeline expects one venue per paper"
        )
    return torch.tensor(rows, dtype=torch.long)


def build_contract(raw_dir: Path, seed: int, min_conf: int) -> Contract:
    al, pa, pc, pt = filter_tables(*read_tables(raw_dir), min_conf)
    # Split the full eligible paper population first, then remove papers that
    # are not valid one-paper/one-area switching blocks. This matches the
    # reference DBLP preprocessing order and keeps the split provenance stable.
    raw_split_papers = split_papers(pc["paper_id"].unique(), seed)
    pr_raw = (
        pa.merge(al[["author_id", "area_id"]], on="author_id")
        [["paper_id", "area_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    area_counts = pr_raw.groupby("paper_id")["area_id"].nunique()
    valid_papers = set(area_counts[area_counts == 1].index.tolist())
    pa = pa[pa["paper_id"].isin(valid_papers)].reset_index(drop=True)
    pc = pc[pc["paper_id"].isin(valid_papers)].reset_index(drop=True)
    pt = pt[pt["paper_id"].isin(valid_papers)].reset_index(drop=True)
    valid_authors = set(pa["author_id"])
    al = al[al["author_id"].isin(valid_authors)].reset_index(drop=True)
    pr = (
        pr_raw[pr_raw["paper_id"].isin(valid_papers)]
        .drop_duplicates("paper_id")
        .reset_index(drop=True)
    )
    maps, offsets, counts, node_type = make_maps(al, pa, pc, pt, pr)
    splits = {
        f"{name}_pos": positive_pairs(pc, papers, maps)
        for name, papers in raw_split_papers.items()
    }
    conference_ids = torch.arange(
        offsets["conference"],
        offsets["conference"] + counts["conference"],
        dtype=torch.long,
    )
    splits["train_neg_tails"] = all_negative_tails(
        splits["train_pos"], conference_ids
    )
    splits["conference_candidates"] = conference_ids
    return Contract(
        al, pa, pc, pt, pr, maps, offsets, counts, node_type, splits,
        raw_split_papers,
    )


def canonical_edge(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def add_edge(
    edges: dict[str, set[tuple[int, int]]],
    relation: str,
    left: int,
    right: int,
) -> None:
    edges.setdefault(relation, set()).add(canonical_edge(left, right))


def build_variant_edges(
    contract: Contract,
    *,
    area_scope: str,
) -> dict[str, dict[str, set[tuple[int, int]]]]:
    if area_scope not in {"baseline", "train"}:
        raise ValueError(f"Unknown DBLP Area scope: {area_scope}")
    maps = contract.maps
    base: dict[str, set[tuple[int, int]]] = {}
    for paper, author in contract.paper_author[
        ["paper_id", "author_id"]
    ].itertuples(index=False):
        add_edge(
            base, "author-paper", maps["author"][int(author)],
            maps["paper"][int(paper)]
        )
    for paper, term in contract.paper_term[
        ["paper_id", "term_id"]
    ].itertuples(index=False):
        add_edge(
            base, "paper-term", maps["paper"][int(paper)],
            maps["term"][int(term)]
        )
    for paper, conf in contract.splits["train_pos"].tolist():
        add_edge(base, "paper-conference", int(paper), int(conf))

    variants = {
        name: {relation: set(rows) for relation, rows in base.items()}
        for name in VARIANTS[:-1]
    }
    train_raw_papers = set(contract.raw_split_papers["train"].tolist())
    area_support_papers = (
        train_raw_papers
        if area_scope == "train"
        else set(contract.paper_area["paper_id"].tolist())
    )
    for paper, area in contract.paper_area[
        ["paper_id", "area_id"]
    ].itertuples(index=False):
        if int(paper) not in area_support_papers:
            continue
        add_edge(
            variants["v1"], "paper-area", maps["paper"][int(paper)],
            maps["area"][int(area)]
        )

    # Leakage-safe v2: conference-area evidence is compiled from training
    # paper-conference blocks only, never from held-out target pairs.
    conference_area = (
        contract.paper_conf[
            contract.paper_conf["paper_id"].isin(train_raw_papers)
        ][["paper_id", "conf_id"]]
        .merge(contract.paper_area, on="paper_id")
        [["conf_id", "area_id"]]
        .drop_duplicates()
    )
    for conf, area in conference_area.itertuples(index=False):
        add_edge(
            variants["v2"], "conference-area",
            maps["conference"][int(conf)], maps["area"][int(area)]
        )

    area_authors = set(
        contract.paper_author[
            contract.paper_author["paper_id"].isin(area_support_papers)
        ]["author_id"].tolist()
    )
    for author, area in contract.author_label[
        ["author_id", "area_id"]
    ].drop_duplicates().itertuples(index=False):
        if int(author) not in area_authors:
            continue
        add_edge(
            variants["v3"], "author-area", maps["author"][int(author)],
            maps["area"][int(area)]
        )
    universal: dict[str, set[tuple[int, int]]] = {}
    for variant in variants.values():
        for relation, rows in variant.items():
            universal.setdefault(relation, set()).update(rows)
    variants["universal"] = universal
    return variants


def compile_invariant_edges(
    contract: Contract,
    physical: dict[str, set[tuple[int, int]]],
    variant: str,
) -> tuple[dict[str, set[tuple[int, int]]], dict[str, int | str]]:
    """Compile one physical realization into the common semantic edge program.

    Area-bearing intermediate nodes are context only. They license the same
    Paper-Area, Conference-Area, and Author-Area propagation edges regardless
    of whether the physical evidence occurs on a paper, conference, or author.
    """
    semantic = {
        relation: set(physical.get(relation, set()))
        for relation in ("author-paper", "paper-term", "paper-conference")
    }
    maps = contract.maps
    train_papers = set(contract.raw_split_papers["train"].tolist())
    paper_area = dict(
        contract.paper_area[["paper_id", "area_id"]].itertuples(index=False)
    )
    authors_by_paper = (
        contract.paper_author.groupby("paper_id")["author_id"].apply(list).to_dict()
    )
    confs_by_paper = (
        contract.paper_conf[
            contract.paper_conf["paper_id"].isin(train_papers)
        ].groupby("paper_id")["conf_id"].apply(list).to_dict()
    )
    physical_adj: dict[str, dict[int, set[int]]] = {}
    for relation, rows in physical.items():
        adjacency: dict[int, set[int]] = {}
        for left, right in rows:
            adjacency.setdefault(int(left), set()).add(int(right))
            adjacency.setdefault(int(right), set()).add(int(left))
        physical_adj[relation] = adjacency
    ambiguous = 0
    rejected = 0
    matched = 0
    for raw_paper in sorted(train_papers & set(paper_area)):
        paper = maps["paper"].get(int(raw_paper))
        area = maps["area"].get(int(paper_area[raw_paper]))
        if paper is None or area is None:
            continue
        authors = [
            maps["author"][int(author)]
            for author in authors_by_paper.get(raw_paper, ())
            if int(author) in maps["author"]
        ]
        conferences = [
            maps["conference"][int(conf)]
            for conf in confs_by_paper.get(raw_paper, ())
            if int(conf) in maps["conference"]
        ]
        if variant == "v1":
            candidates = physical_adj.get("paper-area", {}).get(paper, set())
        elif variant == "v2":
            candidates: set[int] = set()
            for conf in conferences:
                candidates.update(
                    physical_adj.get("conference-area", {}).get(conf, set())
                )
        elif variant == "v3":
            candidates = set()
            for author in authors:
                candidates.update(
                    physical_adj.get("author-area", {}).get(author, set())
                )
        else:
            raise ValueError(f"Invariant compilation requires v1-v3, got {variant}")
        if len(candidates) > 1:
            ambiguous += 1
        rejected += len(candidates - {area})
        if area not in candidates:
            raise RuntimeError(
                f"{variant}: physical context does not license paper={paper}, area={area}; "
                f"candidates={sorted(candidates)}"
            )
        matched += 1
        add_edge(semantic, "paper-area", paper, area)
        for conf in conferences:
            add_edge(semantic, "conference-area", conf, area)
        for author in authors:
            add_edge(semantic, "author-area", author, area)
    audit: dict[str, int | str] = {
        "variant": variant,
        "compiler": "raw_context_with_training_block_local_paper_area_filter",
        "context_matches": matched,
        "ambiguous_raw_contexts": ambiguous,
        "raw_candidates_rejected_by_pair_filter": rejected,
        "pair_filter_model_input": 0,
    }
    return semantic, audit


def graph_tensors(
    relation_edges: dict[str, set[tuple[int, int]]],
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], dict[tuple[int, int], int]]:
    directed: list[tuple[int, int]] = []
    directed_types: list[int] = []
    adjacency = [[] for _ in range(num_nodes)]
    edge_lookup: dict[tuple[int, int], int] = {}
    for relation in EDGE_TYPE_NAMES:
        for left, right in sorted(relation_edges.get(relation, ())):
            adjacency[left].append(right)
            adjacency[right].append(left)
            for source, target in ((left, right), (right, left)):
                edge_lookup[(source, target)] = len(directed)
                directed.append((source, target))
                directed_types.append(EDGE_TYPE[relation])
    for neighbors in adjacency:
        neighbors.sort()
    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(directed_types, dtype=torch.int8)
    return edge_index, edge_type, adjacency, edge_lookup


def path_counts_by_root(adjacency: list[list[int]]) -> np.ndarray:
    """Count exact path lengths per root without materializing the paths."""
    counts = np.zeros((len(adjacency), PATH_LENGTH + 1), dtype=np.int64)
    adjacency_sets = [set(neighbors) for neighbors in adjacency]
    for root, neighbors in enumerate(adjacency):
        counts[root, 0] = 1
        counts[root, 1] = len(neighbors)
        for first in neighbors:
            counts[root, 2] += len(adjacency[first]) - 1
            for second in adjacency[first]:
                if second == root:
                    continue
                # ``first`` is necessarily a neighbor of ``second``. Exclude
                # it, and exclude ``root`` only when the closing triangle edge
                # exists. This is exactly the simple-path test without walking
                # every third-hop neighbor during the census.
                counts[root, 3] += (
                    len(adjacency[second])
                    - 1
                    - int(root in adjacency_sets[second])
                )
    return counts


def count_paths(adjacency: list[list[int]]) -> tuple[int, int, int, int]:
    totals = path_counts_by_root(adjacency).sum(axis=0)
    return tuple(int(value) for value in totals)


def iter_root_paths(root: int, adjacency: list[list[int]]):
    """Yield one root's simple paths in canonical decreasing-length order."""
    neighbors = adjacency[root]
    for first in neighbors:
        for second in adjacency[first]:
            if second == root:
                continue
            for third in adjacency[second]:
                if third != root and third != first:
                    yield (root, first, second, third)
    for first in neighbors:
        for second in adjacency[first]:
            if second != root:
                yield (root, first, second)
    for first in neighbors:
        yield (root, first)
    yield (root,)


def stable_path_key(path: tuple[int, ...], seed: int) -> int:
    """Stable 64-bit FNV-1a key over semantic node identity."""
    value = (1469598103934665603 ^ int(seed)) & ((1 << 64) - 1)
    for item in (len(path), *path):
        value ^= int(item) & ((1 << 64) - 1)
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


UINT64_MASK = (1 << 64) - 1
UINT64_RANGE = 1 << 64


def _splitmix64(state: int) -> tuple[int, int]:
    """Return the next state and value from a stable counter-style PRNG."""
    state = (state + 0x9E3779B97F4A7C15) & UINT64_MASK
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return state, (value ^ (value >> 31)) & UINT64_MASK


def stable_sample_ranks(population: int, sample_size: int, seed: int) -> list[int]:
    """Sample sorted unique integer ranks in O(sample_size) memory and time.

    Floyd's algorithm avoids allocating or walking ``range(population)``.  A
    repository-local SplitMix64 implementation makes the result independent of
    NumPy and Python random-module versions.
    """
    if population < 0 or sample_size < 0 or sample_size > population:
        raise ValueError("Invalid population/sample size")
    if sample_size == 0:
        return []
    if sample_size == population:
        return list(range(population))

    state = int(seed) & UINT64_MASK
    chosen: set[int] = set()

    def randbelow(bound: int) -> int:
        nonlocal state
        limit = UINT64_RANGE - (UINT64_RANGE % bound)
        while True:
            state, value = _splitmix64(state)
            if value < limit:
                return value % bound

    for upper in range(population - sample_size, population):
        candidate = randbelow(upper + 1)
        chosen.add(upper if candidate in chosen else candidate)
    if len(chosen) != sample_size:
        raise AssertionError("Floyd sampler produced duplicate ranks")
    return sorted(chosen)


def allocate_long_path_budget(
    length_two: int,
    length_three: int,
    budget: int,
) -> tuple[int, int]:
    """Allocate a per-root budget proportionally while covering both lengths."""
    counts = [int(length_two), int(length_three)]
    if min(counts) < 0 or budget < 0:
        raise ValueError("Path counts and budget must be non-negative")
    total = sum(counts)
    if total <= budget:
        return counts[0], counts[1]
    if budget == 0:
        return 0, 0

    allocations = [0, 0]
    active = [index for index, count in enumerate(counts) if count]
    # When possible, guarantee that both length strata are represented.  For a
    # one-path budget, prefer the larger stratum (length three wins a tie).
    if budget < len(active):
        selected = sorted(active, key=lambda i: (counts[i], i), reverse=True)[:budget]
        for index in selected:
            allocations[index] = 1
        return allocations[0], allocations[1]

    for index in active:
        allocations[index] = 1
    remaining = budget - len(active)
    capacities = [counts[i] - allocations[i] for i in range(2)]
    capacity_total = sum(capacities)
    if remaining and capacity_total:
        numerators = [remaining * capacity for capacity in capacities]
        extras = [numerator // capacity_total for numerator in numerators]
        for index in range(2):
            allocations[index] += extras[index]
        leftover = remaining - sum(extras)
        order = sorted(
            range(2),
            key=lambda i: (numerators[i] % capacity_total, capacities[i], i),
            reverse=True,
        )
        for index in order:
            if not leftover:
                break
            if allocations[index] < counts[index]:
                allocations[index] += 1
                leftover -= 1
    if sum(allocations) != budget:
        raise AssertionError("Long-path budget allocation is inconsistent")
    return allocations[0], allocations[1]


def selected_path_counts(
    root_counts: np.ndarray,
    long_path_budget: int,
) -> np.ndarray:
    """Return materialized counts by path length without selecting paths."""
    if long_path_budget <= 0:
        return root_counts.sum(axis=0).astype(np.int64, copy=True)
    selected = np.zeros(PATH_LENGTH + 1, dtype=np.int64)
    selected[:2] = root_counts[:, :2].sum(axis=0)
    for row in root_counts:
        count_two, count_three = allocate_long_path_budget(
            int(row[2]), int(row[3]), long_path_budget
        )
        selected[2] += count_two
        selected[3] += count_three
    return selected


def _nth_without(values: list[int], offset: int, excluded: tuple[int, ...]) -> int:
    """Select an item from sorted values after removing known exclusions."""
    index = int(offset)
    for item in sorted(excluded):
        position = bisect_left(values, item)
        if position < len(values) and values[position] == item and position <= index:
            index += 1
    return values[index]


def unrank_length_two_paths(
    root: int,
    adjacency: list[list[int]],
    ranks: list[int],
):
    """Map canonical length-two ranks to paths without enumerating paths."""
    rank_index = 0
    block_start = 0
    for first in adjacency[root]:
        block_size = len(adjacency[first]) - 1
        block_stop = block_start + block_size
        while rank_index < len(ranks) and ranks[rank_index] < block_stop:
            offset = ranks[rank_index] - block_start
            second = _nth_without(adjacency[first], offset, (root,))
            yield (root, first, second)
            rank_index += 1
        block_start = block_stop
        if rank_index == len(ranks):
            return
    if rank_index != len(ranks):
        raise IndexError("Length-two path rank is outside the population")


def unrank_length_three_paths(
    root: int,
    adjacency: list[list[int]],
    adjacency_sets: list[set[int]],
    ranks: list[int],
):
    """Map canonical length-three ranks while walking only length-two prefixes."""
    rank_index = 0
    block_start = 0
    for first in adjacency[root]:
        for second in adjacency[first]:
            if second == root:
                continue
            exclusions = (first, root) if root in adjacency_sets[second] else (first,)
            block_size = len(adjacency[second]) - len(exclusions)
            block_stop = block_start + block_size
            while rank_index < len(ranks) and ranks[rank_index] < block_stop:
                offset = ranks[rank_index] - block_start
                third = _nth_without(adjacency[second], offset, exclusions)
                yield (root, first, second, third)
                rank_index += 1
            block_start = block_stop
            if rank_index == len(ranks):
                return
    if rank_index != len(ranks):
        raise IndexError("Length-three path rank is outside the population")


def selected_root_paths(
    root: int,
    adjacency: list[list[int]],
    adjacency_sets: list[set[int]],
    root_counts: np.ndarray,
    long_path_budget: int,
    seed: int,
):
    """Yield ``(path, weight)`` using direct stratified path sampling.

    Zero- and one-edge paths are exact.  Two- and three-edge paths receive a
    proportional per-root budget and inverse-inclusion-probability weights so
    their weighted sums estimate the exhaustive PAIN path sums.
    """
    if long_path_budget <= 0:
        yield from ((path, 1.0) for path in iter_root_paths(root, adjacency))
        return

    exact_two, exact_three = int(root_counts[2]), int(root_counts[3])
    sample_two, sample_three = allocate_long_path_budget(
        exact_two, exact_three, long_path_budget
    )
    for length, exact, sampled, unrank in (
        (3, exact_three, sample_three, unrank_length_three_paths),
        (2, exact_two, sample_two, unrank_length_two_paths),
    ):
        if not sampled:
            continue
        rank_seed = stable_path_key((root, length), seed)
        ranks = stable_sample_ranks(exact, sampled, rank_seed)
        paths = (
            unrank(root, adjacency, adjacency_sets, ranks)
            if length == 3
            else unrank(root, adjacency, ranks)
        )
        weight = float(exact) / float(sampled)
        yield from ((path, weight) for path in paths)
    for first in adjacency[root]:
        yield (root, first), 1.0
    yield (root,), 1.0


def estimated_compact_bytes(
    num_paths: int,
    width: int = 4,
    *,
    weighted: bool = False,
) -> int:
    # int32 path + edge indices, int8 neighbor + distance + length, int32 root,
    # and optional float32 inverse-probability weight.
    bytes_per_path = width * 4 * 2 + width + width + 1 + 4
    if weighted:
        bytes_per_path += 4
    return int(num_paths * bytes_per_path)


def enumerate_pain_paths(
    adjacency: list[list[int]],
    edge_lookup: dict[tuple[int, int], int],
    root_counts: np.ndarray,
    max_paths_per_root: int,
    sampling_seed: int,
) -> dict[str, torch.Tensor]:
    """Materialize exhaustive paths or a direct weighted per-root sample."""
    selected_counts = selected_path_counts(root_counts, max_paths_per_root)
    total = int(selected_counts.sum())
    width = PATH_LENGTH + 1
    path_index = np.full((width, total), -10, dtype=np.int32)
    path_edge_idx = np.full((width, total), -10, dtype=np.int32)
    neighbor_mask = np.full((total, width), -10, dtype=np.int8)
    distances = np.full((total, width), PATH_LENGTH, dtype=np.int8)
    path_lengths = np.empty(total, dtype=np.int8)
    roots = np.empty(total, dtype=np.int32)
    path_weights = (
        np.empty(total, dtype=np.float32) if max_paths_per_root > 0 else None
    )
    adjacency_sets = [set(neighbors) for neighbors in adjacency]
    cursor = 0

    def emit(path: tuple[int, ...], weight: float) -> None:
        nonlocal cursor
        size, root = len(path), path[0]
        path_index[:size, cursor] = path
        path_edge_idx[0, cursor] = -20
        for position in range(1, size):
            path_edge_idx[position, cursor] = edge_lookup[
                (path[position - 1], path[position])
            ]
        neighbor_mask[cursor, :size] = [
            int(node in adjacency_sets[root]) for node in path
        ]
        for position, node in enumerate(path):
            if node == root:
                distance = 0
            elif node in adjacency_sets[root]:
                distance = 1
            elif position >= 2 and adjacency_sets[root].intersection(
                adjacency_sets[node]
            ):
                distance = 2
            else:
                distance = position
            distances[cursor, position] = distance
        path_lengths[cursor] = size
        roots[cursor] = root
        if path_weights is not None:
            path_weights[cursor] = weight
        cursor += 1

    for root in range(len(adjacency)):
        yield_paths = selected_root_paths(
            root,
            adjacency,
            adjacency_sets,
            root_counts[root],
            max_paths_per_root,
            sampling_seed,
        )
        for path, weight in yield_paths:
            emit(path, weight)
    if cursor != total:
        raise AssertionError(f"filled {cursor} paths, expected {total}")
    result = {
        "path_index": torch.from_numpy(path_index),
        "path_lengths": torch.from_numpy(path_lengths),
        "mask_index": torch.from_numpy(roots),
        "path_edge_idx": torch.from_numpy(path_edge_idx),
        "neighbor_mask": torch.from_numpy(neighbor_mask),
        "distances": torch.from_numpy(distances),
    }
    if path_weights is not None:
        result["path_weights"] = torch.from_numpy(path_weights)
    return result


def tensor_hash(tensors: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def validate_no_target_leakage(
    relation_edges: dict[str, set[tuple[int, int]]],
    splits: dict[str, torch.Tensor],
) -> dict[str, int]:
    target_edges = relation_edges.get("paper-conference", set())
    audit = {}
    for name in ("val_pos", "test_pos"):
        leaked = sum(
            canonical_edge(int(paper), int(conf)) in target_edges
            for paper, conf in splits[name].tolist()
        )
        audit[f"{name}_edges_in_message_graph"] = int(leaked)
        if leaked:
            raise RuntimeError(f"{name}: {leaked} target edges leaked into the graph")
    return audit


def save_shared(
    contract: Contract, output_dir: Path, raw_dir: Path, seed: int
) -> None:
    split_order = (
        "train_pos", "val_pos", "test_pos", "train_neg_tails",
        "conference_candidates",
    )
    supervision_hash = tensor_hash(contract.splits[name] for name in split_order)
    payload = {
        "node_type": contract.node_type,
        "splits": contract.splits,
        "meta": {
            "dataset": "DBLP",
            "source": str(raw_dir),
            "task": "paper_conference_link_prediction",
            "num_nodes": int(contract.node_type.numel()),
            "node_type_names": list(NODE_TYPE_NAMES),
            "node_offsets": contract.offsets,
            "node_counts": contract.counts,
            "split_seed": seed,
            "split_protocol": "paper_disjoint_70_10_20_full_dataset",
            "evaluation_protocol": "filtered_full_conference_ranking",
            "num_candidates": contract.counts["conference"],
            "supervision_sha256": supervision_hash,
        },
    }
    atomic_torch_save(payload, output_dir / "shared.pt")


def preprocess(
    raw_dir: Path,
    output_dir: Path,
    variants_to_build: list[str],
    seed: int,
    min_conf: int,
    count_only: bool,
    mode: str,
    max_paths_per_root: int,
    sampling_seed: int,
) -> None:
    if max_paths_per_root < 0:
        raise ValueError("max_paths_per_root must be non-negative")
    if max_paths_per_root == 1:
        raise ValueError(
            "A sampled long-path budget must be at least 2 so nonempty "
            "length-2 and length-3 strata can both be represented"
        )
    contract = build_contract(raw_dir, seed, min_conf)
    # Ordinary and augmentation artifacts reproduce the existing cross-GNN
    # DBLP variants: all auxiliary P-Area/A-Author labels are visible, while
    # C-Area remains derived from training target blocks to avoid target leak.
    variants = build_variant_edges(contract, area_scope="baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not count_only:
        save_shared(contract, output_dir, raw_dir, seed)
    sampling_mode = (
        "none" if max_paths_per_root == 0
        else "direct_length_stratified_inverse_probability"
    )
    artifact_tag = f"L{PATH_LENGTH}"
    if max_paths_per_root:
        artifact_tag += f"_stratcap{max_paths_per_root}"
    summary: dict = {
        "path_length": PATH_LENGTH,
        "path_semantics": "all_rooted_simple_paths_up_to_L",
        "path_sampling": sampling_mode,
        "sampled_long_paths_per_root": max_paths_per_root,
        "sampling_seed": sampling_seed,
        "sampling_short_path_policy": "retain_all_length_0_and_1",
        "sampling_long_path_allocation": (
            "exhaustive"
            if max_paths_per_root == 0
            else "per_root_proportional_length_2_and_3"
        ),
        "sampling_weighting": (
            "none"
            if max_paths_per_root == 0
            else "inverse_inclusion_probability_by_root_and_length"
        ),
        "artifact_tag": artifact_tag,
        "num_candidates": contract.counts["conference"],
        "node_counts": contract.counts,
        "compact_path_storage": True,
        "variants": {},
    }

    def process_program(
        name: str,
        relation_edges: dict[str, set[tuple[int, int]]],
        *,
        mapping_mode: str,
        extra_meta: dict | None = None,
    ) -> None:
        edge_index, edge_type, adjacency, edge_lookup = graph_tensors(
            relation_edges, int(contract.node_type.numel())
        )
        root_counts = path_counts_by_root(adjacency)
        counts = tuple(int(value) for value in root_counts.sum(axis=0))
        exact_total = sum(counts)
        materialized_counts = selected_path_counts(
            root_counts, max_paths_per_root
        )
        materialized_total = int(materialized_counts.sum())
        leakage = validate_no_target_leakage(relation_edges, contract.splits)
        details = {
            "dataset": "DBLP",
            "variant": name,
            "mapping_mode": mapping_mode,
            "path_length": PATH_LENGTH,
            "path_semantics": "all_rooted_simple_paths_up_to_L",
            "path_sampling": sampling_mode,
            "sampled_long_paths_per_root": max_paths_per_root,
            "sampling_seed": sampling_seed,
            "sampling_short_path_policy": "retain_all_length_0_and_1",
            "sampling_long_path_allocation": (
                "per_root_proportional_length_2_and_3_with_nonempty_coverage"
            ),
            "sampling_selection": (
                "canonical_rank_without_replacement_splitmix64_floyd"
            ),
            "sampling_weighting": (
                "inverse_inclusion_probability_by_root_and_length"
                if max_paths_per_root
                else "none"
            ),
            "path_order": "root_major_descending_length",
            "num_nodes": int(contract.node_type.numel()),
            "num_undirected_edges": int(edge_index.shape[1] // 2),
            "num_directed_edges": int(edge_index.shape[1]),
            "paths_exact_length_0_1_2_3": list(counts),
            "paths_materialized_length_0_1_2_3": materialized_counts.tolist(),
            "num_exact_paths": exact_total,
            "num_paths": materialized_total,
            "estimated_compact_path_bytes": estimated_compact_bytes(
                materialized_total, weighted=max_paths_per_root > 0
            ),
            "edge_type_names": list(EDGE_TYPE_NAMES),
            "message_program_sha256": tensor_hash((edge_index, edge_type)),
            "target_relation_policy": "paper_conference_train_only",
            "v2_area_policy": "training_paper_conference_blocks_only",
            "leakage_audit": leakage,
        }
        if extra_meta:
            details.update(extra_meta)
        summary["variants"][name] = details
        print(
            f"{name:12s} nodes={details['num_nodes']:,} "
            f"edges={details['num_undirected_edges']:,} "
            f"paths={materialized_total:,}/{exact_total:,} "
            f"compact_estimate={details['estimated_compact_path_bytes'] / 2**30:.2f} GiB "
            f"selected_counts={tuple(int(v) for v in materialized_counts)} "
            f"exact_counts={counts}",
            flush=True,
        )
        if count_only:
            return
        paths = enumerate_pain_paths(
            adjacency,
            edge_lookup,
            root_counts,
            max_paths_per_root,
            sampling_seed,
        )
        hash_fields = (
            "path_index", "path_lengths", "mask_index", "path_edge_idx",
            "neighbor_mask", "distances",
        )
        details["selected_path_program_sha256"] = tensor_hash(
            paths[field] for field in hash_fields
        )
        if "path_weights" in paths:
            details["selected_path_weights_sha256"] = tensor_hash(
                (paths["path_weights"],)
            )
        payload = {
            "edge_index": edge_index,
            "edge_type": edge_type,
            **paths,
            "meta": details,
        }
        atomic_torch_save(payload, output_dir / f"{name}_{artifact_tag}.pt")
        del paths, payload, edge_index, edge_type, adjacency, edge_lookup

    if mode in {"baseline", "both"}:
        for variant in variants_to_build:
            physical_tensors = graph_tensors(
                variants[variant], int(contract.node_type.numel())
            )
            process_program(
                variant,
                variants[variant],
                mapping_mode=(
                    "universal_union_graph" if variant == "universal" else "baseline"
                ),
                extra_meta={
                    "physical_graph_sha256": tensor_hash(physical_tensors[:2]),
                    "area_context_scope": "cross_gnn_baseline",
                },
            )

    if mode in {"invariant", "both"}:
        # Mechanism A in INV-RGCN-guide uses training-block context for all
        # three physical realizations so each can compile the same semantic
        # program without held-out paper-conference topology.
        invariant_sources = build_variant_edges(contract, area_scope="train")
        compiled: dict[str, dict[str, set[tuple[int, int]]]] = {}
        compiler_audits = {}
        physical_hashes = {}
        semantic_hashes = {}
        for variant in VARIANTS[:-1]:
            physical_tensors = graph_tensors(
                invariant_sources[variant], int(contract.node_type.numel())
            )
            physical_hashes[variant] = tensor_hash(physical_tensors[:2])
            semantic, audit = compile_invariant_edges(
                contract, invariant_sources[variant], variant
            )
            semantic_tensors = graph_tensors(
                semantic, int(contract.node_type.numel())
            )
            semantic_hashes[variant] = tensor_hash(semantic_tensors[:2])
            compiled[variant] = semantic
            compiler_audits[variant] = audit
        if len(set(physical_hashes.values())) != len(VARIANTS[:-1]):
            raise RuntimeError("Invariant audit failed: physical DBLP graphs are not distinct")
        if len(set(semantic_hashes.values())) != 1:
            raise RuntimeError(
                f"Invariant audit failed: semantic programs differ: {semantic_hashes}"
            )
        first_semantic = compiled[VARIANTS[0]]
        if first_semantic != invariant_sources["universal"]:
            raise RuntimeError(
                "Compiled semantic closure does not match the train-scope universal graph"
            )
        summary["invariance_audit"] = {
            "physical_graph_hashes": physical_hashes,
            "semantic_program_hashes": semantic_hashes,
            "physical_graphs_different": True,
            "semantic_programs_equal": True,
            "compiler_audits": compiler_audits,
        }
        process_program(
            "invariant",
            first_semantic,
            mapping_mode="compiled_invariant_semantic_paths",
            extra_meta={
                "physical_graph_hashes": physical_hashes,
                "semantic_program_hashes": semantic_hashes,
                "compiler_audits": compiler_audits,
                "semantic_programs_equal": True,
                "physical_graphs_different": True,
                "area_context_scope": "training_transformation_blocks",
            },
        )
    atomic_write_json(summary, output_dir / "metadata.json")
    if count_only:
        print("Count-only mode: no tensor artifacts were written.")
    else:
        print(f"Saved DBLP shared contract and path artifacts to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/DBLP"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/preprocessed/DBLP")
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-conf", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("baseline", "invariant", "both"), default="both",
        help="Build original/universal arms, the compiled invariant arm, or both.",
    )
    parser.add_argument(
        "--max-paths-per-root", "--long-path-budget-per-root",
        dest="max_paths_per_root", type=int, default=256,
        help=(
            "Per-root budget for directly sampled length-2/3 paths (default: "
            "256; minimum positive value: 2). 0 requests exhaustive paths. "
            "Length-0/1 paths are retained exactly and sampled paths receive inverse-"
            "probability weights."
        ),
    )
    parser.add_argument("--sampling-seed", type=int, default=SEED)
    parser.add_argument(
        "--count-only", action="store_true",
        help="Report exact path counts and compact-storage estimates without materializing paths.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        args.raw_dir,
        args.output_dir,
        list(args.variants),
        args.seed,
        args.min_conf,
        args.count_only,
        args.mode,
        args.max_paths_per_root,
        args.sampling_seed,
    )
