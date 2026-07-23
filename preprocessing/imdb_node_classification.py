"""Preprocess IMDB for faithful PAIN node-classification experiments.

This adapts the IMDB data contract in ``dhn_nclp`` while replacing DHN's
homomorphism mappings with the tensors consumed by the original PAIN model.
Every graph variant uses the same nodes, features, labels, and splits. Only the
topology (and therefore the rooted simple paths) changes.

Paths are exact, rooted at every node, and include every simple path with zero
through three edges. No path sampling is performed.

Run from the PAIN-nc repository root:

    python -m preprocessing.imdb_node_classification
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


SEED = 1566911444
PATH_LENGTH = 3
LABEL_NAMES = ("Action", "Comedy", "Drama")
ACTOR_COLUMNS = ("actor_1_name", "actor_2_name", "actor_3_name")
NODE_TYPE_NAMES = ("movie", "director", "actor", "imdb_link")
EDGE_TYPE_NAMES = (
    "movie-director",
    "movie-actor",
    "movie-imdb_link",
    "imdb_link-director",
    "imdb_link-actor",
)
VARIANTS = ("v1", "v2", "v3", "v4", "universal")


@dataclass(frozen=True)
class Contract:
    movies: pd.DataFrame
    labels: np.ndarray
    node_types: np.ndarray
    maps: dict[str, dict]
    offsets: dict[str, int]
    counts: dict[str, int]
    features: torch.Tensor
    vocabulary: list[str]
    split_indices: dict[str, np.ndarray]


def load_movies(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply the exact filtering and label priority used by dhn_nclp."""
    movies = pd.read_csv(csv_path, encoding="utf-8")
    required = {
        "movie_imdb_link",
        "director_name",
        "genres",
        "plot_keywords",
        *ACTOR_COLUMNS,
    }
    missing = sorted(required - set(movies.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    movies = (
        movies.drop_duplicates(subset=["movie_imdb_link"])
        .dropna(subset=["actor_1_name", "director_name"])
        .reset_index()
        .rename(columns={"index": "source_row"})
    )

    keep_rows, labels = [], []
    for row_index, genres in movies["genres"].items():
        genre_set = set(str(genres).split("|"))
        label = next(
            (
                label_id
                for label_id, genre in enumerate(LABEL_NAMES)
                if genre in genre_set
            ),
            None,
        )
        if label is not None:
            keep_rows.append(row_index)
            labels.append(label)

    filtered = movies.iloc[keep_rows].reset_index(drop=True)
    return filtered, np.asarray(labels, dtype=np.int64)


def make_node_maps(
    movies: pd.DataFrame,
) -> tuple[dict[str, dict], dict[str, int], dict[str, int], np.ndarray]:
    """Assign deterministic contiguous global IDs, with movies first."""
    values = {
        "movie": list(range(len(movies))),
        "director": sorted(
            movies["director_name"].dropna().astype(str).unique().tolist()
        ),
        "actor": sorted(
            {
                str(value)
                for column in ACTOR_COLUMNS
                for value in movies[column].dropna()
                if str(value)
            }
        ),
        "imdb_link": sorted(
            movies["movie_imdb_link"].dropna().astype(str).unique().tolist()
        ),
    }

    maps: dict[str, dict] = {}
    offsets: dict[str, int] = {}
    counts: dict[str, int] = {}
    node_type_parts = []
    offset = 0
    for type_id, node_type in enumerate(NODE_TYPE_NAMES):
        offsets[node_type] = offset
        counts[node_type] = len(values[node_type])
        maps[node_type] = {
            value: offset + local_id
            for local_id, value in enumerate(values[node_type])
        }
        node_type_parts.append(
            np.full(len(values[node_type]), type_id, dtype=np.int64)
        )
        offset += len(values[node_type])

    return maps, offsets, counts, np.concatenate(node_type_parts)


def canonical_feature_edges(
    movies: pd.DataFrame, maps: dict[str, dict]
) -> set[tuple[int, int]]:
    """Direct movie-entity edges used only for topology-independent features."""
    edges: set[tuple[int, int]] = set()
    for movie_local, row in movies.iterrows():
        movie = maps["movie"][movie_local]
        director = maps["director"][str(row["director_name"])]
        link = maps["imdb_link"][str(row["movie_imdb_link"])]
        edges.add(canonical_edge(movie, director))
        edges.add(canonical_edge(movie, link))
        for column in ACTOR_COLUMNS:
            if pd.notna(row[column]) and str(row[column]):
                actor = maps["actor"][str(row[column])]
                edges.add(canonical_edge(movie, actor))
    return edges


def build_features(
    movies: pd.DataFrame,
    maps: dict[str, dict],
    node_types: np.ndarray,
) -> tuple[torch.Tensor, list[str]]:
    """Reproduce the corrected topology-independent DHN feature contract."""
    vectorizer = CountVectorizer(min_df=2)
    movie_features = vectorizer.fit_transform(
        movies["plot_keywords"]
        .fillna("")
        .astype(str)
        .str.replace("|", " ", regex=False)
    ).astype(np.float32)
    vocabulary = vectorizer.get_feature_names_out().tolist()

    num_nodes = len(node_types)
    num_movies = len(movies)
    row_ids, movie_ids = [], []
    for source, target in canonical_feature_edges(movies, maps):
        if node_types[source] == 0:
            movie, entity = source, target
        elif node_types[target] == 0:
            movie, entity = target, source
        else:
            continue
        row_ids.append(entity)
        movie_ids.append(movie)

    incidence = scipy.sparse.coo_matrix(
        (
            np.ones(len(row_ids), dtype=np.float32),
            (row_ids, movie_ids),
        ),
        shape=(num_nodes, num_movies),
    ).tocsr()
    degrees = np.asarray(incidence.sum(axis=1)).ravel()
    safe_degrees = degrees.copy()
    safe_degrees[safe_degrees == 0] = 1.0
    entity_features = (
        scipy.sparse.diags(1.0 / safe_degrees)
        @ incidence
        @ movie_features
    ).tolil()
    entity_features[:num_movies] = movie_features
    dense = entity_features.tocsr().toarray().astype(np.float32, copy=False)
    return torch.from_numpy(dense), vocabulary


def make_splits(num_movies: int) -> dict[str, np.ndarray]:
    """Reproduce the existing deterministic 70/10/20 IMDB split exactly."""
    all_movies = np.arange(num_movies, dtype=np.int64)
    train, val = train_test_split(
        all_movies,
        test_size=int(0.1 * num_movies),
        random_state=SEED,
    )
    train, test = train_test_split(
        train,
        test_size=int(0.2 * num_movies),
        random_state=SEED,
    )
    return {
        "train": np.sort(train).astype(np.int64),
        "val": np.sort(val).astype(np.int64),
        "test": np.sort(test).astype(np.int64),
    }


def build_contract(csv_path: Path) -> Contract:
    movies, labels = load_movies(csv_path)
    maps, offsets, counts, node_types = make_node_maps(movies)
    features, vocabulary = build_features(movies, maps, node_types)
    return Contract(
        movies=movies,
        labels=labels,
        node_types=node_types,
        maps=maps,
        offsets=offsets,
        counts=counts,
        features=features,
        vocabulary=vocabulary,
        split_indices=make_splits(len(movies)),
    )


def canonical_edge(source: int, target: int) -> tuple[int, int]:
    return (source, target) if source < target else (target, source)


def base_variant_edges(
    movies: pd.DataFrame, maps: dict[str, dict]
) -> dict[str, set[tuple[int, int]]]:
    """Reproduce the four dhn_nclp IMDB topology variants."""
    variants = {name: set() for name in VARIANTS[:-1]}
    for movie_local, row in movies.iterrows():
        movie = maps["movie"][movie_local]
        director = maps["director"][str(row["director_name"])]
        link = maps["imdb_link"][str(row["movie_imdb_link"])]
        actors = [
            maps["actor"][str(row[column])]
            for column in ACTOR_COLUMNS
            if pd.notna(row[column]) and str(row[column])
        ]

        # v1: direct movie-to-entity star.
        variants["v1"].add(canonical_edge(movie, director))
        variants["v1"].add(canonical_edge(movie, link))
        variants["v1"].update(canonical_edge(movie, actor) for actor in actors)

        # v2: link is the hub for movie, director, and every actor.
        variants["v2"].add(canonical_edge(movie, link))
        variants["v2"].add(canonical_edge(link, director))
        variants["v2"].update(canonical_edge(link, actor) for actor in actors)

        # v3: director remains direct; actors are rerouted through link.
        variants["v3"].add(canonical_edge(movie, director))
        variants["v3"].add(canonical_edge(movie, link))
        variants["v3"].update(canonical_edge(link, actor) for actor in actors)

        # v4 intentionally mirrors dhn_nclp: Actor1 is omitted, Actor2/3 stay
        # direct, and the director is rerouted through link.
        variants["v4"].add(canonical_edge(movie, link))
        variants["v4"].add(canonical_edge(link, director))
        variants["v4"].update(
            canonical_edge(movie, actor) for actor in actors[1:]
        )

    variants["universal"] = set().union(*variants.values())
    return variants


def edge_type_id(
    source: int, target: int, node_types: np.ndarray
) -> int:
    pair = tuple(sorted((int(node_types[source]), int(node_types[target]))))
    mapping = {
        (0, 1): 0,
        (0, 2): 1,
        (0, 3): 2,
        (1, 3): 3,
        (2, 3): 4,
    }
    if pair not in mapping:
        raise ValueError(f"Unexpected IMDB endpoint-type pair: {pair}")
    return mapping[pair]


def graph_tensors(
    undirected_edges: set[tuple[int, int]],
    node_types: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], dict[tuple[int, int], int]]:
    """Create deterministic directed edges and adjacency lists."""
    num_nodes = len(node_types)
    adjacency = [[] for _ in range(num_nodes)]
    directed = []
    directed_types = []
    edge_lookup: dict[tuple[int, int], int] = {}

    for source, target in sorted(undirected_edges):
        adjacency[source].append(target)
        adjacency[target].append(source)
        relation = edge_type_id(source, target, node_types)
        for u, v in ((source, target), (target, source)):
            edge_lookup[(u, v)] = len(directed)
            directed.append((u, v))
            directed_types.append(relation)

    for neighbors in adjacency:
        neighbors.sort()
    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(directed_types, dtype=torch.long)
    return edge_index, edge_type, adjacency, edge_lookup


def count_paths(adjacency: list[list[int]]) -> tuple[int, int, int, int]:
    """Count rooted simple paths with exactly 0, 1, 2, and 3 edges."""
    length_0 = len(adjacency)
    length_1 = sum(len(neighbors) for neighbors in adjacency)
    length_2 = sum(
        len(adjacency[middle]) - 1
        for root, neighbors in enumerate(adjacency)
        for middle in neighbors
    )
    length_3 = 0
    for root, neighbors in enumerate(adjacency):
        for first in neighbors:
            for second in adjacency[first]:
                if second == root:
                    continue
                length_3 += sum(
                    third != root and third != first
                    for third in adjacency[second]
                )
    return length_0, length_1, length_2, length_3


def shortest_distance_on_path(
    root: int,
    node: int,
    position: int,
    adjacency_sets: list[set[int]],
) -> int:
    """Return the true graph distance for a node on a path of length <= 3."""
    if node == root:
        return 0
    if node in adjacency_sets[root]:
        return 1
    if position >= 2 and adjacency_sets[root].intersection(
        adjacency_sets[node]
    ):
        return 2
    return position


def enumerate_pain_paths(
    adjacency: list[list[int]],
    edge_lookup: dict[tuple[int, int], int],
) -> tuple[dict[str, torch.Tensor], tuple[int, int, int, int]]:
    """Materialize the original PAIN path fields in descending-length order."""
    counts = count_paths(adjacency)
    total = sum(counts)
    width = PATH_LENGTH + 1
    adjacency_sets = [set(neighbors) for neighbors in adjacency]

    path_index = np.full((width, total), -10, dtype=np.int64)
    path_edge_index = np.full((width, total), -10, dtype=np.int64)
    neighbor_mask = np.full((total, width), -10, dtype=np.int64)
    distances = np.full((total, width), PATH_LENGTH, dtype=np.int64)
    path_lengths = np.empty(total, dtype=np.int64)
    root_index = np.empty(total, dtype=np.int64)
    cursor = 0

    def emit(path: tuple[int, ...]) -> None:
        nonlocal cursor
        size = len(path)
        root = path[0]
        path_index[:size, cursor] = path
        path_edge_index[0, cursor] = -20
        for position in range(1, size):
            path_edge_index[position, cursor] = edge_lookup[
                (path[position - 1], path[position])
            ]
        neighbor_mask[cursor, :size] = [
            int(node in adjacency_sets[root]) for node in path
        ]
        distances[cursor, :size] = [
            shortest_distance_on_path(
                root, node, position, adjacency_sets
            )
            for position, node in enumerate(path)
        ]
        path_lengths[cursor] = size
        root_index[cursor] = root
        cursor += 1

    # The reference implementation sorts by decreasing sequence length before
    # pack_padded_sequence. Generate in that order without a memory-heavy sort.
    for root, neighbors in enumerate(adjacency):
        for first in neighbors:
            for second in adjacency[first]:
                if second == root:
                    continue
                for third in adjacency[second]:
                    if third != root and third != first:
                        emit((root, first, second, third))
    for root, neighbors in enumerate(adjacency):
        for first in neighbors:
            for second in adjacency[first]:
                if second != root:
                    emit((root, first, second))
    for root, neighbors in enumerate(adjacency):
        for first in neighbors:
            emit((root, first))
    for root in range(len(adjacency)):
        emit((root,))

    if cursor != total:
        raise AssertionError(f"filled {cursor} paths, expected {total}")
    tensors = {
        "path_index": torch.from_numpy(path_index),
        "path_lengths": torch.from_numpy(path_lengths),
        "mask_index": torch.from_numpy(root_index),
        "path_edge_idx": torch.from_numpy(path_edge_index),
        "neighbor_mask": torch.from_numpy(neighbor_mask),
        "distances": torch.from_numpy(distances),
    }
    return tensors, counts


def validate_contract(contract: Contract) -> None:
    num_movies = contract.counts["movie"]
    num_nodes = len(contract.node_types)
    if contract.features.shape[0] != num_nodes:
        raise ValueError("Feature rows do not match the node count")
    if len(contract.labels) != num_movies:
        raise ValueError("Every movie must have exactly one target label")
    if sorted(np.unique(contract.labels).tolist()) != [0, 1, 2]:
        raise ValueError("Expected all three contiguous IMDB classes")

    split_sets = {
        name: set(values.tolist())
        for name, values in contract.split_indices.items()
    }
    if any(
        split_sets[left].intersection(split_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Train, validation, and test splits overlap")
    if set().union(*split_sets.values()) != set(range(num_movies)):
        raise ValueError("Splits do not cover every movie exactly once")


def validate_paths(
    path_tensors: dict[str, torch.Tensor],
    adjacency: list[list[int]],
) -> None:
    """Validate every materialized path; intended for preprocessing, not training."""
    path_index = path_tensors["path_index"].numpy()
    lengths = path_tensors["path_lengths"].numpy()
    roots = path_tensors["mask_index"].numpy()
    adjacency_sets = [set(neighbors) for neighbors in adjacency]
    for column, size in enumerate(lengths):
        path = path_index[:size, column].tolist()
        if path[0] != roots[column]:
            raise ValueError(f"Path {column} has the wrong root")
        if len(path) != len(set(path)):
            raise ValueError(f"Path {column} repeats a node")
        if any(
            target not in adjacency_sets[source]
            for source, target in zip(path, path[1:])
        ):
            raise ValueError(f"Path {column} contains a non-edge")


def save_shared(contract: Contract, output_dir: Path, csv_path: Path) -> None:
    num_nodes = len(contract.node_types)
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    y[: contract.counts["movie"]] = torch.from_numpy(contract.labels)
    masks = {}
    for name, indices in contract.split_indices.items():
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[torch.from_numpy(indices)] = True
        masks[f"{name}_mask"] = mask

    payload = {
        "x": contract.features,
        "y": y,
        "node_type": torch.from_numpy(contract.node_types),
        **masks,
        "split_indices": {
            name: torch.from_numpy(values)
            for name, values in contract.split_indices.items()
        },
        "movie_source_rows": torch.from_numpy(
            contract.movies["source_row"].to_numpy(
                dtype=np.int64, copy=True
            )
        ),
        "meta": {
            "dataset": "IMDB",
            "source": str(csv_path),
            "task": "movie_genre_node_classification",
            "label_names": list(LABEL_NAMES),
            "label_priority": list(LABEL_NAMES),
            "class_counts": np.bincount(
                contract.labels, minlength=len(LABEL_NAMES)
            ).tolist(),
            "node_type_names": list(NODE_TYPE_NAMES),
            "node_offsets": contract.offsets,
            "node_counts": contract.counts,
            "num_nodes": num_nodes,
            "num_features": int(contract.features.shape[1]),
            "split_seed": SEED,
            "split_protocol": "dhn_nclp_exact_70_10_20",
            "feature_protocol": (
                "plot_keyword_bow_min_df_2_with_canonical_mean_propagation"
            ),
        },
    }
    torch.save(payload, output_dir / "shared.pt")
    (output_dir / "vocabulary.json").write_text(
        json.dumps(contract.vocabulary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def preprocess(
    csv_path: Path,
    output_dir: Path,
    variants_to_build: list[str],
    validate: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = build_contract(csv_path)
    validate_contract(contract)
    save_shared(contract, output_dir, csv_path)
    all_edges = base_variant_edges(contract.movies, contract.maps)

    summary = {
        "path_length": PATH_LENGTH,
        "path_semantics": "all rooted simple paths with 0..3 edges",
        "path_sampling": "none",
        "path_order": "root_first",
        "variants": {},
    }
    for variant in variants_to_build:
        print(f"\n=== IMDB PAIN-NC | {variant} | L={PATH_LENGTH} ===")
        edge_index, edge_type, adjacency, edge_lookup = graph_tensors(
            all_edges[variant], contract.node_types
        )
        paths, counts = enumerate_pain_paths(adjacency, edge_lookup)
        if validate:
            validate_paths(paths, adjacency)

        payload = {
            "edge_index": edge_index,
            "edge_type": edge_type,
            **paths,
            "meta": {
                "dataset": "IMDB",
                "variant": variant,
                "mapping_mode": (
                    "universal_union_graph"
                    if variant == "universal"
                    else "baseline"
                ),
                "path_length": PATH_LENGTH,
                "path_semantics": "all_rooted_simple_paths_up_to_L",
                "path_sampling": "none",
                "path_order": "root_first",
                "num_nodes": len(contract.node_types),
                "num_undirected_edges": len(all_edges[variant]),
                "num_directed_edges": int(edge_index.shape[1]),
                "paths_exact_length_0_1_2_3": list(counts),
                "num_paths": sum(counts),
                "edge_type_names": list(EDGE_TYPE_NAMES),
            },
        }
        output_path = output_dir / f"{variant}_L{PATH_LENGTH}.pt"
        torch.save(payload, output_path)
        summary["variants"][variant] = payload["meta"]
        print(
            f"nodes={len(contract.node_types):,} "
            f"edges={len(all_edges[variant]):,} "
            f"paths={sum(counts):,} "
            f"counts={counts}"
        )
        print(f"saved {output_path}")
        del paths, payload

    (output_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nsaved shared contract and metadata to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/IMDB/movie_metadata.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/preprocessed/IMDB"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate every generated path (default: true).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.csv, args.output_dir, args.variants, args.validate)
