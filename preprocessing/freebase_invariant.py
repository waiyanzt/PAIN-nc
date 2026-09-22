"""Compile Freebase1-3 into invariant sampled PAIN path programs.

The compiler implements the conditional-graph construction from GNNInvar for
PAIN's rooted-path structural bias.  Each physical variant is read
independently.  Base-relation context occurrences license a canonical set of
36 base, six exact-2, and twenty-two exact-3 semantic propagation relations.
Interior composition witnesses are context-only skip nodes: PAIN receives only
the semantic propagation endpoints.  Sampling is performed after semantic
deduplication and is keyed by semantic relation identity, never physical edge
ids or file order.

The three physical inputs must compile to identical sampled semantic adjacency
and rooted path programs.  A mismatch aborts instead of producing trainable
artifacts.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from preprocessing.freebase_node_classification import (
    array_sha256,
    artifact_tag,
    atomic_torch_save,
    file_sha256,
    make_shared,
    materialize_paths,
    offer_sample,
    parse_info,
    stable_rank,
    write_json,
)


SOURCE_VARIANTS = ("unchanged", "exact_2", "exact_3")
DISPLAY_NAMES = {
    "unchanged": "Freebase_invariant1",
    "exact_2": "Freebase_invariant2",
    "exact_3": "Freebase_invariant3",
}
FORMAT = "freebase_pain_conditional_paths_v1"


def load_registry(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [record for record in records if not str(record.get("skipped_reason", ""))]


def semantic_relation_contract(
    variants_root: Path,
    reference_variant: str,
) -> tuple[list[dict[str, Any]], list[str], str]:
    node_type_names, base_schema = parse_info(
        variants_root / reference_variant / "info.dat"
    )
    type_name_to_id = {name: type_id for type_id, name in node_type_names.items()}
    if len(type_name_to_id) != len(node_type_names):
        raise ValueError("Freebase info.dat contains duplicate node-type names")

    base_by_name = {str(spec["name"]): spec for spec in base_schema.values()}
    if len(base_by_name) != 36:
        raise ValueError(
            f"Expected 36 unchanged Freebase relations, got {len(base_by_name)}"
        )
    relations: list[dict[str, Any]] = [
        {
            "name": name,
            "kind": "base",
            "composition": [name],
            "hop_count": 1,
            "start_type": int(spec["start_type"]),
            "end_type": int(spec["end_type"]),
            "registry_variant": None,
        }
        for name, spec in base_by_name.items()
    ]

    for registry_variant, expected in (("exact_2", 6), ("exact_3", 22)):
        registry_path = (
            variants_root / registry_variant / "new_transitive_relations.jsonl"
        )
        if not registry_path.is_file():
            raise FileNotFoundError(
                f"Missing authoritative Freebase registry: {registry_path}"
            )
        records = load_registry(registry_path)
        if len(records) != expected:
            raise ValueError(
                f"{registry_variant} registry contains {len(records)} selected "
                f"relations; expected {expected}"
            )
        for record in records:
            composition = [
                part.strip()
                for part in str(record["composition"]).split(";")
                if part.strip()
            ]
            if len(composition) != int(record["hop_count"]):
                raise ValueError(
                    f"Bad composition length for {record['new_relation_name']}"
                )
            missing = sorted(set(composition) - set(base_by_name))
            if missing:
                raise KeyError(
                    f"{record['new_relation_name']} depends on unknown base "
                    f"relations: {missing}"
                )
            try:
                start_type = type_name_to_id[str(record["start_type"])]
                end_type = type_name_to_id[str(record["end_type"])]
            except KeyError as error:
                raise KeyError(
                    f"Unknown node type in {registry_path}: {error.args[0]}"
                ) from error
            component_specs = [base_by_name[name] for name in composition]
            if int(component_specs[0]["start_type"]) != int(start_type):
                raise ValueError(
                    f"Registry start type disagrees with composition for "
                    f"{record['new_relation_name']}"
                )
            if int(component_specs[-1]["end_type"]) != int(end_type):
                raise ValueError(
                    f"Registry end type disagrees with composition for "
                    f"{record['new_relation_name']}"
                )
            for left, right in zip(component_specs, component_specs[1:]):
                if int(left["end_type"]) != int(right["start_type"]):
                    raise ValueError(
                        f"Non-composable registry path for "
                        f"{record['new_relation_name']}"
                    )
            relations.append(
                {
                    "name": str(record["new_relation_name"]),
                    "kind": "mapped",
                    "composition": composition,
                    "hop_count": int(record["hop_count"]),
                    "start_type": int(start_type),
                    "end_type": int(end_type),
                    "registry_variant": registry_variant,
                }
            )

    relations.sort(key=lambda item: str(item["name"]))
    names = [str(item["name"]) for item in relations]
    if len(relations) != 64 or len(set(names)) != 64:
        raise ValueError(
            "The Freebase invariant union must contain 64 unique semantic "
            f"relations; got {len(relations)} records and {len(set(names))} names"
        )
    edge_type_names = [
        f"{name}::{direction}"
        for name in names
        for direction in ("forward", "reverse")
    ]
    canonical = json.dumps(relations, sort_keys=True, separators=(",", ":")).encode()
    return relations, edge_type_names, hashlib.sha256(canonical).hexdigest()


def hash_base_program(
    base: Mapping[str, Mapping[int, Sequence[int]]],
) -> str:
    digest = hashlib.sha256()
    for name in sorted(base):
        encoded = name.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
        for source in sorted(base[name]):
            for target in base[name][source]:
                digest.update(struct.pack("<ii", int(source), int(target)))
    return digest.hexdigest()


def read_physical_base_context(
    source_dir: Path,
    *,
    reference_schema: Mapping[str, Mapping[str, Any]],
    num_nodes: int,
    progress_every: int,
) -> tuple[dict[str, dict[int, tuple[int, ...]]], dict[str, Any]]:
    _node_types, input_schema = parse_info(source_dir / "info.dat")
    input_by_name = {str(spec["name"]): spec for spec in input_schema.values()}
    for name, reference in reference_schema.items():
        observed = input_by_name.get(name)
        if observed is None:
            raise KeyError(f"{source_dir.name} is missing base relation {name}")
        observed_types = (int(observed["start_type"]), int(observed["end_type"]))
        reference_types = (
            int(reference["start_type"]),
            int(reference["end_type"]),
        )
        if observed_types != reference_types:
            raise ValueError(
                f"Base relation schema mismatch for {name} in {source_dir.name}"
            )

    relation_id_to_name = {
        relation_id: str(spec["name"])
        for relation_id, spec in input_schema.items()
    }
    needed_names = set(reference_schema)
    base_sets: dict[str, dict[int, set[int]]] = {
        name: defaultdict(set) for name in needed_names
    }
    digest = hashlib.sha256()
    raw_edges = 0
    started = time.perf_counter()
    with (source_dir / "link.dat").open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            digest.update(raw)
            fields = raw.rstrip(b"\r\n").split(b"\t")
            if len(fields) < 3:
                raise ValueError(f"Bad link.dat line {line_number} in {source_dir}")
            source, target, relation_id = (
                int(fields[0]),
                int(fields[1]),
                int(fields[2]),
            )
            if not (0 <= source < num_nodes and 0 <= target < num_nodes):
                raise ValueError(
                    f"Out-of-range node id at {source_dir}/link.dat:{line_number}"
                )
            try:
                relation_name = relation_id_to_name[relation_id]
            except KeyError as error:
                raise KeyError(
                    f"Relation id {relation_id} is absent from {source_dir}/info.dat"
                ) from error
            raw_edges += 1
            if relation_name in needed_names:
                base_sets[relation_name][source].add(target)
            if progress_every > 0 and raw_edges % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  streamed {raw_edges:,} physical edges "
                    f"({raw_edges / max(elapsed, 1e-9):,.0f} edges/s)",
                    flush=True,
                )

    base = {
        name: {
            int(source): tuple(sorted(targets))
            for source, targets in source_table.items()
        }
        for name, source_table in base_sets.items()
    }
    unique_base_edges = sum(
        len(targets)
        for source_table in base.values()
        for targets in source_table.values()
    )
    return base, {
        "raw_physical_edges": raw_edges,
        "physical_graph_sha256": digest.hexdigest(),
        "unique_base_context_edges": int(unique_base_edges),
        "base_context_sha256": hash_base_program(base),
    }


def iter_composed_endpoints(
    base: Mapping[str, Mapping[int, Sequence[int]]],
    composition: Sequence[str],
    source: int,
) -> Iterator[int]:
    if not composition:
        return
    if len(composition) == 1:
        yield from base[composition[0]].get(int(source), ())
        return

    frontier: set[int] = {int(source)}
    for relation_name in composition[:-1]:
        following: set[int] = set()
        table = base[relation_name]
        for node in sorted(frontier):
            following.update(table.get(node, ()))
        frontier = following
        if not frontier:
            return
    final_table = base[composition[-1]]
    for node in sorted(frontier):
        yield from final_table.get(node, ())


def semantic_adjacency_hash(
    adjacency: Sequence[Sequence[tuple[int, int]]],
    edge_type_names: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for name in edge_type_names:
        encoded = name.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    for source, edges in enumerate(adjacency):
        for target, edge_type in edges:
            digest.update(struct.pack("<iii", source, int(target), int(edge_type)))
    return digest.hexdigest()


def compile_sampled_semantic_adjacency(
    base: Mapping[str, Mapping[int, Sequence[int]]],
    semantic_relations: Sequence[Mapping[str, Any]],
    edge_type_names: Sequence[str],
    *,
    num_nodes: int,
    cap_per_relation: int,
    sampling_seed: int,
    progress_every: int,
) -> tuple[list[list[tuple[int, int]]], dict[str, Any]]:
    num_edge_types = len(edge_type_names)
    heaps: dict[int, list[tuple[int, int, int]]] = {}
    selected: dict[int, set[int]] = {}
    candidate_occurrences = 0
    started = time.perf_counter()

    for relation_index, relation in enumerate(semantic_relations):
        name = str(relation["name"])
        composition = [str(value) for value in relation["composition"]]
        forward_type = 2 * relation_index
        reverse_type = forward_type + 1
        sources = sorted(base[composition[0]])
        relation_candidates = 0
        for source in sources:
            for target in iter_composed_endpoints(base, composition, source):
                relation_candidates += 1
                candidate_occurrences += 1
                if source != target:
                    offer_sample(
                        heaps,
                        selected,
                        bucket=source * num_edge_types + forward_type,
                        neighbor=int(target),
                        rank=stable_rank(
                            sampling_seed, source, int(target), forward_type
                        ),
                        cap=cap_per_relation,
                    )
                    offer_sample(
                        heaps,
                        selected,
                        bucket=int(target) * num_edge_types + reverse_type,
                        neighbor=source,
                        rank=stable_rank(
                            sampling_seed, int(target), source, reverse_type
                        ),
                        cap=cap_per_relation,
                    )
                if (
                    progress_every > 0
                    and candidate_occurrences % progress_every == 0
                ):
                    elapsed = time.perf_counter() - started
                    print(
                        f"  resolved {candidate_occurrences:,} semantic endpoint "
                        f"occurrences ({candidate_occurrences / max(elapsed, 1e-9):,.0f}/s)",
                        flush=True,
                    )
        print(
            f"  [{relation_index + 1:02d}/{len(semantic_relations):02d}] "
            f"{name}: context_endpoint_occurrences={relation_candidates:,}",
            flush=True,
        )

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for bucket, heap in heaps.items():
        source, edge_type = divmod(bucket, num_edge_types)
        adjacency[source].extend((entry[2], edge_type) for entry in heap)
    for edges in adjacency:
        edges.sort(key=lambda item: (item[0], item[1]))
    sampled_edges = sum(len(edges) for edges in adjacency)
    return adjacency, {
        "candidate_semantic_endpoint_occurrences": int(candidate_occurrences),
        "sampled_directed_semantic_edges": int(sampled_edges),
        "nonempty_source_semantic_relation_blocks": int(len(heaps)),
        "max_neighbors_per_source_semantic_relation": int(cap_per_relation),
        "semantic_graph_sha256": semantic_adjacency_hash(
            adjacency, edge_type_names
        ),
    }


def shared_contract_hash(shared: Mapping[str, Any]) -> str:
    return array_sha256(
        shared["node_type"].numpy(),
        shared["y"].numpy(),
        shared["train_mask"].numpy(),
        shared["val_mask"].numpy(),
        shared["test_mask"].numpy(),
    )


def metadata_payload(
    *,
    status: str,
    variants: Mapping[str, Any],
    edge_type_names: Sequence[str],
    contract_hash: str,
    shared_provenance: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT,
        "dataset": "Freebase",
        "task": "node_classification",
        "mode": "invariant_conditional_semantic_paths",
        "status": status,
        "source_variants": list(args.variants),
        "compiler": "base_context_to_semantic_endpoints_then_rooted_pain_paths",
        "mapping_semantics": (
            "conditional graph contexts independently matched per physical variant; "
            "interior composition witnesses are context-only skip nodes"
        ),
        "semantic_relation_set": "union_of_36_base_6_exact2_22_exact3",
        "semantic_relation_contract_sha256": contract_hash,
        "edge_type_names": list(edge_type_names),
        "path_length": int(args.path_length),
        "sampling": {
            "policy": (
                "semantic_identity_top_k_per_source_relation_then_"
                "deterministic_prefix_rotation"
            ),
            "sampling_seed": int(args.sampling_seed),
            "max_neighbors_per_source_relation": int(
                args.max_neighbors_per_relation
            ),
            "path_fanout": int(args.path_fanout),
            "max_paths_per_root": int(args.max_paths_per_root),
            "sampling_occurs_after_semantic_projection": True,
            "reporting_label": "sampled invariant PAIN",
        },
        "variants": dict(variants),
        "shared": dict(shared_provenance),
    }


def preprocess(args: argparse.Namespace) -> None:
    variants_root = args.variants_root.resolve()
    output_dir = args.output_dir.resolve()
    reference_dir = variants_root / args.reference_variant
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_path = output_dir / "shared.pt"
    metadata_path = output_dir / "metadata.json"

    if (
        (shared_path.exists() or metadata_path.exists())
        and not args.resume
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Freebase invariant output already exists under {output_dir}; "
            "pass --resume or --overwrite"
        )

    semantic_relations, edge_type_names, relation_contract_hash = (
        semantic_relation_contract(variants_root, args.reference_variant)
    )
    base_names = {
        str(relation["name"]): relation
        for relation in semantic_relations
        if relation["kind"] == "base"
    }
    shared, shared_provenance = make_shared(
        reference_dir,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    if shared_path.exists() and args.resume:
        existing_shared = torch.load(
            shared_path, map_location="cpu", weights_only=False
        )
        for key in (
            "y",
            "node_type",
            "train_mask",
            "val_mask",
            "test_mask",
        ):
            if not torch.equal(existing_shared[key], shared[key]):
                raise ValueError(
                    f"Existing invariant shared.pt differs in {key}"
                )
        shared = existing_shared
    else:
        atomic_torch_save(shared, shared_path)

    existing_variants: dict[str, Any] = {}
    if metadata_path.is_file() and args.resume:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("semantic_relation_contract_sha256") != relation_contract_hash:
            raise ValueError(
                "Existing invariant artifacts use a different semantic relation contract"
            )
        existing_variants.update(existing.get("variants", {}))

    num_nodes = int(shared["x"].shape[0])
    common_shared_hash = shared_contract_hash(shared)
    tag = artifact_tag(
        args.path_length,
        args.max_neighbors_per_relation,
        args.path_fanout,
        args.max_paths_per_root,
    )
    summaries: dict[str, Any] = {}

    for variant in args.variants:
        destination = output_dir / f"invariant_{variant}_{tag}.pt"
        summary_path = destination.with_suffix(".json")
        if destination.is_file() and summary_path.is_file() and args.resume:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected = {
                "path_length": int(args.path_length),
                "max_neighbors_per_source_relation": int(
                    args.max_neighbors_per_relation
                ),
                "path_fanout": int(args.path_fanout),
                "max_paths_per_root": int(args.max_paths_per_root),
                "sampling_seed": int(args.sampling_seed),
                "semantic_relation_contract_sha256": relation_contract_hash,
            }
            observed = {key: summary.get(key) for key in expected}
            if observed != expected:
                raise ValueError(
                    f"Cannot resume {variant}: preprocessing contract differs; "
                    f"observed={observed}, requested={expected}"
                )
            summaries[variant] = summary
            print(f"[{variant}] retaining completed {destination}", flush=True)
            continue

        started = time.perf_counter()
        source_dir = variants_root / variant
        if file_sha256(source_dir / "node.dat") != shared_provenance["node_sha256"]:
            raise ValueError(f"{variant}/node.dat differs from unchanged")
        if file_sha256(source_dir / "label.dat") != shared_provenance["label_sha256"]:
            raise ValueError(f"{variant}/label.dat differs from unchanged")
        print(
            f"\n=== Freebase invariant PAIN | source={variant} | "
            f"semantic_relations={len(semantic_relations)} ===",
            flush=True,
        )
        base, physical_stats = read_physical_base_context(
            source_dir,
            reference_schema=base_names,
            num_nodes=num_nodes,
            progress_every=args.progress_every,
        )
        adjacency, semantic_stats = compile_sampled_semantic_adjacency(
            base,
            semantic_relations,
            edge_type_names,
            num_nodes=num_nodes,
            cap_per_relation=args.max_neighbors_per_relation,
            sampling_seed=args.sampling_seed,
            progress_every=args.semantic_progress_every,
        )
        path_tensors, path_stats = materialize_paths(
            adjacency,
            path_length=args.path_length,
            path_fanout=args.path_fanout,
            max_paths_per_root=args.max_paths_per_root,
            sampling_seed=args.sampling_seed,
        )
        artifact_name = f"invariant_{variant}"
        details = {
            "display_name": DISPLAY_NAMES[variant],
            "artifact": destination.name,
            "source_variant": variant,
            "mapping_mode": "conditional_semantic_path_compile",
            "path_length": int(args.path_length),
            "path_fanout": int(args.path_fanout),
            "max_paths_per_root": int(args.max_paths_per_root),
            "max_neighbors_per_source_relation": int(
                args.max_neighbors_per_relation
            ),
            "sampling_seed": int(args.sampling_seed),
            "semantic_relation_contract_sha256": relation_contract_hash,
            "shared_contract_sha256": common_shared_hash,
            **physical_stats,
            **semantic_stats,
            **path_stats,
        }
        payload = {
            "edge_index": torch.empty((2, 0), dtype=torch.int64),
            "edge_type": torch.arange(len(edge_type_names), dtype=torch.int32),
            **path_tensors,
            "meta": {
                "dataset": "Freebase",
                "variant": artifact_name,
                "source_variant": variant,
                "display_name": DISPLAY_NAMES[variant],
                "mapping_mode": details["mapping_mode"],
                "compiler": (
                    "conditional_base_context_to_canonical_semantic_"
                    "propagation_paths"
                ),
                "context_only_internal_nodes": True,
                "num_nodes": num_nodes,
                "path_length": int(args.path_length),
                "path_semantics": (
                    "bounded_rooted_simple_paths_over_canonical_semantic_"
                    "propagation_edges"
                ),
                "path_sampling": (
                    "semantic_identity_top_k_then_prefix_rotation"
                ),
                "sampling_seed": int(args.sampling_seed),
                "edge_type_names": edge_type_names,
                "physical_graph_sha256": physical_stats[
                    "physical_graph_sha256"
                ],
                "base_context_sha256": physical_stats["base_context_sha256"],
                "semantic_graph_sha256": semantic_stats[
                    "semantic_graph_sha256"
                ],
                "selected_path_program_sha256": path_stats[
                    "selected_path_program_sha256"
                ],
                "semantic_relation_contract_sha256": relation_contract_hash,
                "shared_contract_sha256": common_shared_hash,
            },
        }
        atomic_torch_save(payload, destination)
        details["artifact_bytes"] = int(destination.stat().st_size)
        details["preprocessing_seconds"] = time.perf_counter() - started
        write_json(summary_path, details)
        summaries[variant] = details
        write_json(
            metadata_path,
            metadata_payload(
                status="BUILDING",
                variants=summaries,
                edge_type_names=edge_type_names,
                contract_hash=relation_contract_hash,
                shared_provenance=shared_provenance,
                args=args,
            ),
        )
        print(json.dumps(details, indent=2, sort_keys=True), flush=True)
        del base, adjacency, path_tensors, payload
        gc.collect()

    physical_hashes = {
        variant: summaries[variant]["physical_graph_sha256"]
        for variant in args.variants
    }
    base_hashes = {
        variant: summaries[variant]["base_context_sha256"]
        for variant in args.variants
    }
    semantic_hashes = {
        variant: summaries[variant]["semantic_graph_sha256"]
        for variant in args.variants
    }
    path_hashes = {
        variant: summaries[variant]["selected_path_program_sha256"]
        for variant in args.variants
    }
    if len(args.variants) > 1 and len(set(physical_hashes.values())) != len(
        args.variants
    ):
        raise RuntimeError(
            f"Invariant inputs are not distinct physical graphs: {physical_hashes}"
        )
    if len(set(base_hashes.values())) != 1:
        raise RuntimeError(
            "Freebase variants disagree on the authoritative base context: "
            f"{base_hashes}"
        )
    if len(set(semantic_hashes.values())) != 1:
        raise RuntimeError(
            "Freebase variants did not compile to one sampled semantic graph: "
            f"{semantic_hashes}"
        )
    if len(set(path_hashes.values())) != 1:
        raise RuntimeError(
            "Freebase semantic graphs matched but PAIN path programs differ: "
            f"{path_hashes}"
        )

    final_metadata = metadata_payload(
        status="PASS",
        variants=summaries,
        edge_type_names=edge_type_names,
        contract_hash=relation_contract_hash,
        shared_provenance=shared_provenance,
        args=args,
    )
    final_metadata.update(
        {
            "physical_graph_sha256": physical_hashes,
            "base_context_sha256": base_hashes,
            "semantic_graph_sha256": semantic_hashes,
            "selected_path_program_sha256": path_hashes,
        }
    )
    write_json(metadata_path, final_metadata)
    print(
        "\nFreebase invariant preprocessing PASS: physical graphs differ; "
        "base contexts, sampled semantic graph, and PAIN paths match.",
        flush=True,
    )
    print(f"Wrote {shared_path}", flush=True)
    print(f"Wrote {metadata_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/preprocessed/Freebase_invariant"),
    )
    parser.add_argument("--variants", nargs="+", default=list(SOURCE_VARIANTS))
    parser.add_argument("--reference-variant", default="unchanged")
    parser.add_argument("--path-length", type=int, default=3)
    parser.add_argument("--max-neighbors-per-relation", type=int, default=4)
    parser.add_argument("--path-fanout", type=int, default=8)
    parser.add_argument("--max-paths-per-root", type=int, default=256)
    parser.add_argument("--sampling-seed", type=int, default=1566911444)
    parser.add_argument("--split-seed", type=int, default=1566911444)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--progress-every", type=int, default=10_000_000)
    parser.add_argument(
        "--semantic-progress-every", type=int, default=10_000_000
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.reference_variant != "unchanged":
        parser.error("Freebase invariant compilation requires unchanged as reference")
    unknown = sorted(set(args.variants) - set(SOURCE_VARIANTS))
    if unknown:
        parser.error("Unknown source variants: " + ", ".join(unknown))
    if len(set(args.variants)) != len(args.variants):
        parser.error("--variants contains duplicates")
    if set(args.variants) != set(SOURCE_VARIANTS):
        parser.error(
            "Invariant preprocessing must compile unchanged, exact_2, and "
            "exact_3 together; use --resume to retain completed variants"
        )
    if args.path_length < 1:
        parser.error("--path-length must be positive")
    if args.max_neighbors_per_relation < 1:
        parser.error("--max-neighbors-per-relation must be positive")
    if args.path_fanout < 1 or args.max_paths_per_root < 1:
        parser.error("path fanout and per-root path cap must be positive")
    required = {args.reference_variant, "exact_2", "exact_3", *args.variants}
    missing = [
        args.variants_root / variant
        for variant in sorted(required)
        if not (args.variants_root / variant).is_dir()
    ]
    if missing:
        parser.error("Missing variant directories: " + ", ".join(map(str, missing)))
    return args


if __name__ == "__main__":
    preprocess(parse_args())
