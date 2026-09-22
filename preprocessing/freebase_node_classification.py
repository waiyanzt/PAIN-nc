"""Build bounded PAIN node-classification artifacts for physical Freebase variants.

Freebase's materialized shortcut variants are too large for exhaustive simple-path
enumeration.  This compiler streams each ``link.dat`` once, keeps a deterministic
top-k sample per (source node, directed relation), and then materializes at most a
fixed number of rooted simple paths per node.  Sampling ranks depend only on the
edge's semantic identity and the configured seed, never on file order.

The output uses the ordinary :class:`pain_nc.data.PainGraph` contract, so the
standard PAIN node-classification trainer remains the source of model semantics.
The experiment must be reported as sampled PAIN.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.model_selection import train_test_split


DEFAULT_VARIANTS = ("unchanged", "exact_2", "exact_3")
DISPLAY_NAMES = {
    "unchanged": "Freebase1",
    "exact_2": "Freebase2",
    "exact_3": "Freebase3",
    "union_exact_2_3": "Freebase_universal",
}
MASK64 = (1 << 64) - 1


def artifact_tag(
    path_length: int,
    max_neighbors_per_relation: int,
    path_fanout: int,
    max_paths_per_root: int,
) -> str:
    return (
        f"L{path_length}_rel{max_neighbors_per_relation}_"
        f"fan{path_fanout}_cap{max_paths_per_root}"
    )


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_info(path: Path) -> tuple[dict[int, str], dict[int, dict[str, Any]]]:
    node_types: dict[int, str] = {}
    relations: dict[int, dict[str, Any]] = {}
    section: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        if text == "node.dat":
            section = "nodes"
            continue
        if text == "link.dat":
            section = "links"
            continue
        if text in {"label.dat", "label.dat.test"}:
            section = "labels"
            continue
        if (
            text.startswith("TYPE\tMEANING")
            or text.startswith("LINK\tSTART\tEND\tMEANING")
            or text.startswith("TYPE\tCLASS\tMEANING")
            or text.startswith("Attribute Dimension:")
            or text.startswith("Targeting:")
            or text.startswith("---")
        ):
            continue
        parts = [part for part in text.split("\t") if part != ""]
        if section == "nodes" and len(parts) >= 2:
            node_types[int(parts[0])] = parts[-1]
        elif section == "links" and len(parts) >= 4:
            relation_id = int(parts[0])
            relations[relation_id] = {
                "id": relation_id,
                "start_type": int(parts[1]),
                "end_type": int(parts[2]),
                "name": parts[-1],
            }
    if not node_types or not relations:
        raise ValueError(f"Incomplete Freebase schema: {path}")
    return node_types, relations


def read_nodes(path: Path) -> tuple[np.ndarray, list[int]]:
    rows: list[tuple[int, int]] = []
    raw_types: set[int] = set()
    maximum = -1
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"Bad node.dat line {line_number}")
            node_id, node_type = int(parts[0]), int(parts[2])
            rows.append((node_id, node_type))
            raw_types.add(node_type)
            maximum = max(maximum, node_id)
    ordered_types = sorted(raw_types)
    remap = {value: index for index, value in enumerate(ordered_types)}
    node_types = np.full(maximum + 1, -1, dtype=np.int16)
    for node_id, node_type in rows:
        node_types[node_id] = remap[node_type]
    if np.any(node_types < 0):
        raise ValueError("node.dat ids must be contiguous")
    return node_types, ordered_types


def read_labels(path: Path) -> list[tuple[int, int, int]]:
    labels: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 4:
                raise ValueError(f"Bad label.dat line {line_number}")
            labels.append((int(parts[0]), int(parts[2]), int(parts[3])))
    if not labels:
        raise ValueError(f"No labels in {path}")
    return labels


def make_shared(
    reference_dir: Path,
    *,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_node_type_names, _relations = parse_info(reference_dir / "info.dat")
    node_types, raw_type_ids = read_nodes(reference_dir / "node.dat")
    labels = read_labels(reference_dir / "label.dat")
    labeled_type_counts = Counter(raw_type for _node, raw_type, _label in labels)
    if len(labeled_type_counts) != 1:
        raise ValueError(
            "Freebase label.dat must identify one target node type; got "
            f"{dict(labeled_type_counts)}"
        )
    center_raw_type = next(iter(labeled_type_counts))
    raw_to_compact = {raw: index for index, raw in enumerate(raw_type_ids)}
    center_type = raw_to_compact[center_raw_type]
    labeled = sorted(
        (node_id, label)
        for node_id, raw_type, label in labels
        if raw_type == center_raw_type
    )
    target_nodes = np.asarray([node for node, _label in labeled], dtype=np.int64)
    raw_labels = sorted({label for _node, label in labeled})
    class_remap = {label: index for index, label in enumerate(raw_labels)}
    target_labels = np.asarray(
        [class_remap[label] for _node, label in labeled], dtype=np.int64
    )
    local_indices = np.arange(target_nodes.size, dtype=np.int64)
    test_ratio = 1.0 - train_ratio - val_ratio
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and test_ratio > 0):
        raise ValueError("train/validation/test ratios must all be positive")
    train_local, held_local = train_test_split(
        local_indices,
        test_size=1.0 - train_ratio,
        random_state=split_seed,
        shuffle=True,
        stratify=target_labels,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_local, test_local = train_test_split(
        held_local,
        test_size=relative_test,
        random_state=split_seed,
        shuffle=True,
        stratify=target_labels[held_local],
    )
    split_local = {
        "train": np.asarray(sorted(train_local), dtype=np.int64),
        "val": np.asarray(sorted(val_local), dtype=np.int64),
        "test": np.asarray(sorted(test_local), dtype=np.int64),
    }
    num_nodes = int(node_types.size)
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    y[torch.from_numpy(target_nodes)] = torch.from_numpy(target_labels)
    masks: dict[str, torch.Tensor] = {}
    global_splits: dict[str, np.ndarray] = {}
    for name, local in split_local.items():
        global_ids = target_nodes[local]
        global_splits[name] = global_ids
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[torch.from_numpy(global_ids)] = True
        masks[f"{name}_mask"] = mask
    type_names = [raw_node_type_names.get(raw, str(raw)) for raw in raw_type_ids]
    shared = {
        # Freebase has no dense input feature vectors. A constant scalar keeps
        # the generic PAIN input encoder well-defined; learned node-id and type
        # embeddings supply the actual initial representation.
        "x": torch.ones((num_nodes, 1), dtype=torch.float32),
        "y": y,
        "node_type": torch.from_numpy(node_types.astype(np.int64, copy=False)),
        **masks,
        "split_indices": {
            name: torch.from_numpy(values) for name, values in global_splits.items()
        },
        "target_nodes": torch.from_numpy(target_nodes),
        "meta": {
            "dataset": "Freebase",
            "task": "node_classification",
            "source": str(reference_dir),
            "num_nodes": num_nodes,
            "num_features": 1,
            "num_classes": len(raw_labels),
            "node_type_names": type_names,
            "raw_node_type_ids": raw_type_ids,
            "center_node_type": center_type,
            "center_raw_node_type": center_raw_type,
            "raw_class_ids": raw_labels,
            "split_seed": split_seed,
            "split_protocol": f"stratified_{train_ratio:.3f}_{val_ratio:.3f}_{test_ratio:.3f}",
            "initial_features": "constant_one_plus_learned_node_id_and_type_embeddings",
        },
    }
    provenance = {
        "node_sha256": file_sha256(reference_dir / "node.dat"),
        "label_sha256": file_sha256(reference_dir / "label.dat"),
        "num_nodes": num_nodes,
        "num_target_nodes": int(target_nodes.size),
        "class_counts": np.bincount(target_labels, minlength=len(raw_labels)).tolist(),
        "split_sizes": {name: int(values.size) for name, values in global_splits.items()},
    }
    return shared, provenance


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def stable_rank(seed: int, *values: int) -> int:
    state = int(seed) & MASK64
    for index, value in enumerate(values):
        state = splitmix64(state ^ ((int(value) + index * 0x9E3779B9) & MASK64))
    return state


def offer_sample(
    heaps: dict[int, list[tuple[int, int, int]]],
    selected: dict[int, set[int]],
    *,
    bucket: int,
    neighbor: int,
    rank: int,
    cap: int,
) -> None:
    chosen = selected.get(bucket)
    if chosen is None:
        chosen = set()
        selected[bucket] = chosen
    if neighbor in chosen:
        return
    heap = heaps.get(bucket)
    entry = (-int(rank), -int(neighbor), int(neighbor))
    if heap is None:
        heaps[bucket] = [entry]
        chosen.add(neighbor)
        return
    if len(heap) < cap:
        heapq.heappush(heap, entry)
        chosen.add(neighbor)
        return
    worst_rank, worst_neighbor = -heap[0][0], -heap[0][1]
    if (rank, neighbor) < (worst_rank, worst_neighbor):
        removed = heapq.heapreplace(heap, entry)[2]
        chosen.remove(removed)
        chosen.add(neighbor)


def sampled_adjacency(
    link_path: Path,
    *,
    relation_to_types: dict[int, tuple[int, int]],
    num_nodes: int,
    num_edge_types: int,
    cap_per_relation: int,
    sampling_seed: int,
    progress_every: int,
) -> tuple[list[list[tuple[int, int]]], dict[str, Any]]:
    heaps: dict[int, list[tuple[int, int, int]]] = {}
    selected: dict[int, set[int]] = {}
    digest = hashlib.sha256()
    raw_edges = 0
    started = time.perf_counter()
    with link_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            digest.update(raw)
            parts = raw.rstrip(b"\r\n").split(b"\t")
            if len(parts) < 3:
                raise ValueError(f"Bad link.dat line {line_number}")
            source, target, relation_id = int(parts[0]), int(parts[1]), int(parts[2])
            if not (0 <= source < num_nodes and 0 <= target < num_nodes):
                raise ValueError(f"Out-of-range node id at link.dat line {line_number}")
            try:
                forward_type, reverse_type = relation_to_types[relation_id]
            except KeyError as exc:
                raise KeyError(f"Unknown relation id {relation_id} at line {line_number}") from exc
            raw_edges += 1
            if source != target:
                forward_bucket = source * num_edge_types + forward_type
                reverse_bucket = target * num_edge_types + reverse_type
                offer_sample(
                    heaps,
                    selected,
                    bucket=forward_bucket,
                    neighbor=target,
                    rank=stable_rank(sampling_seed, source, target, forward_type),
                    cap=cap_per_relation,
                )
                offer_sample(
                    heaps,
                    selected,
                    bucket=reverse_bucket,
                    neighbor=source,
                    rank=stable_rank(sampling_seed, target, source, reverse_type),
                    cap=cap_per_relation,
                )
            if progress_every > 0 and raw_edges % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  streamed {raw_edges:,} physical edges "
                    f"({raw_edges / max(elapsed, 1e-9):,.0f} edges/s)",
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
        "raw_physical_edges": raw_edges,
        "raw_link_sha256": digest.hexdigest(),
        "sampled_directed_edges": sampled_edges,
        "nonempty_source_relation_blocks": len(heaps),
        "max_neighbors_per_source_relation": cap_per_relation,
    }


def selected_edges(
    adjacency: list[list[tuple[int, int]]],
    *,
    root: int,
    nodes: tuple[int, ...],
    edge_types: tuple[int, ...],
    depth: int,
    fanout: int,
    sampling_seed: int,
) -> list[tuple[int, int]]:
    candidates = adjacency[nodes[-1]]
    if not candidates:
        return []
    prefix_key = stable_rank(sampling_seed, root, depth, *nodes, *edge_types)
    offset = prefix_key % len(candidates)
    visited = set(nodes)
    result: list[tuple[int, int]] = []
    for index in range(len(candidates)):
        neighbor, edge_type = candidates[(offset + index) % len(candidates)]
        if neighbor in visited:
            continue
        result.append((neighbor, edge_type))
        if len(result) == fanout:
            break
    return result


def rooted_paths(
    root: int,
    adjacency: list[list[tuple[int, int]]],
    *,
    path_length: int,
    path_fanout: int,
    max_paths: int,
    sampling_seed: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    paths: list[tuple[tuple[int, ...], tuple[int, ...]]] = [((root,), ())]
    frontier = [((root,), ())]
    for depth in range(1, path_length + 1):
        following: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for nodes, edge_types in frontier:
            for neighbor, edge_type in selected_edges(
                adjacency,
                root=root,
                nodes=nodes,
                edge_types=edge_types,
                depth=depth,
                fanout=path_fanout,
                sampling_seed=sampling_seed,
            ):
                item = (nodes + (neighbor,), edge_types + (edge_type,))
                paths.append(item)
                following.append(item)
                if len(paths) >= max_paths:
                    return paths
        frontier = following
        if not frontier:
            break
    return paths


def sampled_distances(
    root: int,
    adjacency: list[list[tuple[int, int]]],
    path_length: int,
) -> dict[int, int]:
    distance = {root: 0}
    frontier = [root]
    for depth in range(1, path_length + 1):
        following: list[int] = []
        for node in frontier:
            for neighbor, _edge_type in adjacency[node]:
                if neighbor not in distance:
                    distance[neighbor] = depth
                    following.append(neighbor)
        frontier = following
        if not frontier:
            break
    return distance


def materialize_paths(
    adjacency: list[list[tuple[int, int]]],
    *,
    path_length: int,
    path_fanout: int,
    max_paths_per_root: int,
    sampling_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    num_nodes = len(adjacency)
    counts = np.empty(num_nodes, dtype=np.int32)
    length_counts = np.zeros(path_length + 1, dtype=np.int64)
    for root in range(num_nodes):
        paths = rooted_paths(
            root,
            adjacency,
            path_length=path_length,
            path_fanout=path_fanout,
            max_paths=max_paths_per_root,
            sampling_seed=sampling_seed,
        )
        counts[root] = len(paths)
        for nodes, _edge_types in paths:
            length_counts[len(nodes) - 1] += 1
    total = int(counts.sum())
    width = path_length + 1
    path_index = np.full((width, total), -10, dtype=np.int32)
    path_edge_idx = np.full((width, total), -20, dtype=np.int32)
    path_lengths = np.empty(total, dtype=np.int16)
    mask_index = np.empty(total, dtype=np.int32)
    neighbor_mask = np.full((total, width), -10, dtype=np.int8)
    distances = np.full((total, width), path_length + 1, dtype=np.int8)
    cursor = 0
    for root in range(num_nodes):
        direct_neighbors = {neighbor for neighbor, _edge_type in adjacency[root]}
        distance = sampled_distances(root, adjacency, path_length)
        for nodes, edge_types in rooted_paths(
            root,
            adjacency,
            path_length=path_length,
            path_fanout=path_fanout,
            max_paths=max_paths_per_root,
            sampling_seed=sampling_seed,
        ):
            size = len(nodes)
            path_index[:size, cursor] = nodes
            if edge_types:
                path_edge_idx[1:size, cursor] = edge_types
            path_lengths[cursor] = size
            mask_index[cursor] = root
            neighbor_mask[cursor, :size] = [
                int(node in direct_neighbors) for node in nodes
            ]
            distances[cursor, :size] = [
                distance.get(node, position)
                for position, node in enumerate(nodes)
            ]
            cursor += 1
    if cursor != total:
        raise AssertionError(f"materialized {cursor} paths, expected {total}")
    tensors = {
        "path_index": torch.from_numpy(path_index),
        "path_lengths": torch.from_numpy(path_lengths),
        "mask_index": torch.from_numpy(mask_index),
        "path_edge_idx": torch.from_numpy(path_edge_idx),
        "neighbor_mask": torch.from_numpy(neighbor_mask),
        "distances": torch.from_numpy(distances),
    }
    return tensors, {
        "num_paths": total,
        "path_counts_by_edge_length": length_counts.tolist(),
        "min_paths_per_root": int(counts.min()),
        "max_paths_per_root_observed": int(counts.max()),
        "mean_paths_per_root": float(counts.mean()),
        "selected_path_program_sha256": array_sha256(
            path_index,
            path_edge_idx,
            path_lengths,
            mask_index,
        ),
    }


def relation_vocabulary(
    variants_root: Path, variants: Iterable[str]
) -> tuple[list[str], dict[str, dict[int, tuple[int, int]]]]:
    schemas: dict[str, dict[int, dict[str, Any]]] = {}
    names: set[str] = set()
    for variant in variants:
        _node_types, relations = parse_info(variants_root / variant / "info.dat")
        schemas[variant] = relations
        names.update(str(record["name"]) for record in relations.values())
    ordered_names = sorted(names)
    edge_type_names = [
        f"{name}::{direction}"
        for name in ordered_names
        for direction in ("forward", "reverse")
    ]
    name_to_forward = {name: 2 * index for index, name in enumerate(ordered_names)}
    mappings: dict[str, dict[int, tuple[int, int]]] = {}
    for variant, relations in schemas.items():
        mappings[variant] = {
            relation_id: (
                name_to_forward[str(record["name"])],
                name_to_forward[str(record["name"])] + 1,
            )
            for relation_id, record in relations.items()
        }
    return edge_type_names, mappings


def preprocess(args: argparse.Namespace) -> None:
    variants_root = args.variants_root.resolve()
    output_dir = args.output_dir.resolve()
    reference_dir = variants_root / args.reference_variant
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_path = output_dir / "shared.pt"
    metadata_path = output_dir / "metadata.json"
    if (
        (shared_path.exists() or metadata_path.exists())
        and not args.overwrite
        and not args.resume
    ):
        raise FileExistsError(
            f"Freebase output already exists under {output_dir}; pass --resume "
            "to retain completed variants or --overwrite to replace them"
        )
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
        if (
            existing_shared.get("meta") != shared.get("meta")
            or not torch.equal(existing_shared["y"], shared["y"])
            or not torch.equal(existing_shared["train_mask"], shared["train_mask"])
            or not torch.equal(existing_shared["val_mask"], shared["val_mask"])
            or not torch.equal(existing_shared["test_mask"], shared["test_mask"])
        ):
            raise ValueError(
                "Existing shared.pt does not match the requested Freebase split contract"
            )
        shared = existing_shared
    else:
        atomic_torch_save(shared, shared_path)
    edge_type_names, relation_mappings = relation_vocabulary(
        variants_root, args.variants
    )
    num_nodes = int(shared["x"].shape[0])
    reference_node_hash = shared_provenance["node_sha256"]
    reference_label_hash = shared_provenance["label_sha256"]
    summaries: dict[str, Any] = {}
    if metadata_path.exists() and args.resume:
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summaries.update(existing_metadata.get("variants", {}))
    for variant in args.variants:
        started = time.perf_counter()
        source_dir = variants_root / variant
        if file_sha256(source_dir / "node.dat") != reference_node_hash:
            raise ValueError(f"{variant}/node.dat differs from {args.reference_variant}")
        if file_sha256(source_dir / "label.dat") != reference_label_hash:
            raise ValueError(f"{variant}/label.dat differs from {args.reference_variant}")
        tag = artifact_tag(
            args.path_length,
            args.max_neighbors_per_relation,
            args.path_fanout,
            args.max_paths_per_root,
        )
        destination = output_dir / f"{variant}_{tag}.pt"
        summary_path = destination.with_suffix(".json")
        if destination.exists() and args.resume:
            if not summary_path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume {variant}: artifact exists without {summary_path}"
                )
            print(f"[{variant}] already preprocessed; retaining {destination}")
            prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected_sampling = {
                "path_length": args.path_length,
                "max_neighbors_per_source_relation": args.max_neighbors_per_relation,
                "path_fanout": args.path_fanout,
                "max_paths_per_root": args.max_paths_per_root,
                "sampling_seed": args.sampling_seed,
            }
            observed_sampling = {
                key: prior_summary.get(key) for key in expected_sampling
            }
            if observed_sampling != expected_sampling:
                raise ValueError(
                    f"Cannot resume {variant}: sampling contract differs; "
                    f"observed={observed_sampling}, requested={expected_sampling}"
                )
            summaries[variant] = prior_summary
            continue
        if destination.exists() and not args.overwrite:
            raise FileExistsError(destination)
        print(f"\n=== Freebase PAIN independent | {variant} ===", flush=True)
        adjacency, graph_stats = sampled_adjacency(
            source_dir / "link.dat",
            relation_to_types=relation_mappings[variant],
            num_nodes=num_nodes,
            num_edge_types=len(edge_type_names),
            cap_per_relation=args.max_neighbors_per_relation,
            sampling_seed=args.sampling_seed,
            progress_every=args.progress_every,
        )
        path_tensors, path_stats = materialize_paths(
            adjacency,
            path_length=args.path_length,
            path_fanout=args.path_fanout,
            max_paths_per_root=args.max_paths_per_root,
            sampling_seed=args.sampling_seed,
        )
        payload = {
            # PAIN only indexes relation embeddings through path_edge_idx. Keep
            # the complete physical edge list out of the training artifact.
            "edge_index": torch.empty((2, 0), dtype=torch.int64),
            "edge_type": torch.arange(len(edge_type_names), dtype=torch.int32),
            **path_tensors,
            "meta": {
                "dataset": "Freebase",
                "variant": variant,
                "display_name": DISPLAY_NAMES.get(variant, variant),
                "num_nodes": num_nodes,
                "mapping_mode": "independent_physical_variant",
                "path_length": args.path_length,
                "path_semantics": "bounded_rooted_simple_paths_on_relation_stratified_sample",
                "path_sampling": "canonical_top_k_per_source_relation_then_prefix_rotation",
                "sampling_seed": args.sampling_seed,
                "max_neighbors_per_source_relation": args.max_neighbors_per_relation,
                "path_fanout": args.path_fanout,
                "max_paths_per_root": args.max_paths_per_root,
                "edge_type_names": edge_type_names,
                "physical_graph_sha256": graph_stats["raw_link_sha256"],
                "selected_path_program_sha256": path_stats["selected_path_program_sha256"],
                "shared_contract_sha256": array_sha256(
                    shared["node_type"].numpy(),
                    shared["y"].numpy(),
                    shared["train_mask"].numpy(),
                    shared["val_mask"].numpy(),
                    shared["test_mask"].numpy(),
                ),
            },
        }
        atomic_torch_save(payload, destination)
        summaries[variant] = {
            "display_name": DISPLAY_NAMES.get(variant, variant),
            "artifact": destination.name,
            "path_length": args.path_length,
            "path_fanout": args.path_fanout,
            "max_paths_per_root": args.max_paths_per_root,
            "sampling_seed": args.sampling_seed,
            **graph_stats,
            **path_stats,
            "artifact_bytes": int(destination.stat().st_size),
            "preprocessing_seconds": time.perf_counter() - started,
        }
        write_json(summary_path, summaries[variant])
        print(json.dumps(summaries[variant], indent=2), flush=True)
        del adjacency, path_tensors, payload

    metadata = {
        "format_version": "freebase_pain_independent_v1",
        "dataset": "Freebase",
        "task": "node_classification",
        "mode": "independent_physical_variants",
        "variants": summaries,
        "edge_type_names": edge_type_names,
        "path_length": args.path_length,
        "sampling": {
            "policy": "relation_stratified_canonical_edge_sample_and_bounded_rooted_paths",
            "sampling_seed": args.sampling_seed,
            "max_neighbors_per_source_relation": args.max_neighbors_per_relation,
            "path_fanout": args.path_fanout,
            "max_paths_per_root": args.max_paths_per_root,
            "reporting_label": "sampled PAIN",
        },
        "shared": shared_provenance,
    }
    write_json(metadata_path, metadata)
    print(f"\nWrote {shared_path}")
    print(f"Wrote {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/preprocessed/Freebase"))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.path_length < 1:
        parser.error("--path-length must be positive")
    if args.max_neighbors_per_relation < 1:
        parser.error("--max-neighbors-per-relation must be positive")
    if args.path_fanout < 1:
        parser.error("--path-fanout must be positive")
    if args.max_paths_per_root < 1:
        parser.error("--max-paths-per-root must be positive")
    missing = [
        args.variants_root / variant
        for variant in [args.reference_variant, *args.variants]
        if not (args.variants_root / variant).is_dir()
    ]
    if missing:
        parser.error("Missing variant directories: " + ", ".join(map(str, missing)))
    return args


if __name__ == "__main__":
    preprocess(parse_args())
