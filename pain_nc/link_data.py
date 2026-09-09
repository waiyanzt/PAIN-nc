"""Data containers and loading for PAIN link-prediction graphs."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from .data import PATH_FIELDS, _torch_load, group_paths_by_root, reverse_valid_paths


@dataclass
class PainLinkGraph:
    """One physical graph variant and its exact rooted-path program."""

    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    path_index: torch.Tensor
    path_lengths: torch.Tensor
    mask_index: torch.Tensor
    path_edge_idx: torch.Tensor
    neighbor_mask: torch.Tensor
    distances: torch.Tensor
    shared_meta: dict[str, Any]
    variant_meta: dict[str, Any]

    @property
    def num_nodes(self) -> int:
        return int(self.node_type.numel())

    @property
    def num_node_types(self) -> int:
        names = self.shared_meta.get("node_type_names", ())
        return len(names) if names else int(self.node_type.max().item() + 1)

    @property
    def num_edge_types(self) -> int:
        names = self.variant_meta.get("edge_type_names", ())
        return len(names) if names else int(self.edge_type.max().item() + 1)

    @property
    def num_paths(self) -> int:
        return int(self.path_lengths.numel())

    def to(
        self,
        device: torch.device | str,
        *,
        move_paths: bool = False,
    ) -> "PainLinkGraph":
        """Move small tensors eagerly and optionally retain the path store on CPU."""
        device = torch.device(device)
        path_names = set(PATH_FIELDS)
        values: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if not isinstance(value, torch.Tensor):
                values[item.name] = value
            elif item.name == "path_lengths":
                # pack_padded_sequence requires CPU lengths.
                values[item.name] = value.cpu()
            elif item.name in path_names and not move_paths:
                values[item.name] = value.cpu()
            elif item.name == "edge_index":
                # PAIN addresses relation ids through path_edge_idx.
                values[item.name] = value.cpu()
            else:
                values[item.name] = value.to(device, non_blocking=True)
        return PainLinkGraph(**values)


def validate_link_graph(
    graph: PainLinkGraph,
    splits: dict[str, torch.Tensor],
) -> None:
    n, p = graph.num_nodes, graph.num_paths
    if graph.node_type.shape != (n,):
        raise ValueError("node_type must contain one value per node")
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if graph.edge_type.shape != (graph.edge_index.shape[1],):
        raise ValueError("edge_type and edge_index disagree on the edge count")
    if graph.path_index.ndim != 2:
        raise ValueError("path_index must have shape [L+1, num_paths]")
    width = int(graph.path_index.shape[0])
    expected = {
        "path_lengths": (p,),
        "mask_index": (p,),
        "path_edge_idx": (width, p),
        "neighbor_mask": (p, width),
        "distances": (p, width),
    }
    for name, shape in expected.items():
        actual = tuple(getattr(graph, name).shape)
        if actual != shape:
            raise ValueError(f"{name} has shape {actual}, expected {shape}")
    if p == 0:
        raise ValueError("A PAIN graph must contain at least its zero-edge paths")
    if int(graph.path_lengths.min()) < 1 or int(graph.path_lengths.max()) > width:
        raise ValueError("path_lengths contains an invalid sequence length")
    if int(graph.mask_index.min()) < 0 or int(graph.mask_index.max()) >= n:
        raise ValueError("mask_index contains an invalid root")
    if graph.mask_index.numel() > 1 and bool(
        torch.any(graph.mask_index[1:] < graph.mask_index[:-1])
    ):
        raise ValueError("mask_index must be root-sorted")
    for name in ("train_pos", "val_pos", "test_pos"):
        pairs = splits.get(name)
        if pairs is None or pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"{name} must have shape [num_pairs, 2]")
        if pairs.numel() and (
            int(pairs.min()) < 0 or int(pairs.max()) >= n
        ):
            raise ValueError(f"{name} contains an out-of-range node id")
    train_neg = splits.get("train_neg_tails")
    if train_neg is None or train_neg.ndim != 2:
        raise ValueError("train_neg_tails must have shape [num_train_pos, K]")
    if train_neg.shape[0] != splits["train_pos"].shape[0]:
        raise ValueError("train_neg_tails must align row-wise with train_pos")


def load_dblp_link_graph(
    shared_path: str | Path,
    variant_path: str | Path,
    *,
    reverse_paths: bool = True,
    validate: bool = True,
) -> tuple[PainLinkGraph, dict[str, torch.Tensor]]:
    """Load one DBLP topology variant plus the shared supervision contract."""
    shared = _torch_load(Path(shared_path))
    variant = _torch_load(Path(variant_path))
    required_shared = {"node_type", "splits", "meta"}
    required_variant = {"edge_index", "edge_type", "meta", *PATH_FIELDS}
    missing_shared = sorted(required_shared - shared.keys())
    missing_variant = sorted(required_variant - variant.keys())
    if missing_shared or missing_variant:
        raise ValueError(
            f"Missing fields: shared={missing_shared}, variant={missing_variant}"
        )
    if reverse_paths:
        reverse_valid_paths(variant)
    group_paths_by_root(variant)
    graph = PainLinkGraph(
        node_type=shared["node_type"],
        edge_index=variant["edge_index"],
        edge_type=variant["edge_type"],
        path_index=variant["path_index"],
        path_lengths=variant["path_lengths"],
        mask_index=variant["mask_index"],
        path_edge_idx=variant["path_edge_idx"],
        neighbor_mask=variant["neighbor_mask"],
        distances=variant["distances"],
        shared_meta=shared["meta"],
        variant_meta=variant["meta"],
    )
    if int(shared["meta"]["num_nodes"]) != int(variant["meta"]["num_nodes"]):
        raise ValueError("Shared and variant artifacts have different node counts")
    splits = {name: value for name, value in shared["splits"].items()}
    if validate:
        validate_link_graph(graph, splits)
    return graph, splits
