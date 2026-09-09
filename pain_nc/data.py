"""Data loading and validation for preprocessed PAIN node-classification graphs."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import torch


PATH_FIELDS = (
    "path_index",
    "path_lengths",
    "mask_index",
    "path_edge_idx",
    "neighbor_mask",
    "distances",
)

SHARED_TENSOR_FIELDS = (
    "x",
    "y",
    "node_type",
    "train_mask",
    "val_mask",
    "test_mask",
)


@dataclass
class PainGraph:
    x: torch.Tensor
    y: torch.Tensor
    node_type: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    path_index: torch.Tensor
    path_lengths: torch.Tensor
    mask_index: torch.Tensor
    path_edge_idx: torch.Tensor
    neighbor_mask: torch.Tensor
    distances: torch.Tensor
    shared_meta: dict
    variant_meta: dict

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_classes(self) -> int:
        labels = self.y[self.y >= 0]
        return int(labels.max().item() + 1)

    @property
    def num_node_types(self) -> int:
        names = self.shared_meta.get("node_type_names", ())
        return len(names) if names else int(self.node_type.max().item() + 1)

    @property
    def num_edge_types(self) -> int:
        # Use the dataset-wide relation vocabulary rather than the maximum
        # relation observed in this particular physical variant. This keeps
        # independently trained and jointly augmented models shape-identical.
        names = self.variant_meta.get("edge_type_names", ())
        return len(names) if names else int(self.edge_type.max().item() + 1)

    @property
    def num_paths(self) -> int:
        return int(self.path_lengths.numel())

    def to(
        self,
        device: torch.device | str,
        move_paths: bool = True,
        shared_from: "PainGraph | None" = None,
    ) -> "PainGraph":
        """Move model inputs to a device.

        ``path_lengths`` intentionally remains on CPU because PyTorch's packed
        sequence API requires CPU lengths. If ``move_paths`` is false, path
        chunks are transferred lazily by the model.
        """
        device = torch.device(device)
        path_names = set(PATH_FIELDS)
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if shared_from is not None and item.name in SHARED_TENSOR_FIELDS:
                values[item.name] = getattr(shared_from, item.name)
            elif not isinstance(value, torch.Tensor):
                values[item.name] = value
            elif item.name == "path_lengths":
                values[item.name] = value.cpu()
            elif item.name in path_names and not move_paths:
                values[item.name] = value.cpu()
            elif item.name == "edge_index":
                # PAIN consumes edge types through path_edge_idx, not edge_index.
                values[item.name] = value.cpu()
            else:
                values[item.name] = value.to(device, non_blocking=True)
        return PainGraph(**values)


def _torch_load(path: Path) -> dict:
    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except (TypeError, RuntimeError):
        return torch.load(path, **kwargs)


def reverse_valid_paths(payload: dict) -> None:
    """Reverse each valid root-first path exactly as the PAIN experiments do."""
    path_index = payload["path_index"]
    width, num_paths = path_index.shape
    lengths = payload["path_lengths"]
    positions = torch.arange(width).view(width, 1)
    valid = positions < lengths.view(1, num_paths)
    source = lengths.view(1, num_paths) - 1 - positions
    source = torch.where(valid, source, positions).clamp(min=0)

    for name in ("path_index", "path_edge_idx"):
        payload[name] = torch.gather(payload[name], 0, source)
    for name in ("neighbor_mask", "distances"):
        transposed = payload[name].t().contiguous()
        payload[name] = torch.gather(transposed, 0, source).t().contiguous()

    payload["meta"] = dict(payload.get("meta", {}))
    payload["meta"]["runtime_path_order"] = "leaf_first"


def group_paths_by_root(payload: dict) -> None:
    """Canonically group paths by root for deterministic segment reduction.

    Early PAIN-NC preprocessing artifacts were emitted in decreasing path-length
    order, so the same root appeared in several disjoint blocks.  The LSTM is
    evaluated independently for every path and aggregation is a sum/mean, hence
    reordering path columns does not change the represented graph.  A stable
    root grouping also preserves the original decreasing-length order within
    each root.
    """
    roots = payload["mask_index"]
    if roots.numel() < 2 or not bool(torch.any(roots[1:] < roots[:-1])):
        return

    order = torch.argsort(roots, stable=True)
    for name in ("path_index", "path_edge_idx"):
        payload[name] = payload[name].index_select(1, order)
    row_fields = ["path_lengths", "mask_index", "neighbor_mask", "distances"]
    if "path_weights" in payload:
        row_fields.append("path_weights")
    for name in row_fields:
        payload[name] = payload[name].index_select(0, order)

    payload["meta"] = dict(payload.get("meta", {}))
    payload["meta"]["runtime_path_grouping"] = "root_stable"


def validate_graph(graph: PainGraph) -> None:
    n, p = graph.num_nodes, graph.num_paths
    if graph.y.shape != (n,) or graph.node_type.shape != (n,):
        raise ValueError("x, y, and node_type disagree on the node count")
    if graph.path_index.ndim != 2:
        raise ValueError("path_index must have shape [L+1, num_paths]")
    width = graph.path_index.shape[0]
    expected = {
        "path_lengths": (p,),
        "mask_index": (p,),
        "path_edge_idx": (width, p),
        "neighbor_mask": (p, width),
        "distances": (p, width),
    }
    for name, shape in expected.items():
        if tuple(getattr(graph, name).shape) != shape:
            raise ValueError(f"{name} has shape {tuple(getattr(graph, name).shape)}, expected {shape}")
    if int(graph.path_lengths.min()) < 1 or int(graph.path_lengths.max()) > width:
        raise ValueError("path_lengths contains an invalid sequence length")
    if int(graph.mask_index.min()) < 0 or int(graph.mask_index.max()) >= n:
        raise ValueError("mask_index contains an invalid root node")
    if graph.mask_index.numel() > 1 and torch.any(
        graph.mask_index[1:] < graph.mask_index[:-1]
    ):
        raise ValueError(
            "mask_index must be root-sorted for deterministic segment reduction"
        )
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(graph, name)
        if mask.dtype != torch.bool or mask.shape != (n,):
            raise ValueError(f"{name} must be a Boolean node mask")
        if torch.any(graph.y[mask] < 0):
            raise ValueError(f"{name} selects unlabeled nodes")
    if torch.any(graph.train_mask & graph.val_mask) or torch.any(
        graph.train_mask & graph.test_mask
    ) or torch.any(graph.val_mask & graph.test_mask):
        raise ValueError("train, validation, and test masks overlap")


def load_imdb_graph(
    shared_path: str | Path,
    variant_path: str | Path,
    *,
    reverse_paths: bool = True,
    validate: bool = True,
    shared_payload: dict | None = None,
) -> PainGraph:
    """Load one topology variant together with the shared IMDb contract."""
    shared_path, variant_path = Path(shared_path), Path(variant_path)
    shared = _torch_load(shared_path) if shared_payload is None else shared_payload
    variant = _torch_load(variant_path)
    required_shared = {
        "x", "y", "node_type", "train_mask", "val_mask", "test_mask", "meta"
    }
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
    graph = PainGraph(
        x=shared["x"],
        y=shared["y"],
        node_type=shared["node_type"],
        train_mask=shared["train_mask"],
        val_mask=shared["val_mask"],
        test_mask=shared["test_mask"],
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
    if shared["meta"].get("num_nodes") != variant["meta"].get("num_nodes"):
        raise ValueError("Shared and variant artifacts have different node counts")
    if validate:
        validate_graph(graph)
    return graph

