from pathlib import Path

import torch

from experiments.node_classification.imdb_augmentation import (
    parse_variants,
    prepare_graphs,
)


def _save_shared(path: Path) -> None:
    torch.save(
        {
            "x": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "y": torch.tensor([0, 1, -1]),
            "node_type": torch.tensor([0, 0, 1]),
            "train_mask": torch.tensor([True, False, False]),
            "val_mask": torch.tensor([False, True, False]),
            "test_mask": torch.tensor([False, False, False]),
            "meta": {
                "num_nodes": 3,
                "node_type_names": ["movie", "entity"],
            },
        },
        path,
    )


def _save_variant(path: Path, variant: str, edge_type: int) -> None:
    torch.save(
        {
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
            "edge_type": torch.full((2,), edge_type, dtype=torch.long),
            "path_index": torch.tensor([[0, 1, 2]]),
            "path_lengths": torch.ones(3, dtype=torch.long),
            "mask_index": torch.tensor([0, 1, 2]),
            "path_edge_idx": torch.full((1, 3), -1, dtype=torch.long),
            "neighbor_mask": torch.zeros(3, 1),
            "distances": torch.zeros(3, 1, dtype=torch.long),
            "meta": {
                "variant": variant,
                "num_nodes": 3,
                "path_length": 3,
                "edge_type_names": ["r0", "r1", "r2", "r3", "r4"],
            },
        },
        path,
    )


def test_variant_aliases_are_canonicalized():
    assert parse_variants(["IMDb1", "v2", "IMDB3", "v4"]) == [
        "v1",
        "v2",
        "v3",
        "v4",
    ]


def test_preflight_uses_one_shared_contract_and_global_relations(tmp_path):
    _save_shared(tmp_path / "shared.pt")
    _save_variant(tmp_path / "v1_L3.pt", "v1", 0)
    _save_variant(tmp_path / "v2_L3.pt", "v2", 4)
    config = {
        "data": {
            "preprocessed_dir": str(tmp_path),
            "shared_path": str(tmp_path / "shared.pt"),
        },
        "model": {"path_length": 3, "reverse_paths": False},
    }
    graphs, _ = prepare_graphs(config, ["v1", "v2"])
    assert graphs["v1"].x.data_ptr() == graphs["v2"].x.data_ptr()
    assert graphs["v1"].num_edge_types == graphs["v2"].num_edge_types == 5
