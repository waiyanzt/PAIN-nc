"""Structural contracts inherited from the authors' PAIN implementation."""

import torch
from torch import nn

from pain_nc.data import group_paths_by_root
from pain_nc.model import PainNodeClassifier, PathAggregator


def build_small_model(dropout: float = 0.0) -> PainNodeClassifier:
    return PainNodeClassifier(
        input_dim=4,
        num_classes=3,
        num_node_types=2,
        num_edge_types=5,
        hidden_dim=8,
        num_layers=1,
        lstm_depth=1,
        head_layers=2,
        path_length=2,
        dropout=dropout,
        share_lstm=True,
    )


def test_layer_mlp_matches_reference_block_shape():
    model = build_small_model()
    modules = list(model.layers[0].mlp)
    assert [type(module) for module in modules] == [
        nn.Linear,
        nn.BatchNorm1d,
        nn.ReLU,
        nn.Linear,
        nn.BatchNorm1d,
        nn.ReLU,
    ]


def test_prediction_head_matches_reference_operation_order():
    model = build_small_model(dropout=0.2)
    modules = list(model.classifier)
    assert [type(module) for module in modules] == [
        nn.Linear,
        nn.Dropout,
        nn.ReLU,
        nn.Linear,
    ]


def test_learned_position_parameter_matches_reference_shape():
    model = build_small_model()
    assert model.layers[0].aggregator.position_embedding.shape == (3, 1, 8)
    assert model.layers[0].aggregator.edge_embedding.num_embeddings == 5


def test_deterministic_segment_add_matches_index_add_on_cpu():
    aggregate = torch.zeros(4, 2)
    roots = torch.tensor([0, 0, 2, 2, 2, 3])
    messages = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    expected = aggregate.index_add(0, roots, messages)
    actual, counts = PathAggregator._deterministic_segment_add(
        aggregate, roots, messages
    )
    assert torch.equal(actual, expected)
    assert torch.equal(counts, torch.tensor([[2.0], [3.0], [1.0]]))


def test_legacy_length_grouped_paths_are_stably_grouped_by_root():
    payload = {
        "path_index": torch.tensor([[20, 10, 21, 11]]),
        "path_lengths": torch.tensor([2, 2, 1, 1]),
        "mask_index": torch.tensor([2, 1, 2, 1]),
        "path_edge_idx": torch.tensor([[200, 100, 201, 101]]),
        "neighbor_mask": torch.tensor([[20], [10], [21], [11]]),
        "distances": torch.tensor([[2], [1], [2], [1]]),
        "meta": {},
    }

    group_paths_by_root(payload)

    assert torch.equal(payload["mask_index"], torch.tensor([1, 1, 2, 2]))
    assert torch.equal(payload["path_index"], torch.tensor([[10, 11, 20, 21]]))
    assert torch.equal(payload["path_edge_idx"], torch.tensor([[100, 101, 200, 201]]))
    assert torch.equal(payload["neighbor_mask"].flatten(), torch.tensor([10, 11, 20, 21]))
    assert payload["meta"]["runtime_path_grouping"] == "root_stable"


def test_every_variant_uses_declared_relation_vocabulary():
    # This guards against physical variants silently constructing different
    # model shapes merely because some relation types are absent.
    from pain_nc.data import PainGraph

    graph = PainGraph(
        x=torch.zeros(2, 4),
        y=torch.tensor([0, 1]),
        node_type=torch.tensor([0, 1]),
        train_mask=torch.tensor([True, False]),
        val_mask=torch.tensor([False, True]),
        test_mask=torch.tensor([False, False]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_type=torch.zeros(2, dtype=torch.long),
        path_index=torch.tensor([[0, 1]]),
        path_lengths=torch.ones(2, dtype=torch.long),
        mask_index=torch.tensor([0, 1]),
        path_edge_idx=torch.full((1, 2), -1),
        neighbor_mask=torch.zeros(2, 1),
        distances=torch.zeros(2, 1, dtype=torch.long),
        shared_meta={"node_type_names": ["a", "b", "c"]},
        variant_meta={"edge_type_names": ["r0", "r1", "r2", "r3", "r4"]},
    )
    assert graph.num_node_types == 3
    assert graph.num_edge_types == 5
