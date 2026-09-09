from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pain_nc.model import PathAggregator
from preprocessing.dblp_link_prediction import (
    allocate_long_path_budget,
    enumerate_pain_paths,
    iter_root_paths,
    path_counts_by_root,
    stable_sample_ranks,
)


def toy_graph() -> tuple[list[list[int]], dict[tuple[int, int], int]]:
    adjacency = [
        [1, 2],
        [0, 2, 3],
        [0, 1, 3],
        [1, 2],
    ]
    directed = [
        (source, target)
        for source, neighbors in enumerate(adjacency)
        for target in neighbors
    ]
    return adjacency, {edge: index for index, edge in enumerate(directed)}


def decoded_paths(payload: dict[str, torch.Tensor]) -> list[tuple[int, ...]]:
    return [
        tuple(int(value) for value in payload["path_index"][:length, index])
        for index, length in enumerate(payload["path_lengths"].tolist())
    ]


def test_stable_rank_sampler_handles_huge_populations_without_population_storage() -> None:
    ranks = stable_sample_ranks(2_892_530_930, 256, seed=1234)
    assert len(ranks) == len(set(ranks)) == 256
    assert ranks == sorted(ranks)
    assert ranks[0] >= 0
    assert ranks[-1] < 2_892_530_930
    assert ranks == stable_sample_ranks(2_892_530_930, 256, seed=1234)


def test_budget_allocation_covers_lengths_and_respects_budget() -> None:
    assert allocate_long_path_budget(100, 900, 100) == (11, 89)
    assert allocate_long_path_budget(1, 100, 2) == (1, 1)
    assert allocate_long_path_budget(1, 2, 20) == (1, 2)
    assert sum(allocate_long_path_budget(100, 900, 101)) == 101


def test_direct_sampler_preserves_short_paths_and_weights_long_strata() -> None:
    adjacency, edge_lookup = toy_graph()
    root_counts = path_counts_by_root(adjacency)
    with patch(
        "preprocessing.dblp_link_prediction.iter_root_paths",
        side_effect=AssertionError("capped sampling must not enumerate all paths"),
    ):
        payload = enumerate_pain_paths(
            adjacency,
            edge_lookup,
            root_counts,
            max_paths_per_root=2,
            sampling_seed=99,
        )

    paths = decoded_paths(payload)
    assert len(paths) == len(set(paths))
    assert {(root,) for root in range(len(adjacency))}.issubset(paths)
    assert {
        (root, neighbor)
        for root, neighbors in enumerate(adjacency)
        for neighbor in neighbors
    }.issubset(paths)

    weights = payload["path_weights"].tolist()
    for root in range(len(adjacency)):
        for length in (3, 4):
            indices = [
                index
                for index, path in enumerate(paths)
                if path[0] == root and len(path) == length
            ]
            exact = int(root_counts[root, length - 1])
            sampled = len(indices)
            expected = float(exact) / float(sampled) if sampled else None
            for index in indices:
                assert np.isclose(weights[index], expected)
    assert all(
        np.isclose(weight, 1.0)
        for path, weight in zip(paths, weights, strict=True)
        if len(path) <= 2
    )
    for root in range(len(adjacency)):
        estimated_count = sum(
            weight
            for path, weight in zip(paths, weights, strict=True)
            if path[0] == root
        )
        assert np.isclose(estimated_count, root_counts[root].sum())


def test_full_budget_matches_exact_path_set() -> None:
    adjacency, edge_lookup = toy_graph()
    root_counts = path_counts_by_root(adjacency)
    payload = enumerate_pain_paths(
        adjacency,
        edge_lookup,
        root_counts,
        max_paths_per_root=100,
        sampling_seed=7,
    )
    expected = {
        path
        for root in range(len(adjacency))
        for path in iter_root_paths(root, adjacency)
    }
    assert set(decoded_paths(payload)) == expected
    assert torch.equal(payload["path_weights"], torch.ones_like(payload["path_weights"]))


def test_path_aggregator_applies_inverse_probability_weights() -> None:
    aggregator = PathAggregator(
        hidden_dim=2,
        path_length=1,
        lstm_depth=1,
        num_edge_types=1,
        mark_neighbors=False,
        shortest_path_encoding=False,
        aggregation="sum",
        dropout=0.0,
        chunk_size=10,
        checkpoint_chunks=False,
    )
    aggregator._encode_chunk = lambda *args: torch.ones(  # type: ignore[method-assign]
        (args[2].shape[1], 2), dtype=args[0].dtype, device=args[0].device
    )
    graph = SimpleNamespace(
        num_paths=3,
        num_nodes=2,
        edge_type=torch.zeros(1, dtype=torch.int8),
        path_index=torch.zeros((2, 3), dtype=torch.int32),
        path_lengths=torch.ones(3, dtype=torch.int8),
        path_edge_idx=torch.zeros((2, 3), dtype=torch.int32),
        neighbor_mask=torch.zeros((3, 2), dtype=torch.int8),
        distances=torch.zeros((3, 2), dtype=torch.int8),
        mask_index=torch.tensor([0, 0, 1], dtype=torch.int32),
        path_weights=torch.tensor([2.0, 3.0, 4.0]),
    )
    actual = aggregator(torch.zeros((2, 2)), graph)
    expected = torch.tensor([[5.0, 5.0], [4.0, 4.0]])
    assert torch.equal(actual, expected)

    aggregator.aggregation = "mean"
    normalized = aggregator(torch.zeros((2, 2)), graph)
    assert torch.equal(normalized, torch.ones((2, 2)))
