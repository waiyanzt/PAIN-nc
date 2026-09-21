"""Exhaustive checks for the four-variant IMDb invariant PAIN compiler."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from pain_nc.data import PATH_FIELDS
from preprocessing.imdb_invariant import tensor_hash
from preprocessing.imdb_node_classification import (
    base_variant_edges,
    canonical_edge,
    compile_invariant_edges,
    enumerate_pain_paths,
    graph_tensors,
)


M1, M2, D1, D2, A1, A2, L1, L2 = range(8)
NODE_TYPES = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)


def physical_variants() -> dict[str, set[tuple[int, int]]]:
    variants = {name: set() for name in ("v1", "v2", "v3", "v4")}
    for movie, link, director, actor in (
        (M1, L1, D1, A1),
        (M2, L2, D2, A2),
    ):
        for graph in variants.values():
            graph.add(canonical_edge(movie, link))
        variants["v1"].update(
            (canonical_edge(movie, director), canonical_edge(movie, actor))
        )
        variants["v2"].update(
            (canonical_edge(link, director), canonical_edge(link, actor))
        )
        variants["v3"].update(
            (canonical_edge(movie, director), canonical_edge(link, actor))
        )
        variants["v4"].update(
            (canonical_edge(link, director), canonical_edge(movie, actor))
        )
    return variants


def path_hash(edges: set[tuple[int, int]]) -> str:
    edge_index, edge_type, adjacency, lookup = graph_tensors(edges, NODE_TYPES)
    paths, _counts = enumerate_pain_paths(adjacency, lookup)
    return tensor_hash(
        (edge_index, edge_type, *(paths[name] for name in PATH_FIELDS))
    )


def test_four_physical_variants_compile_to_identical_exact_pain_paths() -> None:
    compiled = []
    raw_hashes = set()
    compiled_hashes = set()
    for variant, physical in physical_variants().items():
        raw_hashes.add(path_hash(physical))
        semantic, audit = compile_invariant_edges(
            physical, variant, NODE_TYPES
        )
        assert audit["movie_link_contexts"] == 2
        assert audit["director_contexts"] == 2
        assert audit["actor_contexts"] == 2
        compiled.append(semantic)
        compiled_hashes.add(path_hash(semantic))

    assert len(raw_hashes) == 4
    assert len(compiled_hashes) == 1
    assert compiled[0] == compiled[1] == compiled[2] == compiled[3]


def test_missing_imdb4_actor_witness_breaks_cross_variant_equality() -> None:
    physical = physical_variants()
    physical["v4"].remove(canonical_edge(M1, A1))
    with pytest.raises(RuntimeError, match="no Actor witness"):
        compile_invariant_edges(physical["v4"], "v4", NODE_TYPES)


def test_physical_imdb4_keeps_actor1_and_all_other_actors() -> None:
    movies = pd.DataFrame(
        [
            {
                "director_name": "director",
                "movie_imdb_link": "link",
                "actor_1_name": "actor1",
                "actor_2_name": "actor2",
                "actor_3_name": "actor3",
            }
        ]
    )
    maps = {
        "movie": {0: 0},
        "director": {"director": 1},
        "actor": {"actor1": 2, "actor2": 3, "actor3": 4},
        "imdb_link": {"link": 5},
    }
    edges = base_variant_edges(movies, maps)["v4"]
    assert canonical_edge(0, 2) in edges
    assert canonical_edge(0, 3) in edges
    assert canonical_edge(0, 4) in edges


def test_invariant_training_config_differs_only_in_input_directory() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    ordinary = yaml.safe_load((configs / "imdb_nc.yaml").read_text())
    invariant = yaml.safe_load(
        (configs / "imdb_nc_invariant.yaml").read_text()
    )
    invariant["data"] = ordinary["data"]
    assert invariant == ordinary
