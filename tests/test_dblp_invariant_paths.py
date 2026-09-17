"""Small exhaustive checks for DBLP conditional PAIN path construction."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from pain_nc.model import PathAggregator
from preprocessing.dblp_link_prediction import (
    EDGE_TYPE_NAMES,
    add_edge,
    compile_invariant_edges,
    enumerate_pain_paths,
    graph_tensors,
    path_counts_by_root,
    tensor_hash,
)


NUM_NODES = 8
P1, P2, A1, A2, VENUE, AREA1, AREA2, TERM = range(NUM_NODES)
CERTIFICATE = {P1: AREA1, P2: AREA2}
PATH_FIELDS = (
    "path_index", "path_lengths", "mask_index", "path_edge_idx",
    "neighbor_mask", "distances", "path_weights",
)


def physical_variants() -> dict[str, dict[str, set[tuple[int, int]]]]:
    base: dict[str, set[tuple[int, int]]] = {}
    for paper, author in ((P1, A1), (P2, A2)):
        add_edge(base, "author-paper", paper, author)
        add_edge(base, "paper-conference", paper, VENUE)
        add_edge(base, "paper-term", paper, TERM)
    variants = {
        name: {relation: set(edges) for relation, edges in base.items()}
        for name in ("v1", "v2", "v3")
    }
    for paper, area in CERTIFICATE.items():
        add_edge(variants["v1"], "paper-area", paper, area)
        add_edge(variants["v2"], "conference-area", VENUE, area)
    for author, area in ((A1, AREA1), (A2, AREA2)):
        add_edge(variants["v3"], "author-area", author, area)
    return variants


def path_payload(edges: dict[str, set[tuple[int, int]]]) -> tuple[torch.Tensor, dict]:
    _edge_index, edge_type, adjacency, lookup = graph_tensors(edges, NUM_NODES)
    counts = path_counts_by_root(adjacency)
    paths = enumerate_pain_paths(adjacency, lookup, counts, 4, 1566911444)
    return edge_type, paths


def test_each_physical_graph_independently_produces_the_same_pain_paths() -> None:
    physical = physical_variants()
    raw_path_hashes = set()
    compiled_path_hashes = set()
    compiled_edges = []
    for variant, graph in physical.items():
        _raw_types, raw_paths = path_payload(graph)
        raw_path_hashes.add(tensor_hash(raw_paths[field] for field in PATH_FIELDS))

        semantic, audit = compile_invariant_edges(
            graph, variant, paper_area_certificate=CERTIFICATE
        )
        assert audit["context_matches"] == 2
        compiled_edges.append(semantic)
        edge_type, paths = path_payload(semantic)
        compiled_path_hashes.add(
            tensor_hash((edge_type, *(paths[field] for field in PATH_FIELDS)))
        )
        if variant == "v2":
            assert audit["ambiguous_raw_contexts"] == 2
            assert audit["raw_candidates_rejected_by_pair_filter"] == 2

    assert len(raw_path_hashes) == 3
    assert len(compiled_path_hashes) == 1
    assert compiled_edges[0] == compiled_edges[1] == compiled_edges[2]

    # The emitted semantic edges, including virtual ones, are precisely the
    # training-scope closure licensed by these three physical realizations.
    for paper, area, author in ((P1, AREA1, A1), (P2, AREA2, A2)):
        assert (paper, area) in compiled_edges[0]["paper-area"]
        assert (author, area) in compiled_edges[0]["author-area"]
        assert (VENUE, area) in compiled_edges[0]["conference-area"]


def test_missing_physical_context_cannot_be_supplied_by_certificate() -> None:
    physical = physical_variants()
    physical["v2"]["conference-area"].remove((VENUE, AREA1))
    with pytest.raises(RuntimeError, match="does not license"):
        compile_invariant_edges(
            physical["v2"], "v2", paper_area_certificate=CERTIFICATE
        )


def test_union_graph_cannot_masquerade_as_a_physical_variant() -> None:
    physical = physical_variants()
    union = {
        relation: set().union(
            *(graph.get(relation, set()) for graph in physical.values())
        )
        for relation in (
            "author-paper", "paper-term", "paper-conference",
            "paper-area", "conference-area", "author-area",
        )
    }
    with pytest.raises(RuntimeError, match="unexpected conference-area"):
        compile_invariant_edges(
            union, "v1", paper_area_certificate=CERTIFICATE
        )


def test_pain_encoder_receives_equal_sequences_after_projection() -> None:
    physical = physical_variants()
    torch.manual_seed(7)
    encoder = PathAggregator(
        hidden_dim=4,
        path_length=3,
        lstm_depth=1,
        num_edge_types=len(EDGE_TYPE_NAMES),
        mark_neighbors=True,
        shortest_path_encoding=False,
        aggregation="sum",
        dropout=0.0,
        chunk_size=32,
        checkpoint_chunks=False,
    ).eval()
    features = torch.randn(NUM_NODES, 4)
    outputs = []
    for variant, graph in physical.items():
        semantic, _audit = compile_invariant_edges(
            graph, variant, paper_area_certificate=CERTIFICATE
        )
        edge_type, paths = path_payload(semantic)
        model_graph = type("Graph", (), {
            "num_nodes": NUM_NODES,
            "num_paths": paths["path_lengths"].numel(),
            "edge_type": edge_type,
            **paths,
        })()
        outputs.append(encoder(features, model_graph))
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[1], outputs[2])


def test_invariant_training_config_differs_only_in_input_directory() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    ordinary = yaml.safe_load((configs / "dblp_lp.yaml").read_text())
    invariant = yaml.safe_load((configs / "dblp_lp_invariant.yaml").read_text())
    invariant["data"] = ordinary["data"]
    assert invariant == ordinary
