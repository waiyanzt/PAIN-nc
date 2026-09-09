"""Train one PAIN run for DBLP paper-conference link prediction."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from pain_nc.config import load_config, merged_config
from pain_nc.experiment import atomic_torch_save, cpu_state_dict
from pain_nc.link_data import PainLinkGraph, load_dblp_link_graph
from pain_nc.link_prediction import (
    binary_metrics_from_full_candidates,
    candidate_scores,
    filtered_ranks,
    known_true_tails,
    ranking_metrics,
    score_pairs,
)
from pain_nc.model import PainLinkPredictor
from pain_nc.telemetry import (
    PeakRSSMonitor,
    artifact_sizes,
    cuda_memory_stats,
    environment_metadata,
    model_memory_bytes,
    reset_cuda_peak,
    serialized_torch_bytes,
    validate_resource_metrics,
)
from experiments.node_classification.train import resolve_device, set_seed


def build_model(
    graph: PainLinkGraph, config: dict[str, Any]
) -> PainLinkPredictor:
    model_config = dict(config["model"])
    model_config.pop("reverse_paths", None)
    model_config.pop("paths_on_device", None)
    return PainLinkPredictor(
        num_nodes=graph.num_nodes,
        num_node_types=graph.num_node_types,
        num_edge_types=graph.num_edge_types,
        **model_config,
    )


def pointwise_pairwise_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    """Equal-weight positive and negative log-sigmoid terms."""
    return -(
        F.logsigmoid(positive_scores).mean()
        + F.logsigmoid(-negative_scores).mean()
    )


def training_scores(
    embeddings: torch.Tensor,
    positives: torch.Tensor,
    negative_tails: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive_scores = score_pairs(embeddings, positives)
    heads = positives[:, 0:1].expand_as(negative_tails)
    negative_pairs = torch.stack((heads.reshape(-1), negative_tails.reshape(-1)), dim=1)
    negative_scores = score_pairs(embeddings, negative_pairs)
    return positive_scores, negative_scores


def candidate_loss(
    scores: torch.Tensor,
    positives: torch.Tensor,
    candidates: torch.Tensor,
    truth: dict[int, set[int]],
) -> torch.Tensor:
    candidate_list = [int(value) for value in candidates.detach().cpu().tolist()]
    positions = {node: index for index, node in enumerate(candidate_list)}
    positive_columns = torch.tensor(
        [positions[int(tail)] for _, tail in positives.detach().cpu().tolist()],
        dtype=torch.long,
        device=scores.device,
    )
    positive_scores = scores[
        torch.arange(scores.shape[0], device=scores.device), positive_columns
    ]
    negative_mask = torch.ones_like(scores, dtype=torch.bool)
    for row, (head, _tail) in enumerate(positives.detach().cpu().tolist()):
        for true_tail in truth.get(int(head), ()):
            column = positions.get(int(true_tail))
            if column is not None:
                negative_mask[row, column] = False
    return pointwise_pairwise_loss(positive_scores, scores[negative_mask])


@torch.no_grad()
def validation_statistics(
    model: PainLinkPredictor,
    graph: PainLinkGraph,
    positives: torch.Tensor,
    candidates: torch.Tensor,
    truth: dict[int, set[int]],
) -> tuple[float, float]:
    model.eval()
    embeddings = model.node_embeddings(graph)
    scores = candidate_scores(embeddings, positives, candidates)
    ranks = filtered_ranks(scores, positives, candidates, truth)
    return float(candidate_loss(scores, positives, candidates, truth)), float(
        np.mean(1.0 / ranks)
    )


@torch.no_grad()
def test_predictions(
    model: PainLinkPredictor,
    graph: PainLinkGraph,
    positives: torch.Tensor,
    candidates: torch.Tensor,
    truth: dict[int, set[int]],
    hits_k: tuple[int, ...],
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    embeddings = model.node_embeddings(graph)
    scores = candidate_scores(embeddings, positives, candidates)
    ranks = filtered_ranks(scores, positives, candidates, truth)
    return {
        "test_queries": positives[:, 0].detach().cpu(),
        "test_positive_tails": positives[:, 1].detach().cpu(),
        "candidate_ids": candidates.detach().cpu(),
        "candidate_scores": scores.detach().cpu(),
        "test_ranks": torch.from_numpy(ranks),
        **ranking_metrics(ranks, hits_k),
        **binary_metrics_from_full_candidates(
            scores, positives, candidates, truth, threshold
        ),
    }


def train_one_run(
    config: dict[str, Any],
    *,
    variant: str,
    variant_path: str | Path,
    seed: int,
    verbose: bool = True,
) -> dict[str, Any]:
    rss_monitor = PeakRSSMonitor().start()
    set_seed(seed, bool(config["training"].get("deterministic", True)))
    device = resolve_device(str(config["device"]))
    model_config = config["model"]
    graph, cpu_splits = load_dblp_link_graph(
        config["data"]["shared_path"],
        variant_path,
        reverse_paths=bool(model_config.get("reverse_paths", True)),
    )
    if int(graph.variant_meta["path_length"]) != int(model_config["path_length"]):
        raise ValueError(
            f"Artifact L={graph.variant_meta['path_length']} does not match "
            f"model L={model_config['path_length']}"
        )
    graph = graph.to(
        device, move_paths=bool(model_config.get("paths_on_device", False))
    )
    splits = {
        name: value.to(device, non_blocking=True)
        for name, value in cpu_splits.items()
    }
    truth = known_true_tails(cpu_splits)
    model = build_model(graph, config).to(device)
    training = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epochs = int(training["epochs"])
    patience = int(training.get("early_stopping_patience", epochs))
    max_hours = float(training.get("max_hours", 0.0))
    gradient_clip = float(training.get("gradient_clip_norm", 0.0))
    embedding_regularization = float(training.get("embedding_regularization", 0.0))
    candidates = splits["conference_candidates"]
    expected_candidates = int(config["evaluation"].get("num_candidates", len(candidates)))
    if len(candidates) != expected_candidates:
        raise ValueError(
            f"Config expects {expected_candidates} conference candidates, "
            f"but the shared artifact contains {len(candidates)}"
        )
    hits_k = tuple(int(k) for k in config["evaluation"].get("hits_k", (1, 3, 5, 10)))
    threshold = float(config["evaluation"].get("threshold", 0.5))

    best_state = None
    best_epoch = -1
    best_val_mrr = -1.0
    best_val_loss = float("inf")
    no_improvement = 0
    history = []
    started = time.monotonic()
    time_to_best = 0.0
    reset_cuda_peak(device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = model.node_embeddings(graph)
        positive_scores, negative_scores = training_scores(
            embeddings, splits["train_pos"], splits["train_neg_tails"]
        )
        loss = pointwise_pairwise_loss(positive_scores, negative_scores)
        if embedding_regularization:
            loss = loss + embedding_regularization * embeddings.square().mean()
        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        val_loss, val_mrr = validation_statistics(
            model, graph, splits["val_pos"], candidates, truth
        )
        elapsed = time.monotonic() - started
        row = {
            "epoch": epoch,
            "train_loss": float(loss.detach()),
            "val_loss": val_loss,
            "val_mrr": val_mrr,
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        improved = val_mrr > best_val_mrr + 1e-12 or (
            abs(val_mrr - best_val_mrr) <= 1e-12 and val_loss < best_val_loss
        )
        if improved:
            best_state = cpu_state_dict(model)
            best_epoch = epoch
            best_val_mrr = val_mrr
            best_val_loss = val_loss
            no_improvement = 0
            time_to_best = elapsed
        else:
            no_improvement += 1
        if verbose:
            print(
                f"{variant} seed={seed} epoch={epoch:04d} "
                f"train_loss={float(loss):.5f} val_loss={val_loss:.5f} "
                f"val_mrr={val_mrr:.5f} bad={no_improvement}",
                flush=True,
            )
        if no_improvement >= patience:
            if verbose:
                print(f"Early stopping after {no_improvement} unimproved epochs")
            break
        if max_hours > 0 and elapsed >= max_hours * 3600:
            if verbose:
                print(f"Stopping at configured max_hours={max_hours:g}")
            break

    if best_state is None:
        raise RuntimeError("Training completed without a validation checkpoint")
    training_gpu = cuda_memory_stats(device)
    model.load_state_dict(best_state)
    model.to(device)
    reset_cuda_peak(device)
    predictions = test_predictions(
        model, graph, splits["test_pos"], candidates, truth, hits_k, threshold
    )
    inference_gpu = cuda_memory_stats(device)
    elapsed = time.monotonic() - started
    peak_rss = rss_monitor.stop()
    resources = {
        **model_memory_bytes(model),
        "checkpoint_bytes": serialized_torch_bytes({"model": best_state}),
        "process_peak_rss_bytes": peak_rss,
        "training_gpu": training_gpu,
        "inference_gpu": inference_gpu,
        "artifacts": artifact_sizes(
            {"shared": config["data"]["shared_path"], "variant": variant_path}
        ),
        "environment": environment_metadata(device),
    }
    validate_resource_metrics(resources)
    return {
        "dataset": "DBLP",
        "task": "paper_conference_link_prediction",
        "model": "PAIN-LP",
        "variant": variant,
        "variant_source_name": graph.variant_meta.get("variant"),
        "message_program_sha256": graph.variant_meta.get("message_program_sha256"),
        "selected_path_program_sha256": graph.variant_meta.get(
            "selected_path_program_sha256"
        ),
        "selected_path_weights_sha256": graph.variant_meta.get(
            "selected_path_weights_sha256"
        ),
        "path_sampling": graph.variant_meta.get("path_sampling", "none"),
        "sampling_seed": graph.variant_meta.get("sampling_seed"),
        "sampled_long_paths_per_root": graph.variant_meta.get(
            "sampled_long_paths_per_root", 0
        ),
        "physical_graph_hashes": graph.variant_meta.get("physical_graph_hashes"),
        "semantic_program_hashes": graph.variant_meta.get(
            "semantic_program_hashes"
        ),
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "best_val_mrr": best_val_mrr,
        "best_val_loss": best_val_loss,
        "selection_metric": "validation_filtered_full_conference_MRR",
        "evaluation_protocol": "filtered_full_conference_ranking",
        "training_negatives_per_positive": int(splits["train_neg_tails"].shape[1]),
        "time_to_best_seconds": time_to_best,
        "elapsed_seconds": elapsed,
        "num_nodes": graph.num_nodes,
        "num_paths": graph.num_paths,
        "history": history,
        "config": copy.deepcopy(config),
        "model_state_dict": best_state,
        "resources": resources,
        **predictions,
    }


def save_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    validate_resource_metrics(artifact.get("resources", {}))
    atomic_torch_save(artifact, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dblp_lp.yaml")
    parser.add_argument("--variant", default="DBLP1")
    parser.add_argument(
        "--variant-path",
        default="data/preprocessed/DBLP/v1_L3_stratcap256.pt",
    )
    parser.add_argument("--seed", type=int, default=1566911444)
    parser.add_argument("--device")
    parser.add_argument("--output", default="results/dblp_lp/DBLP1/seed1566911444.pt")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config = merged_config(config, {"device": args.device})
    artifact = train_one_run(
        config,
        variant=args.variant,
        variant_path=args.variant_path,
        seed=args.seed,
    )
    save_artifact(artifact, args.output)
    metrics = {
        key: value
        for key, value in artifact.items()
        if key.startswith("test_") and isinstance(value, float)
    }
    print(json.dumps(metrics, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
