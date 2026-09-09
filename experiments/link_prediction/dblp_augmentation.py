"""Joint graph-variant data augmentation for DBLP PAIN link prediction."""
from __future__ import annotations

import argparse
import copy
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

from experiments.link_prediction.train_dblp import (
    build_model,
    pointwise_pairwise_loss,
    test_predictions,
    training_scores,
    validation_statistics,
)
from experiments.node_classification.train import resolve_device, set_seed
from pain_nc.config import load_config, merged_config
from pain_nc.experiment import (
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    capture_rng_state,
    cpu_state_dict,
    restore_rng_state,
)
from pain_nc.link_data import PainLinkGraph, load_dblp_link_graph
from pain_nc.link_prediction import known_true_tails
from pain_nc.telemetry import (
    PeakRSSMonitor,
    artifact_sizes,
    cuda_memory_stats,
    environment_metadata,
    merge_cuda_memory_stats,
    model_memory_bytes,
    reset_cuda_peak,
    validate_resource_metrics,
)


VARIANTS = ("v1", "v2", "v3")
DEFAULT_SEEDS = (1566911444, 20241017, 20251017)


def query_macro_kendall(
    scores_a: np.ndarray, scores_b: np.ndarray
) -> tuple[float, int]:
    values = []
    for row_a, row_b in zip(scores_a, scores_b, strict=True):
        if np.array_equal(row_a, row_b):
            values.append(1.0)
            continue
        value = kendalltau(
            row_a, row_b, variant="b", nan_policy="omit"
        ).statistic
        if not np.isnan(value):
            values.append(float(value))
    return (
        float(np.mean(values)) if values else float("nan"),
        len(values),
    )


def flat_kendall(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    if np.array_equal(scores_a, scores_b):
        return 1.0
    return float(
        kendalltau(
            scores_a.ravel(), scores_b.ravel(),
            variant="b", nan_policy="omit",
        ).statistic
    )


def _canonical_config(config: Mapping[str, Any]) -> str:
    return json.dumps(dict(config), sort_keys=True, separators=(",", ":"), default=str)


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def prepare_graphs(
    config: Mapping[str, Any], device: torch.device
) -> tuple[dict[str, PainLinkGraph], dict[str, torch.Tensor], dict[str, Path]]:
    data_dir = Path(config["data"]["preprocessed_dir"])
    shared_path = Path(config["data"]["shared_path"])
    artifact_tag = str(config["data"].get("artifact_tag", "L3"))
    paths = {
        variant: data_dir / f"{variant}_{artifact_tag}.pt"
        for variant in VARIANTS
    }
    missing = [path for path in (shared_path, *paths.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing DBLP augmentation artifacts:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    reverse = bool(config["model"].get("reverse_paths", True))
    cpu_graphs = {}
    reference_splits = None
    for variant, path in paths.items():
        graph, splits = load_dblp_link_graph(
            shared_path, path, reverse_paths=reverse
        )
        cpu_graphs[variant] = graph
        if reference_splits is None:
            reference_splits = splits
        else:
            for name in reference_splits:
                if not torch.equal(reference_splits[name], splits[name]):
                    raise ValueError(f"Shared supervision differs for {variant}: {name}")
    reference = cpu_graphs[VARIANTS[0]]
    for variant, graph in cpu_graphs.items():
        if (
            graph.num_nodes != reference.num_nodes
            or graph.num_node_types != reference.num_node_types
            or graph.num_edge_types != reference.num_edge_types
        ):
            raise ValueError(f"Graph vocabulary differs for {variant}")
    move_paths = bool(config["model"].get("paths_on_device", False))
    graphs = {
        variant: graph.to(device, move_paths=move_paths)
        for variant, graph in cpu_graphs.items()
    }
    splits = {
        name: value.to(device, non_blocking=True)
        for name, value in reference_splits.items()
    }
    return graphs, splits, {"shared": shared_path, **paths}


def pairwise_rows(outputs: dict[str, dict[str, torch.Tensor]]) -> list[dict]:
    rows = []
    for left_name, right_name in combinations(VARIANTS, 2):
        left, right = outputs[left_name], outputs[right_name]
        for field in ("candidate_ids", "test_queries", "test_positive_tails"):
            if not torch.equal(left[field], right[field]):
                raise ValueError(f"Augmentation outputs are misaligned on {field}")
        left_scores = left["candidate_scores"].numpy()
        right_scores = right["candidate_scores"].numpy()
        difference = left_scores - right_scores
        tau, valid_queries = query_macro_kendall(left_scores, right_scores)
        rows.append(
            {
                "variant_a": left_name,
                "variant_b": right_name,
                "kendall_tau": tau,
                "kendall_tau_flat_candidate_scores": flat_kendall(
                    left_scores, right_scores
                ),
                "num_test_queries": int(left_scores.shape[0]),
                "valid_query_taus": valid_queries,
                "max_abs_score_diff": float(np.max(np.abs(difference))),
                "mean_abs_score_diff": float(np.mean(np.abs(difference))),
                "candidate_scores_exact": bool(
                    np.array_equal(left_scores, right_scores)
                ),
            }
        )
    return rows


def run_seed(
    config: dict[str, Any],
    seed: int,
    output_root: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    rss_monitor = PeakRSSMonitor().start()
    set_seed(seed, bool(config["training"].get("deterministic", True)))
    device = resolve_device(str(config["device"]))
    graphs, splits, input_paths = prepare_graphs(config, device)
    cpu_truth = {
        name: value.detach().cpu() for name, value in splits.items()
    }
    truth = known_true_tails(cpu_truth)
    model = build_model(graphs[VARIANTS[0]], config).to(device)
    training = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    target_steps = int(training.get("optimizer_steps", training["epochs"]))
    patience = int(training.get("early_stopping_patience", target_steps))
    gradient_clip = float(training.get("gradient_clip_norm", 0.0))
    regularization = float(training.get("embedding_regularization", 0.0))
    max_hours = float(training.get("max_hours", 0.0))
    candidates = splits["conference_candidates"]
    hits_k = tuple(int(k) for k in config["evaluation"]["hits_k"])
    threshold = float(config["evaluation"].get("threshold", 0.5))

    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    best_path = seed_dir / "best_checkpoint.pt"
    state_path = seed_dir / "latest_training_state.pt"
    rng = np.random.RandomState(seed)
    run_config = {
        "seed": seed,
        "variants": list(VARIANTS),
        "config": copy.deepcopy(config),
        "input_paths": {name: str(path.resolve()) for name, path in input_paths.items()},
    }
    history = []
    completed_steps = completed_super_epochs = 0
    best_mrr = -1.0
    best_loss = float("inf")
    no_improvement = 0
    prior_seconds = 0.0
    prior_peak_rss = 0
    prior_gpu = {}
    if state_path.exists():
        if not resume:
            raise RuntimeError(f"{state_path} exists; pass --resume or use a new output root")
        state = torch.load(state_path, map_location=device, weights_only=False)
        if _canonical_config(state["run_config"]) != _canonical_config(run_config):
            raise ValueError("Resume configuration differs from the saved run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        _optimizer_to_device(optimizer, device)
        restore_rng_state(state["rng_state"], rng)
        history = list(state["history"])
        completed_steps = int(state["completed_steps"])
        completed_super_epochs = int(state["completed_super_epochs"])
        best_mrr = float(state["best_mrr"])
        best_loss = float(state["best_loss"])
        no_improvement = int(state["no_improvement"])
        prior_seconds = float(state["training_seconds"])
        prior_peak_rss = int(state.get("process_peak_rss_bytes", 0))
        prior_gpu = dict(state.get("training_gpu", {}))
    elif resume:
        print(f"[resume] No state at {state_path}; starting a new run")

    reset_cuda_peak(device)
    started = time.monotonic()
    while completed_steps < target_steps and no_improvement < patience:
        order = [VARIANTS[index] for index in rng.permutation(len(VARIANTS))]
        train_losses = {}
        visited = []
        for variant in order:
            if completed_steps >= target_steps:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            embeddings = model.node_embeddings(graphs[variant])
            positive, negative = training_scores(
                embeddings, splits["train_pos"], splits["train_neg_tails"]
            )
            loss = pointwise_pairwise_loss(positive, negative)
            if regularization:
                loss = loss + regularization * embeddings.square().mean()
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            completed_steps += 1
            visited.append(variant)
            train_losses[variant] = float(loss.detach())
        completed_super_epochs += 1

        val_losses, val_mrrs = {}, {}
        for variant in VARIANTS:
            val_loss, val_mrr = validation_statistics(
                model, graphs[variant], splits["val_pos"], candidates, truth
            )
            val_losses[variant] = val_loss
            val_mrrs[variant] = val_mrr
        mean_loss = float(np.mean(list(val_losses.values())))
        mean_mrr = float(np.mean(list(val_mrrs.values())))
        improved = mean_mrr > best_mrr + 1e-12 or (
            abs(mean_mrr - best_mrr) <= 1e-12 and mean_loss < best_loss
        )
        if improved:
            best_mrr, best_loss, no_improvement = mean_mrr, mean_loss, 0
            atomic_torch_save(
                {
                    "model": cpu_state_dict(model),
                    "best_super_epoch": completed_super_epochs,
                    "optimizer_steps": completed_steps,
                },
                best_path,
            )
        else:
            no_improvement += 1
        row = {
            "super_epoch": completed_super_epochs,
            "optimizer_steps": completed_steps,
            "variant_order": ",".join(visited),
            "mean_train_loss": float(np.mean(list(train_losses.values()))),
            "mean_val_loss": mean_loss,
            "mean_val_mrr": mean_mrr,
            "best_mean_val_mrr": best_mrr,
        }
        for variant in VARIANTS:
            row[f"val_loss_{variant}"] = val_losses[variant]
            row[f"val_mrr_{variant}"] = val_mrrs[variant]
            row[f"train_loss_{variant}"] = train_losses.get(variant, float("nan"))
        history.append(row)
        atomic_write_csv(pd.DataFrame(history), seed_dir / "training_history.csv")
        segment_seconds = time.monotonic() - started
        gpu = merge_cuda_memory_stats(prior_gpu, cuda_memory_stats(device))
        peak_rss = max(prior_peak_rss, rss_monitor.peak_bytes)
        atomic_torch_save(
            {
                "state_version": 1,
                "run_config": run_config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": capture_rng_state(rng),
                "history": history,
                "completed_steps": completed_steps,
                "completed_super_epochs": completed_super_epochs,
                "best_mrr": best_mrr,
                "best_loss": best_loss,
                "no_improvement": no_improvement,
                "training_seconds": prior_seconds + segment_seconds,
                "process_peak_rss_bytes": peak_rss,
                "training_gpu": gpu,
            },
            state_path,
        )
        print(
            f"seed={seed} super_epoch={completed_super_epochs:04d} "
            f"steps={completed_steps}/{target_steps} val_mrr={mean_mrr:.6f} "
            f"order={','.join(visited)}",
            flush=True,
        )
        if max_hours > 0 and prior_seconds + segment_seconds >= max_hours * 3600:
            print(f"Stopping at configured max_hours={max_hours:g}", flush=True)
            break

    training_seconds = prior_seconds + (time.monotonic() - started)
    training_gpu = merge_cuda_memory_stats(prior_gpu, cuda_memory_stats(device))
    peak_rss = max(prior_peak_rss, rss_monitor.peak_bytes)
    if not best_path.is_file():
        raise RuntimeError("No validation-selected augmentation checkpoint was saved")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    reset_cuda_peak(device)
    outputs = {}
    per_variant_metrics = {}
    for variant in VARIANTS:
        prediction = test_predictions(
            model,
            graphs[variant],
            splits["test_pos"],
            candidates,
            truth,
            hits_k,
            threshold,
        )
        outputs[variant] = {
            name: prediction[name]
            for name in (
                "candidate_ids", "test_queries", "test_positive_tails",
                "candidate_scores",
            )
        }
        per_variant_metrics[variant] = {
            name: value for name, value in prediction.items()
            if isinstance(value, float)
        }
        atomic_torch_save(prediction, seed_dir / f"test_{variant}.pt")
    inference_gpu = cuda_memory_stats(device)
    peak_rss = max(peak_rss, rss_monitor.stop())
    pairwise = pairwise_rows(outputs)
    atomic_write_csv(pd.DataFrame(pairwise), seed_dir / "pairwise_kendall_tau.csv")
    resources = {
        **model_memory_bytes(model),
        "checkpoint_bytes": int(best_path.stat().st_size),
        "process_peak_rss_bytes": peak_rss,
        "training_gpu": training_gpu,
        "inference_gpu": inference_gpu,
        "artifacts": artifact_sizes(input_paths),
        "environment": environment_metadata(device),
    }
    validate_resource_metrics(resources)
    mean_metrics = {
        name: float(np.mean([metrics[name] for metrics in per_variant_metrics.values()]))
        for name in next(iter(per_variant_metrics.values()))
    }
    summary = {
        "dataset": "DBLP",
        "model": "PAIN-LP",
        "protocol": "joint_physical_variant_augmentation",
        "seed": seed,
        "variants": list(VARIANTS),
        "selection_metric": "mean_validation_filtered_full_conference_MRR",
        "best_mean_val_mrr": best_mrr,
        "best_mean_val_loss": best_loss,
        "optimizer_steps": completed_steps,
        "target_optimizer_steps": target_steps,
        "super_epochs_ran": completed_super_epochs,
        "training_seconds": training_seconds,
        "mean_test_metrics": mean_metrics,
        "per_variant_test_metrics": per_variant_metrics,
        "pairwise_kendall_tau": pairwise,
        "resources": resources,
        "run_config": run_config,
    }
    atomic_write_json(summary, seed_dir / "summary.json")
    atomic_torch_save(summary, seed_dir / "summary.pt")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dblp_lp_augmentation.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device")
    parser.add_argument("--artifact-tag")
    parser.add_argument("--output-root", type=Path, default=Path("results/dblp_lp_augmentation"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    overrides = {}
    if args.device:
        overrides["device"] = args.device
    if args.artifact_tag:
        overrides["data.artifact_tag"] = args.artifact_tag
    if overrides:
        config = merged_config(config, overrides)
    if args.preflight_only:
        device = resolve_device(str(config["device"]))
        graphs, _splits, paths = prepare_graphs(config, device)
        print(
            f"[OK] DBLP PAIN augmentation: variants={','.join(VARIANTS)} "
            f"nodes={graphs[VARIANTS[0]].num_nodes:,}"
        )
        for name, path in paths.items():
            print(f"  {name}: {path} ({path.stat().st_size / 2**30:.2f} GiB)")
        return
    summaries = [
        run_seed(config, seed, args.output_root, resume=args.resume)
        for seed in args.seeds
    ]
    seed_rows = []
    kendall_rows = []
    for summary in summaries:
        seed_rows.append(
            {
                "seed": summary["seed"],
                "optimizer_steps": summary["optimizer_steps"],
                "super_epochs_ran": summary["super_epochs_ran"],
                "training_seconds": summary["training_seconds"],
                "best_mean_val_mrr": summary["best_mean_val_mrr"],
                **{
                    f"mean_{name}": value
                    for name, value in summary["mean_test_metrics"].items()
                },
            }
        )
        kendall_rows.extend(
            {"seed": summary["seed"], **row}
            for row in summary["pairwise_kendall_tau"]
        )
    atomic_write_csv(pd.DataFrame(seed_rows), args.output_root / "seed_summary.csv")
    kendall_frame = pd.DataFrame(kendall_rows)
    atomic_write_csv(
        kendall_frame, args.output_root / "kendall_tau_per_seed.csv"
    )
    aggregate_rows = []
    for (left, right), group in kendall_frame.groupby(
        ["variant_a", "variant_b"], sort=True
    ):
        row = {
            "variant_a": left,
            "variant_b": right,
            "seed_count": int(len(group)),
        }
        for metric in (
            "kendall_tau", "kendall_tau_flat_candidate_scores",
            "max_abs_score_diff", "mean_abs_score_diff",
        ):
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        aggregate_rows.append(row)
    atomic_write_csv(
        pd.DataFrame(aggregate_rows),
        args.output_root / "kendall_tau_summary.csv",
    )
    atomic_write_json({"runs": summaries}, args.output_root / "all_seed_summaries.json")
    print(f"[OK] Results written under {args.output_root}")

if __name__ == "__main__":
    main()
