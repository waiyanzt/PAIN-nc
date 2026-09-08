"""Joint IMDb graph-variant data augmentation for PAIN node classification."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from experiments.node_classification.train import build_model, resolve_device, set_seed
from pain_nc.config import load_config, merged_config
from pain_nc.data import PainGraph, load_imdb_graph
from pain_nc.experiment import (
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    capture_rng_state,
    classification_invariance_rows,
    classification_metrics,
    cpu_state_dict,
    mean_dict,
    restore_rng_state,
)
from pain_nc.telemetry import (
    artifact_sizes,
    cuda_memory_stats,
    environment_metadata,
    merge_cuda_memory_stats,
    model_memory_bytes,
    PeakRSSMonitor,
    reset_cuda_peak,
    validate_resource_metrics,
)


VARIANT_FILES = {
    "v1": "v1_L3.pt",
    "v2": "v2_L3.pt",
    "v3": "v3_L3.pt",
    "v4": "v4_L3.pt",
}
VARIANT_ALIASES = {
    **{name: name for name in VARIANT_FILES},
    **{f"imdb{index}": f"v{index}" for index in range(1, 5)},
}
SHARED_FIELDS = (
    "x",
    "y",
    "node_type",
    "train_mask",
    "val_mask",
    "test_mask",
)


def parse_variants(values: list[str]) -> list[str]:
    variants = []
    for value in values:
        canonical = VARIANT_ALIASES.get(value.lower())
        if canonical is None:
            raise ValueError(f"Unknown IMDb variant {value!r}")
        variants.append(canonical)
    if len(variants) != len(set(variants)):
        raise ValueError("Duplicate IMDb variants are not allowed")
    if len(variants) < 2:
        raise ValueError("Data augmentation requires at least two variants")
    return variants


def _torch_load(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def prepare_graphs(
    config: Mapping[str, Any], variants: list[str]
) -> tuple[dict[str, PainGraph], dict[str, Path]]:
    data_dir = Path(config["data"]["preprocessed_dir"])
    shared_path = Path(config["data"]["shared_path"])
    paths = {variant: data_dir / VARIANT_FILES[variant] for variant in variants}
    missing = [path for path in (shared_path, *paths.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing PAIN IMDb artifacts:\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\nRun: python -m preprocessing.imdb_node_classification"
        )

    reverse_paths = bool(config["model"].get("reverse_paths", True))
    shared_payload = _torch_load(shared_path)
    graphs = {
        variant: load_imdb_graph(
            shared_path,
            path,
            reverse_paths=reverse_paths,
            shared_payload=shared_payload,
        )
        for variant, path in paths.items()
    }
    reference_name = variants[0]
    reference = graphs[reference_name]
    expected_path_length = int(config["model"]["path_length"])
    for variant, graph in graphs.items():
        if graph.num_nodes != reference.num_nodes:
            raise ValueError(f"Node count differs for {variant}")
        if graph.num_features != reference.num_features:
            raise ValueError(f"Feature count differs for {variant}")
        if graph.num_classes != reference.num_classes:
            raise ValueError(f"Class count differs for {variant}")
        if graph.num_node_types != reference.num_node_types:
            raise ValueError(f"Node-type vocabulary differs for {variant}")
        if graph.num_edge_types != reference.num_edge_types:
            raise ValueError(f"Edge-type vocabulary differs for {variant}")
        if int(graph.variant_meta["path_length"]) != expected_path_length:
            raise ValueError(
                f"{variant} path length does not match the model configuration"
            )
        for field in SHARED_FIELDS:
            if not torch.equal(getattr(reference, field), getattr(graph, field)):
                raise ValueError(
                    f"Shared field {field} differs between {reference_name} and {variant}"
                )
    return graphs, {"shared": shared_path, **paths}


def move_graphs(
    graphs: Mapping[str, PainGraph],
    variants: list[str],
    device: torch.device,
    move_paths: bool,
) -> dict[str, PainGraph]:
    first = variants[0]
    moved = {first: graphs[first].to(device, move_paths=move_paths)}
    for variant in variants[1:]:
        moved[variant] = graphs[variant].to(
            device,
            move_paths=move_paths,
            shared_from=moved[first],
        )
    return moved


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    graph: PainGraph,
    mask_name: str,
) -> tuple[float, dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    logits = model(graph)
    mask = getattr(graph, f"{mask_name}_mask")
    indices = mask.nonzero(as_tuple=False).view(-1)
    selected = logits[indices]
    labels = graph.y[indices]
    loss = torch.nn.functional.cross_entropy(selected, labels)
    metrics = classification_metrics(selected, labels)
    return float(loss), metrics, selected.detach().cpu(), indices.detach().cpu()


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _canonical_config(config: Mapping[str, Any]) -> str:
    return json.dumps(dict(config), sort_keys=True, separators=(",", ":"), default=str)


def run_seed(
    config: dict[str, Any],
    variants: list[str],
    seed: int,
    output_root: Path,
    *,
    resume: bool,
    super_epochs_override: int | None,
    patience_override: int | None,
) -> dict[str, Any]:
    rss_monitor = PeakRSSMonitor().start()
    set_seed(seed, bool(config["training"].get("deterministic", True)))
    device = resolve_device(str(config["device"]))
    cpu_graphs, input_paths = prepare_graphs(config, variants)
    graphs = move_graphs(
        cpu_graphs,
        variants,
        device,
        move_paths=bool(config["model"].get("paths_on_device", True)),
    )
    del cpu_graphs
    model = build_model(graphs[variants[0]], config).to(device)
    training = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training.get("lr_factor", 0.5)),
        patience=int(training.get("lr_patience", 20)),
        min_lr=float(training.get("min_learning_rate", 1e-5)),
    )
    super_epochs = int(
        super_epochs_override
        if super_epochs_override is not None
        else (
            training["super_epochs"]
            if "super_epochs" in training
            else training["epochs"]
        )
    )
    patience = int(
        patience_override
        if patience_override is not None
        else training.get("early_stopping_patience", super_epochs)
    )
    if super_epochs < 1 or patience < 1:
        raise ValueError("super-epochs and patience must be positive")
    gradient_clip = float(training.get("gradient_clip_norm", 0.0))
    max_hours = float(training.get("max_hours", 0.0))

    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "shared_checkpoint.pt"
    state_path = seed_dir / "latest_training_state.pt"
    history_path = seed_dir / "training_history.csv"
    rng = np.random.RandomState(seed)
    run_config = {
        "dataset": "IMDB",
        "model": "PAIN-NC",
        "protocol": "joint_variant_augmentation",
        "seed": seed,
        "variants": variants,
        "model_config": copy.deepcopy(config["model"]),
        "training_config": copy.deepcopy(training),
        "patience": patience,
        "input_paths": {key: str(path.resolve()) for key, path in input_paths.items()},
    }

    history: list[dict[str, Any]] = []
    completed = optimizer_steps = variant_epochs = 0
    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    no_improvement = 0
    prior_seconds = 0.0
    prior_peak_rss = 0
    prior_training_gpu: dict[str, int] = {}
    if state_path.exists():
        if not resume:
            raise RuntimeError(
                f"{state_path} already exists; pass --resume or choose a new output root"
            )
        state = _torch_load(state_path, map_location=device)
        if _canonical_config(state["run_config"]) != _canonical_config(run_config):
            raise ValueError("Resume configuration differs from the saved run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        _optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(state["rng_state"], rng)
        history = list(state["history"])
        completed = int(state["completed_super_epoch"])
        optimizer_steps = int(state["optimizer_steps"])
        variant_epochs = int(state["variant_epochs"])
        best_macro_f1 = float(state["best_macro_f1"])
        best_val_loss = float(state["best_val_loss"])
        no_improvement = int(state["no_improvement"])
        prior_seconds = float(state["training_seconds"])
        prior_peak_rss = int(state.get("process_peak_rss_bytes", 0))
        prior_training_gpu = dict(state.get("training_gpu", {}))
    elif resume:
        print(f"[resume] No state at {state_path}; starting a new run.")
    elif checkpoint_path.exists() or history_path.exists():
        raise RuntimeError(
            f"Incomplete prior outputs exist under {seed_dir}; choose a new output root"
        )

    reset_cuda_peak(device)
    started = time.monotonic()
    for super_epoch in range(completed, super_epochs):
        if no_improvement >= patience:
            break
        order = [variants[index] for index in rng.permutation(len(variants))]
        train_losses = {}
        for variant in order:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            graph = graphs[variant]
            logits = model(graph)
            loss = torch.nn.functional.cross_entropy(
                logits[graph.train_mask], graph.y[graph.train_mask]
            )
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            optimizer_steps += 1
            variant_epochs += 1
            train_losses[variant] = float(loss.detach().cpu())
            del logits, loss

        validation_losses = {}
        validation_metrics = {}
        for variant in variants:
            loss_value, metrics, _, _ = evaluate(model, graphs[variant], "val")
            validation_losses[variant] = loss_value
            validation_metrics[variant] = metrics
        mean_val_macro_f1 = float(
            np.mean([validation_metrics[name]["Macro_F1"] for name in variants])
        )
        mean_val_loss = float(np.mean(list(validation_losses.values())))
        scheduler.step(mean_val_loss)
        completed = super_epoch + 1
        improved = mean_val_macro_f1 > best_macro_f1 + 1e-12 or (
            abs(mean_val_macro_f1 - best_macro_f1) <= 1e-12
            and mean_val_loss < best_val_loss
        )
        if improved:
            best_macro_f1 = mean_val_macro_f1
            best_val_loss = mean_val_loss
            no_improvement = 0
            atomic_torch_save(
                {
                    "model": cpu_state_dict(model),
                    "metadata": {
                        "seed": seed,
                        "variants": variants,
                        "best_super_epoch": completed,
                        "optimizer_steps": optimizer_steps,
                        "selection_metric": "mean_validation_Macro_F1",
                    },
                },
                checkpoint_path,
            )
        else:
            no_improvement += 1

        row: dict[str, Any] = {
            "super_epoch": completed,
            "variant_order": ",".join(order),
            "optimizer_steps_cumulative": optimizer_steps,
            "variant_epochs_cumulative": variant_epochs,
            "mean_train_loss": float(np.mean(list(train_losses.values()))),
            "mean_val_loss": mean_val_loss,
            "mean_val_macro_f1": mean_val_macro_f1,
            "best_mean_val_macro_f1": best_macro_f1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        for variant in variants:
            row[f"train_loss_{variant}"] = train_losses[variant]
            row[f"val_loss_{variant}"] = validation_losses[variant]
            row[f"val_macro_f1_{variant}"] = validation_metrics[variant]["Macro_F1"]
        history.append(row)
        atomic_write_csv(pd.DataFrame(history), history_path)

        segment_seconds = time.monotonic() - started
        training_gpu = merge_cuda_memory_stats(
            prior_training_gpu, cuda_memory_stats(device)
        )
        peak_rss = max(prior_peak_rss, rss_monitor.peak_bytes)
        atomic_torch_save(
            {
                "state_version": 1,
                "run_config": run_config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_state": capture_rng_state(rng),
                "history": history,
                "completed_super_epoch": completed,
                "optimizer_steps": optimizer_steps,
                "variant_epochs": variant_epochs,
                "best_macro_f1": best_macro_f1,
                "best_val_loss": best_val_loss,
                "no_improvement": no_improvement,
                "training_seconds": prior_seconds + segment_seconds,
                "process_peak_rss_bytes": peak_rss,
                "training_gpu": training_gpu,
            },
            state_path,
        )
        print(
            f"seed={seed} super_epoch={completed:04d} steps={optimizer_steps} "
            f"mean_val_macro_f1={mean_val_macro_f1:.6f} "
            f"best={best_macro_f1:.6f} order={','.join(order)}",
            flush=True,
        )
        if max_hours > 0 and prior_seconds + segment_seconds >= max_hours * 3600:
            print(f"Stopping at configured max_hours={max_hours:g}", flush=True)
            break

    training_seconds = prior_seconds + (time.monotonic() - started)
    training_gpu = merge_cuda_memory_stats(
        prior_training_gpu, cuda_memory_stats(device)
    )
    peak_rss = max(prior_peak_rss, rss_monitor.peak_bytes)
    if not checkpoint_path.is_file():
        raise RuntimeError("No validation-selected checkpoint was saved")
    checkpoint = _torch_load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    reset_cuda_peak(device)
    per_variant_metrics = {}
    outputs = {}
    for variant in variants:
        loss_value, metrics, logits, indices = evaluate(
            model, graphs[variant], "test"
        )
        probabilities = logits.softmax(dim=-1).numpy()
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        labels = graphs[variant].y[indices.to(device)].detach().cpu().numpy()
        metrics["CrossEntropy"] = loss_value
        metrics["num_paths"] = float(graphs[variant].num_paths)
        per_variant_metrics[variant] = metrics
        outputs[variant] = {
            "item_id": indices.numpy(),
            "label": labels,
            "logits": logits.numpy(),
            "prediction": predictions,
            "confidence": confidence,
        }
        frame = pd.DataFrame(
            {
                "node_id": indices.numpy(),
                "label": labels,
                "prediction": predictions,
                "confidence": confidence,
            }
        )
        for class_id in range(probabilities.shape[1]):
            frame[f"prob_class_{class_id}"] = probabilities[:, class_id]
            frame[f"logit_class_{class_id}"] = logits[:, class_id].numpy()
        atomic_write_csv(frame, seed_dir / f"test_scores_{variant}.csv")

    inference_gpu = cuda_memory_stats(device)
    peak_rss = max(peak_rss, rss_monitor.stop())
    pairwise = classification_invariance_rows(outputs)
    atomic_write_csv(pd.DataFrame(pairwise), seed_dir / "pairwise_invariance.csv")
    atomic_write_csv(
        pd.DataFrame(
            [
                {"variant": variant, **per_variant_metrics[variant]}
                for variant in variants
            ]
        ),
        seed_dir / "test_metrics_by_variant.csv",
    )
    resources = {
        **model_memory_bytes(model),
        "checkpoint_bytes": int(checkpoint_path.stat().st_size),
        "process_peak_rss_bytes": peak_rss,
        "training_gpu": training_gpu,
        "inference_gpu": inference_gpu,
        "artifacts": artifact_sizes(input_paths),
        "environment": environment_metadata(device),
    }
    validate_resource_metrics(resources)
    expected_steps = completed * len(variants)
    if optimizer_steps != expected_steps or variant_epochs != expected_steps:
        raise AssertionError(
            f"Optimizer-step accounting mismatch: {optimizer_steps} != {expected_steps}"
        )
    summary = {
        "dataset": "IMDB",
        "model": "PAIN-NC",
        "protocol": "joint_variant_augmentation",
        "seed": seed,
        "variants": variants,
        "selection_metric": "mean_validation_Macro_F1",
        "best_mean_val_macro_f1": best_macro_f1,
        "epoch_accounting": {
            "definition": (
                "one super-epoch performs one full-batch optimizer update "
                "on every selected physical variant"
            ),
            "super_epochs_ran": completed,
            "variant_epochs_ran": variant_epochs,
            "updates_per_super_epoch": len(variants),
            "optimizer_steps": optimizer_steps,
            "compute_budget_warning": (
                "At equal super-epochs this arm receives V times the optimizer "
                "updates of one independent-variant run. Compare matched-update "
                "results or disclose the asymmetry."
            ),
        },
        "training_seconds": training_seconds,
        "mean_test_metrics": mean_dict(list(per_variant_metrics.values())),
        "per_variant_test_metrics": per_variant_metrics,
        "pairwise_invariance": pairwise,
        "resources": resources,
        "run_config": run_config,
    }
    atomic_write_json(summary, seed_dir / "summary.json")
    atomic_torch_save(summary, seed_dir / "summary.pt")
    return summary


def summary_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    resources = summary["resources"]
    row = {
        "seed": summary["seed"],
        "variants": ",".join(summary["variants"]),
        "super_epochs_ran": summary["epoch_accounting"]["super_epochs_ran"],
        "optimizer_steps": summary["epoch_accounting"]["optimizer_steps"],
        "training_seconds": summary["training_seconds"],
        "best_mean_val_macro_f1": summary["best_mean_val_macro_f1"],
        **{
            f"mean_test_{key}": value
            for key, value in summary["mean_test_metrics"].items()
        },
        **{
            key: resources[key]
            for key in (
                "parameter_bytes",
                "buffer_bytes",
                "static_model_bytes",
                "checkpoint_bytes",
                "process_peak_rss_bytes",
            )
        },
    }
    for phase in ("training_gpu", "inference_gpu"):
        prefix = phase.removesuffix("_gpu")
        for key, value in resources[phase].items():
            row[f"{prefix}_{key}"] = value
    for key, value in resources["artifacts"].items():
        row[key] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/imdb_nc_augmentation.yaml")
    parser.add_argument(
        "--variants", nargs="+", default=["v1", "v2", "v3", "v4"]
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[1566911444, 20241017, 20251017],
    )
    parser.add_argument("--device")
    parser.add_argument("--super-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--output-root", default="results/imdb_nc_augmentation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config = merged_config(config, {"device": args.device})
    variants = parse_variants(args.variants)
    if args.preflight_only:
        graphs, paths = prepare_graphs(config, variants)
        print(
            f"[OK] IMDb PAIN augmentation: variants={','.join(variants)} "
            f"nodes={graphs[variants[0]].num_nodes:,}"
        )
        for name, path in paths.items():
            print(f"  {name}: {path} ({path.stat().st_size / 2**20:.1f} MiB)")
        return

    output_root = Path(args.output_root)
    summaries = [
        run_seed(
            config,
            variants,
            seed,
            output_root,
            resume=args.resume,
            super_epochs_override=args.super_epochs,
            patience_override=args.patience,
        )
        for seed in args.seeds
    ]
    atomic_write_csv(
        pd.DataFrame([summary_row(summary) for summary in summaries]),
        output_root / "seed_summary.csv",
    )
    atomic_write_json({"runs": summaries}, output_root / "all_seed_summaries.json")
    print(f"[OK] Results written under {output_root}")


if __name__ == "__main__":
    main()
