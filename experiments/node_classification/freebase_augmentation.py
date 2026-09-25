"""Train one shared PAIN-NC model per seed across Freebase1-3."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

from experiments.node_classification.benchmark_freebase import DEFAULT_SEEDS
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
    PeakRSSMonitor,
    artifact_sizes,
    cuda_memory_stats,
    environment_metadata,
    merge_cuda_memory_stats,
    model_memory_bytes,
    reset_cuda_peak,
    validate_resource_metrics,
)


VARIANTS = {
    "Freebase1": "unchanged",
    "Freebase2": "exact_2",
    "Freebase3": "exact_3",
}
SHARED_FIELDS = ("x", "y", "node_type", "train_mask", "val_mask", "test_mask")


def parse_variants(values: list[str]) -> list[str]:
    aliases = {name.lower(): name for name in VARIANTS}
    variants = []
    for value in values:
        if value.lower() not in aliases:
            raise ValueError(f"Unknown Freebase variant {value!r}")
        variants.append(aliases[value.lower()])
    if len(variants) != len(set(variants)):
        raise ValueError("Duplicate Freebase variants are not allowed")
    if len(variants) < 2:
        raise ValueError("Augmentation requires at least two variants")
    return variants


def artifact_paths(
    config: Mapping[str, Any], variants: list[str]
) -> dict[str, Path]:
    directory = Path(config["data"]["preprocessed_dir"])
    tag = str(config["data"].get("artifact_tag", "L3_rel4_fan8_cap256"))
    paths = {"shared": Path(config["data"]["shared_path"])}
    paths.update(
        {
            variant: directory / f"{VARIANTS[variant]}_{tag}.pt"
            for variant in variants
        }
    )
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Freebase PAIN artifacts:\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\nRun: python -m preprocessing.freebase_node_classification"
        )
    return paths


def _load(path: Path, *, mmap: bool = False, map_location: Any = "cpu") -> Any:
    return torch.load(
        path, map_location=map_location, mmap=mmap, weights_only=False
    )


def inspect_artifacts(
    config: Mapping[str, Any], variants: list[str]
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Check the shared contract and common relation vocabulary without path transforms."""
    paths = artifact_paths(config, variants)
    shared = _load(paths["shared"], mmap=True)
    expected_length = int(config["model"]["path_length"])
    reference_relations = None
    reference_contract = None
    details: dict[str, dict[str, Any]] = {}
    for variant in variants:
        payload = _load(paths[variant], mmap=True)
        meta = payload["meta"]
        relations = meta.get("edge_type_names")
        contract = meta.get("shared_contract_sha256")
        if not relations or not contract:
            raise ValueError(f"{variant} is missing relation or shared-contract metadata")
        if int(meta["path_length"]) != expected_length:
            raise ValueError(f"{variant} path length differs from configuration")
        if reference_relations is None:
            reference_relations = relations
            reference_contract = contract
        elif relations != reference_relations or contract != reference_contract:
            raise ValueError(f"{variant} has a different relation vocabulary or split contract")
        if int(payload["mask_index"].max()) >= int(shared["x"].shape[0]):
            raise ValueError(f"{variant} contains a root beyond the shared node range")
        details[variant] = {
            "num_paths": int(payload["path_lengths"].numel()),
            "num_relations": len(relations),
            "physical_graph_sha256": meta.get("physical_graph_sha256"),
            "selected_path_program_sha256": meta.get(
                "selected_path_program_sha256"
            ),
            "shared_contract_sha256": contract,
        }
        del payload
    return paths, details


def prepare_graphs(
    config: Mapping[str, Any], variants: list[str]
) -> tuple[dict[str, PainGraph], dict[str, Path], dict[str, dict[str, Any]]]:
    paths, details = inspect_artifacts(config, variants)
    shared_payload = _load(paths["shared"], mmap=True)
    graphs = {
        variant: load_imdb_graph(
            paths["shared"],
            paths[variant],
            reverse_paths=bool(config["model"].get("reverse_paths", True)),
            shared_payload=shared_payload,
        )
        for variant in variants
    }
    reference = graphs[variants[0]]
    for variant in variants[1:]:
        graph = graphs[variant]
        if (
            graph.num_nodes != reference.num_nodes
            or graph.num_classes != reference.num_classes
            or graph.num_node_types != reference.num_node_types
            or graph.num_edge_types != reference.num_edge_types
        ):
            raise ValueError(f"{variant} has an incompatible model shape")
        for field in SHARED_FIELDS:
            if not torch.equal(getattr(reference, field), getattr(graph, field)):
                raise ValueError(f"Shared field {field} differs for {variant}")
    return graphs, paths, details


def move_graphs(
    graphs: Mapping[str, PainGraph],
    variants: list[str],
    device: torch.device,
    *,
    move_paths: bool,
) -> dict[str, PainGraph]:
    first = variants[0]
    moved = {first: graphs[first].to(device, move_paths=move_paths)}
    for variant in variants[1:]:
        moved[variant] = graphs[variant].to(
            device, move_paths=move_paths, shared_from=moved[first]
        )
    return moved


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, graph: PainGraph, split: str
) -> tuple[float, dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    indices = getattr(graph, f"{split}_mask").nonzero(as_tuple=False).view(-1)
    logits = model(graph)[indices]
    labels = graph.y[indices]
    loss = torch.nn.functional.cross_entropy(logits, labels)
    metrics = classification_metrics(logits, labels)
    return float(loss), metrics, logits.detach().cpu(), indices.detach().cpu()


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)


def rowwise_tau(left: np.ndarray, right: np.ndarray) -> float:
    """Mean test-node Kendall tau-b, matching the paper-table definition."""
    if left.shape != right.shape:
        raise ValueError("Variant class-score shapes differ")
    values = []
    for left_row, right_row in zip(left, right, strict=True):
        if np.array_equal(left_row, right_row):
            values.append(1.0)
            continue
        value = kendalltau(left_row, right_row, variant="b").statistic
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def run_seed(
    config: dict[str, Any],
    variants: list[str],
    graphs: Mapping[str, PainGraph],
    input_paths: Mapping[str, Path],
    fingerprints: Mapping[str, Mapping[str, Any]],
    seed: int,
    output_root: Path,
    *,
    resume: bool,
    super_epochs_override: int | None = None,
    patience_override: int | None = None,
) -> dict[str, Any] | None:
    """Return a completed summary, or None when the invocation time cap fires."""
    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    best_path = seed_dir / "shared_checkpoint.pt"
    state_path = seed_dir / "latest_training_state.pt"
    summary_path = seed_dir / "summary.pt"
    training = config["training"]
    super_epochs = int(
        super_epochs_override
        if super_epochs_override is not None
        else training.get("super_epochs", training["epochs"] // len(variants))
    )
    patience = int(
        patience_override
        if patience_override is not None
        else training.get("early_stopping_patience", super_epochs)
    )
    if super_epochs < 1 or patience < 1:
        raise ValueError("super-epochs and patience must be positive")
    run_config = {
        "dataset": "Freebase",
        "model": "PAIN-NC",
        "protocol": "joint_variant_augmentation",
        "seed": seed,
        "variants": variants,
        "model_config": copy.deepcopy(config["model"]),
        "training_config": {
            key: value for key, value in training.items() if key != "max_hours"
        },
        "super_epochs": super_epochs,
        "patience": patience,
        "fingerprints": dict(fingerprints),
    }
    if summary_path.exists():
        summary = _load(summary_path)
        if _canonical(summary["run_config"]) != _canonical(run_config):
            raise ValueError(f"Completed seed {seed} used a different configuration")
        print(f"Loading completed {summary_path}", flush=True)
        return summary
    if state_path.exists() and not resume:
        raise RuntimeError(
            f"{state_path} exists; pass --resume to continue the seed"
        )

    rss_monitor = PeakRSSMonitor().start()
    set_seed(seed, bool(training.get("deterministic", True)))
    device = resolve_device(str(config["device"]))
    model = build_model(graphs[variants[0]], config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training.get("lr_factor", 0.5)),
        patience=int(training.get("lr_patience", 3)),
        min_lr=float(training.get("min_learning_rate", 1e-5)),
    )
    rng = np.random.RandomState(seed)
    history: list[dict[str, Any]] = []
    completed = optimizer_steps = 0
    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_super_epoch = 0
    time_to_best = 0.0
    no_improvement = 0
    prior_seconds = 0.0
    prior_peak_rss = 0
    prior_training_gpu: dict[str, int] = {}
    if state_path.exists():
        state = _load(state_path, map_location=device)
        if _canonical(state["run_config"]) != _canonical(run_config):
            raise ValueError("Resume configuration differs from the saved run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        _optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(state["rng_state"], rng)
        history = list(state["history"])
        completed = int(state["completed_super_epoch"])
        optimizer_steps = int(state["optimizer_steps"])
        best_macro_f1 = float(state["best_macro_f1"])
        best_val_loss = float(state["best_val_loss"])
        best_super_epoch = int(state["best_super_epoch"])
        time_to_best = float(state["time_to_best_seconds"])
        no_improvement = int(state["no_improvement"])
        prior_seconds = float(state["training_seconds"])
        prior_peak_rss = int(state.get("process_peak_rss_bytes", 0))
        prior_training_gpu = dict(state.get("training_gpu", {}))
    elif resume:
        print(f"[resume] No state for seed={seed}; starting from scratch", flush=True)

    reset_cuda_peak(device)
    started = time.monotonic()
    max_hours = float(training.get("max_hours", 0.0))
    gradient_clip = float(training.get("gradient_clip_norm", 0.0))
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
            train_losses[variant] = float(loss.detach().cpu())
            del logits, loss

        validation_losses = {}
        validation_metrics = {}
        for variant in variants:
            val_loss, metrics, _, _ = evaluate(model, graphs[variant], "val")
            validation_losses[variant] = val_loss
            validation_metrics[variant] = metrics
        mean_val_macro_f1 = float(
            np.mean([validation_metrics[name]["Macro_F1"] for name in variants])
        )
        mean_val_loss = float(np.mean(list(validation_losses.values())))
        scheduler.step(mean_val_loss)
        completed = super_epoch + 1
        elapsed = prior_seconds + time.monotonic() - started
        improved = mean_val_macro_f1 > best_macro_f1 + 1e-12 or (
            abs(mean_val_macro_f1 - best_macro_f1) <= 1e-12
            and mean_val_loss < best_val_loss
        )
        if improved:
            best_macro_f1 = mean_val_macro_f1
            best_val_loss = mean_val_loss
            best_super_epoch = completed
            time_to_best = elapsed
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
                best_path,
            )
        else:
            no_improvement += 1
        row: dict[str, Any] = {
            "super_epoch": completed,
            "variant_order": ",".join(order),
            "optimizer_steps_cumulative": optimizer_steps,
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
        atomic_write_csv(pd.DataFrame(history), seed_dir / "training_history.csv")
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
                "best_macro_f1": best_macro_f1,
                "best_val_loss": best_val_loss,
                "best_super_epoch": best_super_epoch,
                "time_to_best_seconds": time_to_best,
                "no_improvement": no_improvement,
                "training_seconds": prior_seconds + time.monotonic() - started,
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
        if (
            max_hours > 0
            and completed < super_epochs
            and no_improvement < patience
            and time.monotonic() - started >= max_hours * 3600
        ):
            rss_monitor.stop()
            print(
                f"Reached max_hours={max_hours:g} after a saved super-epoch; "
                "rerun with --resume",
                flush=True,
            )
            return None

    training_seconds = prior_seconds + time.monotonic() - started
    training_gpu = merge_cuda_memory_stats(
        prior_training_gpu, cuda_memory_stats(device)
    )
    peak_rss = max(prior_peak_rss, rss_monitor.peak_bytes)
    if optimizer_steps != completed * len(variants):
        raise AssertionError("Optimizer-step accounting mismatch")
    checkpoint = _load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    reset_cuda_peak(device)
    per_variant_metrics = {}
    outputs = {}
    for variant in variants:
        loss_value, metrics, logits, indices = evaluate(
            model, graphs[variant], "test"
        )
        probabilities = logits.softmax(dim=-1).numpy()
        labels = graphs[variant].y[indices.to(device)].detach().cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        metrics["CrossEntropy"] = loss_value
        metrics["num_paths"] = float(graphs[variant].num_paths)
        per_variant_metrics[variant] = metrics
        outputs[variant] = {
            "item_id": indices.numpy(),
            "label": labels,
            "logits": logits.numpy(),
            "probabilities": probabilities,
            "prediction": predictions,
            "confidence": probabilities.max(axis=1),
        }
        frame = pd.DataFrame(
            {
                "node_id": indices.numpy(),
                "label": labels,
                "prediction": predictions,
                "confidence": probabilities.max(axis=1),
            }
        )
        for class_id in range(probabilities.shape[1]):
            frame[f"prob_class_{class_id}"] = probabilities[:, class_id]
            frame[f"logit_class_{class_id}"] = logits[:, class_id].numpy()
        atomic_write_csv(frame, seed_dir / f"test_scores_{variant}.csv")
    inference_gpu = cuda_memory_stats(device)
    peak_rss = max(peak_rss, rss_monitor.stop())
    pairwise = classification_invariance_rows(outputs)
    for row in pairwise:
        left = row["variant_a"]
        right = row["variant_b"]
        if not np.array_equal(outputs[left]["item_id"], outputs[right]["item_id"]):
            raise ValueError("Test node order differs across variants")
        row["kendall_tau_b"] = rowwise_tau(
            outputs[left]["probabilities"], outputs[right]["probabilities"]
        )
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
        "checkpoint_bytes": int(best_path.stat().st_size),
        "process_peak_rss_bytes": peak_rss,
        "training_gpu": training_gpu,
        "inference_gpu": inference_gpu,
        "artifacts": artifact_sizes(input_paths),
        "environment": environment_metadata(device),
    }
    validate_resource_metrics(resources)
    summary = {
        "dataset": "Freebase",
        "model": "PAIN-NC",
        "protocol": "joint_variant_augmentation",
        "seed": seed,
        "variants": variants,
        "selection_metric": "mean_validation_Macro_F1",
        "best_mean_val_macro_f1": best_macro_f1,
        "best_super_epoch": best_super_epoch,
        "time_to_best_seconds": time_to_best,
        "epoch_accounting": {
            "definition": "one optimizer update on every variant per super-epoch",
            "super_epochs_ran": completed,
            "variant_epochs_ran": optimizer_steps,
            "updates_per_super_epoch": len(variants),
            "optimizer_steps": optimizer_steps,
        },
        "training_seconds": training_seconds,
        "mean_test_metrics": mean_dict(list(per_variant_metrics.values())),
        "per_variant_test_metrics": per_variant_metrics,
        "pairwise_invariance": pairwise,
        "resources": resources,
        "run_config": run_config,
    }
    atomic_write_json(summary, seed_dir / "summary.json")
    atomic_torch_save(summary, summary_path)
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
    row.update(resources["artifacts"])
    return row


def write_aggregates(summaries: list[dict[str, Any]], output_root: Path) -> None:
    atomic_write_csv(
        pd.DataFrame([summary_row(summary) for summary in summaries]),
        output_root / "seed_summary.csv",
    )
    atomic_write_json({"runs": summaries}, output_root / "all_seed_summaries.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/freebase_nc_augmentation.yaml")
    parser.add_argument(
        "--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device")
    parser.add_argument("--super-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("results/freebase_nc_augmentation"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    variants = parse_variants(args.variants)
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Duplicate seeds are not allowed")
    config = load_config(args.config)
    if args.device:
        config = merged_config(config, {"device": args.device})
    paths, fingerprints = inspect_artifacts(config, variants)
    if args.preflight_only:
        print(f"[OK] Freebase PAIN augmentation: {','.join(variants)}")
        for variant in variants:
            print(
                f"  {variant}: {fingerprints[variant]['num_paths']:,} sampled paths; "
                f"{paths[variant].stat().st_size / 2**30:.2f} GiB artifact"
            )
        return
    device = resolve_device(str(config["device"]))
    cpu_graphs, paths, fingerprints = prepare_graphs(config, variants)
    graphs = move_graphs(
        cpu_graphs,
        variants,
        device,
        move_paths=bool(config["model"].get("paths_on_device", False)),
    )
    del cpu_graphs
    gc.collect()
    summaries = []
    for seed in args.seeds:
        summary = run_seed(
            config,
            variants,
            graphs,
            paths,
            fingerprints,
            seed,
            args.output_root,
            resume=args.resume,
            super_epochs_override=args.super_epochs,
            patience_override=args.patience,
        )
        if summary is None:
            print("Training paused. Re-run the same command with --resume.")
            return
        summaries.append(summary)
        write_aggregates(summaries, args.output_root)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"[OK] Freebase augmentation results written under {args.output_root}")


if __name__ == "__main__":
    main()
