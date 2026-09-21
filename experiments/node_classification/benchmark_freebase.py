"""Benchmark sampled PAIN-NC independently on physical Freebase variants."""
from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.node_classification.train import save_artifact, train_one_run
from pain_nc.config import load_config, merged_config
from pain_nc.telemetry import CUDA_MEMORY_KEYS, validate_resource_metrics


DEFAULT_SEEDS = (1566911444, 20241017, 20251017)
VARIANTS = {
    "Freebase1": "unchanged",
    "Freebase2": "exact_2",
    "Freebase3": "exact_3",
}
METRICS = (
    "best_val_accuracy",
    "best_val_macro_f1",
    "test_accuracy",
    "test_precision_macro",
    "test_recall_macro",
    "test_f1_micro",
    "test_f1_macro",
    "elapsed_seconds",
    "time_to_best_seconds",
)
RESOURCE_METRICS = (
    "parameter_bytes",
    "buffer_bytes",
    "static_model_bytes",
    "checkpoint_bytes",
    "process_peak_rss_bytes",
    "training_gpu_allocated_bytes",
    "training_gpu_reserved_bytes",
    "training_gpu_peak_allocated_bytes",
    "training_gpu_peak_reserved_bytes",
    "inference_gpu_allocated_bytes",
    "inference_gpu_reserved_bytes",
    "inference_gpu_peak_allocated_bytes",
    "inference_gpu_peak_reserved_bytes",
    "shared_bytes",
    "variant_bytes",
)


def artifact_name(source_variant: str, tag: str) -> str:
    return f"{source_variant}_{tag}.pt"


def scalar_row(artifact: dict[str, Any]) -> dict[str, Any]:
    resources = artifact.get("resources", {})
    validate_resource_metrics(resources)
    resource_columns = {
        key: int(resources[key])
        for key in (
            "parameter_bytes",
            "buffer_bytes",
            "static_model_bytes",
            "checkpoint_bytes",
            "process_peak_rss_bytes",
        )
    }
    for phase in ("training_gpu", "inference_gpu"):
        prefix = phase.removesuffix("_gpu")
        for key in CUDA_MEMORY_KEYS:
            resource_columns[f"{prefix}_{key}"] = int(resources[phase][key])
    resource_columns.update(
        {key: int(value) for key, value in resources["artifacts"].items()}
    )
    return {
        "variant": artifact["variant"],
        "seed": artifact["seed"],
        "best_epoch": artifact["best_epoch"],
        "epochs_trained": artifact["epochs_trained"],
        "num_paths": artifact["num_paths"],
        **{metric: artifact[metric] for metric in METRICS},
        **resource_columns,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(rows: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    summary = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        record: dict[str, Any] = {"variant": variant, "n_seeds": len(selected)}
        for metric in (*METRICS, *RESOURCE_METRICS):
            values = np.asarray([item[metric] for item in selected], dtype=np.float64)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summary.append(record)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/freebase_nc.yaml")
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device")
    parser.add_argument("--output-root", type=Path, default=Path("results/freebase_nc"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config = merged_config(config, {"device": args.device})
    data_dir = Path(config["data"]["preprocessed_dir"])
    shared_path = Path(config["data"]["shared_path"])
    tag = str(config["data"].get("artifact_tag", "L3_rel4_fan8_cap256"))
    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        source_variant = VARIANTS[variant]
        variant_path = data_dir / artifact_name(source_variant, tag)
        if not shared_path.is_file() or not variant_path.is_file():
            raise FileNotFoundError(
                f"Missing Freebase PAIN artifact: {variant_path}. Run "
                "python -m preprocessing.freebase_node_classification first."
            )
        for seed in args.seeds:
            destination = args.output_root / variant / f"seed{seed}.pt"
            if destination.exists() and not args.overwrite:
                print(f"Loading existing {destination}", flush=True)
                artifact = torch.load(
                    destination, map_location="cpu", weights_only=False
                )
                validate_resource_metrics(artifact.get("resources", {}))
            else:
                print(f"\n=== {variant} ({source_variant}) | seed={seed} ===", flush=True)
                artifact = train_one_run(
                    config,
                    variant=variant,
                    variant_path=variant_path,
                    seed=seed,
                )
                save_artifact(artifact, destination)
                print(f"Saved {destination}", flush=True)
            rows.append(scalar_row(artifact))
            del artifact
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raw_path = args.output_root / "freebase_nc_runs.csv"
    summary_path = args.output_root / "freebase_nc_summary.csv"
    write_csv(rows, raw_path)
    summary = summarize(rows, args.variants)
    write_csv(summary, summary_path)
    print("\nMean +/- sample standard deviation across seeds")
    for row in summary:
        print(
            f"{row['variant']:16s} accuracy="
            f"{row['test_accuracy_mean']:.4f} +/- {row['test_accuracy_std']:.4f}  "
            f"macro-F1={row['test_f1_macro_mean']:.4f} +/- "
            f"{row['test_f1_macro_std']:.4f}"
        )
    print(f"Wrote {raw_path} and {summary_path}")


if __name__ == "__main__":
    main()
