"""Benchmark PAIN-NC on physical, universal, or invariant IMDb artifacts."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pain_nc.config import load_config, merged_config
from experiments.node_classification.train import save_artifact, train_one_run
from pain_nc.telemetry import CUDA_MEMORY_KEYS, validate_resource_metrics


DEFAULT_SEEDS = (1566911444, 20241017, 20251017)
ORDINARY_VARIANTS = {
    "IMDb1": "v1_L3.pt",
    "IMDb2": "v2_L3.pt",
    "IMDb3": "v3_L3.pt",
    "IMDb4": "v4_L3.pt",
    "IMDb_universal": "universal_L3.pt",
}
INVARIANT_VARIANTS = {
    "IMDb_invariant_v1": "invariant_v1_L3.pt",
    "IMDb_invariant_v2": "invariant_v2_L3.pt",
    "IMDb_invariant_v3": "invariant_v3_L3.pt",
    "IMDb_invariant_v4": "invariant_v4_L3.pt",
}
VARIANTS = {**ORDINARY_VARIANTS, **INVARIANT_VARIANTS}
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


def preflight(config: dict[str, Any], variants: list[str]) -> None:
    data_dir = Path(config["data"]["preprocessed_dir"])
    shared = Path(config["data"]["shared_path"])
    metadata_path = data_dir / "metadata.json"
    missing = [
        path
        for path in [shared, metadata_path, *(data_dir / VARIANTS[v] for v in variants)]
        if not path.is_file()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        preprocessor = (
            "preprocessing.imdb_invariant"
            if any(name in INVARIANT_VARIANTS for name in variants)
            else "preprocessing.imdb_node_classification"
        )
        raise FileNotFoundError(
            f"Preprocessed IMDb artifacts are missing:\n{formatted}\n"
            f"Run: python -m {preprocessor}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if any(name in INVARIANT_VARIANTS for name in variants):
        physical = metadata.get("physical_graph_sha256", {})
        semantic = metadata.get("semantic_graph_sha256", {})
        paths = metadata.get("selected_path_program_sha256", {})
        expected_sources = {"v1", "v2", "v3", "v4"}
        if (
            metadata.get("mode") != "invariant"
            or metadata.get("status") != "PASS"
            or set(physical) != expected_sources
            or set(semantic) != expected_sources
            or set(paths) != expected_sources
            or len(set(physical.values())) != 4
            or len(set(semantic.values())) != 1
            or len(set(paths.values())) != 1
            or not metadata.get("semantic_equals_universal", False)
            or set(semantic.values())
            != {metadata.get("universal_graph_sha256")}
        ):
            raise ValueError(
                "IMDb invariant preprocessing proof is absent or invalid; "
                "re-run python -m preprocessing.imdb_invariant"
            )
    print(f"Shared: {shared} ({shared.stat().st_size / 2**20:.1f} MiB)")
    for display_name in variants:
        source_name = Path(VARIANTS[display_name]).stem.split("_L")[0]
        path = data_dir / VARIANTS[display_name]
        details = metadata["variants"][source_name]
        if "num_physical_undirected_edges" in details:
            edge_summary = (
                f"physical_edges={details['num_physical_undirected_edges']:,} "
                f"semantic_edges={details['num_semantic_undirected_edges']:,}"
            )
        else:
            edge_summary = (
                f"undirected_edges={details['num_undirected_edges']:,}"
            )
        print(
            f"{display_name:20s} {path.stat().st_size / 2**20:8.1f} MiB  "
            f"paths={details['num_paths']:,} {edge_summary}"
        )


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    summary = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        row: dict[str, Any] = {"variant": variant, "n_seeds": len(selected)}
        for metric in (*METRICS, *RESOURCE_METRICS):
            values = np.asarray([item[metric] for item in selected], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/imdb_nc.yaml")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(ORDINARY_VARIANTS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device")
    parser.add_argument("--output-root", default="results/imdb_nc")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config = merged_config(config, {"device": args.device})
    preflight(config, args.variants)
    if args.preflight_only:
        return

    data_dir = Path(config["data"]["preprocessed_dir"])
    output_root = Path(args.output_root)
    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            destination = output_root / variant / f"seed{seed}.pt"
            if destination.exists() and not args.overwrite:
                print(f"Loading existing {destination}")
                artifact = torch.load(destination, map_location="cpu", weights_only=False)
                try:
                    validate_resource_metrics(artifact.get("resources", {}))
                except ValueError as error:
                    raise RuntimeError(
                        f"Existing artifact {destination} predates mandatory "
                        "memory telemetry. Re-run with --overwrite."
                    ) from error
            else:
                print(f"\n=== {variant} | seed={seed} ===")
                artifact = train_one_run(
                    config,
                    variant=variant,
                    variant_path=data_dir / VARIANTS[variant],
                    seed=seed,
                )
                save_artifact(artifact, destination)
                print(f"Saved {destination}")
            rows.append(scalar_row(artifact))
            del artifact
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raw_path = output_root / "imdb_nc_runs.csv"
    summary_path = output_root / "imdb_nc_summary.csv"
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

