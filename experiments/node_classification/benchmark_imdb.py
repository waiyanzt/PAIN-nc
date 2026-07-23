"""Benchmark PAIN-NC on IMDb1-4 and the universal union mapping."""
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


DEFAULT_SEEDS = (1566911444, 20241017, 20251017)
VARIANTS = {
    "IMDb1": "v1_L3.pt",
    "IMDb2": "v2_L3.pt",
    "IMDb3": "v3_L3.pt",
    "IMDb4": "v4_L3.pt",
    "IMDb_universal": "universal_L3.pt",
}
METRICS = (
    "best_val_accuracy",
    "test_accuracy",
    "test_precision_macro",
    "test_recall_macro",
    "test_f1_micro",
    "test_f1_macro",
    "elapsed_seconds",
    "time_to_best_seconds",
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
        raise FileNotFoundError(
            f"Preprocessed IMDb artifacts are missing:\n{formatted}\n"
            "Run: python -m preprocessing.imdb_node_classification"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print(f"Shared: {shared} ({shared.stat().st_size / 2**20:.1f} MiB)")
    for display_name in variants:
        source_name = Path(VARIANTS[display_name]).stem.split("_L")[0]
        path = data_dir / VARIANTS[display_name]
        details = metadata["variants"][source_name]
        print(
            f"{display_name:16s} {path.stat().st_size / 2**20:8.1f} MiB  "
            f"paths={details['num_paths']:,} "
            f"undirected_edges={details['num_undirected_edges']:,}"
        )


def scalar_row(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": artifact["variant"],
        "seed": artifact["seed"],
        "best_epoch": artifact["best_epoch"],
        "epochs_trained": artifact["epochs_trained"],
        "num_paths": artifact["num_paths"],
        **{metric: artifact[metric] for metric in METRICS},
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
        for metric in METRICS:
            values = np.asarray([item[metric] for item in selected], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/imdb_nc.yaml")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
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

