"""Benchmark PAIN-LP on DBLP1-3 and the universal union graph."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from experiments.link_prediction.train_dblp import save_artifact, train_one_run
from pain_nc.config import load_config, merged_config
from pain_nc.experiment import atomic_write_csv
from pain_nc.telemetry import CUDA_MEMORY_KEYS, validate_resource_metrics


DEFAULT_SEEDS = (1566911444, 20241017, 20251017)
VARIANTS = {
    "DBLP1": "v1",
    "DBLP2": "v2",
    "DBLP3": "v3",
    "DBLP_universal": "universal",
    # The three entries deliberately load the same deduplicated semantic-path
    # artifact. Preprocessing independently compiles it from each distinct
    # physical graph and refuses to save unless all semantic hashes match.
    "DBLP_invariant_v1": "invariant",
    "DBLP_invariant_v2": "invariant",
    "DBLP_invariant_v3": "invariant",
}
METRICS = (
    "best_val_mrr",
    "best_val_loss",
    "test_auc",
    "test_average_precision",
    "test_mrr",
    "test_mean_rank",
    "test_hits_at_1",
    "test_hits_at_3",
    "test_hits_at_5",
    "test_hits_at_10",
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
    artifact_tag = str(config["data"].get("artifact_tag", "L3"))
    metadata_path = data_dir / "metadata.json"
    required = [
        shared,
        metadata_path,
        *(data_dir / f"{VARIANTS[v]}_{artifact_tag}.pt" for v in variants),
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Preprocessed DBLP artifacts are missing:\n{formatted}\n"
            "Run: python -m preprocessing.dblp_link_prediction"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_tag") != artifact_tag:
        raise ValueError(
            f"Config requests artifact_tag={artifact_tag!r}, but metadata records "
            f"{metadata.get('artifact_tag')!r}. Re-run matching preprocessing or fix the config."
        )
    if any(name.startswith("DBLP_invariant_") for name in variants):
        audit = metadata.get("invariance_audit", {})
        if not audit.get("physical_graphs_different") or not audit.get(
            "semantic_programs_equal"
        ):
            raise ValueError(
                "Invariant DBLP artifacts lack a passing physical/semantic hash audit"
            )
    print(f"Shared: {shared} ({shared.stat().st_size / 2**20:.1f} MiB)")
    for display_name in variants:
        source_name = VARIANTS[display_name]
        path = data_dir / f"{source_name}_{artifact_tag}.pt"
        details = metadata["variants"][source_name]
        print(
            f"{display_name:16s} {path.stat().st_size / 2**30:8.2f} GiB  "
            f"paths={details['num_paths']:,} "
            f"undirected_edges={details['num_undirected_edges']:,}"
        )


def scalar_row(artifact: dict[str, Any]) -> dict[str, Any]:
    resources = artifact.get("resources", {})
    validate_resource_metrics(resources)
    resource_columns = {
        key: int(resources[key])
        for key in (
            "parameter_bytes", "buffer_bytes", "static_model_bytes",
            "checkpoint_bytes", "process_peak_rss_bytes",
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


def summarize(rows: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    result = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        summary: dict[str, Any] = {"variant": variant, "n_seeds": len(selected)}
        for metric in (*METRICS, *RESOURCE_METRICS):
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        result.append(summary)
    return result


def random_mrr(num_candidates: int) -> float:
    return float(np.mean(1.0 / np.arange(1, num_candidates + 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dblp_lp.yaml")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device")
    parser.add_argument(
        "--artifact-tag",
        help="Override data.artifact_tag (for example L3_cap100000).",
    )
    parser.add_argument("--output-root", default="results/dblp_lp")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Load completed artifacts and write tables without starting missing runs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config = merged_config(config, {"device": args.device})
    if args.artifact_tag:
        config = merged_config(
            config, {"data.artifact_tag": args.artifact_tag}
        )
    preflight(config, args.variants)
    if args.preflight_only:
        return

    data_dir = Path(config["data"]["preprocessed_dir"])
    artifact_tag = str(config["data"].get("artifact_tag", "L3"))
    output_root = Path(args.output_root)
    rows = []
    for variant in args.variants:
        variant_path = data_dir / f"{VARIANTS[variant]}_{artifact_tag}.pt"
        for seed in args.seeds:
            destination = output_root / variant / f"seed{seed}.pt"
            if destination.exists() and not args.overwrite:
                print(f"Loading existing {destination}")
                artifact = torch.load(destination, map_location="cpu", weights_only=False)
                try:
                    validate_resource_metrics(artifact.get("resources", {}))
                except ValueError as error:
                    raise RuntimeError(
                        f"Existing artifact {destination} predates mandatory resource "
                        "telemetry. Re-run with --overwrite."
                    ) from error
            elif args.aggregate_only:
                raise FileNotFoundError(
                    f"Missing completed run required by --aggregate-only: {destination}"
                )
            else:
                print(f"\n=== {variant} | seed={seed} ===")
                artifact = train_one_run(
                    config,
                    variant=variant,
                    variant_path=variant_path,
                    seed=seed,
                )
                save_artifact(artifact, destination)
            rows.append(scalar_row(artifact))
            del artifact
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    num_candidates = int(config["evaluation"]["num_candidates"])
    chance = random_mrr(num_candidates)
    degenerate = [
        row for row in rows if float(row["test_mrr"]) <= chance
    ]
    atomic_write_csv(pd.DataFrame(rows), output_root / "dblp_lp_raw.csv")
    atomic_write_csv(
        pd.DataFrame(summarize(rows, args.variants)),
        output_root / "dblp_lp_summary.csv",
    )
    if degenerate:
        details = ", ".join(
            f"{row['variant']}/seed{row['seed']}={row['test_mrr']:.4f}"
            for row in degenerate
        )
        raise RuntimeError(
            f"At/below-chance MRR runs detected (chance={chance:.4f}): {details}"
        )
    print(f"Wrote {output_root / 'dblp_lp_raw.csv'}")
    print(f"Wrote {output_root / 'dblp_lp_summary.csv'}")


if __name__ == "__main__":
    main()
