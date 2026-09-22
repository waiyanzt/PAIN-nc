"""Train PAIN-NC independently on invariant Freebase1-3 path programs."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from experiments.node_classification.benchmark_freebase import (
    DEFAULT_SEEDS,
    scalar_row,
    summarize,
    write_csv,
)
from experiments.node_classification.train import save_artifact, train_one_run
from pain_nc.config import load_config, merged_config
from pain_nc.telemetry import validate_resource_metrics


VARIANTS = {
    "Freebase_invariant1": "unchanged",
    "Freebase_invariant2": "exact_2",
    "Freebase_invariant3": "exact_3",
}


def artifact_name(source_variant: str, tag: str) -> str:
    return f"invariant_{source_variant}_{tag}.pt"


def require_invariant_contract(
    data_dir: Path,
    variants: list[str],
    tag: str,
) -> None:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {metadata_path}; run python -m preprocessing.freebase_invariant"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sources = [VARIANTS[name] for name in variants]
    missing = [
        data_dir / artifact_name(source, tag)
        for source in sources
        if not (data_dir / artifact_name(source, tag)).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Freebase invariant artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    if (
        metadata.get("mode") != "invariant_conditional_semantic_paths"
        or metadata.get("status") != "PASS"
    ):
        raise ValueError(
            "Freebase invariant metadata is not a completed conditional-path compile"
        )
    semantic = metadata.get("semantic_graph_sha256", {})
    paths = metadata.get("selected_path_program_sha256", {})
    if any(source not in semantic or source not in paths for source in sources):
        raise ValueError("Invariant metadata is missing a requested source variant")
    if len({semantic[source] for source in sources}) != 1:
        raise ValueError("Requested variants do not share one semantic graph hash")
    if len({paths[source] for source in sources}) != 1:
        raise ValueError("Requested variants do not share one PAIN path-program hash")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", default="configs/freebase_nc_invariant.yaml"
    )
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/freebase_nc_invariant"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config = merged_config(config, {"device": args.device})
    data_dir = Path(config["data"]["preprocessed_dir"])
    shared_path = Path(config["data"]["shared_path"])
    tag = str(config["data"].get("artifact_tag", "L3_rel4_fan8_cap256"))
    if not shared_path.is_file():
        raise FileNotFoundError(shared_path)
    require_invariant_contract(data_dir, args.variants, tag)

    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        source_variant = VARIANTS[variant]
        variant_path = data_dir / artifact_name(source_variant, tag)
        for seed in args.seeds:
            destination = args.output_root / variant / f"seed{seed}.pt"
            if destination.exists() and not args.overwrite:
                print(f"Loading existing {destination}", flush=True)
                artifact = torch.load(
                    destination, map_location="cpu", weights_only=False
                )
                validate_resource_metrics(artifact.get("resources", {}))
            else:
                print(
                    f"\n=== {variant} ({source_variant}) | seed={seed} ===",
                    flush=True,
                )
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

    raw_path = args.output_root / "freebase_nc_invariant_runs.csv"
    summary_path = args.output_root / "freebase_nc_invariant_summary.csv"
    write_csv(rows, raw_path)
    summary = summarize(rows, args.variants)
    write_csv(summary, summary_path)
    print("\nMean +/- sample standard deviation across seeds")
    for row in summary:
        print(
            f"{row['variant']:23s} accuracy="
            f"{row['test_accuracy_mean']:.4f} +/- {row['test_accuracy_std']:.4f}  "
            f"macro-F1={row['test_f1_macro_mean']:.4f} +/- "
            f"{row['test_f1_macro_std']:.4f}"
        )
    print(f"Wrote {raw_path} and {summary_path}")


if __name__ == "__main__":
    main()
