"""Fail-closed structural and matched-seed audit for invariant IMDb PAIN."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

from pain_nc.experiment import atomic_write_csv


DEFAULT_SEEDS = (1566911444, 20241017, 20251017)
DEFAULT_VARIANTS = (
    "IMDb_invariant_v1",
    "IMDb_invariant_v2",
    "IMDb_invariant_v3",
    "IMDb_invariant_v4",
)
SOURCE_NAMES = {
    "IMDb_invariant_v1": "invariant_v1",
    "IMDb_invariant_v2": "invariant_v2",
    "IMDb_invariant_v3": "invariant_v3",
    "IMDb_invariant_v4": "invariant_v4",
}


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def node_macro_kendall(left: np.ndarray, right: np.ndarray) -> float:
    if np.array_equal(left, right):
        return 1.0
    values = []
    for row_left, row_right in zip(left, right, strict=True):
        if np.array_equal(row_left, row_right):
            values.append(1.0)
            continue
        value = kendalltau(
            row_left, row_right, variant="b", nan_policy="omit"
        ).statistic
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def load_run(root: Path, variant: str, seed: int) -> dict:
    path = root / variant / f"seed{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "test_node_ids",
        "y_true",
        "y_prob",
        "model_state_dict",
        "variant_source_name",
        "mapping_mode",
        "physical_graph_sha256",
        "semantic_graph_sha256",
        "selected_path_program_sha256",
        "shared_contract_sha256",
        "resources",
    }
    missing = sorted(required - artifact.keys())
    if missing:
        raise ValueError(f"{path} is missing invariant provenance: {missing}")
    return artifact


def as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def validate_preprocessing(metadata_path: Path) -> dict:
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("mode") != "invariant" or metadata.get("status") != "PASS":
        raise ValueError("Invariant preprocessing metadata is not PASS")
    physical = metadata.get("physical_graph_sha256", {})
    semantic = metadata.get("semantic_graph_sha256", {})
    paths = metadata.get("selected_path_program_sha256", {})
    expected = {"v1", "v2", "v3", "v4"}
    for name, values in (
        ("physical", physical),
        ("semantic", semantic),
        ("path", paths),
    ):
        if set(values) != expected:
            raise ValueError(f"{name} hashes do not cover IMDb1-4: {values}")
    if len(set(physical.values())) != 4:
        raise ValueError("The four source physical graph hashes are not distinct")
    if len(set(semantic.values())) != 1:
        raise ValueError("The four semantic graph hashes differ")
    if len(set(paths.values())) != 1:
        raise ValueError("The four semantic PAIN path hashes differ")
    if (
        not metadata.get("semantic_equals_universal", False)
        or set(semantic.values())
        != {metadata.get("universal_graph_sha256")}
    ):
        raise ValueError("Semantic/universal parity was not established")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/preprocessed/IMDB_invariant/metadata.json"),
    )
    parser.add_argument(
        "--root", type=Path, default=Path("results/imdb_nc_invariant")
    )
    parser.add_argument(
        "--variants", nargs="+", default=list(DEFAULT_VARIANTS)
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/imdb_nc_invariant_audit.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Defaults to <output_stem>_summary.csv.",
    )
    parser.add_argument(
        "--allow-nonexact",
        action="store_true",
        help="Write reports without failing on non-identical outputs/checkpoints.",
    )
    args = parser.parse_args()
    unknown = set(args.variants) - set(SOURCE_NAMES)
    if unknown:
        raise ValueError(f"Unknown invariant variants: {sorted(unknown)}")

    metadata = validate_preprocessing(args.metadata)
    rows = []
    failures = []
    for seed in args.seeds:
        runs = {
            variant: load_run(args.root, variant, seed)
            for variant in args.variants
        }
        devices = {
            run["resources"]["environment"]["device_name"]
            for run in runs.values()
        }
        if len(devices) != 1:
            raise ValueError(
                f"seed={seed}: invariant runs used different GPU architectures: "
                f"{sorted(devices)}"
            )
        for variant, run in runs.items():
            source_name = SOURCE_NAMES[variant]
            source_variant = source_name.removeprefix("invariant_")
            expected = metadata["variants"][source_name]
            provenance = {
                "variant_source_name": source_name,
                "mapping_mode": "conditional_semantic_path_compile",
                "physical_graph_sha256": metadata["physical_graph_sha256"][
                    source_variant
                ],
                "semantic_graph_sha256": expected["semantic_graph_sha256"],
                "selected_path_program_sha256": expected[
                    "selected_path_program_sha256"
                ],
                "shared_contract_sha256": metadata["shared_contract_sha256"],
            }
            for field, value in provenance.items():
                if run.get(field) != value:
                    raise ValueError(
                        f"{variant} seed={seed}: {field}={run.get(field)!r}, "
                        f"expected {value!r}"
                    )

        for left_name, right_name in combinations(args.variants, 2):
            left, right = runs[left_name], runs[right_name]
            for field in ("test_node_ids", "y_true"):
                if not np.array_equal(
                    as_numpy(left[field]), as_numpy(right[field])
                ):
                    raise ValueError(
                        f"seed={seed}: {field} differs for "
                        f"{left_name} and {right_name}"
                    )
            left_scores = as_numpy(left["y_prob"])
            right_scores = as_numpy(right["y_prob"])
            difference = left_scores - right_scores
            left_checkpoint = state_hash(left["model_state_dict"])
            right_checkpoint = state_hash(right["model_state_dict"])
            exact_scores = bool(np.array_equal(left_scores, right_scores))
            exact_checkpoint = left_checkpoint == right_checkpoint
            tau = node_macro_kendall(left_scores, right_scores)
            row = {
                "seed": seed,
                "variant_a": left_name,
                "variant_b": right_name,
                "gpu_architecture": next(iter(devices)),
                "kendall_tau": tau,
                "max_abs_score_diff": float(np.max(np.abs(difference))),
                "mean_abs_score_diff": float(np.mean(np.abs(difference))),
                "probabilities_exact": exact_scores,
                "checkpoint_hash_equal": exact_checkpoint,
                "checkpoint_hash_a": left_checkpoint,
                "checkpoint_hash_b": right_checkpoint,
            }
            rows.append(row)
            if not exact_scores or not exact_checkpoint or tau != 1.0:
                failures.append(row)

    frame = pd.DataFrame(rows)
    atomic_write_csv(frame, args.output)
    summary_rows = []
    for (left_name, right_name), group in frame.groupby(
        ["variant_a", "variant_b"], sort=True
    ):
        values = group["kendall_tau"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "variant_a": left_name,
                "variant_b": right_name,
                "seed_count": int(len(group)),
                "probabilities_exact_all_seeds": bool(
                    group["probabilities_exact"].all()
                ),
                "checkpoint_hash_equal_all_seeds": bool(
                    group["checkpoint_hash_equal"].all()
                ),
                "max_abs_score_diff": float(
                    group["max_abs_score_diff"].max()
                ),
                "kendall_tau_mean": float(np.mean(values)),
                "kendall_tau_std": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
            }
        )
    summary_output = args.summary_output or args.output.with_name(
        f"{args.output.stem}_summary.csv"
    )
    atomic_write_csv(pd.DataFrame(summary_rows), summary_output)
    print(frame.to_string(index=False))
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_output}")
    if failures and not args.allow_nonexact:
        raise RuntimeError(
            f"Invariant audit failed for {len(failures)} matched-seed pairs"
        )
    print("IMDb invariant audit PASS", flush=True)


if __name__ == "__main__":
    main()
