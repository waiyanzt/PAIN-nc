"""Audit matched-seed DBLP link-ranking invariance across PAIN runs."""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

from pain_nc.experiment import atomic_write_csv


DEFAULT_SEEDS = (1566911444, 20241017, 20251017)
DEFAULT_VARIANTS = (
    "DBLP_invariant_v1",
    "DBLP_invariant_v2",
    "DBLP_invariant_v3",
)


def query_macro_kendall(
    scores_a: np.ndarray, scores_b: np.ndarray
) -> tuple[float, int]:
    """Mean per-query Kendall tau-b, matching the structural RGCN guide."""
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


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_run(root: Path, variant: str, seed: int) -> dict:
    path = root / variant / f"seed{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "candidate_scores", "candidate_ids", "test_queries",
        "test_positive_tails", "model_state_dict", "message_program_sha256",
        "selected_path_program_sha256", "path_sampling", "sampling_seed",
        "sampled_long_paths_per_root", "physical_graph_hashes",
        "semantic_program_hashes",
    }
    missing = sorted(required - artifact.keys())
    if missing:
        raise ValueError(f"{path} is missing invariance fields: {missing}")
    return artifact


def aligned(left: dict, right: dict, name: str) -> None:
    if not torch.equal(left[name], right[name]):
        raise ValueError(f"Run alignment failed: {name} differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("results/dblp_lp"))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/kendall_tau_dblp_lp_invariant.csv"),
    )
    parser.add_argument(
        "--summary-output", type=Path,
        help="Mean/std table; defaults to <output_stem>_summary.csv.",
    )
    parser.add_argument(
        "--allow-nonexact", action="store_true",
        help="Report without failing when scores/checkpoints are not bit-identical.",
    )
    args = parser.parse_args()
    rows = []
    failures = []
    for seed in args.seeds:
        runs = {
            variant: load_run(args.root, variant, seed)
            for variant in args.variants
        }
        for left_name, right_name in combinations(args.variants, 2):
            left, right = runs[left_name], runs[right_name]
            physical_hashes = left["physical_graph_hashes"]
            semantic_hashes = left["semantic_program_hashes"]
            if (
                not isinstance(physical_hashes, dict)
                or len(physical_hashes) != 3
                or len(set(physical_hashes.values())) != 3
            ):
                raise ValueError("Invariant source physical graph hashes are not distinct")
            if (
                not isinstance(semantic_hashes, dict)
                or len(semantic_hashes) != 3
                or len(set(semantic_hashes.values())) != 1
            ):
                raise ValueError("Invariant compiled semantic graph hashes do not match")
            for field in ("candidate_ids", "test_queries", "test_positive_tails"):
                aligned(left, right, field)
            provenance_fields = (
                "message_program_sha256",
                "selected_path_program_sha256",
                "selected_path_weights_sha256",
                "path_sampling",
                "sampling_seed",
                "sampled_long_paths_per_root",
                "physical_graph_hashes",
                "semantic_program_hashes",
            )
            for field in provenance_fields:
                if left.get(field) != right.get(field):
                    raise ValueError(
                        f"Run provenance failed: {field} differs between "
                        f"{left_name} and {right_name}"
                    )
            left_scores = left["candidate_scores"].numpy()
            right_scores = right["candidate_scores"].numpy()
            difference = left_scores - right_scores
            tau, valid_queries = query_macro_kendall(left_scores, right_scores)
            flat_tau = flat_kendall(left_scores, right_scores)
            left_hash = state_hash(left["model_state_dict"])
            right_hash = state_hash(right["model_state_dict"])
            exact_scores = bool(np.array_equal(left_scores, right_scores))
            exact_checkpoint = left_hash == right_hash
            row = {
                "seed": seed,
                "variant_a": left_name,
                "variant_b": right_name,
                "kendall_tau": tau,
                "kendall_tau_flat_candidate_scores": flat_tau,
                "num_test_queries": int(left_scores.shape[0]),
                "valid_query_taus": valid_queries,
                "max_abs_score_diff": float(np.max(np.abs(difference))),
                "mean_abs_score_diff": float(np.mean(np.abs(difference))),
                "candidate_scores_exact": exact_scores,
                "checkpoint_hash_equal": exact_checkpoint,
                "checkpoint_hash_a": left_hash,
                "checkpoint_hash_b": right_hash,
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
        record = {
            "variant_a": left_name,
            "variant_b": right_name,
            "seed_count": int(len(group)),
            "candidate_scores_exact_all_seeds": bool(
                group["candidate_scores_exact"].all()
            ),
            "checkpoint_hash_equal_all_seeds": bool(
                group["checkpoint_hash_equal"].all()
            ),
            "max_abs_score_diff": float(group["max_abs_score_diff"].max()),
        }
        for metric in (
            "kendall_tau", "kendall_tau_flat_candidate_scores"
        ):
            values = group[metric].to_numpy(dtype=np.float64)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        summary_rows.append(record)
    summary_output = args.summary_output or args.output.with_name(
        f"{args.output.stem}_summary.csv"
    )
    atomic_write_csv(pd.DataFrame(summary_rows), summary_output)
    print(frame.to_string(index=False))
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_output}")
    if failures and not args.allow_nonexact:
        raise RuntimeError(
            f"Invariant audit failed for {len(failures)} matched-seed pair(s); "
            "inspect the CSV and confirm all runs used one GPU architecture."
        )


if __name__ == "__main__":
    main()
