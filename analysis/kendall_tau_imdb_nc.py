"""Pairwise test-set Kendall tau-b for PAIN-NC IMDb predictions."""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau


DEFAULT_VARIANTS = (
    "IMDb1",
    "IMDb2",
    "IMDb3",
    "IMDb4",
    "IMDb_universal",
)
DEFAULT_SEEDS = (1566911444, 20241017, 20251017)


def load_artifact(
    root: Path, variant: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = root / variant / f"seed{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    scores = np.asarray(artifact["y_prob"])
    labels = np.asarray(artifact["y_true"])
    node_ids = np.asarray(artifact["test_node_ids"])
    if scores.ndim != 2 or labels.shape != (scores.shape[0],):
        raise ValueError(f"Malformed predictions in {path}")
    return scores, labels, node_ids


def mean_node_tau(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    if scores_a.shape != scores_b.shape:
        raise ValueError(f"Score shapes differ: {scores_a.shape} vs {scores_b.shape}")
    taus = np.asarray(
        [
            kendalltau(a, b, variant="b", nan_policy="omit").statistic
            for a, b in zip(scores_a, scores_b)
        ],
        dtype=np.float64,
    )
    if np.isnan(taus).all():
        raise ValueError("Every node-level Kendall tau is undefined")
    return float(np.nanmean(taus))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="results/imdb_nc")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output", default="reports/kendall_tau_imdb_nc_test.csv"
    )
    args = parser.parse_args()

    root = Path(args.root)
    loaded = {
        (variant, seed): load_artifact(root, variant, seed)
        for variant in args.variants
        for seed in args.seeds
    }
    rows = []
    for variant_a, variant_b in itertools.combinations(args.variants, 2):
        seed_values = []
        for seed in args.seeds:
            scores_a, labels_a, nodes_a = loaded[(variant_a, seed)]
            scores_b, labels_b, nodes_b = loaded[(variant_b, seed)]
            if not np.array_equal(nodes_a, nodes_b):
                raise ValueError(
                    f"Test node order differs for {variant_a}, {variant_b}, seed={seed}"
                )
            if not np.array_equal(labels_a, labels_b):
                raise ValueError(
                    f"Test labels differ for {variant_a}, {variant_b}, seed={seed}"
                )
            value = mean_node_tau(scores_a, scores_b)
            seed_values.append(value)
            print(f"{variant_a} vs {variant_b} seed={seed}: tau-b={value:.6f}")
        values = np.asarray(seed_values, dtype=np.float64)
        rows.append(
            {
                "variant_a": variant_a,
                "variant_b": variant_b,
                "n_seeds": len(values),
                "kendall_tau_test_mean": float(np.nanmean(values)),
                "kendall_tau_test_std": (
                    float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
                ),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nMean +/- sample standard deviation across matched seeds")
    for row in rows:
        print(
            f"{row['variant_a']} vs {row['variant_b']}: "
            f"{row['kendall_tau_test_mean']:.4f} +/- "
            f"{row['kendall_tau_test_std']:.4f}"
        )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

