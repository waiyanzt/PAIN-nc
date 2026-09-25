"""Generate explicitly simulated Freebase augmentation Kendall-tau placeholders.

Accuracy and Macro-F1 cannot determine Kendall tau: they do not specify the
per-book ordering of all seven class scores. This script uses the real test
labels only as a simulation scaffold, constructs synthetic score matrices
whose predictions approximate the draft classification metric entries, then
computes rowwise Kendall tau-b exactly as build_paper_tables.py does. The
shared-score/noise model is an assumption, so its output is a layout
simulation, not an estimate or an experimental result.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau


SEEDS = (1566911444, 20241017, 20251017)
VARIANTS = ("Freebase1", "Freebase2", "Freebase3")
STAT = re.compile(r"([0-9]+\.[0-9]+) \$\\pm\$ ([0-9]+\.[0-9]+)")


@dataclass(frozen=True)
class Target:
    accuracy: float
    accuracy_std: float
    precision: float
    precision_std: float
    recall: float
    recall_std: float
    macro_f1: float
    macro_f1_std: float


def read_targets(path: Path) -> dict[str, Target]:
    """Read the Freebase augmentation row values already in the draft table."""
    text = path.read_text(encoding="utf-8")
    if "SIMULATED DRAFT" not in text:
        raise ValueError("Expected a results table explicitly labeled as a simulated draft")
    block = text.split(r"\multirow{11}{*}{Freebase}", 1)[1]
    block = block.split(r"\multirow{3}{*}{Augmentation}", 1)[1]
    block = block.split(r"\cline{2-8}", 1)[0]
    targets = {}
    for line in block.splitlines():
        match = re.search(r"Freebase[123]", line)
        if match is None:
            continue
        stats = [(float(mean), float(std)) for mean, std in STAT.findall(line)]
        if len(stats) != 5:
            raise ValueError(f"Expected five metric cells in {match.group(0)}")
        if stats[0] != stats[3]:
            raise ValueError(
                f"Accuracy and Micro-F1 disagree in {match.group(0)}"
            )
        targets[match.group(0)] = Target(
            *stats[0], *stats[1], *stats[2], *stats[4]
        )
    if set(targets) != set(VARIANTS):
        raise ValueError("Expected Freebase1, Freebase2, and Freebase3 targets")
    return targets


def read_test_labels(path: Path) -> np.ndarray:
    shared = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    labels = shared["y"][shared["test_mask"]].numpy().astype(np.int64)
    if len(labels) == 0 or labels.min() < 0:
        raise ValueError("The shared artifact has no valid test labels")
    return labels


def metrics(
    labels: np.ndarray, prediction: np.ndarray, classes: int
) -> tuple[float, float, float, float]:
    confusion = np.bincount(
        labels * classes + prediction, minlength=classes * classes
    ).reshape(classes, classes)
    correct = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(axis=0)
    actual = confusion.sum(axis=1)
    precision = np.divide(
        correct, predicted, out=np.zeros(classes, dtype=np.float64),
        where=predicted != 0,
    )
    recall = np.divide(
        correct, actual, out=np.zeros(classes, dtype=np.float64),
        where=actual != 0,
    )
    denominator = predicted + actual
    macro_f1 = np.divide(
        2 * correct,
        denominator,
        out=np.zeros(classes, dtype=np.float64),
        where=denominator != 0,
    ).mean()
    return (
        float(correct.sum() / len(labels)),
        float(precision.mean()),
        float(recall.mean()),
        float(macro_f1),
    )


def fit_scores(
    labels: np.ndarray,
    raw_scores: np.ndarray,
    target: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Fit synthetic score controls to the draft classification metrics."""
    classes = raw_scores.shape[1]
    counts = np.bincount(labels, minlength=classes)
    log_frequency = np.log(counts / counts.sum())
    rarity = -np.log(counts / counts.sum())
    rarity = (rarity - rarity.mean()) / rarity.std()
    row_ids = np.arange(len(labels))

    def search(
        strengths: np.ndarray,
        rare_weights: np.ndarray,
        biases: np.ndarray,
        best=None,
    ):
        for bias in biases:
            biased = raw_scores + bias * log_frequency
            for strength in strengths:
                for rare_weight in rare_weights:
                    scores = biased.copy()
                    scores[row_ids, labels] += strength + rare_weight * rarity[labels]
                    observed = metrics(labels, scores.argmax(axis=1), classes)
                    error = sum(
                        (value - desired) ** 2
                        for value, desired in zip(observed, target, strict=True)
                    )
                    if best is None or error < best[0]:
                        best = (
                            error, float(strength), float(rare_weight),
                            float(bias), observed,
                        )
        return best

    best = search(
        np.linspace(0.0, 4.0, 21),
        np.linspace(-1.0, 1.5, 16),
        np.linspace(0.0, 1.2, 9),
    )
    best = search(
        np.linspace(max(0.0, best[1] - 0.2), best[1] + 0.2, 11),
        np.linspace(best[2] - 0.2, best[2] + 0.2, 11),
        np.linspace(max(0.0, best[3] - 0.15), best[3] + 0.15, 7),
        best,
    )
    scores = raw_scores + best[3] * log_frequency
    scores[row_ids, labels] += best[1] + best[2] * rarity[labels]
    if any(
        abs(value - desired) > 0.01
        for value, desired in zip(best[4], target, strict=True)
    ):
        raise ValueError(
            "Synthetic score model did not fit the draft metrics closely enough: "
            f"target={target}, observed={best[4]}"
        )
    return scores, best[4]


def rowwise_kendall(left: np.ndarray, right: np.ndarray) -> float:
    """Mean Kendall tau-b across test books, matching build_paper_tables.py."""
    if left.shape != right.shape:
        raise ValueError("Synthetic class score shapes differ")
    values = []
    for left_row, right_row in zip(left, right, strict=True):
        if np.array_equal(left_row, right_row):
            values.append(1.0)
            continue
        value = kendalltau(left_row, right_row, variant="b", nan_policy="omit").statistic
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def simulate(
    labels: np.ndarray, targets: dict[str, Target], shared_weight: float
) -> dict[tuple[str, str], list[float]]:
    classes = int(labels.max() + 1)
    offsets = (-1.0, 0.0, 1.0)
    pairs = (
        ("Freebase1", "Freebase2"),
        ("Freebase1", "Freebase3"),
        ("Freebase2", "Freebase3"),
    )
    values = {pair: [] for pair in pairs}
    for seed_index, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        shared = rng.standard_normal((len(labels), classes))
        outputs = {}
        for variant_index, variant in enumerate(VARIANTS):
            target = targets[variant]
            offset = offsets[(seed_index + variant_index) % len(offsets)]
            target_metrics = (
                target.accuracy + offset * target.accuracy_std,
                target.precision + offset * target.precision_std,
                target.recall + offset * target.recall_std,
                target.macro_f1 + offset * target.macro_f1_std,
            )
            individual = rng.standard_normal((len(labels), classes))
            base = shared_weight * shared + individual
            scores, observed = fit_scores(labels, base, target_metrics)
            outputs[variant] = scores
            print(
                f"seed={seed} {variant}: target A/P/R/F1 "
                + "/".join(f"{value:.4f}" for value in target_metrics)
                + "; synthetic "
                + "/".join(f"{value:.4f}" for value in observed)
            )
        for pair in pairs:
            values[pair].append(rowwise_kendall(outputs[pair[0]], outputs[pair[1]]))
    return values


def update_tex(
    path: Path,
    values: dict[tuple[str, str], list[float]],
    *,
    shared_weight: float,
    dry_run: bool = False,
) -> None:
    text = path.read_text(encoding="utf-8")
    if "SIMULATED DRAFT" not in text:
        raise ValueError("Refusing to edit an invariance table without a simulated-draft label")
    text, note_count = re.subn(
        r"shared-score strength \([0-9]+(?:\.[0-9]+)?\) is an assumption",
        f"shared-score strength ({shared_weight:g}) is an assumption",
        text,
    )
    if note_count != 1:
        raise ValueError("Expected one synthetic-score assumption note")
    for (left, right), per_seed in values.items():
        mean = float(np.mean(per_seed))
        std = float(np.std(per_seed, ddof=1))
        pattern = re.compile(
            rf"({re.escape(left)} vs\. {re.escape(right)} & )"
            rf"[0-9]+\.[0-9]+ \$\\pm\$ [0-9]+\.[0-9]+( \\\\)"
        )
        # Limit the replacement to the Augmentation section of the Freebase block.
        start = text.index(r"\multirow{12}{*}{Freebase}")
        start = text.index(r"\multirow{3}{*}{Augmentation}", start)
        stop = text.index(r"\cline{2-4}", start)
        block = text[start:stop]
        updated, count = pattern.subn(
            lambda match: f"{match.group(1)}{mean:.4f} $\\pm$ {std:.4f}{match.group(2)}",
            block,
        )
        if count != 1:
            raise ValueError(f"Expected one simulated Augmentation row for {left}/{right}")
        text = text[:start] + updated + text[stop:]
        print(f"{left} vs. {right}: {mean:.4f} +/- {std:.4f} (synthetic)")
    if not dry_run:
        path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("results/pain_nc_results.tex"))
    parser.add_argument("--invariance", type=Path, default=Path("results/pain_nc_invariance.tex"))
    parser.add_argument("--shared", type=Path, default=Path("data/preprocessed/Freebase/shared.pt"))
    parser.add_argument(
        "--shared-weight", type=float, default=1.5,
        help="Assumed shared-score strength; changing it changes tau without changing target metrics.",
    )
    parser.add_argument(
        "--max-simulated-tau", type=float, default=0.71,
        help="Reject a simulated pair whose mean plus sample standard deviation exceeds this cap.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print values without changing LaTeX")
    args = parser.parse_args()
    if not 0 <= args.shared_weight <= 2:
        parser.error("--shared-weight must be between 0 and 2")
    labels = read_test_labels(args.shared)
    targets = read_targets(args.results)
    values = simulate(labels, targets, args.shared_weight)
    for pair, per_seed in values.items():
        upper = float(np.mean(per_seed) + np.std(per_seed, ddof=1))
        if not np.isfinite(upper) or upper > args.max_simulated_tau:
            raise ValueError(
                f"Synthetic tau for {pair} exceeds cap {args.max_simulated_tau:g}: "
                f"mean plus SD = {upper:.4f}"
            )
    update_tex(
        args.invariance, values, shared_weight=args.shared_weight,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
