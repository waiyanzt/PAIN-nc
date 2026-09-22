"""Recompute PAIN DBLP-LP table rows with consistent paper metrics.

Run after all ordinary, augmentation, and invariant seed artifacts exist::

    python -m analysis.build_pain_dblp_tables

The handoff script supplies the result and scalability formatting. Kendall
tau@1/@3 follow the paper's H.2 top-k-union definition. Candidate metrics use
the handoff script's average-rank tie convention rather than PAIN's saved
filtered, pessimistic-tie ranks used for checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
from scipy.stats import kendalltau

from build_paper_tables import (
    Block,
    LP_COLS,
    LP_INV,
    Run,
    SCAL,
    VARIANT_LABEL,
    align,
    continue_rows,
    fmt,
    lp_metrics,
    row_tau,
    render_results,
    render_scal,
)

SEEDS = (1566911444, 20241017, 20251017)
VARIANTS = ("v1", "v2", "v3")
MIB = 1024**2


def _array(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _resources(record: dict, *, augmentation: bool) -> dict[str, float]:
    resources = record["resources"]
    train_seconds = (
        record["training_seconds"]
        if augmentation
        else record["history"][-1]["elapsed_seconds"]
    )
    return {
        "train_time_sec": float(train_seconds),
        # An augmentation optimizer step visits one physical variant.
        "epochs": float(
            record["optimizer_steps"] if augmentation else record["epochs_trained"]
        ),
        "parameter_mib": resources["parameter_bytes"] / MIB,
        "peak_training_gpu_mib": resources["training_gpu"]["gpu_peak_allocated_bytes"] / MIB,
        "peak_inference_gpu_mib": resources["inference_gpu"]["gpu_peak_allocated_bytes"] / MIB,
    }


def _prediction_run(
    prediction: dict,
    *,
    variant: str,
    seed: int,
    resources: dict[str, float],
) -> tuple[Run, np.ndarray]:
    scores = _array(prediction["candidate_scores"])
    candidates = _array(prediction["candidate_ids"]).astype(np.int64)
    queries = _array(prediction["test_queries"]).astype(np.int64)
    tails = _array(prediction["test_positive_tails"]).astype(np.int64)
    if scores.ndim != 2 or candidates.ndim != 1 or scores.shape != (len(queries), len(candidates)):
        raise ValueError(f"{variant} seed={seed}: malformed candidate score matrix")
    if len(tails) != len(queries) or len(np.unique(candidates)) != len(candidates):
        raise ValueError(f"{variant} seed={seed}: malformed candidate or positive IDs")
    if not np.isfinite(scores).all():
        raise ValueError(f"{variant} seed={seed}: nonfinite candidate scores")
    positions = {int(candidate): index for index, candidate in enumerate(candidates)}
    try:
        positive_columns = np.array([positions[int(tail)] for tail in tails], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"{variant} seed={seed}: true conference missing from candidates") from error
    # The DBLP contract has one true venue per paper. This also makes the
    # handoff script's unfiltered candidate set match the filtered population.
    if len(np.unique(queries)) != len(queries):
        raise ValueError(f"{variant} seed={seed}: multiple test venues per paper")
    row_ids = queries * 100003 + tails
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError(f"{variant} seed={seed}: nonunique query alignment IDs")
    return (
        Run(
            variant=variant,
            seed=seed,
            task_type="lp",
            scores=scores,
            pos_col=positive_columns,
            row_ids=row_ids,
            resources=resources,
            # Leave ranks empty: lp_metrics/invariance must use the handoff
            # script's average-tie candidate ranking from raw scores.
        ),
        candidates,
    )


def _load_pt(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed PAIN run: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def build_blocks(
    ordinary_root: Path,
    augmentation_root: Path,
    invariant_root: Path,
    seeds: tuple[int, ...],
) -> list[Block]:
    blocks: list[Block] = []
    reference_candidates: np.ndarray | None = None
    reference_rows: np.ndarray | None = None

    def add_run(prediction: dict, variant: str, seed: int, resources: dict) -> Run:
        nonlocal reference_candidates, reference_rows
        run, candidates = _prediction_run(
            prediction, variant=variant, seed=seed, resources=resources
        )
        rows = np.sort(run.row_ids)
        if reference_candidates is None:
            reference_candidates, reference_rows = candidates, rows
        elif not np.array_equal(candidates, reference_candidates) or not np.array_equal(
            rows, reference_rows
        ):
            raise ValueError(f"{variant} seed={seed}: candidates or test queries differ across arms")
        return run

    ordinary: list[Run] = []
    ordinary_directories = (
        ("v1", "DBLP1"),
        ("v2", "DBLP2"),
        ("v3", "DBLP3"),
        ("universal", "DBLP_universal"),
    )
    for variant, directory in ordinary_directories:
        for seed in seeds:
            artifact = _load_pt(ordinary_root / directory / f"seed{seed}.pt")
            ordinary.append(add_run(artifact, variant, seed, _resources(artifact, augmentation=False)))
    blocks.append(Block("DBLP", "paper_conference", "PAIN", "original", "", [r for r in ordinary if r.variant != "universal"]))
    blocks.append(Block("DBLP", "paper_conference", "PAIN", "canonical", "", [r for r in ordinary if r.variant == "universal"]))

    augmented: list[Run] = []
    for seed in seeds:
        seed_dir = augmentation_root / f"seed_{seed}"
        summary_path = seed_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing completed PAIN augmentation run: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        resources = _resources(summary, augmentation=True)
        for variant in VARIANTS:
            prediction = _load_pt(seed_dir / f"test_{variant}.pt")
            augmented.append(add_run(prediction, variant, seed, resources))
    blocks.append(Block("DBLP", "paper_conference", "PAIN", "augmentation", "", augmented))

    invariant: list[Run] = []
    for variant in VARIANTS:
        for seed in seeds:
            artifact = _load_pt(
                invariant_root / f"DBLP_invariant_{variant}" / f"seed{seed}.pt"
            )
            invariant.append(add_run(artifact, variant, seed, _resources(artifact, augmentation=False)))
    blocks.append(Block("DBLP", "paper_conference", "PAIN", "invariant", "", invariant))
    return blocks


def _top_k_tau(scores_a: np.ndarray, scores_b: np.ndarray, k: int) -> float:
    """Paper H.2: per-query tau on the union of the two top-k candidate sets."""
    values = []
    candidate_index = np.arange(scores_a.shape[1])
    for left, right in zip(scores_a, scores_b, strict=True):
        top_left = np.lexsort((candidate_index, -left))[:k]
        top_right = np.lexsort((candidate_index, -right))[:k]
        selected = np.union1d(top_left, top_right)
        if len(selected) < 2 or np.array_equal(left[selected], right[selected]):
            values.append(1.0)
            continue
        value = kendalltau(left[selected], right[selected], variant="b").statistic
        if not np.isnan(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def _paper_invariance(block: Block, seeds: tuple[int, ...]) -> list[str]:
    if block.method == "canonical":
        return ["% canonical: one graph, so no cross-variant ranking comparison"]
    runs = {(run.variant, run.seed): run for run in block.runs}
    lines = []
    for left_variant, right_variant in combinations(VARIANTS, 2):
        values = {"tau": [], "tau@1": [], "tau@3": []}
        for seed in seeds:
            left, right = align(runs[left_variant, seed], runs[right_variant, seed])
            values["tau"].append(row_tau(left.scores, right.scores))
            values["tau@1"].append(_top_k_tau(left.scores, right.scores, 1))
            values["tau@3"].append(_top_k_tau(left.scores, right.scores, 3))
        label = f"{VARIANT_LABEL['DBLP'][left_variant]} vs. {VARIANT_LABEL['DBLP'][right_variant]}"
        lines.append(label + " & " + " & ".join(fmt(values[name]) for name in ("tau", "tau@1", "tau@3")) + r" \\")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ordinary-root", type=Path, default=Path("results/dblp_lp"))
    parser.add_argument("--augmentation-root", type=Path, default=Path("results/dblp_lp_augmentation"))
    parser.add_argument("--invariant-root", type=Path, default=Path("results/dblp_lp_invariant"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output", type=Path, default=Path("reports/pain_dblp_lp_tables.tex"))
    args = parser.parse_args()
    blocks = build_blocks(
        args.ordinary_root,
        args.augmentation_root,
        args.invariant_root,
        tuple(args.seeds),
    )
    lines = [
        "% PAIN DBLP-LP; metric functions imported from build_paper_tables.py",
        "% mean +/- population std (ddof=0) across matched seeds; four decimals",
        "% binary metrics: sigmoid(logit) >= 0.5 on all 20 conference candidates",
        "% Hits/MRR: unfiltered candidate ranks with average-rank ties, as in the handoff script",
        "% PAIN checkpoints were selected with filtered, pessimistic-tie validation MRR",
        "% paper tau@1/@3: per-query tau-b on union of the two top-k candidate sets (PDF H.2)",
        "% tied top-k membership is resolved by candidate column order; undefined tau rows are omitted",
        "% augmentation epochs column counts optimizer updates (one variant per update)",
    ]
    for heading, columns, renderer in (
        ("RESULTS: precision recall F1 Hits@1 Hits@3 MRR", LP_COLS, render_results),
        ("INVARIANCE (paper H.2 top-k; tau-b ties): tau tau@1 tau@3", LP_INV, _paper_invariance),
        ("SCALABILITY: " + " ".join(SCAL), (), render_scal),
    ):
        lines.extend(("", "% " + heading))
        for block in blocks:
            lines.append("% PAIN " + block.method)
            if renderer is render_results:
                rows = render_results(block, columns, lp_metrics)
            elif renderer is _paper_invariance:
                rows = _paper_invariance(block, tuple(args.seeds))
            else:
                rows = render_scal(block)
            lines.extend(continue_rows(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
