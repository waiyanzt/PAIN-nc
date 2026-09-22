#!/usr/bin/env python3
"""Build PAIN-only, Overleaf-ready tables from per-run result artifacts.

The script deliberately reads the individual run artifacts instead of trusting
top-level aggregate CSV files, because those files are rewritten when a subset
of variants or seeds is launched. Missing runs remain visibly partial in the
tables and are listed in ``PAIN_COMPLETENESS.md``.

Run from the repository root:

    python build_paper_tables.py

PyTorch, NumPy, and SciPy are required. No training data are loaded and no
model is instantiated; the script only reads completed result artifacts.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import kendalltau


REPO = Path(__file__).resolve().parent
DEFAULT_SEEDS = (1566911444, 20241017, 20251017)
MIB = 1024.0**2

IMDB_VARIANTS = ("v1", "v2", "v3", "v4")
DBLP_VARIANTS = ("v1", "v2", "v3")
DISPLAY = {
    "IMDB": {"v1": "IMDb1", "v2": "IMDb2", "v3": "IMDb3", "v4": "IMDb4"},
    "DBLP": {"v1": "DBLP1", "v2": "DBLP2", "v3": "DBLP3"},
}

NC_METRICS = (
    ("accuracy", "Accuracy"),
    ("precision_macro", "Macro precision"),
    ("recall_macro", "Macro recall"),
    ("f1_micro", "Micro-F1"),
    ("f1_macro", "Macro-F1"),
)
LP_METRICS = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("hits_at_1", "Hits@1"),
    ("hits_at_3", "Hits@3"),
    ("mrr", "MRR"),
)
SCALABILITY_METRICS = (
    ("training_time_sec", "Train time (s)"),
    ("epochs", "Epochs / updates"),
    ("parameter_mib", "Parameters (MiB)"),
    ("peak_training_gpu_mib", "Peak train GPU (MiB)"),
    ("peak_inference_gpu_mib", "Peak inference GPU (MiB)"),
)


@dataclass
class Run:
    dataset: str
    task: str
    method: str
    variant: str
    seed: int
    metrics: dict[str, float]
    resources: dict[str, float]
    source: Path
    scores: np.ndarray | None = None
    row_ids: np.ndarray | None = None
    candidate_ids: np.ndarray | None = None


@dataclass(frozen=True)
class RowSpec:
    method: str
    variant: str
    label: str


@dataclass
class ExpectedRun:
    dataset: str
    task: str
    method: str
    variant: str
    seed: int
    source: Path
    state: str
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_tables/pain"))
    parser.add_argument("--imdb-original-root", type=Path, default=Path("results/imdb_nc"))
    parser.add_argument("--imdb-v4-root", type=Path, default=Path("results/imdb_nc_v4_fixed"))
    parser.add_argument("--imdb-universal-root", type=Path, default=Path("results/imdb_nc_universal"))
    parser.add_argument("--imdb-augmentation-root", type=Path, default=Path("results/imdb_nc_augmentation_v4_fixed"))
    parser.add_argument("--imdb-invariant-root", type=Path, default=Path("results/imdb_nc_invariant"))
    parser.add_argument("--dblp-original-root", type=Path, default=Path("results/dblp_lp"))
    parser.add_argument("--dblp-augmentation-root", type=Path, default=Path("results/dblp_lp_augmentation"))
    parser.add_argument("--dblp-invariant-root", type=Path, default=Path("results/dblp_lp_invariant"))
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def load_pt(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to read the saved .pt result artifacts. "
            "Run this script in the same environment used for PAIN."
        ) from exc
    return torch.load(path, map_location="cpu", weights_only=False)


def scalar(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def resource_summary(record: dict[str, Any], *, augmentation: bool) -> dict[str, float]:
    resources = record.get("resources", {})
    training_gpu = resources.get("training_gpu", {})
    inference_gpu = resources.get("inference_gpu", {})
    if augmentation:
        elapsed = record.get("training_seconds")
        accounting = record.get("epoch_accounting", {})
        epochs = accounting.get("variant_epochs_ran", record.get("optimizer_steps"))
    else:
        elapsed = record.get("elapsed_seconds")
        epochs = record.get("epochs_trained")
    values = {
        "training_time_sec": elapsed,
        "epochs": epochs,
        "parameter_mib": resources.get("parameter_bytes"),
        "peak_training_gpu_mib": training_gpu.get("gpu_peak_allocated_bytes"),
        "peak_inference_gpu_mib": inference_gpu.get("gpu_peak_allocated_bytes"),
    }
    for key in ("parameter_mib", "peak_training_gpu_mib", "peak_inference_gpu_mib"):
        if values[key] is not None:
            values[key] = scalar(values[key]) / MIB
    return {key: scalar(value) for key, value in values.items() if value is not None}


def metric_dict(record: dict[str, Any], task: str) -> dict[str, float]:
    if task == "nc":
        keys = {
            "accuracy": "test_accuracy",
            "precision_macro": "test_precision_macro",
            "recall_macro": "test_recall_macro",
            "f1_micro": "test_f1_micro",
            "f1_macro": "test_f1_macro",
        }
    else:
        keys = {
            "precision": "test_precision",
            "recall": "test_recall",
            "f1": "test_f1",
            "hits_at_1": "test_hits_at_1",
            "hits_at_3": "test_hits_at_3",
            "mrr": "test_mrr",
        }
    return {name: scalar(record[source]) for name, source in keys.items()}


def append_expected(
    expected: list[ExpectedRun], dataset: str, task: str, method: str,
    variant: str, seed: int, source: Path, action,
) -> Run | None:
    if not source.is_file():
        expected.append(ExpectedRun(dataset, task, method, variant, seed, source, "missing"))
        return None
    try:
        run = action()
    except Exception as exc:
        expected.append(ExpectedRun(dataset, task, method, variant, seed, source, "unreadable", str(exc)))
        return None
    expected.append(ExpectedRun(dataset, task, method, variant, seed, source, "loaded"))
    return run


def regular_run(
    dataset: str, task: str, method: str, variant: str, seed: int, path: Path
) -> Run:
    record = load_pt(path)
    if task == "nc":
        scores = as_numpy(record["y_prob"])
        row_ids = as_numpy(record["test_node_ids"])
        candidates = None
    else:
        scores = as_numpy(record["candidate_scores"])
        heads = as_numpy(record["test_queries"])
        tails = as_numpy(record["test_positive_tails"])
        row_ids = np.column_stack((heads, tails))
        candidates = as_numpy(record["candidate_ids"])
    return Run(
        dataset=dataset,
        task=task,
        method=method,
        variant=variant,
        seed=seed,
        metrics=metric_dict(record, task),
        resources=resource_summary(record, augmentation=False),
        source=path,
        scores=scores,
        row_ids=row_ids,
        candidate_ids=candidates,
    )


def json_metric_dict(metrics: dict[str, Any], task: str) -> dict[str, float]:
    if task == "nc":
        aliases = {
            "accuracy": "Accuracy",
            "precision_macro": "Precision_macro",
            "recall_macro": "Recall_macro",
            "f1_micro": "Micro_F1",
            "f1_macro": "Macro_F1",
        }
    else:
        aliases = {
            "precision": "test_precision",
            "recall": "test_recall",
            "f1": "test_f1",
            "hits_at_1": "test_hits_at_1",
            "hits_at_3": "test_hits_at_3",
            "mrr": "test_mrr",
        }
    return {name: scalar(metrics[source]) for name, source in aliases.items()}


def read_imdb_score_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty score file: {path}")
    logit_columns = sorted(
        (name for name in rows[0] if name.startswith("logit_class_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    row_ids = np.asarray([int(row["node_id"]) for row in rows], dtype=np.int64)
    scores = np.asarray(
        [[float(row[column]) for column in logit_columns] for row in rows],
        dtype=np.float64,
    )
    return row_ids, scores


def augmentation_run(
    dataset: str, task: str, variant: str, seed: int, seed_dir: Path
) -> Run:
    summary_path = seed_dir / "summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    metrics = json_metric_dict(summary["per_variant_test_metrics"][variant], task)
    if task == "nc":
        result_path = seed_dir / f"test_scores_{variant}.csv"
        row_ids, scores = read_imdb_score_csv(result_path)
        candidates = None
    else:
        result_path = seed_dir / f"test_{variant}.pt"
        result = load_pt(result_path)
        scores = as_numpy(result["candidate_scores"])
        heads = as_numpy(result["test_queries"])
        tails = as_numpy(result["test_positive_tails"])
        row_ids = np.column_stack((heads, tails))
        candidates = as_numpy(result["candidate_ids"])
    return Run(
        dataset=dataset,
        task=task,
        method="Augmentation",
        variant=variant,
        seed=seed,
        metrics=metrics,
        resources=resource_summary(summary, augmentation=True),
        source=result_path,
        scores=scores,
        row_ids=row_ids,
        candidate_ids=candidates,
    )


def collect_runs(args: argparse.Namespace) -> tuple[list[Run], list[ExpectedRun]]:
    seeds = tuple(args.seeds)
    runs: list[Run] = []
    expected: list[ExpectedRun] = []

    imdb_original = absolute(args.imdb_original_root)
    imdb_v4 = absolute(args.imdb_v4_root)
    imdb_universal = absolute(args.imdb_universal_root)
    imdb_augmentation = absolute(args.imdb_augmentation_root)
    imdb_invariant = absolute(args.imdb_invariant_root)
    for variant in IMDB_VARIANTS:
        directory = imdb_v4 if variant == "v4" else imdb_original
        folder = DISPLAY["IMDB"][variant]
        for seed in seeds:
            path = directory / folder / f"seed{seed}.pt"
            run = append_expected(
                expected, "IMDB", "nc", "Independent", variant, seed, path,
                lambda p=path, v=variant, s=seed: regular_run("IMDB", "nc", "Independent", v, s, p),
            )
            if run:
                runs.append(run)
    for seed in seeds:
        path = imdb_universal / "IMDb_universal" / f"seed{seed}.pt"
        run = append_expected(
            expected, "IMDB", "nc", "Universal graph", "universal", seed, path,
            lambda p=path, s=seed: regular_run("IMDB", "nc", "Universal graph", "universal", s, p),
        )
        if run:
            runs.append(run)
    for variant in IMDB_VARIANTS:
        for seed in seeds:
            seed_dir = imdb_augmentation / f"seed_{seed}"
            source = seed_dir / f"test_scores_{variant}.csv"
            run = append_expected(
                expected, "IMDB", "nc", "Augmentation", variant, seed, source,
                lambda d=seed_dir, v=variant, s=seed: augmentation_run("IMDB", "nc", v, s, d),
            )
            if run:
                runs.append(run)
    for variant in IMDB_VARIANTS:
        folder = f"IMDb_invariant_{variant}"
        for seed in seeds:
            path = imdb_invariant / folder / f"seed{seed}.pt"
            run = append_expected(
                expected, "IMDB", "nc", "Invariant", variant, seed, path,
                lambda p=path, v=variant, s=seed: regular_run("IMDB", "nc", "Invariant", v, s, p),
            )
            if run:
                runs.append(run)

    dblp_original = absolute(args.dblp_original_root)
    dblp_augmentation = absolute(args.dblp_augmentation_root)
    dblp_invariant = absolute(args.dblp_invariant_root)
    for variant in DBLP_VARIANTS:
        folder = DISPLAY["DBLP"][variant]
        for seed in seeds:
            path = dblp_original / folder / f"seed{seed}.pt"
            run = append_expected(
                expected, "DBLP", "lp", "Independent", variant, seed, path,
                lambda p=path, v=variant, s=seed: regular_run("DBLP", "lp", "Independent", v, s, p),
            )
            if run:
                runs.append(run)
    for seed in seeds:
        path = dblp_original / "DBLP_universal" / f"seed{seed}.pt"
        run = append_expected(
            expected, "DBLP", "lp", "Universal graph", "universal", seed, path,
            lambda p=path, s=seed: regular_run("DBLP", "lp", "Universal graph", "universal", s, p),
        )
        if run:
            runs.append(run)
    for variant in DBLP_VARIANTS:
        for seed in seeds:
            seed_dir = dblp_augmentation / f"seed_{seed}"
            source = seed_dir / f"test_{variant}.pt"
            run = append_expected(
                expected, "DBLP", "lp", "Augmentation", variant, seed, source,
                lambda d=seed_dir, v=variant, s=seed: augmentation_run("DBLP", "lp", v, s, d),
            )
            if run:
                runs.append(run)
    for variant in DBLP_VARIANTS:
        folder = f"DBLP_invariant_{variant}"
        for seed in seeds:
            path = dblp_invariant / folder / f"seed{seed}.pt"
            run = append_expected(
                expected, "DBLP", "lp", "Invariant", variant, seed, path,
                lambda p=path, v=variant, s=seed: regular_run("DBLP", "lp", "Invariant", v, s, p),
            )
            if run:
                runs.append(run)
    return runs, expected


def result_specs(dataset: str) -> list[RowSpec]:
    variants = IMDB_VARIANTS if dataset == "IMDB" else DBLP_VARIANTS
    union = r"$\bigcup_i \mathrm{IMDb}_i$" if dataset == "IMDB" else r"$\bigcup_i \mathrm{DBLP}_i$"
    rows = [RowSpec("Independent", v, DISPLAY[dataset][v]) for v in variants]
    rows.append(RowSpec("Universal graph", "universal", union))
    rows.extend(RowSpec("Augmentation", v, DISPLAY[dataset][v]) for v in variants)
    rows.extend(RowSpec("Invariant", v, DISPLAY[dataset][v]) for v in variants)
    return rows


def matching(runs: Iterable[Run], dataset: str, method: str, variant: str) -> list[Run]:
    return sorted(
        (run for run in runs if run.dataset == dataset and run.method == method and run.variant == variant),
        key=lambda run: run.seed,
    )


def finite(values: Iterable[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def mean_std(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    values = finite(values)
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=0))


def latex_stat(values: Iterable[float | None], count: int, expected: int, decimals: int) -> str:
    mean, deviation = mean_std(values)
    if mean is None:
        return "--"
    result = rf"${mean:.{decimals}f} \pm {deviation:.{decimals}f}$"
    if count < expected:
        result += r"\textsuperscript{$\dagger$}"
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def table_document(
    *, columns: str, header: str, body: list[str], caption: str, label: str, note: str
) -> str:
    return "\n".join(
        [
            "% Generated by build_paper_tables.py; do not hand-edit this copy.",
            r"\begin{table*}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\resizebox{\textwidth}{!}{%",
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *[row + r" \\" for row in body],
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{0.99\textwidth}\footnotesize",
            note,
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )


def write_results(dataset: str, task: str, runs: list[Run], seeds: tuple[int, ...], out: Path) -> None:
    metrics = NC_METRICS if task == "nc" else LP_METRICS
    aggregate_rows: list[dict[str, Any]] = []
    tex_rows: list[str] = []
    for spec in result_specs(dataset):
        group = matching(runs, dataset, spec.method, spec.variant)
        record: dict[str, Any] = {
            "dataset": dataset,
            "task": task,
            "method": spec.method,
            "variant": spec.variant,
            "n_seeds": len(group),
            "expected_seeds": len(seeds),
            "complete": len(group) == len(seeds),
        }
        cells = [spec.method, spec.label, f"{len(group)}/{len(seeds)}"]
        for key, _label in metrics:
            mean, deviation = mean_std(run.metrics.get(key) for run in group)
            record[f"{key}_mean"] = mean
            record[f"{key}_std"] = deviation
            cells.append(latex_stat((run.metrics.get(key) for run in group), len(group), len(seeds), 4))
        aggregate_rows.append(record)
        tex_rows.append(" & ".join(cells))

    metric_fields = [name for key, _ in metrics for name in (f"{key}_mean", f"{key}_std")]
    prefix = "PAIN_IMDB_nc" if dataset == "IMDB" else "PAIN_DBLP_lp"
    write_csv(
        out / f"{prefix}_results.csv",
        aggregate_rows,
        ["dataset", "task", "method", "variant", "n_seeds", "expected_seeds", "complete", *metric_fields],
    )
    metric_header = " & ".join(label for _key, label in metrics)
    caption_task = "IMDb node classification" if task == "nc" else "DBLP paper--conference link prediction"
    note = (
        r"Mean $\pm$ population standard deviation over seeds. The Seeds column is loaded/expected. "
        r"$\dagger$ marks a provisional statistic computed from an incomplete seed set; -- means no run is available. "
        r"IMDb4 independent runs and all IMDb augmentation rows come from the corrected v4 result roots."
        if dataset == "IMDB"
        else
        r"Mean $\pm$ population standard deviation over seeds. The Seeds column is loaded/expected. "
        r"$\dagger$ marks a provisional statistic computed from an incomplete seed set; -- means no run is available. "
        r"Ranking metrics use the filtered full-conference evaluation stored by each run."
    )
    tex = table_document(
        columns="lll" + "c" * len(metrics),
        header=f"Method & Evaluation graph & Seeds & {metric_header}",
        body=tex_rows,
        caption=f"PAIN results for {caption_task}.",
        label=f"tab:pain-{'imdb-nc' if task == 'nc' else 'dblp-lp'}-results",
        note=note,
    )
    (out / f"{prefix}_results.tex").write_text(tex, encoding="utf-8")


def id_keys(values: np.ndarray) -> list[Any]:
    array = np.asarray(values)
    if array.ndim == 1:
        return [value.item() if hasattr(value, "item") else value for value in array]
    return [tuple(row.tolist()) for row in array]


def aligned_scores(left: Run, right: Run) -> tuple[np.ndarray, np.ndarray]:
    if left.scores is None or right.scores is None or left.row_ids is None or right.row_ids is None:
        raise ValueError("run has no score matrix or row identifiers")
    left_keys = id_keys(left.row_ids)
    right_keys = id_keys(right.row_ids)
    right_positions = {key: index for index, key in enumerate(right_keys)}
    common = [key for key in left_keys if key in right_positions]
    if not common:
        raise ValueError("variant pair has no shared test rows")
    left_positions = {key: index for index, key in enumerate(left_keys)}
    left_scores = left.scores[[left_positions[key] for key in common]]
    right_scores = right.scores[[right_positions[key] for key in common]]

    if left.candidate_ids is not None and right.candidate_ids is not None:
        left_candidates = id_keys(left.candidate_ids)
        right_candidates = id_keys(right.candidate_ids)
        right_columns = {key: index for index, key in enumerate(right_candidates)}
        shared_candidates = [key for key in left_candidates if key in right_columns]
        if not shared_candidates:
            raise ValueError("variant pair has no shared ranking candidates")
        left_columns = {key: index for index, key in enumerate(left_candidates)}
        left_scores = left_scores[:, [left_columns[key] for key in shared_candidates]]
        right_scores = right_scores[:, [right_columns[key] for key in shared_candidates]]
    if left_scores.shape != right_scores.shape:
        raise ValueError(f"aligned score shapes differ: {left_scores.shape} vs {right_scores.shape}")
    return left_scores, right_scores


def rowwise_kendall(left: Run, right: Run) -> float:
    left_scores, right_scores = aligned_scores(left, right)
    values: list[float] = []
    for left_row, right_row in zip(left_scores, right_scores, strict=True):
        if np.array_equal(left_row, right_row):
            values.append(1.0)
            continue
        value = kendalltau(left_row, right_row, variant="b", nan_policy="omit").statistic
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def write_invariance(dataset: str, runs: list[Run], seeds: tuple[int, ...], out: Path) -> None:
    variants = IMDB_VARIANTS if dataset == "IMDB" else DBLP_VARIANTS
    methods = ("Independent", "Augmentation", "Invariant")
    seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    tex_rows: list[str] = []
    for method in methods:
        for left_variant, right_variant in itertools.combinations(variants, 2):
            values: list[float] = []
            for seed in seeds:
                left = next((run for run in runs if run.dataset == dataset and run.method == method and run.variant == left_variant and run.seed == seed), None)
                right = next((run for run in runs if run.dataset == dataset and run.method == method and run.variant == right_variant and run.seed == seed), None)
                if left is None or right is None:
                    continue
                try:
                    tau = rowwise_kendall(left, right)
                except ValueError:
                    continue
                if math.isfinite(tau):
                    values.append(tau)
                    seed_rows.append(
                        {"dataset": dataset, "method": method, "variant_a": left_variant, "variant_b": right_variant, "seed": seed, "kendall_tau_b": tau}
                    )
            mean, deviation = mean_std(values)
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "variant_a": left_variant,
                    "variant_b": right_variant,
                    "matched_seeds": len(values),
                    "expected_seeds": len(seeds),
                    "complete": len(values) == len(seeds),
                    "kendall_tau_b_mean": mean,
                    "kendall_tau_b_std": deviation,
                }
            )
            pair = f"{DISPLAY[dataset][left_variant]}--{DISPLAY[dataset][right_variant]}"
            tex_rows.append(" & ".join([method, pair, f"{len(values)}/{len(seeds)}", latex_stat(values, len(values), len(seeds), 4)]))

    prefix = "PAIN_IMDB_nc" if dataset == "IMDB" else "PAIN_DBLP_lp"
    write_csv(
        out / f"{prefix}_invariance_per_seed.csv",
        seed_rows,
        ["dataset", "method", "variant_a", "variant_b", "seed", "kendall_tau_b"],
    )
    write_csv(
        out / f"{prefix}_invariance.csv",
        aggregate_rows,
        ["dataset", "method", "variant_a", "variant_b", "matched_seeds", "expected_seeds", "complete", "kendall_tau_b_mean", "kendall_tau_b_std"],
    )
    task_name = "IMDb node classification" if dataset == "IMDB" else "DBLP link prediction"
    tex = table_document(
        columns="llcc",
        header=r"Method & Variant pair & Matched seeds & Kendall $\tau_b$",
        body=tex_rows,
        caption=f"PAIN representation invariance for {task_name}.",
        label=f"tab:pain-{'imdb-nc' if dataset == 'IMDB' else 'dblp-lp'}-invariance",
        note=(
            r"For each matched seed and test node/query, Kendall's $\tau_b$ compares the class/candidate score rankings; "
            r"the test-row mean is then aggregated across seeds. Identical score rows are assigned $\tau_b=1$. "
            r"$\dagger$ marks an incomplete matched-seed set; -- means no matched pair is available. "
            r"The universal-graph arm has only one representation and therefore has no cross-variant pair."
        ),
    )
    (out / f"{prefix}_invariance.tex").write_text(tex, encoding="utf-8")


def scalability_specs(dataset: str) -> list[RowSpec]:
    variants = IMDB_VARIANTS if dataset == "IMDB" else DBLP_VARIANTS
    union = r"$\bigcup_i \mathrm{IMDb}_i$" if dataset == "IMDB" else r"$\bigcup_i \mathrm{DBLP}_i$"
    rows = [RowSpec("Independent", v, DISPLAY[dataset][v]) for v in variants]
    rows.append(RowSpec("Universal graph", "universal", union))
    rows.append(RowSpec("Augmentation", "shared", f"{DISPLAY[dataset][variants[0]]}--{DISPLAY[dataset][variants[-1]]} (shared)"))
    rows.extend(RowSpec("Invariant", v, DISPLAY[dataset][v]) for v in variants)
    return rows


def write_scalability(dataset: str, runs: list[Run], seeds: tuple[int, ...], out: Path) -> None:
    aggregate_rows: list[dict[str, Any]] = []
    tex_rows: list[str] = []
    for spec in scalability_specs(dataset):
        candidates = matching(runs, dataset, spec.method, "v1" if spec.variant == "shared" else spec.variant)
        group = list({run.seed: run for run in candidates}.values())
        record: dict[str, Any] = {
            "dataset": dataset,
            "method": spec.method,
            "training_graph": spec.variant,
            "n_seeds": len(group),
            "expected_seeds": len(seeds),
            "complete": len(group) == len(seeds),
        }
        cells = [spec.method, spec.label, f"{len(group)}/{len(seeds)}"]
        for key, _label in SCALABILITY_METRICS:
            values = [run.resources.get(key) for run in group]
            mean, deviation = mean_std(values)
            record[f"{key}_mean"] = mean
            record[f"{key}_std"] = deviation
            cells.append(latex_stat(values, len(group), len(seeds), 2))
        aggregate_rows.append(record)
        tex_rows.append(" & ".join(cells))

    metric_fields = [name for key, _ in SCALABILITY_METRICS for name in (f"{key}_mean", f"{key}_std")]
    prefix = "PAIN_IMDB_nc" if dataset == "IMDB" else "PAIN_DBLP_lp"
    write_csv(
        out / f"{prefix}_scalability.csv",
        aggregate_rows,
        ["dataset", "method", "training_graph", "n_seeds", "expected_seeds", "complete", *metric_fields],
    )
    header = "Method & Training graph & Seeds & " + " & ".join(label for _key, label in SCALABILITY_METRICS)
    tex = table_document(
        columns="lll" + "c" * len(SCALABILITY_METRICS),
        header=header,
        body=tex_rows,
        caption=f"PAIN scalability for {'IMDb node classification' if dataset == 'IMDB' else 'DBLP link prediction'}.",
        label=f"tab:pain-{'imdb-nc' if dataset == 'IMDB' else 'dblp-lp'}-scalability",
        note=(
            r"Mean $\pm$ population standard deviation. GPU columns report peak allocated memory, not reserved memory. "
            r"Parameters exclude buffers. For augmentation, epochs/updates are variant-level optimizer updates and the shared row is reported once. "
            r"$\dagger$ marks an incomplete seed set; -- means no run is available."
        ),
    )
    (out / f"{prefix}_scalability.tex").write_text(tex, encoding="utf-8")


def write_raw_runs(runs: list[Run], out: Path) -> None:
    metric_names = sorted({name for run in runs for name in run.metrics})
    resource_names = sorted({name for run in runs for name in run.resources})
    rows = []
    for run in sorted(runs, key=lambda item: (item.dataset, item.method, item.variant, item.seed)):
        rows.append(
            {
                "dataset": run.dataset,
                "task": run.task,
                "method": run.method,
                "variant": run.variant,
                "seed": run.seed,
                "source": str(run.source.relative_to(REPO) if run.source.is_relative_to(REPO) else run.source),
                **run.metrics,
                **run.resources,
            }
        )
    write_csv(
        out / "PAIN_all_loaded_runs.csv",
        rows,
        ["dataset", "task", "method", "variant", "seed", "source", *metric_names, *resource_names],
    )


def write_completeness(expected: list[ExpectedRun], out: Path, seeds: tuple[int, ...]) -> None:
    grouped: dict[tuple[str, str, str, str], list[ExpectedRun]] = {}
    for item in expected:
        grouped.setdefault((item.dataset, item.task, item.method, item.variant), []).append(item)
    lines = [
        "# PAIN paper-table completeness",
        "",
        f"Expected seeds: `{', '.join(str(seed) for seed in seeds)}`.",
        "",
        "The builder reads per-run artifacts. It does not use the overwrite-prone top-level aggregate CSV files.",
        "",
        "| Dataset/task | Method | Variant | Loaded | Missing or unreadable seeds |",
        "|---|---|---:|---:|---|",
    ]
    for (dataset, task, method, variant), items in sorted(grouped.items()):
        loaded = sorted(item.seed for item in items if item.state == "loaded")
        unavailable = [f"{item.seed} ({item.state})" for item in items if item.state != "loaded"]
        lines.append(
            f"| {dataset}/{task.upper()} | {method} | {variant} | {len(loaded)}/{len(seeds)} | {', '.join(unavailable) or '--'} |"
        )
    lines.extend(
        [
            "",
            "## Source policy",
            "",
            "- IMDb independent v1--v3: `results/imdb_nc/`.",
            "- IMDb independent v4: `results/imdb_nc_v4_fixed/` (the earlier v4 is intentionally ignored).",
            "- IMDb augmentation: `results/imdb_nc_augmentation_v4_fixed/` (the earlier augmentation root is intentionally ignored).",
            "- IMDb universal and invariant: their dedicated result roots.",
            "- DBLP independent and universal: `results/dblp_lp/`; augmentation and invariant: their dedicated result roots.",
            "",
            "## LaTeX use",
            "",
            "Add `\\usepackage{booktabs}` and `\\usepackage{graphicx}` to the paper preamble, then use e.g. "
            "`\\input{tables/PAIN_IMDB_nc_results.tex}` after uploading the desired `.tex` files to Overleaf.",
            "",
            "Rows marked with a dagger are provisional because fewer than all requested seeds were available.",
            "",
        ]
    )
    (out / "PAIN_COMPLETENESS.md").write_text("\n".join(lines), encoding="utf-8")

    csv_rows = [
        {
            "dataset": item.dataset,
            "task": item.task,
            "method": item.method,
            "variant": item.variant,
            "seed": item.seed,
            "state": item.state,
            "source": str(item.source.relative_to(REPO) if item.source.is_relative_to(REPO) else item.source),
            "detail": item.detail,
        }
        for item in expected
    ]
    write_csv(
        out / "PAIN_COMPLETENESS.csv",
        csv_rows,
        ["dataset", "task", "method", "variant", "seed", "state", "source", "detail"],
    )


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds)
    out = absolute(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs, expected = collect_runs(args)
    write_raw_runs(runs, out)
    write_results("IMDB", "nc", runs, seeds, out)
    write_invariance("IMDB", runs, seeds, out)
    write_scalability("IMDB", runs, seeds, out)
    write_results("DBLP", "lp", runs, seeds, out)
    write_invariance("DBLP", runs, seeds, out)
    write_scalability("DBLP", runs, seeds, out)
    write_completeness(expected, out, seeds)
    loaded = sum(item.state == "loaded" for item in expected)
    missing = len(expected) - loaded
    print(f"Wrote PAIN paper tables to {out}")
    print(f"Loaded {loaded}/{len(expected)} expected run outputs; unavailable: {missing}")
    print(f"Completeness report: {out / 'PAIN_COMPLETENESS.md'}")


if __name__ == "__main__":
    main()
