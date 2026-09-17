#!/usr/bin/env python3
"""Emit LaTeX table rows (results / invariance / scalability) for every dataset x model x method.

    python paper_tables/build_paper_tables.py            # writes paper_tables/<Dataset>.tex + README.md

Every number is recomputed from the raw per-run outputs (test logits / candidate scores) with ONE
definition per metric, so the models are comparable; only the resource columns (time, epochs,
parameters, GPU peaks) are taken from the runs' own records. See README.md for definitions and
for the exact result root behind every block.  Conventions:

* mean $\\pm$ population std (ddof=0) over the three seeds, 4 decimals (scalability: 2 decimals, zero-padded).
* within a block, every data row after the first is prefixed with '& & ' (empty model / method cells).
* NC results: accuracy, macro precision, macro recall, micro-F1, macro-F1 (test nodes).
* LP results: precision, recall, F1 (sigmoid(score) >= .5 over every candidate: positive=1, sampled
  negatives=0), Hits@1, Hits@3, MRR (averaged-rank tie convention); WN18RR Hits/MRR are the
  filtered full-entity ranks stored by the runs.
* invariance: per seed, for every variant pair, the mean over test nodes/queries of Kendall tau-b
  between the two score rows (1 when identical); LP adds tau over the per-query Hit@1 / Hit@3
  indicator vectors. Augmentation = the same pairwise statistic on the shared model's per-variant
  outputs. The canonical (union-graph) arm is a single model, so it is skipped.
* scalability: train time (s), epochs (augmentation: variant-epochs), parameters (MiB), peak
  training GPU (MiB), peak inference GPU (MiB).
"""
from __future__ import annotations

import csv
import glob
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "src" / "baselines"
DATA = Path("/nfs/hpc/share/mousavij/gnn/data/preprocessed")
OUT = ROOT / "paper_tables"
SEEDS = (1566911444, 20241017, 20251017)
MIB = 1024.0 ** 2

VARIANT_LABEL = {
    "IMDB": {"v1": "IMDb1", "v2": "IMDb2", "v3": "IMDb3", "v4": "IMDb4"},
    "DBLP": {"v1": "DBLP1", "v2": "DBLP2", "v3": "DBLP3"},
    "WORDNET": {"no_changes": "WN18RR1", "all_inverse_edges": "WN18RR2", "transitive_edges": "WN18RR3"},
    "FREEBASE": {"unchanged": "Freebase1", "exact_2": "Freebase2", "exact_3": "Freebase3"},
}
UNION_LABEL = {"IMDB": r"\multirow{2}{*}{$\bigcup_i \text{IMDb}_i$}", "DBLP": r"\multirow{2}{*}{$\bigcup_i \text{DBLP}_i$}",
               "WORDNET": r"\multirow{2}{*}{$\bigcup_i \text{WN18RR}_i$}", "FREEBASE": r"\multirow{2}{*}{$\bigcup_i \text{Freebase}_i$}"}
NC_COLS = ["accuracy", "precision", "recall", "micro_f1", "macro_f1"]
LP_COLS = ["precision", "recall", "f1", "hits@1", "hits@3", "mrr"]
NC_INV = ["tau"]
LP_INV = ["tau", "tau@1", "tau@3"]
SCAL = ["train_time_sec", "epochs", "parameter_mib", "peak_training_gpu_mib", "peak_inference_gpu_mib"]


# ----------------------------------------------------------------------------- records
@dataclass
class Run:
    variant: str
    seed: int
    task_type: str                      # "nc" | "lp"
    scores: np.ndarray                  # nc: [n, C] logits; lp: [q, K+1] candidate scores
    labels: np.ndarray | None = None    # nc: [n]
    pos_col: np.ndarray | None = None   # lp: positive column per query (default 0)
    ranks: np.ndarray | None = None     # lp: precomputed 1-based ranks (WN18RR filtered)
    row_ids: np.ndarray | None = None   # alignment key across variants (nc node ids / lp query ids)
    resources: dict = field(default_factory=dict)
    stored: dict = field(default_factory=dict)


@dataclass
class Block:
    dataset: str; task: str; model: str; method: str; title: str; runs: list; note: str = ""; sources: list = field(default_factory=list)


def std(x): return float(np.std(np.asarray(x, dtype=float), ddof=0))
def fmt(vals, nd=4):
    """mean $\\pm$ std with exactly `nd` decimals (zero-padded): 4 for results/invariance, 2 for scalability."""
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals: return "--"
    return f"{np.mean(vals):.{nd}f} $\\pm$ {std(vals):.{nd}f}"
SCAL_DECIMALS = 2


def continue_rows(lines):
    """Prefix every data row after the first one in a block with '& & ' (the model / method cells are
    left empty on continuation rows). Comment lines are passed through and do not count."""
    out, seen = [], False
    for ln in lines:
        if ln.startswith("%") or not ln.strip():
            out.append(ln)
        else:
            out.append(("& & " + ln) if seen else ln); seen = True
    return out


# ----------------------------------------------------------------------------- metrics
def nc_metrics(r: Run) -> dict:
    pred = r.scores.argmax(1); y = r.labels
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, average="macro", zero_division=0),
            "recall": recall_score(y, pred, average="macro", zero_division=0), "micro_f1": f1_score(y, pred, average="micro"),
            "macro_f1": f1_score(y, pred, average="macro")}


def lp_ranks(r: Run) -> np.ndarray:
    if r.ranks is not None: return r.ranks.astype(float)
    S = r.scores; pc = r.pos_col if r.pos_col is not None else np.zeros(len(S), dtype=int)
    pos = S[np.arange(len(S)), pc]
    better = (S > pos[:, None]).sum(1); tied = (S == pos[:, None]).sum(1) - 1
    return 1.0 + better + tied / 2.0


def lp_metrics(r: Run) -> dict:
    S = r.scores; pc = r.pos_col if r.pos_col is not None else np.zeros(len(S), dtype=int)
    lab = np.zeros_like(S, dtype=int); lab[np.arange(len(S)), pc] = 1
    prob = 1.0 / (1.0 + np.exp(-S.astype(np.float64))); pred = (prob >= 0.5).astype(int)
    y = lab.ravel(); p = pred.ravel()
    ranks = lp_ranks(r)
    out = {"precision": precision_score(y, p, zero_division=0), "recall": recall_score(y, p, zero_division=0), "f1": f1_score(y, p, zero_division=0),
           "hits@1": float(np.mean(ranks <= 1)), "hits@3": float(np.mean(ranks <= 3)), "mrr": float(np.mean(1.0 / ranks))}
    for k, v in r.stored.items():                                   # explicit overrides (WN18RR filtered protocol)
        out[k] = v
    return out


def row_tau(a: np.ndarray, b: np.ndarray) -> float:
    vals = []
    for x, y in zip(a, b):
        if np.array_equal(x, y): vals.append(1.0); continue
        t = kendalltau(x, y, nan_policy="omit")[0]
        if not np.isnan(t): vals.append(float(t))
    return float(np.mean(vals)) if vals else float("nan")


def vec_tau(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if np.array_equal(a, b): return 1.0
    t = kendalltau(a, b, nan_policy="omit")[0]
    return float(t) if t is not None and not np.isnan(t) else float("nan")


def align(ra: Run, rb: Run):
    """Order both runs' rows by their alignment ids (they must cover the same set)."""
    if ra.row_ids is None or rb.row_ids is None:
        return ra, rb
    ia = np.argsort(ra.row_ids, kind="stable"); ib = np.argsort(rb.row_ids, kind="stable")
    if not np.array_equal(ra.row_ids[ia], rb.row_ids[ib]):
        raise ValueError("row ids differ between variants")
    def sub(r, idx):
        return Run(r.variant, r.seed, r.task_type, r.scores[idx], None if r.labels is None else r.labels[idx],
                   None if r.pos_col is None else r.pos_col[idx], None if r.ranks is None else r.ranks[idx], r.row_ids[idx], r.resources, r.stored)
    return sub(ra, ia), sub(rb, ib)


def invariance(ra: Run, rb: Run) -> dict:
    ra, rb = align(ra, rb)
    out = {"tau": row_tau(ra.scores, rb.scores)}
    if ra.task_type == "lp":
        # tau@k over per-query hit indicators on the CANDIDATE ranking (same candidates in every variant)
        ka = lp_ranks(Run(ra.variant, ra.seed, "lp", ra.scores, pos_col=ra.pos_col)); kb = lp_ranks(Run(rb.variant, rb.seed, "lp", rb.scores, pos_col=rb.pos_col))
        out["tau@1"] = vec_tau(ka <= 1, kb <= 1); out["tau@3"] = vec_tau(ka <= 3, kb <= 3)
    return out


# ----------------------------------------------------------------------------- resource helpers
def res_structural(m: dict) -> dict:
    return {"train_time_sec": m.get("train_time_sec"), "epochs": m.get("epochs"), "parameter_mib": m.get("parameter_mib"),
            "peak_training_gpu_mib": m.get("peak_training_gpu_mib"), "peak_inference_gpu_mib": m.get("peak_inference_gpu_mib")}


# ----------------------------------------------------------------------------- loaders
def structural_runs(root: Path, task_type: str, dataset_dir: str, task_dir: str, variants, label_fn=None, seeds=SEEDS) -> list[Run]:
    """RGCN-layout roots: <root>/<DATASET>/<task>/<variant>/seed_<s>/{metrics.json,scores.npz} (RGCN, MAGNN v2, CMPNN, SeHGNN)."""
    runs = []
    for v in variants:
        for s in seeds:
            d = root / dataset_dir / task_dir / v / f"seed_{s}"
            if not (d / "metrics.json").exists(): continue
            m = json.load(open(d / "metrics.json")); z = np.load(d / "scores.npz")
            res = res_structural(m["metrics"])
            if task_type == "nc":
                ids = z["ids"]; runs.append(Run(v, s, "nc", z["scores"], label_fn(ids), row_ids=ids, resources=res))
            else:
                S = z["scores"]
                if "candidate_ids" in z.files:
                    cid = z["candidate_ids"]; pc = np.array([int(np.where(cid == t)[0][0]) for t in z["positive_tails"]])
                else:
                    pc = np.zeros(len(S), dtype=int)
                runs.append(Run(v, s, "lp", S, pos_col=pc, row_ids=z["queries"] * 100003 + z["positive_tails"], resources=res))
    return runs


def imdb_nc_labels_rgcn():
    import torch
    sys.path.insert(0, str(B / "RGCN"))
    from inv_rgcn.data import load_graph_data
    d = load_graph_data(DATA / "RGCN_structural_v10_year_canonfeat/IMDB/nc/v1"); y = d.y.numpy()
    return lambda ids: y[ids]


def bundle_labels(labels_path: Path, targets_path: Path):
    lab = np.load(labels_path); tg = np.load(targets_path); pos = {int(t): i for i, t in enumerate(tg)}
    return lambda ids: np.array([lab[pos[int(i)]] for i in ids])


def slotgat_runs(root: Path, condition: str, variants: dict, seeds=SEEDS) -> list[Run]:
    runs = []
    for v, dirname in variants.items():
        for s in seeds:
            d = root / condition / dirname / f"seed_{s}"
            if not (d / "result.json").exists(): continue
            j = json.load(open(d / "result.json")); z = np.load(d / "test_logits.npz"); mem = j["memory"]
            res = {"train_time_sec": j["training_time_seconds"], "epochs": j["epochs_trained"], "parameter_mib": j["num_parameters"] * 4 / MIB,
                   "peak_training_gpu_mib": mem["training_peak_allocated_bytes"] / MIB, "peak_inference_gpu_mib": mem["inference_peak_allocated_bytes"] / MIB}
            runs.append(Run(v, s, "nc", z["logits"], z["labels"], row_ids=z["node_ids"], resources=res))
    return runs


def slotgat_aug_runs(root: Path, variants: dict, seeds=SEEDS) -> list[Run]:
    runs = []
    for s in seeds:
        d = root / f"seed_{s}"
        if not (d / "result.json").exists(): continue
        j = json.load(open(d / "result.json")); mem = j["memory"]
        res = {"train_time_sec": j["training_time_seconds"], "epochs": j["variant_epochs_run"], "parameter_mib": j["parameter_mib"],
               "peak_training_gpu_mib": mem["training_peak_allocated_bytes"] / MIB, "peak_inference_gpu_mib": mem["inference_peak_allocated_bytes"] / MIB}
        for v, dirname in variants.items():
            z = np.load(d / f"{dirname}_test_logits.npz")
            runs.append(Run(v, s, "nc", z["logits"], z["labels"], row_ids=z["node_ids"], resources=res))
    return runs


def rgcn_freebase_runs(root: Path, subdir_fn, variants, seeds=SEEDS) -> list[Run]:
    runs = []
    for v in variants:
        for s in seeds:
            d = root / subdir_fn(v) / f"seed_{s}"
            if not (d / "result.json").exists(): continue
            j = json.load(open(d / "result.json")); z = np.load(d / "logits.npz"); mem = j["memory"]
            res = {"train_time_sec": j["train_time_sec"], "epochs": j["epochs_run"], "parameter_mib": mem["parameter_mib"],
                   "peak_training_gpu_mib": mem["peak_training_gpu_mib"], "peak_inference_gpu_mib": mem["peak_inference_gpu_mib"]}
            runs.append(Run(v, s, "nc", z["test_logits"], z["test_labels"], row_ids=z["test_global_ids"], resources=res))
    return runs


def rgcn_freebase_aug_runs(root: Path, variants, label_src: list[Run], seeds=SEEDS) -> list[Run]:
    runs = []
    lab = {(r.seed): r for r in label_src if r.variant == variants[0]}
    for s in seeds:
        d = root / f"seed_{s}"
        if not (d / "summary.json").exists(): continue
        j = json.load(open(d / "summary.json")); z = np.load(d / "test_logits.npz")
        res = {"train_time_sec": j["train_seconds"], "epochs": j["variant_epochs_ran"], "parameter_mib": j["parameter_mib"],
               "peak_training_gpu_mib": j["peak_training_gpu_mib"], "peak_inference_gpu_mib": j["peak_inference_gpu_mib"]}
        ref = lab[s]
        for v in variants:
            runs.append(Run(v, s, "nc", z[v], ref.labels, row_ids=ref.row_ids, resources=res))
    return runs


def rgcn_dblp_aug_runs(root: Path, variants, seeds=SEEDS) -> list[Run]:
    runs = []
    for s in seeds:
        d = root / f"seed_{s}"
        if not (d / "summary.json").exists(): continue
        j = json.load(open(d / "summary.json")); mem = j["memory"]
        res = {"train_time_sec": j["training_seconds"], "epochs": j["epoch_accounting"]["variant_epochs_ran"], "parameter_mib": mem["parameter_bytes"] / MIB,
               "peak_training_gpu_mib": mem["training_gpu"]["gpu_peak_allocated_bytes"] / MIB, "peak_inference_gpu_mib": mem["inference_gpu"]["gpu_peak_allocated_bytes"] / MIB}
        for v in variants:
            rows = list(csv.DictReader(open(d / f"test_scores_{v}.csv")))
            by = {}
            for r in rows:
                pr = min(max(float(r["score"]), 1e-7), 1 - 1e-7)          # the CSV stores sigmoid probabilities -> back to logits
                by.setdefault(int(r["paper_id"]), []).append((int(r["label"]), math.log(pr / (1 - pr)), int(r["conf_id"])))
            qids = sorted(by); S = []; pc = []
            for q in qids:
                items = sorted(by[q], key=lambda t: (-t[0], t[2]))     # positive first, then negatives by conf id
                S.append([t[1] for t in items]); pc.append(0)
            runs.append(Run(v, s, "lp", np.array(S, dtype=np.float32), pos_col=np.array(pc), row_ids=np.array(qids), resources=res))
    return runs


def wordnet_structural_runs(root: Path, variants, seeds=SEEDS) -> list[Run]:
    """RGCN WordNet structural layout: <root>/full/intersection/<variant>/seed_<s>/{metrics.json,test_scores.npz}.
    Candidate matrix = true entity + 50 fixed negatives (column 0 = true); Hits/MRR from the FILTERED full-entity ranks."""
    runs = []
    for v in variants:
        for s in seeds:
            d = root / "full" / "intersection" / v / f"seed_{s}"
            if not (d / "metrics.json").exists(): continue
            j = json.load(open(d / "metrics.json")); z = np.load(d / "test_scores.npz")
            side = z["side"]; n = len(side)
            fr = np.where(side == 0, z["filtered_tail_ranks"][np.arange(n) // 2], z["filtered_head_ranks"][np.arange(n) // 2]).astype(float)
            tm = j["test_metrics"]
            stored = {"hits@1": tm["Hits@1"], "hits@3": tm["Hits@3"], "mrr": tm["filtered_MRR"]}
            res = {"train_time_sec": j.get("train_time_sec"), "epochs": j.get("epochs_completed"), "parameter_mib": j["parameter_bytes"] / MIB,
                   "peak_training_gpu_mib": j["peak_train_gpu_bytes"] / MIB, "peak_inference_gpu_mib": j["peak_inference_gpu_bytes"] / MIB}
            q = z["query_triples"]; rid = (q[:, 0] * 1000003 + q[:, 1]) * 1000003 + q[:, 2] * 2 + side
            runs.append(Run(v, s, "lp", z["candidate_scores"], pos_col=np.zeros(n, dtype=int), ranks=fr, row_ids=rid, resources=res, stored=stored))
    return runs


def wordnet_aug_runs(root: Path, variants, seeds=SEEDS) -> list[Run]:
    runs = []
    for s in seeds:
        d = root / f"seed_{s}"
        if not (d / "summary.json").exists(): continue
        j = json.load(open(d / "summary.json")); mem = j["memory"]
        res = {"train_time_sec": j["training_seconds"], "epochs": j["epoch_accounting"]["variant_epochs_ran"], "parameter_mib": mem["parameter_bytes"] / MIB,
               "peak_training_gpu_mib": mem["training_gpu"]["gpu_peak_allocated_bytes"] / MIB, "peak_inference_gpu_mib": mem["inference_gpu"]["gpu_peak_allocated_bytes"] / MIB}
        legacy = {r["variant"]: r for r in csv.DictReader(open(d / "legacy_test_metrics_by_variant.csv"))}
        for v in variants:
            rows = list(csv.DictReader(open(d / f"shared_candidate_test_scores_{v}.csv")))
            by = {}
            for r in rows:
                by.setdefault(int(r["query_id"]), []).append((int(r["label"]), float(r["logit"]), int(r["tail"]), int(r["head"]), int(r["relation"])))
            qids = sorted(by); S = []
            for q in qids:
                items = sorted(by[q], key=lambda t: (-t[0], t[2], t[3]))
                S.append([t[1] for t in items])
            lm = legacy[v]
            stored = {"hits@1": float(lm["Hits@1"]), "hits@3": float(lm["Hits@3"]), "mrr": float(lm["filtered_MRR"])}
            runs.append(Run(v, s, "lp", np.array(S, dtype=np.float32), pos_col=np.zeros(len(S), dtype=int), row_ids=np.array(qids), resources=res, stored=stored))
    return runs


# ----------------------------------------------------------------------------- registry
def blocks() -> list[Block]:
    out = []
    def add(dataset, task, model, method, title, runs, note="", sources=()):
        out.append(Block(dataset, task, model, method, title, runs, note, list(sources)))

    # ---------------- IMDb NC
    imdb = ["v1", "v2", "v3", "v4"]
    lab = imdb_nc_labels_rgcn()
    r = B / "RGCN/results/IMDB_year_nc_canonfeat"
    for meth, arm in (("original", "originals"), ("canonical", "universal"), ("augmentation", "augmentation"), ("invariant", "invariant")):
        add("IMDB", "nc", "RGCN", meth, "", structural_runs(r / arm, "nc", "IMDB", "nc", imdb if meth != "canonical" else ["universal"], lab), sources=[str(r / arm)])
    r = B / "MAGNN/results/MAGNN_imdb_v2/nc"; labm = bundle_labels(DATA / "imdb_magnn_v2/nc/stock/v1/labels.npy", DATA / "imdb_magnn_v2/nc/stock/v1/target_nodes.npy")
    for meth, arm, title in (("original", "originals", ""), ("canonical", "universal", ""), ("augmentation", "augmentation", ""),
                             ("invariant", "invariant", "invariant (intersection metapaths)"), ("invariant", "invariant_union", "invariant + union metapaths")):
        add("IMDB", "nc", "MAGNN", meth, title, structural_runs(r / arm, "nc", "IMDB", "nc", imdb if meth != "canonical" else ["universal"], labm), sources=[str(r / arm)])
    sg = B / "SlotGAT/results"; sv = {v: f"IMDB_var{i}_year" for i, v in enumerate(imdb, 1)}
    add("IMDB", "nc", "SlotGAT", "original", "original (upstream stock SlotGAT, 3 layers, 3.25M params)", slotgat_runs(sg / "imdb_nc_year_v2", "original", sv), sources=[str(sg / "imdb_nc_year_v2/original")])
    add("IMDB", "nc", "SlotGAT", "original", "original (matched control: same architecture as the invariant arm on the physical graph)", slotgat_runs(sg / "imdb_nc_year_v2", "physical", sv), sources=[str(sg / "imdb_nc_year_v2/physical")])
    add("IMDB", "nc", "SlotGAT", "canonical", "canonical (upstream stock SlotGAT on the union graph)", slotgat_runs(sg / "imdb_nc_year_v2_universal", "original", {"universal": "IMDB_var5_year"}), sources=[str(sg / "imdb_nc_year_v2_universal/original")])
    add("IMDB", "nc", "SlotGAT", "canonical", "canonical (matched architecture on the union graph)", slotgat_runs(sg / "imdb_nc_year_v2_universal", "physical", {"universal": "IMDB_var5_year"}), sources=[str(sg / "imdb_nc_year_v2_universal/physical")])
    add("IMDB", "nc", "SlotGAT", "augmentation", "", slotgat_aug_runs(sg / "imdb_nc_year_augmentation", sv), sources=[str(sg / "imdb_nc_year_augmentation")])
    add("IMDB", "nc", "SlotGAT", "invariant", "", slotgat_runs(sg / "imdb_nc_year_v2", "conditional", sv), sources=[str(sg / "imdb_nc_year_v2/conditional")])
    r = B / "SeHGNN/results/SeHGNN_v2/IMDB_nc"; labs = bundle_labels(DATA / "sehgnn_v2/IMDB/full/physical/v1/labels.npy", DATA / "sehgnn_v2/IMDB/full/physical/v1/targets.npy")
    for meth, arm in (("original", "originals"), ("canonical", "universal"), ("augmentation", "augmentation"), ("invariant", "invariant")):
        flavs = ("full", "restricted", "restricted_union") if meth == "invariant" else ("full", "restricted")
        for fl in flavs:
            title = {"full": f"{meth} (full-K channel set: every type path $\\le 4$)", "restricted": f"{meth} (restricted-K: MAGNN intersection metapaths)",
                     "restricted_union": f"{meth} (restricted-K + union: MAGNN union metapaths)"}[fl]
            add("IMDB", "nc", "SeHGNN", meth, title, structural_runs(r / f"{arm}_{fl}", "nc", "IMDB", "nc", imdb if meth != "canonical" else ["universal"], labs), sources=[str(r / f"{arm}_{fl}")])
    add("IMDB", "nc", "SeHGNN", "invariant", "invariant (full-K WITHOUT skip nodes -- ablation: 143 channels, paths through Link kept)",
        structural_runs(r / "invariant_full_noskip", "nc", "IMDB", "nc", imdb, labs), sources=[str(r / "invariant_full_noskip")])

    # ---------------- IMDb LP (movie_year v1-4, movie_director v1,v3)
    for task, variants in (("movie_year", imdb), ("movie_director", ["v1", "v3"])):
        r = B / f"RGCN/results/IMDB_year_{task}_canonfeat"
        for meth, arm in (("original", "originals"), ("canonical", "universal"), ("augmentation", "augmentation"), ("invariant", "invariant")):
            add("IMDB", task, "RGCN", meth, "", structural_runs(r / arm, "lp", "IMDB", task, variants if meth != "canonical" else ["universal"]), sources=[str(r / arm)])
        r = B / f"MAGNN/results/MAGNN_imdb_v2/{task}"
        for meth, arm, title in (("original", "originals", ""), ("canonical", "universal", ""), ("invariant", "invariant", "invariant (intersection metapaths)"), ("invariant", "invariant_union", "invariant + union metapaths")):
            add("IMDB", task, "MAGNN", meth, title, structural_runs(r / arm, "lp", "IMDB", task, variants if meth != "canonical" else ["universal"]), sources=[str(r / arm)])
        add("IMDB", task, "MAGNN", "augmentation", "", [], note="no joint-training LP runner exists for MAGNN (MAGNN_IMDB_HANDOFF.md §3)")
        r = B / f"CMPNN/results/CMPNN_structural/IMDB_{task}"
        for meth, arm in (("original", "originals"), ("canonical", "universal"), ("augmentation", "augmentation"), ("invariant", "invariant")):
            add("IMDB", task, "CMPNN", meth, "", structural_runs(r / arm, "lp", "IMDB", task, variants if meth != "canonical" else ["universal"]), sources=[str(r / arm)])

    # ---------------- DBLP LP
    dblp = ["v1", "v2", "v3"]
    r = B / "RGCN/results/DBLP_one_runner"
    for meth, arm in (("original", "originals"), ("canonical", "universal"), ("invariant", "invariant")):
        add("DBLP", "paper_conference", "RGCN", meth, "", structural_runs(r / arm, "lp", "DBLP", "paper_conference", dblp if meth != "canonical" else ["universal"]), sources=[str(r / arm)])
    add("DBLP", "paper_conference", "RGCN", "augmentation", "", rgcn_dblp_aug_runs(B / "rgcn_data_augmentation/results/rgcn_augmentation/DBLP_instr", dblp), sources=[str(B / "rgcn_data_augmentation/results/rgcn_augmentation/DBLP_instr")])
    r = B / "MAGNN/results/MAGNN_dblp_v2"   # docs/MAGNN_DBLP_HANDOFF.md: v2 bundles from RGCN_structural_v8_dblp, four arms + invariant union
    for meth, arm, title in (("original", "originals", ""), ("canonical", "universal", ""), ("augmentation", "augmentation", ""),
                             ("invariant", "invariant", "invariant (intersection metapaths)"), ("invariant", "invariant_union", "invariant + union metapaths")):
        add("DBLP", "paper_conference", "MAGNN", meth, title, structural_runs(r / arm, "lp", "DBLP", "paper_conference", dblp if meth != "canonical" else ["universal"]),
            note="MAGNN DBLP v2 suite not yet run (docs/MAGNN_DBLP_HANDOFF.md)", sources=[str(r / arm)])
    r = B / "CMPNN/results/CMPNN_structural/DBLP_paper_conference"
    for meth, arm in (("original", "originals"), ("canonical", "universal"), ("augmentation", "augmentation"), ("invariant", "invariant")):
        add("DBLP", "paper_conference", "CMPNN", meth, "", structural_runs(r / arm, "lp", "DBLP", "paper_conference", dblp if meth != "canonical" else ["universal"]), sources=[str(r / arm)])

    # ---------------- WN18RR LP
    wn = ["no_changes", "all_inverse_edges", "transitive_edges"]
    add("WORDNET", "link_prediction", "RGCN", "original", "", wordnet_structural_runs(B / "RGCN/results/WORDNET_one_runner/originals", wn), sources=[str(B / "RGCN/results/WORDNET_one_runner/originals")])
    add("WORDNET", "link_prediction", "RGCN", "canonical", "", wordnet_structural_runs(B / "RGCN/results/WORDNET_one_runner/universal", ["universal_edges"]), sources=[str(B / "RGCN/results/WORDNET_one_runner/universal")])
    add("WORDNET", "link_prediction", "RGCN", "augmentation", "", wordnet_aug_runs(B / "rgcn_data_augmentation/results/WORDNET_augmentation_patience", wn), sources=[str(B / "rgcn_data_augmentation/results/WORDNET_augmentation_patience")])
    add("WORDNET", "link_prediction", "RGCN", "invariant", "invariant (task_only objective, intersection catalog)", wordnet_structural_runs(B / "RGCN/results/RGCN_structural/WORDNET/patience_task_only", wn), sources=[str(B / "RGCN/results/RGCN_structural/WORDNET/patience_task_only")])
    for meth in ("original", "canonical", "augmentation", "invariant"):
        add("WORDNET", "link_prediction", "CMPNN", meth, "", [], note="CMPNN WN18RR array 21265369 not yet run (pending on dgxh)")

    # ---------------- Freebase NC
    fb = ["unchanged", "exact_2", "exact_3"]
    inv = rgcn_freebase_runs(B / "RGCN/results/freebase_rgcn_block_conditional_select_val_f1", lambda v: f"range_2_3/union/all_graph/{v}", fb)
    add("FREEBASE", "nc", "RGCN", "original", "", rgcn_freebase_runs(B / "RGCN/results/freebase_rgcn_physical_baselines_select_val_f1", lambda v: f"physical_{v}/union/all_graph/{v}", fb), sources=[str(B / "RGCN/results/freebase_rgcn_physical_baselines_select_val_f1")])
    add("FREEBASE", "nc", "RGCN", "canonical", "", rgcn_freebase_runs(B / "RGCN/results/freebase_rgcn_physical_baselines_select_val_f1", lambda v: f"physical_{v}/union/all_graph/{v}", ["union_exact_2_3"]), sources=[str(B / "RGCN/results/freebase_rgcn_physical_baselines_select_val_f1/physical_union_exact_2_3")])
    add("FREEBASE", "nc", "RGCN", "augmentation", "", rgcn_freebase_aug_runs(B / "RGCN/results/freebase_rgcn_augmentation_sampled_instr", fb, inv), sources=[str(B / "RGCN/results/freebase_rgcn_augmentation_sampled_instr")])
    add("FREEBASE", "nc", "RGCN", "invariant", "", inv, sources=[str(B / "RGCN/results/freebase_rgcn_block_conditional_select_val_f1")])
    # MAGNN / SeHGNN Freebase: loaders registered; blocks stay "(no runs)" until the dgxh arrays land
    r = B / "MAGNN/results/MAGNN_freebase_v2"
    labm = bundle_labels(DATA / "freebase_magnn_v2/stock/unchanged/labels.npy", DATA / "freebase_magnn_v2/stock/unchanged/target_nodes.npy") if (DATA / "freebase_magnn_v2/stock/unchanged/labels.npy").exists() else (lambda ids: ids)
    for meth, arm, title in (("original", "originals", ""), ("canonical", "universal", ""), ("augmentation", "augmentation", ""),
                             ("invariant", "invariant", "invariant (intersection metapaths)"), ("invariant", "invariant_union", "invariant + union metapaths")):
        add("FREEBASE", "nc", "MAGNN", meth, title, structural_runs(r / arm, "nc", "FREEBASE", "nc", fb if meth != "canonical" else ["union_exact_2_3"], labm),
            note="MAGNN Freebase v2 array 21259884 not yet run (pending on dgxh)", sources=[str(r / arm)])
    for meth in ("original", "canonical", "augmentation", "invariant"):
        add("FREEBASE", "nc", "SlotGAT", meth, "", [], note="SlotGAT Freebase phase-2 arms incomplete (dgxh continuations pending)")
    r = B / "SeHGNN/results/SeHGNN_v2/FREEBASE_nc"
    labs = bundle_labels(DATA / "sehgnn_v2/FREEBASE/full/physical/unchanged/labels.npy", DATA / "sehgnn_v2/FREEBASE/full/physical/unchanged/targets.npy")
    for meth, arm in (("original", "originals"), ("canonical", "universal"), ("augmentation", "augmentation"), ("invariant", "invariant")):
        flavs = ("full", "restricted", "restricted_union") if meth == "invariant" else ("full", "restricted")
        for fl in flavs:
            title = {"full": f"{meth} (full-K channel set: every type path $\\le 2$)", "restricted": f"{meth} (restricted-K: MAGNN intersection metapaths)",
                     "restricted_union": f"{meth} (restricted-K + union: MAGNN union metapaths)"}[fl]
            add("FREEBASE", "nc", "SeHGNN", meth, title, structural_runs(r / f"{arm}_{fl}", "nc", "FREEBASE", "nc", fb if meth != "canonical" else ["union_exact_2_3"], labs),
                note="SeHGNN Freebase array 21264894 not yet run (pending on dgxh)", sources=[str(r / f"{arm}_{fl}")])
    return out


# ----------------------------------------------------------------------------- rendering
def per_variant_table(b: Block, cols, metric_fn):
    """rows: variant label -> list over seeds of metric dicts."""
    by = {}
    for r in b.runs:
        by.setdefault(r.variant, []).append(metric_fn(r))
    return by


def union_variants(b: Block):
    vs = sorted({r.variant for r in b.runs}, key=lambda v: list(VARIANT_LABEL[b.dataset]).index(v) if v in VARIANT_LABEL[b.dataset] else 99)
    return vs


def aug_label(dataset, variants):
    labels = [VARIANT_LABEL[dataset][v] for v in variants]
    base = labels[0].rstrip("0123456789"); nums = [l[len(base):] for l in labels]
    if len(nums) > 1 and [int(n) for n in nums] == list(range(int(nums[0]), int(nums[0]) + len(nums))):
        return f"{base}{nums[0]}-{nums[-1]}"
    return base + ",".join(nums)


def render_results(b: Block, cols, metric_fn):
    lines = []
    if not b.runs:
        return [f"% (no runs) {b.note}"]
    by = per_variant_table(b, cols, metric_fn)
    if b.method == "canonical":
        v = list(by)[0]; cells = [fmt([m[c] for m in by[v]]) for c in cols]
        lines.append(UNION_LABEL[b.dataset] + " & " + " & ".join(f"\\multirow{{2}}{{*}}{{{c}}}" for c in cells) + r" \\")
    elif b.method == "augmentation":
        vs = union_variants(b)
        for v in vs:
            lines.append(f"% {VARIANT_LABEL[b.dataset][v]}: " + " & ".join(fmt([m[c] for m in by[v]]) for c in cols))
        # the row: mean over variants of each seed's metric, then mean +/- std over seeds
        seeds = sorted({r.seed for r in b.runs}); cells = []
        for c in cols:
            per_seed = []
            for s in seeds:
                vals = [metric_fn(r)[c] for r in b.runs if r.seed == s]
                per_seed.append(float(np.mean(vals)))
            cells.append(fmt(per_seed))
        lines.append(aug_label(b.dataset, vs) + " & " + " & ".join(cells) + r" \\")
    else:
        for v in union_variants(b):
            lines.append(VARIANT_LABEL[b.dataset][v] + " & " + " & ".join(fmt([m[c] for m in by[v]]) for c in cols) + r" \\")
    return lines


def render_invariance(b: Block, cols):
    if b.method == "canonical":
        return ["% canonical: single model on the union graph, invariance is trivially 1 (skipped)"]
    if not b.runs:
        return [f"% (no runs) {b.note}"]
    vs = union_variants(b); seeds = sorted({r.seed for r in b.runs}); lines = []
    for va, vb in itertools.combinations(vs, 2):
        per = {c: [] for c in cols}
        for s in seeds:
            ra = [r for r in b.runs if r.variant == va and r.seed == s]; rb = [r for r in b.runs if r.variant == vb and r.seed == s]
            if not ra or not rb: continue
            inv = invariance(ra[0], rb[0])
            for c in cols: per[c].append(inv[c])
        lines.append(f"{VARIANT_LABEL[b.dataset][va]} vs. {VARIANT_LABEL[b.dataset][vb]} & " + " & ".join(fmt(per[c]) for c in cols) + r" \\")
    return lines


def render_scal(b: Block):
    if not b.runs:
        return [f"% (no runs) {b.note}"]
    def cells(runs):
        out = []
        for c in SCAL:
            vals = [r.resources.get(c) for r in runs]
            out.append(fmt(vals, SCAL_DECIMALS))
        return out
    by = {}
    for r in b.runs: by.setdefault(r.variant, []).append(r)
    if b.method == "canonical":
        v = list(by)[0]
        return [UNION_LABEL[b.dataset] + " & " + " & ".join(f"\\multirow{{2}}{{*}}{{{c}}}" for c in cells(by[v])) + r" \\"]
    if b.method == "augmentation":
        vs = union_variants(b); seeds = sorted({r.seed for r in b.runs})
        one = [next(r for r in b.runs if r.seed == s) for s in seeds]       # shared model: one record per seed
        return [aug_label(b.dataset, vs) + " & " + " & ".join(cells(one)) + r" \\"]
    return [VARIANT_LABEL[b.dataset][v] + " & " + " & ".join(cells(by[v])) + r" \\" for v in union_variants(b)]


DATASET_FILES = {"IMDB_nc": ("IMDb node classification", [("IMDB", "nc")]),
                 "IMDB_lp": ("IMDb link prediction", [("IMDB", "movie_year"), ("IMDB", "movie_director")]),
                 "DBLP_lp": ("DBLP link prediction (paper--conference)", [("DBLP", "paper_conference")]),
                 "WN18RR_lp": ("WN18RR link prediction", [("WORDNET", "link_prediction")]),
                 "Freebase_nc": ("Freebase node classification", [("FREEBASE", "nc")])}
TASK_TITLE = {"nc": "node classification", "movie_year": "Movie--Year (MY)", "movie_director": "Movie--Director (MD)", "paper_conference": "Paper--Conference", "link_prediction": "link prediction"}
MODELS = ["RGCN", "MAGNN", "CMPNN", "SlotGAT", "SeHGNN"]
METHODS = ["original", "canonical", "augmentation", "invariant"]


def main():
    all_blocks = blocks()
    OUT.mkdir(exist_ok=True)
    index = []
    for fname, (title, tasks) in DATASET_FILES.items():
        lines = [f"% ===================================================================", f"% {title} -- generated by paper_tables/build_paper_tables.py",
                 f"% mean $\\pm$ std (ddof=0) over seeds {SEEDS}; see paper_tables/README.md for metric definitions and sources",
                 f"% ==================================================================="]
        for section, header in (("RESULTS", None), ("INVARIANCE", None), ("SCALABILITY", None)):
            lines += ["", "", f"%% ################################ {section} ################################"]
            if section == "RESULTS":
                lines.append("% NC columns: variant & accuracy & precision (macro) & recall (macro) & micro-F1 & macro-F1" if tasks[0][1] == "nc" or tasks[0][0] == "FREEBASE"
                             else "% LP columns: variant & precision & recall & F1 & Hits@1 & Hits@3 & MRR")
            elif section == "INVARIANCE":
                lines.append("% NC columns: comparison & Kendall-tau" if tasks[0][1] == "nc" else "% LP columns: comparison & Kendall-tau & Kendall-tau@1 & Kendall-tau@3")
            else:
                lines.append("% columns: variant & train time (s) & epochs & parameters (MiB) & peak train GPU (MiB) & peak inference GPU (MiB)  [2 decimals]")
            for model in MODELS:
                mb = [b for b in all_blocks if b.model == model and (b.dataset, b.task) in tasks]
                if not mb: continue
                lines += ["", f"%% ======== {model} ========"]
                for meth in METHODS:
                    for (ds, task) in tasks:
                        for b in [x for x in mb if x.method == meth and x.task == task]:
                            head = f"% ---- {model} / {b.title or meth}" + (f" / {TASK_TITLE[task]}" if len(tasks) > 1 else "")
                            lines.append(head)
                            is_nc = b.runs and b.runs[0].task_type == "nc" or (not b.runs and (task == "nc"))
                            if section == "RESULTS":
                                rows = render_results(b, NC_COLS if is_nc else LP_COLS, nc_metrics if is_nc else lp_metrics)
                            elif section == "INVARIANCE":
                                rows = render_invariance(b, NC_INV if is_nc else LP_INV)
                            else:
                                rows = render_scal(b)
                            lines += continue_rows(rows)
                            lines.append("")
        (OUT / f"{fname}.tex").write_text("\n".join(lines) + "\n")
        index.append((fname, title))
        print(f"wrote {OUT / f'{fname}.tex'}")
    # sources appendix
    src = ["# Result roots behind every block", ""]
    for b in all_blocks:
        src.append(f"- {b.dataset}/{b.task} / {b.model} / {b.title or b.method}: " + (", ".join(os.path.relpath(s, ROOT) for s in b.sources) if b.sources else f"(none) {b.note}") + f"  [{len(b.runs)} runs]")
    (OUT / "SOURCES.md").write_text("\n".join(src) + "\n")
    print(f"wrote {OUT / 'SOURCES.md'}")


if __name__ == "__main__":
    main()
