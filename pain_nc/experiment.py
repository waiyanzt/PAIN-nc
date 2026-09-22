"""Reusable deterministic experiment and reporting helpers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def capture_rng_state(rng: np.random.RandomState) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_local": rng.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(
    state: Mapping[str, Any], rng: np.random.RandomState
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    rng.set_state(state["numpy_local"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("torch_cuda")
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    truth = labels.detach().cpu().numpy()
    prediction = logits.argmax(dim=-1).detach().cpu().numpy()
    return {
        "Accuracy": float(accuracy_score(truth, prediction)),
        "Precision_macro": float(
            precision_score(truth, prediction, average="macro", zero_division=0)
        ),
        "Recall_macro": float(
            recall_score(truth, prediction, average="macro", zero_division=0)
        ),
        "Micro_F1": float(f1_score(truth, prediction, average="micro")),
        "Macro_F1": float(f1_score(truth, prediction, average="macro")),
    }


def _safe_tau(left: np.ndarray, right: np.ndarray) -> float:
    result = kendalltau(left, right, nan_policy="omit").statistic
    return float(result) if result is not None and np.isfinite(result) else float("nan")


def classification_invariance_rows(
    outputs: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    variants = list(outputs)
    rows = []
    for index, variant_a in enumerate(variants):
        for variant_b in variants[index + 1 :]:
            left, right = outputs[variant_a], outputs[variant_b]
            if not np.array_equal(left["item_id"], right["item_id"]):
                raise ValueError(
                    f"Test node IDs differ between {variant_a} and {variant_b}"
                )
            if not np.array_equal(left["label"], right["label"]):
                raise ValueError(
                    f"Test labels differ between {variant_a} and {variant_b}"
                )
            difference = left["logits"] - right["logits"]
            rows.append(
                {
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "kendall_tau_flat_logits": _safe_tau(
                        left["logits"].ravel(), right["logits"].ravel()
                    ),
                    "kendall_tau_confidence": _safe_tau(
                        left["confidence"], right["confidence"]
                    ),
                    "prediction_agreement": float(
                        np.mean(left["prediction"] == right["prediction"])
                    ),
                    "max_abs_logit_diff": float(np.max(np.abs(difference))),
                    "mean_abs_logit_diff": float(np.mean(np.abs(difference))),
                    "mean_l2_logit_diff": float(
                        np.linalg.norm(difference, axis=1).mean()
                    ),
                }
            )
    return rows


def mean_dict(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }
