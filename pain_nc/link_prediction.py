"""Shared scoring and filtered-ranking utilities for PAIN link prediction."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def score_pairs(embeddings: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    return (embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]]).sum(dim=-1)


def candidate_scores(
    embeddings: torch.Tensor,
    positives: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    heads = embeddings[positives[:, 0]]
    tails = embeddings[candidates]
    return heads @ tails.t()


def known_true_tails(
    splits: Mapping[str, torch.Tensor],
) -> dict[int, set[int]]:
    truth: dict[int, set[int]] = {}
    for name in ("train_pos", "val_pos", "test_pos"):
        for head, tail in splits[name].detach().cpu().tolist():
            truth.setdefault(int(head), set()).add(int(tail))
    return truth


def filtered_ranks(
    scores: torch.Tensor,
    positives: torch.Tensor,
    candidates: torch.Tensor,
    truth: Mapping[int, set[int]],
) -> np.ndarray:
    """Pessimistic filtered ranks (ties rank behind every equal score)."""
    rows = scores.detach().cpu().numpy()
    positive_rows = positives.detach().cpu().tolist()
    candidate_ids = candidates.detach().cpu().tolist()
    positions = {int(node): index for index, node in enumerate(candidate_ids)}
    ranks = np.empty(len(positive_rows), dtype=np.int64)
    for index, (head, tail) in enumerate(positive_rows):
        column = positions.get(int(tail))
        if column is None:
            raise ValueError(f"Positive tail {tail} is outside the candidate set")
        row = rows[index].copy()
        for other in truth.get(int(head), ()):
            other_column = positions.get(int(other))
            if other_column is not None and other_column != column:
                row[other_column] = -np.inf
        ranks[index] = int(np.count_nonzero(row >= row[column]))
    return ranks


def ranking_metrics(ranks: np.ndarray, hits_k: tuple[int, ...]) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.float64)
    result = {
        "test_mrr": float(np.mean(1.0 / ranks)),
        "test_mean_rank": float(np.mean(ranks)),
    }
    for k in hits_k:
        result[f"test_hits_at_{k}"] = float(np.mean(ranks <= k))
    return result


def binary_metrics_from_full_candidates(
    scores: torch.Tensor,
    positives: torch.Tensor,
    candidates: torch.Tensor,
    truth: Mapping[int, set[int]],
    threshold: float,
) -> dict[str, float]:
    rows = scores.detach().cpu().numpy()
    positive_rows = positives.detach().cpu().tolist()
    candidate_ids = [int(value) for value in candidates.detach().cpu().tolist()]
    positions = {node: index for index, node in enumerate(candidate_ids)}
    positive_scores, negative_scores = [], []
    for row, (head, tail) in zip(rows, positive_rows):
        positive_scores.append(float(row[positions[int(tail)]]))
        true_tails = truth.get(int(head), set())
        negative_scores.extend(
            float(score)
            for score, candidate in zip(row, candidate_ids)
            if candidate not in true_tails
        )
    logits = np.asarray(positive_scores + negative_scores, dtype=np.float64)
    labels = np.concatenate(
        (
            np.ones(len(positive_scores), dtype=np.int8),
            np.zeros(len(negative_scores), dtype=np.int8),
        )
    )
    probabilities = np.empty_like(logits)
    nonnegative = logits >= 0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exponent = np.exp(logits[~nonnegative])
    probabilities[~nonnegative] = exponent / (1.0 + exponent)
    predictions = (probabilities > threshold).astype(np.int8)
    return {
        "test_auc": float(roc_auc_score(labels, logits)),
        "test_average_precision": float(average_precision_score(labels, logits)),
        "test_precision": float(precision_score(labels, predictions, zero_division=0)),
        "test_recall": float(recall_score(labels, predictions, zero_division=0)),
        "test_f1": float(f1_score(labels, predictions, zero_division=0)),
        "test_accuracy": float(accuracy_score(labels, predictions)),
    }
