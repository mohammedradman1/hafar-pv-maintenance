"""Aggregation and reporting helpers for pipeline outputs."""

from __future__ import annotations

import statistics
from typing import Iterable

import torch


def summarize_fault_scores(scores: Iterable[torch.Tensor]) -> dict[str, float]:
    """Compute simple statistics over fault probabilities."""

    flattened = torch.stack(list(scores), dim=0)
    mean_scores = flattened.mean(dim=0)
    max_scores = flattened.max(dim=0).values
    return {
        "mean": float(mean_scores.max().item()),
        "max": float(max_scores.max().item()),
    }


def majority_vote(labels: Iterable[int]) -> int:
    """Return the most common label or raise if empty."""

    values = list(labels)
    if not values:
        raise ValueError("No labels provided for voting")
    return statistics.mode(values)
