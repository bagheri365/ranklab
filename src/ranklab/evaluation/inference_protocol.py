"""Frozen M0.20 statistical interpretation policy.

This module contains protocol constants and deterministic classification helpers.
Bootstrap execution over M1 per-user metrics is implemented separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


PRIMARY_SEEDS = (0, 1, 2, 3, 4)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 365
CONFIDENCE_LEVEL = 0.95
PRIMARY_PAIRWISE_COMPARISONS = 12
BONFERRONI_INDIVIDUAL_CONFIDENCE = 1.0 - (
    (1.0 - CONFIDENCE_LEVEL) / PRIMARY_PAIRWISE_COMPARISONS
)


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float


def seed_mean(values: Mapping[int, float]) -> float:
    """Mean over the five frozen stochastic-model primary seeds."""
    if set(values) != set(PRIMARY_SEEDS):
        raise ValueError(f"expected exactly seeds {PRIMARY_SEEDS}")
    return float(np.mean([float(values[s]) for s in PRIMARY_SEEDS]))


def descriptive_order(scores: Mapping[str, float]) -> tuple[str, ...]:
    """Descending model order with model id as deterministic exact tie-break."""
    return tuple(
        model for model, _ in sorted(
            ((str(model), float(score)) for model, score in scores.items()),
            key=lambda pair: (-pair[1], pair[0]),
        )
    )


def pairwise_margin(score_a: float, score_b: float) -> float:
    """D_AB: positive means A has higher macro NDCG than B."""
    return float(score_a - score_b)


def logging_regime_contrast(
    randomized_margin: float,
    standard_margin: float,
) -> float:
    """G_AB = D_AB(randomized) - D_AB(standard)."""
    return float(randomized_margin - standard_margin)


def behavioral_target_contrast(
    long_view_margin: float,
    click_margin: float,
) -> float:
    """T_AB = D_AB(long_view) - D_AB(is_click)."""
    return float(long_view_margin - click_margin)


def interval_sign(interval: Interval) -> int:
    """Return +1/-1 only when the full interval is strictly above/below zero."""
    if interval.lower > 0.0:
        return 1
    if interval.upper < 0.0:
        return -1
    return 0


def decisive_winner(
    *,
    scores: Mapping[str, float],
    pair_intervals: Mapping[tuple[str, str], Interval],
) -> str | None:
    """A winner must beat every other model with a simultaneous CI above zero."""
    if len(scores) < 2:
        raise ValueError("at least two models are required")

    top = descriptive_order(scores)[0]
    for other in scores:
        if other == top:
            continue
        if (top, other) in pair_intervals:
            interval = pair_intervals[(top, other)]
        elif (other, top) in pair_intervals:
            reverse = pair_intervals[(other, top)]
            interval = Interval(lower=-reverse.upper, upper=-reverse.lower)
        else:
            raise ValueError(f"missing interval for {top} vs {other}")
        if interval_sign(interval) != 1:
            return None
    return top


def selection_change(
    winner_a: str | None,
    winner_b: str | None,
) -> str:
    """Classify decisive selection stability between two cells."""
    if winner_a is None or winner_b is None:
        return "indeterminate"
    if winner_a == winner_b:
        return "stable"
    return "decisive_change"


def seed_order_agreement(
    seed_orders: Sequence[Sequence[str]],
) -> bool:
    """Descriptive stability flag: all five stochastic-seed orders identical."""
    if len(seed_orders) != len(PRIMARY_SEEDS):
        raise ValueError("expected one model order for each frozen primary seed")
    first = tuple(seed_orders[0])
    return all(tuple(order) == first for order in seed_orders[1:])
