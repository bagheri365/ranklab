from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairwiseDecision:
    model_a: str
    model_b: str
    point_difference: float
    ci_low: float
    ci_high: float
    practical_delta: float
    seed_stable: bool
    decisive: bool


def exceeds_practical_threshold(point_difference: float, delta: float) -> bool:
    """Mechanical threshold check; uncertainty logic is frozen during M0."""
    if delta < 0:
        raise ValueError("delta must be non-negative")
    return point_difference >= delta
