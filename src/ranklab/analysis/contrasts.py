from __future__ import annotations


def pairwise_margin(metric_a: float, metric_b: float) -> float:
    """D_AB(c) = Metric_A(c) - Metric_B(c)."""
    return metric_a - metric_b


def regime_contrast(standard_margin: float, randomized_margin: float) -> float:
    """G_AB(t) = D_AB(standard, t) - D_AB(randomized, t)."""
    return standard_margin - randomized_margin


def target_contrast(target_a_margin: float, target_b_margin: float) -> float:
    """T_AB(r) = D_AB(r, target_A) - D_AB(r, target_B)."""
    return target_a_margin - target_b_margin
