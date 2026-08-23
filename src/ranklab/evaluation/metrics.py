from __future__ import annotations

from collections.abc import Callable, Sequence
import math

GainFn = Callable[[float], float]
DiscountFn = Callable[[int], float]


def identity_gain(relevance: float) -> float:
    return relevance


def exp2_gain(relevance: float) -> float:
    return (2.0**relevance) - 1.0


def log2_discount(rank_one_indexed: int) -> float:
    if rank_one_indexed < 1:
        raise ValueError("rank must be one-indexed and >= 1")
    return 1.0 / math.log2(rank_one_indexed + 1.0)


def dcg_at_k(
    relevance: Sequence[float],
    k: int,
    *,
    gain_fn: GainFn,
    discount_fn: DiscountFn,
) -> float:
    """Compute DCG@k without imposing RankLab's yet-unfrozen M0 semantics."""
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(
        gain_fn(float(rel)) * discount_fn(rank)
        for rank, rel in enumerate(relevance[:k], start=1)
    )


def ndcg_at_k(
    relevance: Sequence[float],
    k: int,
    *,
    gain_fn: GainFn,
    discount_fn: DiscountFn,
    zero_relevance_value: float | None,
) -> float:
    """Compute NDCG@k with explicit zero-relevance semantics.

    Passing ``zero_relevance_value=None`` raises on a zero-IDCG unit. This is
    intentional: M0 must freeze the project's exact edge-case policy.
    """
    observed = dcg_at_k(relevance, k, gain_fn=gain_fn, discount_fn=discount_fn)
    ideal = dcg_at_k(
        sorted(relevance, reverse=True),
        k,
        gain_fn=gain_fn,
        discount_fn=discount_fn,
    )
    if ideal == 0.0:
        if zero_relevance_value is None:
            raise ValueError("zero-relevance handling is not frozen")
        return float(zero_relevance_value)
    return observed / ideal
