"""Frozen M0.19 evaluation candidate and NDCG semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ranklab.evaluation.metrics import (
    identity_gain,
    log2_discount,
    ndcg_at_k,
)
from ranklab.evaluation.ranking import deterministic_rank_indices


PRIMARY_K = 10
MIN_CANDIDATES = 2


@dataclass(frozen=True)
class UserNDCGResult:
    ndcg: float | None
    candidates: int
    relevant: int
    included_in_macro: bool
    exclusion_reason: str | None


def target_any_positive(values: Sequence[int | bool | float]) -> int:
    """Binary relevance after pair collapse: 1 iff any retained row is positive."""
    numeric = [int(v) for v in values]
    if any(v not in (0, 1) for v in numeric):
        raise ValueError("target relevance values must be binary")
    return int(any(numeric))


def evaluate_user_ndcg(
    *,
    item_ids: Sequence[int],
    scores: Sequence[float],
    relevance: Sequence[int | bool | float],
    k: int = PRIMARY_K,
) -> UserNDCGResult:
    """Apply frozen per-user NDCG eligibility and ranking semantics.

    Users with fewer than two unique candidates are excluded.
    Users with zero relevant candidates are excluded from target-specific macro
    NDCG because IDCG is zero. Their counts must be reported separately.
    """
    if len(item_ids) != len(scores) or len(item_ids) != len(relevance):
        raise ValueError("item_ids, scores, and relevance must have the same length")
    if len(set(int(item) for item in item_ids)) != len(item_ids):
        raise ValueError("candidate item_ids must be unique")
    if k <= 0:
        raise ValueError("k must be positive")

    rel = [int(v) for v in relevance]
    if any(v not in (0, 1) for v in rel):
        raise ValueError("relevance must be binary")

    n = len(item_ids)
    relevant = int(sum(rel))

    if n < MIN_CANDIDATES:
        return UserNDCGResult(
            ndcg=None,
            candidates=n,
            relevant=relevant,
            included_in_macro=False,
            exclusion_reason="fewer_than_2_candidates",
        )

    if relevant == 0:
        return UserNDCGResult(
            ndcg=None,
            candidates=n,
            relevant=0,
            included_in_macro=False,
            exclusion_reason="zero_relevance",
        )

    order = deterministic_rank_indices(item_ids, scores)
    ranked_relevance = [rel[idx] for idx in order]

    value = ndcg_at_k(
        ranked_relevance,
        k,
        gain_fn=identity_gain,
        discount_fn=log2_discount,
        zero_relevance_value=None,
    )
    return UserNDCGResult(
        ndcg=float(value),
        candidates=n,
        relevant=relevant,
        included_in_macro=True,
        exclusion_reason=None,
    )


def macro_average_ndcg(results: Sequence[UserNDCGResult]) -> float:
    """Equal-weight macro average over users eligible for target-specific NDCG."""
    values = [
        float(result.ndcg)
        for result in results
        if result.included_in_macro and result.ndcg is not None
    ]
    if not values:
        raise ValueError("no users eligible for macro NDCG")
    return float(sum(values) / len(values))
