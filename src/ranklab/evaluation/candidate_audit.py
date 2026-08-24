"""Candidate-structure audit under the frozen M0.15 support policy.

This audit uses evaluation labels only to describe candidate/relevance
structure. It does not load, score, compare, or rank any recommender model.

The purpose is to freeze later edge-case rules (<K, minimum candidate count,
zero-relevance users) without observing model performance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ranklab.evaluation.support import (
    derive_evaluation_support,
    restrict_primary_support,
    restrict_shared_tab_sensitivity,
    restrict_tab1_sensitivity,
)


EVAL_COLUMNS = ("user_id", "video_id", "tab", "is_click", "long_view")


@dataclass(frozen=True)
class TargetCandidateSummary:
    target: str
    users: int
    users_with_relevance: int
    users_zero_relevance: int
    users_with_relevance_and_ge2_candidates: int
    users_with_relevance_and_ge10_candidates: int
    relevant_pairs: int
    candidate_pairs: int
    pair_relevance_prevalence: float | None


@dataclass(frozen=True)
class CandidateSummary:
    regime: str
    support: str
    rows: int
    unique_pairs: int
    duplicate_rows_removed_by_pair_collapse: int
    users: int
    candidate_count: dict[str, float | int | None]
    users_ge2_candidates: int
    users_ge10_candidates: int
    targets: list[TargetCandidateSummary]


def _quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "min": None,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "min": int(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": int(arr.max()),
    }


def collapse_logged_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated exposure rows to one user-video candidate.

    Target-specific relevance is binary any-positive over all retained rows for
    that user-video pair.
    """
    required = set(EVAL_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    work = frame.loc[:, list(EVAL_COLUMNS)].copy()
    for target in ("is_click", "long_view"):
        numeric = pd.to_numeric(work[target], errors="raise")
        if not numeric.isin([0, 1]).all():
            bad = sorted(set(numeric.loc[~numeric.isin([0, 1])].tolist()))
            raise ValueError(f"{target} must be binary; observed {bad[:5]}")
        work[target] = numeric.astype("int8")

    collapsed = (
        work.groupby(["user_id", "video_id"], sort=True, as_index=False)
        .agg(
            tab=("tab", "first"),
            is_click=("is_click", "max"),
            long_view=("long_view", "max"),
        )
    )
    return collapsed


def summarize_candidate_structure(
    frame: pd.DataFrame,
    *,
    regime: str,
    support_name: str,
) -> CandidateSummary:
    collapsed = collapse_logged_candidates(frame)
    per_user = collapsed.groupby("user_id", sort=True).size()
    counts = [int(v) for v in per_user.tolist()]

    target_summaries: list[TargetCandidateSummary] = []
    for target in ("is_click", "long_view"):
        relevant_per_user = collapsed.groupby("user_id", sort=True)[target].sum()
        # align to every user represented in the collapsed candidate table
        relevant_per_user = relevant_per_user.reindex(per_user.index, fill_value=0)
        positive_mask = relevant_per_user > 0
        ge2 = per_user >= 2
        ge10 = per_user >= 10
        relevant_pairs = int(collapsed[target].sum())
        candidate_pairs = int(len(collapsed))
        target_summaries.append(
            TargetCandidateSummary(
                target=target,
                users=int(len(per_user)),
                users_with_relevance=int(positive_mask.sum()),
                users_zero_relevance=int((~positive_mask).sum()),
                users_with_relevance_and_ge2_candidates=int((positive_mask & ge2).sum()),
                users_with_relevance_and_ge10_candidates=int((positive_mask & ge10).sum()),
                relevant_pairs=relevant_pairs,
                candidate_pairs=candidate_pairs,
                pair_relevance_prevalence=(
                    float(relevant_pairs / candidate_pairs)
                    if candidate_pairs
                    else None
                ),
            )
        )

    return CandidateSummary(
        regime=regime,
        support=support_name,
        rows=int(len(frame)),
        unique_pairs=int(len(collapsed)),
        duplicate_rows_removed_by_pair_collapse=int(len(frame) - len(collapsed)),
        users=int(len(per_user)),
        candidate_count=_quantiles(counts),
        users_ge2_candidates=int((per_user >= 2).sum()),
        users_ge10_candidates=int((per_user >= 10).sum()),
        targets=target_summaries,
    )


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(EVAL_COLUMNS))


def build_candidate_structure_audit(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir)
    standard_path = root / "log_standard_4_22_to_5_08_pure.csv"
    randomized_path = root / "log_random_4_22_to_5_08_pure.csv"

    standard = _read(standard_path)
    randomized = _read(randomized_path)
    support = derive_evaluation_support(standard, randomized)

    variants = [
        (
            "primary_shared_users_and_videos",
            restrict_primary_support,
        ),
        (
            "sensitivity_shared_users_videos_and_tabs",
            restrict_shared_tab_sensitivity,
        ),
        (
            "sensitivity_shared_users_videos_tab_1",
            restrict_tab1_sensitivity,
        ),
    ]

    summaries: list[CandidateSummary] = []
    for support_name, restrict_fn in variants:
        for regime, frame in (("standard", standard), ("randomized", randomized)):
            restricted = restrict_fn(frame, support)
            summaries.append(
                summarize_candidate_structure(
                    restricted,
                    regime=regime,
                    support_name=support_name,
                )
            )

    return {
        "status": "M0_CANDIDATE_STRUCTURE_AUDIT_ONLY",
        "guardrail": (
            "No recommender checkpoints, model scores, rankings, or model metrics "
            "are loaded or computed by this audit."
        ),
        "pair_collapse": "unique_user_video_pair_with_target_relevance_any_positive",
        "support": {
            "shared_users": len(support.shared_users),
            "shared_videos": len(support.shared_videos),
            "shared_tabs": sorted(support.shared_tabs),
        },
        "summaries": [
            {
                **{
                    key: value
                    for key, value in asdict(summary).items()
                    if key != "targets"
                },
                "targets": [asdict(target) for target in summary.targets],
            }
            for summary in summaries
        ],
        "interpretation": (
            "Descriptive label/candidate audit only. Minimum candidate count, "
            "<K handling, and zero-relevance-user handling remain unfrozen."
        ),
    }
