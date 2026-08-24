"""Frozen M0.15 evaluation support policy.

Primary support aligns the user and item universes across standard-policy and
randomized-policy evaluation while preserving each regime's own logged
user-item exposures.

This module intentionally does not collapse duplicate user-item rows or define
target relevance. Those candidate semantics are frozen separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


PRIMARY_SUPPORT_NAME = "shared_users_and_videos"
SHARED_TAB_SENSITIVITY_NAME = "shared_users_videos_and_tabs"
TAB1_SENSITIVITY_NAME = "shared_users_videos_tab_1"


@dataclass(frozen=True)
class EvaluationSupport:
    shared_users: frozenset[int]
    shared_videos: frozenset[int]
    shared_tabs: frozenset[int]


def derive_evaluation_support(
    standard: pd.DataFrame,
    randomized: pd.DataFrame,
) -> EvaluationSupport:
    """Derive entity/scenario intersections from the two evaluation regimes."""
    required = {"user_id", "video_id", "tab"}
    for name, frame in (("standard", standard), ("randomized", randomized)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")

    return EvaluationSupport(
        shared_users=frozenset(
            set(standard["user_id"].dropna().astype(int))
            & set(randomized["user_id"].dropna().astype(int))
        ),
        shared_videos=frozenset(
            set(standard["video_id"].dropna().astype(int))
            & set(randomized["video_id"].dropna().astype(int))
        ),
        shared_tabs=frozenset(
            set(standard["tab"].dropna().astype(int))
            & set(randomized["tab"].dropna().astype(int))
        ),
    )


def restrict_primary_support(
    frame: pd.DataFrame,
    support: EvaluationSupport,
) -> pd.DataFrame:
    """Keep shared users/items only; do not intersect exposed user-item pairs."""
    mask = (
        frame["user_id"].astype(int).isin(support.shared_users)
        & frame["video_id"].astype(int).isin(support.shared_videos)
    )
    return frame.loc[mask].copy()


def restrict_shared_tab_sensitivity(
    frame: pd.DataFrame,
    support: EvaluationSupport,
) -> pd.DataFrame:
    """Primary shared entity support plus the shared tab set."""
    primary = restrict_primary_support(frame, support)
    return primary.loc[primary["tab"].astype(int).isin(support.shared_tabs)].copy()


def restrict_tab1_sensitivity(
    frame: pd.DataFrame,
    support: EvaluationSupport,
) -> pd.DataFrame:
    """Primary shared entity support plus the descriptive tab=1 slice."""
    primary = restrict_primary_support(frame, support)
    return primary.loc[primary["tab"].astype(int) == 1].copy()
