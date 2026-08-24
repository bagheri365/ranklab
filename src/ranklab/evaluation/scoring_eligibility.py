"""Frozen M0.18 scoring-eligibility rule.

All models are evaluated on the same subset of M0.15 support:
users and items must both belong to the frozen BPR/LightGCN training index
universe. This avoids introducing untrained cold-start representations and
prevents Popularity from receiving a broader evaluation population.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ranklab.training.primary_multiseed import load_primary_training_data


@dataclass(frozen=True)
class TrainingIndexUniverse:
    users: frozenset[int]
    items: frozenset[int]


def derive_training_index_universe(
    training_path: str | Path,
) -> TrainingIndexUniverse:
    training = load_primary_training_data(training_path)
    return TrainingIndexUniverse(
        users=frozenset(int(v) for v in training.eligible_users),
        items=frozenset(int(v) for v in training.training_items),
    )


def restrict_to_training_indexed_entities(
    frame: pd.DataFrame,
    universe: TrainingIndexUniverse,
) -> pd.DataFrame:
    required = {"user_id", "video_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    mask = (
        frame["user_id"].astype(int).isin(universe.users)
        & frame["video_id"].astype(int).isin(universe.items)
    )
    return frame.loc[mask].copy()
