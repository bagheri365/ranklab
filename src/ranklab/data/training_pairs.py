from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("user_id", "video_id", "date", "is_click")


@dataclass(frozen=True)
class TrainingPairSummary:
    rows: int
    unique_pairs: int
    positive_pairs: int
    negative_pairs: int
    users_with_positive: int
    users_with_negative: int
    users_with_both: int


def collapse_click_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse exposures into one click-labelled row per user-video pair.

    A pair is positive when any exposure in the supplied training slice has
    ``is_click == 1``. Otherwise it is an eligible logged negative pair.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    pairs = (
        frame.loc[:, ["user_id", "video_id", "is_click"]]
        .assign(is_click=lambda df: pd.to_numeric(df["is_click"], errors="raise").astype(int))
        .groupby(["user_id", "video_id"], as_index=False, sort=True)["is_click"]
        .max()
        .rename(columns={"is_click": "is_positive"})
    )
    pairs["is_positive"] = pairs["is_positive"].astype(bool)
    return pairs


def load_training_slice(
    path: str | Path,
    *,
    start_date: int = 20220409,
    end_date: int = 20220416,
) -> pd.DataFrame:
    """Load only the frozen standard-history training dates and core pair fields."""
    frame = pd.read_csv(Path(path), usecols=list(REQUIRED_COLUMNS))
    dates = pd.to_numeric(frame["date"], errors="raise").astype(int)
    return frame.loc[dates.between(start_date, end_date)].copy()


def build_training_pairs(
    path: str | Path,
    *,
    start_date: int = 20220409,
    end_date: int = 20220416,
) -> pd.DataFrame:
    return collapse_click_pairs(load_training_slice(path, start_date=start_date, end_date=end_date))


def summarize_training_pairs(pairs: pd.DataFrame) -> TrainingPairSummary:
    required = {"user_id", "video_id", "is_positive"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"pair table must contain {sorted(required)}")

    positive = pairs.loc[pairs["is_positive"].astype(bool)]
    negative = pairs.loc[~pairs["is_positive"].astype(bool)]
    positive_users = set(positive["user_id"].tolist())
    negative_users = set(negative["user_id"].tolist())
    return TrainingPairSummary(
        rows=len(pairs),
        unique_pairs=len(pairs),
        positive_pairs=len(positive),
        negative_pairs=len(negative),
        users_with_positive=len(positive_users),
        users_with_negative=len(negative_users),
        users_with_both=len(positive_users & negative_users),
    )
