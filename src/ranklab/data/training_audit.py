from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

STANDARD_TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
POSITIVE_SOCIAL_FIELDS = ("is_like", "is_follow", "is_comment", "is_forward")
DOWNSTREAM_ENGAGEMENT_FIELDS = ("is_profile_enter", *POSITIVE_SOCIAL_FIELDS)
RAW_AUDIT_FIELDS = POSITIVE_SOCIAL_FIELDS + ("is_hate", "is_profile_enter")


def _binary_rate(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    return float((numeric == 1).mean())


def _frame_stats(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "unique_users": int(frame["user_id"].nunique()),
        "unique_videos": int(frame["video_id"].nunique()),
    }


def _positive_user_stats(frame: pd.DataFrame, positive_mask: pd.Series) -> dict[str, Any]:
    positive = frame.loc[positive_mask, ["user_id", "video_id"]]
    if positive.empty:
        return {
            "positive_rows": 0,
            "positive_prevalence": 0.0,
            "users_with_1_plus": 0,
            "users_with_2_plus": 0,
            "users_with_5_plus": 0,
            "unique_positive_videos": 0,
        }

    per_user = positive.groupby("user_id", sort=False).size()
    return {
        "positive_rows": int(len(positive)),
        "positive_prevalence": float(positive_mask.mean()),
        "users_with_1_plus": int((per_user >= 1).sum()),
        "users_with_2_plus": int((per_user >= 2).sum()),
        "users_with_5_plus": int((per_user >= 5).sum()),
        "unique_positive_videos": int(positive["video_id"].nunique()),
    }


def _candidate_signals(frame: pd.DataFrame) -> dict[str, Any]:
    positive_social = pd.Series(False, index=frame.index)
    for field in POSITIVE_SOCIAL_FIELDS:
        positive_social |= pd.to_numeric(frame[field], errors="coerce").eq(1)

    downstream_engagement = pd.Series(False, index=frame.index)
    for field in DOWNSTREAM_ENGAGEMENT_FIELDS:
        downstream_engagement |= pd.to_numeric(frame[field], errors="coerce").eq(1)

    return {
        "all_logged_exposures": {
            "definition": "Every standard-policy logged exposure is a positive interaction.",
            "caveat": "This learns historical exposure structure rather than direct behavioral preference.",
            **_positive_user_stats(frame, pd.Series(True, index=frame.index)),
        },
        "any_positive_social": {
            "definition": "Positive when any of is_like, is_follow, is_comment, or is_forward equals 1.",
            "caveat": "This is behaviorally meaningful and distinct from is_click/long_view, but may be sparse and policy-conditioned.",
            **_positive_user_stats(frame, positive_social),
        },
        "downstream_engagement": {
            "definition": (
                "Positive when is_profile_enter equals 1 or any of is_like, is_follow, "
                "is_comment, or is_forward equals 1."
            ),
            "caveat": (
                "This remains distinct from is_click/long_view and may improve coverage over social-only positives, "
                "but it is still policy-conditioned and must not be promoted without density/support review."
            ),
            **_positive_user_stats(frame, downstream_engagement),
        },
    }


def _daily_stats(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("date", sort=True):
        record = {
            "date": str(date),
            **_frame_stats(group),
        }
        for field in RAW_AUDIT_FIELDS:
            record[f"{field}_prevalence"] = _binary_rate(group[field])
        rows.append(record)
    return rows


def _last_n_dates(dates: Iterable[str], n: int) -> tuple[list[str], list[str]]:
    ordered = sorted({str(value) for value in dates})
    if len(ordered) <= n:
        raise ValueError("not enough distinct dates to create an earlier train / later validation split")
    return ordered[:-n], ordered[-n:]


def _split_overlap(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, Any]:
    train_users = set(train["user_id"].unique())
    val_users = set(validation["user_id"].unique())
    train_videos = set(train["video_id"].unique())
    val_videos = set(validation["video_id"].unique())
    return {
        "validation_users_seen_in_train": len(train_users & val_users),
        "validation_user_coverage": len(train_users & val_users) / len(val_users) if val_users else 0.0,
        "validation_videos_seen_in_train": len(train_videos & val_videos),
        "validation_video_coverage": len(train_videos & val_videos) / len(val_videos) if val_videos else 0.0,
    }


def build_training_audit(data_dir: Path, validation_days: int = 3) -> dict[str, Any]:
    path = data_dir / STANDARD_TRAIN_LOG
    usecols = ["user_id", "video_id", "date", *RAW_AUDIT_FIELDS]
    frame = pd.read_csv(path, usecols=usecols, dtype={"date": "string"})
    frame["date"] = frame["date"].astype("string")

    train_dates, validation_dates = _last_n_dates(frame["date"].dropna().tolist(), validation_days)
    train = frame[frame["date"].isin(train_dates)].copy()
    validation = frame[frame["date"].isin(validation_dates)].copy()

    raw_signal_prevalence = {field: _binary_rate(frame[field]) for field in RAW_AUDIT_FIELDS}

    return {
        "dataset": "KuaiRand-Pure",
        "status": "M0_TRAINING_CONTRACT_AUDIT_ONLY",
        "interpretation": (
            "Descriptive training-signal and temporal-split audit only. No training interaction, "
            "negative-sampling rule, validation objective, or cutoff is frozen by this artifact."
        ),
        "source": {
            "file": STANDARD_TRAIN_LOG,
            "logging_regime": "standard-policy history only",
            "observed_dates": sorted(frame["date"].dropna().unique().tolist()),
            **_frame_stats(frame),
        },
        "raw_signal_prevalence": raw_signal_prevalence,
        "candidate_training_interactions": _candidate_signals(frame),
        "daily_stats": _daily_stats(frame),
        "presumptive_temporal_split": {
            "rule": f"Use the last {validation_days} observed standard-history dates as validation; earlier observed dates as training.",
            "status": "DESCRIPTIVE_ONLY_NOT_FROZEN",
            "train_dates": train_dates,
            "validation_dates": validation_dates,
            "train": {
                **_frame_stats(train),
                "candidate_training_interactions": _candidate_signals(train),
            },
            "validation": {
                **_frame_stats(validation),
                "candidate_training_interactions": _candidate_signals(validation),
            },
            "overlap": _split_overlap(train, validation),
        },
        "decision_guardrails": [
            "The training interaction must remain distinct from is_click and long_view.",
            "Do not select a training signal because it produces a more interesting M1 reversal.",
            "Do not touch Apr 22-May 8 evaluation labels while choosing the training interaction or validation cutoff.",
            "Freeze negative sampling, validation objective, and model-specific loss only after the interaction definition is selected.",
        ],
    }
