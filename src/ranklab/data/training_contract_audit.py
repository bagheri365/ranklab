from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

STANDARD_TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
PRIMARY_K = 10
VALIDATION_WINDOW_CANDIDATES = (3, 4, 5)
PROVISIONAL_VALIDATION_DAYS = 3
NEGATIVES_PER_POSITIVE = 1


def _last_n_dates(values: pd.Series, n: int) -> tuple[list[str], list[str]]:
    dates = sorted(values.dropna().astype(str).unique().tolist())
    if len(dates) <= n:
        raise ValueError("not enough distinct dates to create train/validation split")
    return dates[:-n], dates[-n:]


def _collapse_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[["user_id", "video_id", "is_click"]].copy()
    work["is_click"] = pd.to_numeric(work["is_click"], errors="coerce").fillna(0).astype(int)
    return (
        work.groupby(["user_id", "video_id"], as_index=False, sort=False)["is_click"]
        .max()
        .rename(columns={"is_click": "relevance"})
    )


def _pair_stats(frame: pd.DataFrame) -> dict[str, Any]:
    pairs = _collapse_pairs(frame)
    positives = pairs[pairs["relevance"] == 1]
    negatives = pairs[pairs["relevance"] == 0]

    pos_per_user = positives.groupby("user_id", sort=False).size()
    neg_per_user = negatives.groupby("user_id", sort=False).size()
    users = pd.Index(pairs["user_id"].unique())
    per_user = pd.DataFrame(index=users)
    per_user["positives"] = pos_per_user.reindex(users, fill_value=0)
    per_user["negatives"] = neg_per_user.reindex(users, fill_value=0)
    per_user["candidates"] = per_user["positives"] + per_user["negatives"]

    has_positive = per_user["positives"] > 0
    has_negative = per_user["negatives"] > 0
    pairwise_eligible = has_positive & has_negative
    ranking_eligible = has_positive & (per_user["candidates"] >= 2)

    users_with_positive = int(has_positive.sum())
    users_pairwise_eligible = int(pairwise_eligible.sum())

    return {
        "rows": int(len(frame)),
        "unique_pairs": int(len(pairs)),
        "positive_pairs": int(len(positives)),
        "negative_only_pairs": int(len(negatives)),
        "positive_pair_prevalence": float(len(positives) / len(pairs)) if len(pairs) else 0.0,
        "users": int(len(users)),
        "users_with_positive": users_with_positive,
        "users_with_negative": int(has_negative.sum()),
        "users_with_both": users_pairwise_eligible,
        "users_eligible_for_same_user_logged_negative_sampling": users_pairwise_eligible,
        "positive_user_retention_for_pairwise_training": (
            float(users_pairwise_eligible / users_with_positive) if users_with_positive else 0.0
        ),
        "users_excluded_no_logged_negative": int((has_positive & ~has_negative).sum()),
        "users_with_positive_and_2_plus_candidates": int(ranking_eligible.sum()),
        "users_with_10_plus_candidates": int((ranking_eligible & (per_user["candidates"] >= PRIMARY_K)).sum()),
        "candidate_count_median": float(per_user["candidates"].median()) if len(per_user) else 0.0,
        "candidate_count_p90": float(per_user["candidates"].quantile(0.9)) if len(per_user) else 0.0,
    }


def _overlap_stats(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, Any]:
    train_users = set(train["user_id"].unique().tolist())
    val_users = set(validation["user_id"].unique().tolist())
    train_videos = set(train["video_id"].unique().tolist())
    val_videos = set(validation["video_id"].unique().tolist())
    return {
        "validation_users": len(val_users),
        "validation_users_seen_in_train": len(train_users & val_users),
        "validation_user_coverage": float(len(train_users & val_users) / len(val_users)) if val_users else 0.0,
        "validation_videos": len(val_videos),
        "validation_videos_seen_in_train": len(train_videos & val_videos),
        "validation_video_coverage": float(len(train_videos & val_videos) / len(val_videos)) if val_videos else 0.0,
    }


def _window_audit(frame: pd.DataFrame, validation_days: int) -> dict[str, Any]:
    train_dates, validation_dates = _last_n_dates(frame["date"], validation_days)
    train = frame[frame["date"].isin(train_dates)].copy()
    validation = frame[frame["date"].isin(validation_dates)].copy()
    return {
        "validation_days": validation_days,
        "train_dates": train_dates,
        "validation_dates": validation_dates,
        "train": _pair_stats(train),
        "validation": _pair_stats(validation),
        "overlap": _overlap_stats(train, validation),
    }


def build_training_contract_audit(data_dir: Path) -> dict[str, Any]:
    path = data_dir / STANDARD_TRAIN_LOG
    frame = pd.read_csv(
        path,
        usecols=["user_id", "video_id", "date", "is_click"],
        dtype={"date": "string"},
    )
    frame["date"] = frame["date"].astype("string")
    frame["is_click"] = pd.to_numeric(frame["is_click"], errors="coerce")

    windows = {
        str(days): _window_audit(frame, days)
        for days in VALIDATION_WINDOW_CANDIDATES
    }
    provisional = windows[str(PROVISIONAL_VALIDATION_DAYS)]

    return {
        "dataset": "KuaiRand-Pure",
        "status": "M0_TRAINING_CONTRACT_FREEZE_CANDIDATE_ONLY",
        "interpretation": (
            "This audit tests the click-based implicit-feedback training contract and compares validation-window depth "
            "using standard-policy history only. It does not freeze M1 and does not inspect Apr 22-May 8 test labels."
        ),
        "proposed_contract": {
            "training_source": STANDARD_TRAIN_LOG,
            "provisional_training_dates": provisional["train_dates"],
            "provisional_validation_dates": provisional["validation_dates"],
            "positive_pair_rule": (
                "A user-video pair is positive if any standard-policy training exposure for that pair has is_click=1."
            ),
            "negative_pair_rule": (
                "A user-video pair is an eligible logged negative only if it was exposed in the standard-policy training slice "
                "and never has is_click=1 in that slice."
            ),
            "negative_sampling": (
                "For BPR-style objectives, sample one eligible logged negative per positive pair per epoch from the same user's "
                "negative pool, uniformly with replacement. Resample each epoch using the model training seed."
            ),
            "pairwise_training_eligibility": (
                "A user must have at least one positive pair and at least one eligible same-user logged negative. Users with "
                "positive pairs but zero eligible logged negatives are excluded from pairwise BPR/LightGCN training and reported."
            ),
            "lightgcn_graph": (
                "Build user-item graph edges from positive click pairs only; use the same-user logged-negative pool for pairwise loss."
            ),
            "validation_window_selection_rule": (
                "Choose among the prespecified 3-, 4-, and 5-day trailing validation windows using only ranking-depth, overlap, "
                "and sample-size adequacy. Do not use Apr 22-May 8 labels or downstream model performance to choose the window."
            ),
            "validation_ranking_unit": (
                "One validation user. Candidate set is that user's unique logged user-video pairs in the frozen validation window; "
                "relevance is binary any-click."
            ),
            "validation_metric": (
                "Macro-average NDCG@10 across validation users with at least one positive pair and at least two candidate pairs; "
                "candidate sets shorter than K use the frozen <K metric semantics."
            ),
            "primary_k": PRIMARY_K,
            "negatives_per_positive": NEGATIVES_PER_POSITIVE,
            "checkpoint_rule": (
                "Tune on validation only; refit/freeze each selected model configuration on the frozen training slice per frozen "
                "seed before M1."
            ),
        },
        "validation_window_candidates": windows,
        "guardrails": [
            "Do not use Apr 22-May 8 standard or randomized evaluation labels to choose this contract.",
            "Do not use long_view for training or hyperparameter selection in the primary study.",
            "Do not allow the same user-video pair to be both positive and negative after pair collapse.",
            "Do not require a distinct negative for every positive; same-user logged negatives are sampled with replacement.",
            "Report the count and fraction of positive users excluded because they have zero eligible logged negatives.",
            "Choose validation-window length from prespecified depth/coverage criteria only, not downstream model performance.",
            "M1 remains blocked until this contract and the remaining M0 decisions are promoted into the frozen protocol artifact.",
        ],
    }
