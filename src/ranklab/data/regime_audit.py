from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

AUDIT_COLUMNS: tuple[str, ...] = (
    "user_id",
    "video_id",
    "date",
    "is_click",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "is_rand",
    "tab",
)


@dataclass(frozen=True)
class LogSummary:
    name: str
    rows: int
    date_min: str | None
    date_max: str | None
    unique_users: int
    unique_videos: int
    is_rand_counts: dict[str, int]
    tab_counts: dict[str, int]
    target_counts: dict[str, dict[str, int]]
    target_prevalence: dict[str, float | None]
    missing: dict[str, int]
    duration_sanity: dict[str, int]


@dataclass(frozen=True)
class OverlapSummary:
    standard_name: str
    randomized_name: str
    shared_users: int
    standard_users: int
    randomized_users: int
    user_jaccard: float
    standard_user_coverage: float
    randomized_user_coverage: float
    shared_videos: int
    standard_videos: int
    randomized_videos: int
    video_jaccard: float
    standard_video_coverage: float
    randomized_video_coverage: float


def _key(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def _ratio(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _prevalence(counts: Counter[str]) -> float | None:
    positive = counts.get("1", 0) + counts.get("1.0", 0)
    observed = sum(value for key, value in counts.items() if key != "<NA>")
    if observed == 0:
        return None
    return float(positive / observed)


def summarize_log(path: str | Path, *, chunksize: int = 250_000) -> LogSummary:
    """Compute descriptive M0 statistics without assigning study semantics."""
    csv_path = Path(path)
    rows = 0
    users: set[Any] = set()
    videos: set[Any] = set()
    date_min: Any | None = None
    date_max: Any | None = None
    is_rand_counts: Counter[str] = Counter()
    tab_counts: Counter[str] = Counter()
    target_counts: dict[str, Counter[str]] = {
        "is_click": Counter(),
        "long_view": Counter(),
    }
    missing: Counter[str] = Counter()
    duration_sanity: Counter[str] = Counter()

    for chunk in pd.read_csv(csv_path, usecols=list(AUDIT_COLUMNS), chunksize=chunksize):
        rows += len(chunk)
        users.update(chunk["user_id"].dropna().tolist())
        videos.update(chunk["video_id"].dropna().tolist())

        dates = chunk["date"].dropna()
        if not dates.empty:
            chunk_min = dates.min()
            chunk_max = dates.max()
            date_min = chunk_min if date_min is None or chunk_min < date_min else date_min
            date_max = chunk_max if date_max is None or chunk_max > date_max else date_max

        for value, count in chunk["is_rand"].map(_key).value_counts(dropna=False).items():
            is_rand_counts[str(value)] += int(count)
        for value, count in chunk["tab"].map(_key).value_counts(dropna=False).items():
            tab_counts[str(value)] += int(count)

        for target in target_counts:
            for value, count in chunk[target].map(_key).value_counts(dropna=False).items():
                target_counts[target][str(value)] += int(count)

        for column in AUDIT_COLUMNS:
            missing[column] += int(chunk[column].isna().sum())

        duration = pd.to_numeric(chunk["duration_ms"], errors="coerce")
        play = pd.to_numeric(chunk["play_time_ms"], errors="coerce")
        duration_sanity["duration_missing_or_non_numeric"] += int(duration.isna().sum())
        duration_sanity["duration_nonpositive"] += int((duration <= 0).fillna(False).sum())
        duration_sanity["play_missing_or_non_numeric"] += int(play.isna().sum())
        duration_sanity["play_negative"] += int((play < 0).fillna(False).sum())
        duration_sanity["play_exceeds_duration"] += int(
            ((play > duration) & play.notna() & duration.notna()).sum()
        )

    prevalences = {target: _prevalence(counts) for target, counts in target_counts.items()}
    return LogSummary(
        name=csv_path.name,
        rows=rows,
        date_min=None if date_min is None else str(date_min),
        date_max=None if date_max is None else str(date_max),
        unique_users=len(users),
        unique_videos=len(videos),
        is_rand_counts=dict(sorted(is_rand_counts.items())),
        tab_counts=dict(sorted(tab_counts.items())),
        target_counts={key: dict(sorted(value.items())) for key, value in target_counts.items()},
        target_prevalence=prevalences,
        missing={column: int(missing[column]) for column in AUDIT_COLUMNS},
        duration_sanity=dict(sorted(duration_sanity.items())),
    )


def _id_sets(path: Path, *, chunksize: int) -> tuple[set[Any], set[Any]]:
    users: set[Any] = set()
    videos: set[Any] = set()
    for chunk in pd.read_csv(path, usecols=["user_id", "video_id"], chunksize=chunksize):
        users.update(chunk["user_id"].dropna().tolist())
        videos.update(chunk["video_id"].dropna().tolist())
    return users, videos


def summarize_overlap(
    standard_path: str | Path,
    randomized_path: str | Path,
    *,
    chunksize: int = 250_000,
) -> OverlapSummary:
    standard = Path(standard_path)
    randomized = Path(randomized_path)
    std_users, std_videos = _id_sets(standard, chunksize=chunksize)
    rnd_users, rnd_videos = _id_sets(randomized, chunksize=chunksize)

    shared_users = std_users & rnd_users
    shared_videos = std_videos & rnd_videos
    union_users = std_users | rnd_users
    union_videos = std_videos | rnd_videos

    return OverlapSummary(
        standard_name=standard.name,
        randomized_name=randomized.name,
        shared_users=len(shared_users),
        standard_users=len(std_users),
        randomized_users=len(rnd_users),
        user_jaccard=_ratio(len(shared_users), len(union_users)),
        standard_user_coverage=_ratio(len(shared_users), len(std_users)),
        randomized_user_coverage=_ratio(len(shared_users), len(rnd_users)),
        shared_videos=len(shared_videos),
        standard_videos=len(std_videos),
        randomized_videos=len(rnd_videos),
        video_jaccard=_ratio(len(shared_videos), len(union_videos)),
        standard_video_coverage=_ratio(len(shared_videos), len(std_videos)),
        randomized_video_coverage=_ratio(len(shared_videos), len(rnd_videos)),
    )


def build_regime_audit(data_dir: str | Path, *, chunksize: int = 250_000) -> dict[str, Any]:
    root = Path(data_dir)
    names = (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "log_random_4_22_to_5_08_pure.csv",
    )
    summaries = [summarize_log(root / name, chunksize=chunksize) for name in names]
    overlap = summarize_overlap(root / names[1], root / names[2], chunksize=chunksize)
    return {
        "dataset": "KuaiRand-Pure",
        "status": "M0_DESCRIPTIVE_AUDIT_ONLY",
        "logs": [asdict(summary) for summary in summaries],
        "evaluation_regime_overlap": asdict(overlap),
        "interpretation": (
            "Descriptive audit only. These statistics do not freeze target semantics, "
            "ranking units, matched support, or M1 conclusions."
        ),
    }
