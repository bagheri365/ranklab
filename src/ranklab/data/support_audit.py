from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

EVAL_COLUMNS: tuple[str, ...] = (
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
class RestrictedSummary:
    name: str
    restriction: str
    rows: int
    row_retention: float
    unique_users: int
    unique_videos: int
    tabs: dict[str, int]
    target_prevalence: dict[str, float | None]
    exposure_per_user: dict[str, float | int | None]
    duration_sanity: dict[str, int]


def _ratio(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _prevalence(positive: int, observed: int) -> float | None:
    return float(positive / observed) if observed else None


def _summary_from_chunks(
    path: Path,
    *,
    restriction: str,
    total_rows: int,
    chunksize: int,
    users: set[Any] | None = None,
    videos: set[Any] | None = None,
    tabs: set[Any] | None = None,
    exact_tab: Any | None = None,
) -> RestrictedSummary:
    rows = 0
    seen_users: set[Any] = set()
    seen_videos: set[Any] = set()
    tab_counts: Counter[str] = Counter()
    positive = Counter()
    observed = Counter()
    user_exposures: Counter[Any] = Counter()
    sanity = Counter()

    for chunk in pd.read_csv(path, usecols=list(EVAL_COLUMNS), chunksize=chunksize):
        mask = pd.Series(True, index=chunk.index)
        if users is not None:
            mask &= chunk["user_id"].isin(users)
        if videos is not None:
            mask &= chunk["video_id"].isin(videos)
        if tabs is not None:
            mask &= chunk["tab"].isin(tabs)
        if exact_tab is not None:
            mask &= chunk["tab"] == exact_tab
        part = chunk.loc[mask]
        if part.empty:
            continue

        rows += len(part)
        seen_users.update(part["user_id"].dropna().tolist())
        seen_videos.update(part["video_id"].dropna().tolist())
        user_exposures.update(part["user_id"].dropna().tolist())

        for value, count in part["tab"].value_counts(dropna=False).items():
            tab_counts[str(value)] += int(count)

        for target in ("is_click", "long_view"):
            values = pd.to_numeric(part[target], errors="coerce")
            observed[target] += int(values.notna().sum())
            positive[target] += int((values == 1).sum())

        duration = pd.to_numeric(part["duration_ms"], errors="coerce")
        play = pd.to_numeric(part["play_time_ms"], errors="coerce")
        sanity["duration_nonpositive"] += int((duration <= 0).fillna(False).sum())
        sanity["play_exceeds_duration"] += int(
            ((play > duration) & play.notna() & duration.notna()).sum()
        )

    counts = np.asarray(list(user_exposures.values()), dtype=float)
    if counts.size:
        exposure_per_user: dict[str, float | int | None] = {
            "users": int(counts.size),
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "p90": float(np.quantile(counts, 0.90)),
            "max": int(counts.max()),
        }
    else:
        exposure_per_user = {"users": 0, "mean": None, "median": None, "p90": None, "max": None}

    return RestrictedSummary(
        name=path.name,
        restriction=restriction,
        rows=rows,
        row_retention=_ratio(rows, total_rows),
        unique_users=len(seen_users),
        unique_videos=len(seen_videos),
        tabs=dict(sorted(tab_counts.items())),
        target_prevalence={
            target: _prevalence(positive[target], observed[target])
            for target in ("is_click", "long_view")
        },
        exposure_per_user=exposure_per_user,
        duration_sanity=dict(sorted(sanity.items())),
    )


def _sets_and_rows(path: Path, *, chunksize: int) -> tuple[set[Any], set[Any], set[Any], int]:
    users: set[Any] = set()
    videos: set[Any] = set()
    tabs: set[Any] = set()
    rows = 0
    for chunk in pd.read_csv(path, usecols=["user_id", "video_id", "tab"], chunksize=chunksize):
        rows += len(chunk)
        users.update(chunk["user_id"].dropna().tolist())
        videos.update(chunk["video_id"].dropna().tolist())
        tabs.update(chunk["tab"].dropna().tolist())
    return users, videos, tabs, rows


def _as_payload(items: Iterable[RestrictedSummary]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]


def build_support_audit(data_dir: str | Path, *, chunksize: int = 250_000) -> dict[str, Any]:
    """Describe common-support restrictions without freezing the matched-support policy."""
    root = Path(data_dir)
    standard = root / "log_standard_4_22_to_5_08_pure.csv"
    randomized = root / "log_random_4_22_to_5_08_pure.csv"

    std_users, std_videos, std_tabs, std_rows = _sets_and_rows(standard, chunksize=chunksize)
    rnd_users, rnd_videos, rnd_tabs, rnd_rows = _sets_and_rows(randomized, chunksize=chunksize)
    shared_users = std_users & rnd_users
    shared_videos = std_videos & rnd_videos
    shared_tabs = std_tabs & rnd_tabs

    standard_summaries = [
        _summary_from_chunks(
            standard,
            restriction="native",
            total_rows=std_rows,
            chunksize=chunksize,
        ),
        _summary_from_chunks(
            standard,
            restriction="shared_users_and_videos",
            total_rows=std_rows,
            chunksize=chunksize,
            users=shared_users,
            videos=shared_videos,
        ),
        _summary_from_chunks(
            standard,
            restriction="shared_users_videos_and_tabs",
            total_rows=std_rows,
            chunksize=chunksize,
            users=shared_users,
            videos=shared_videos,
            tabs=shared_tabs,
        ),
    ]
    randomized_summaries = [
        _summary_from_chunks(
            randomized,
            restriction="native",
            total_rows=rnd_rows,
            chunksize=chunksize,
        ),
        _summary_from_chunks(
            randomized,
            restriction="shared_users_and_videos",
            total_rows=rnd_rows,
            chunksize=chunksize,
            users=shared_users,
            videos=shared_videos,
        ),
        _summary_from_chunks(
            randomized,
            restriction="shared_users_videos_and_tabs",
            total_rows=rnd_rows,
            chunksize=chunksize,
            users=shared_users,
            videos=shared_videos,
            tabs=shared_tabs,
        ),
    ]

    tab1 = None
    if 1 in shared_tabs or 1.0 in shared_tabs:
        tab1 = {
            "note": (
                "tab=1 is reported as a descriptive shared-scenario slice only. "
                "The official KuaiRand README documents `tab` as a scenario identifier but "
                "does not assign public semantic names to each numeric value."
            ),
            "standard": asdict(
                _summary_from_chunks(
                    standard,
                    restriction="shared_users_videos_tab_1",
                    total_rows=std_rows,
                    chunksize=chunksize,
                    users=shared_users,
                    videos=shared_videos,
                    exact_tab=1,
                )
            ),
            "randomized": asdict(
                _summary_from_chunks(
                    randomized,
                    restriction="shared_users_videos_tab_1",
                    total_rows=rnd_rows,
                    chunksize=chunksize,
                    users=shared_users,
                    videos=shared_videos,
                    exact_tab=1,
                )
            ),
        }

    return {
        "dataset": "KuaiRand-Pure",
        "status": "M0_SUPPORT_SCENARIO_AUDIT_ONLY",
        "official_semantics": {
            "tab": (
                "Scenario identifier in the range [0, 14]; the public README gives examples "
                "such as recommendation/main-page scenarios but does not map numeric tab IDs "
                "to semantic names."
            ),
            "random_intervention": (
                "The official README states that random-exposure interactions result from "
                "random intervention in standard recommendation feeds."
            ),
        },
        "common_support": {
            "shared_users": len(shared_users),
            "shared_videos": len(shared_videos),
            "shared_tabs": sorted(str(value) for value in shared_tabs),
        },
        "standard": _as_payload(standard_summaries),
        "randomized": _as_payload(randomized_summaries),
        "tab_1_slice": tab1,
        "interpretation": (
            "Descriptive support/scenario audit only. No restriction in this artifact is "
            "promoted to the frozen matched-support policy."
        ),
    }
