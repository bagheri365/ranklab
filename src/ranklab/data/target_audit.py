from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

TARGET_COLUMNS: tuple[str, ...] = (
    "is_click",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "tab",
)

LOGS: tuple[tuple[str, str], ...] = (
    ("standard_train", "log_standard_4_08_to_4_21_pure.csv"),
    ("standard_eval", "log_standard_4_22_to_5_08_pure.csv"),
    ("randomized_eval", "log_random_4_22_to_5_08_pure.csv"),
)

DURATION_BUCKET_ORDER: tuple[str, ...] = (
    "nonpositive",
    "0_7s",
    "7_18s",
    "18_30s",
    "30s_plus",
)


def _bucket_duration(duration_ms: pd.Series) -> pd.Series:
    d = pd.to_numeric(duration_ms, errors="coerce")
    result = pd.Series("missing", index=d.index, dtype="object")
    result.loc[d <= 0] = "nonpositive"
    result.loc[(d > 0) & (d <= 7_000)] = "0_7s"
    result.loc[(d > 7_000) & (d <= 18_000)] = "7_18s"
    result.loc[(d > 18_000) & (d <= 30_000)] = "18_30s"
    result.loc[d > 30_000] = "30s_plus"
    return result


def _safe_rate(positive: int, observed: int) -> float | None:
    return float(positive / observed) if observed else None


def _long_view_expected(play: pd.Series, duration: pd.Series) -> pd.Series:
    p = pd.to_numeric(play, errors="coerce")
    d = pd.to_numeric(duration, errors="coerce")
    valid = p.notna() & d.notna() & (d > 0)
    expected = pd.Series(pd.NA, index=p.index, dtype="Int64")
    expected.loc[valid & (d <= 18_000)] = (
        p.loc[valid & (d <= 18_000)] >= d.loc[valid & (d <= 18_000)]
    ).astype(int)
    expected.loc[valid & (d > 18_000)] = (p.loc[valid & (d > 18_000)] >= 18_000).astype(int)
    return expected


def _summarize_log(path: Path, *, chunksize: int) -> dict[str, Any]:
    rows = 0
    joint = Counter()
    anomalies = Counter()
    long_rule = Counter()
    by_duration: dict[str, Counter[str]] = defaultdict(Counter)
    by_tab: dict[str, Counter[str]] = defaultdict(Counter)

    for chunk in pd.read_csv(path, usecols=list(TARGET_COLUMNS), chunksize=chunksize):
        rows += len(chunk)
        click = pd.to_numeric(chunk["is_click"], errors="coerce")
        long_view = pd.to_numeric(chunk["long_view"], errors="coerce")
        play = pd.to_numeric(chunk["play_time_ms"], errors="coerce")
        duration = pd.to_numeric(chunk["duration_ms"], errors="coerce")
        buckets = _bucket_duration(duration)

        joint["click_observed"] += int(click.notna().sum())
        joint["click_positive"] += int((click == 1).sum())
        joint["long_observed"] += int(long_view.notna().sum())
        joint["long_positive"] += int((long_view == 1).sum())
        joint["both_observed"] += int((click.notna() & long_view.notna()).sum())
        joint["click_and_long"] += int(((click == 1) & (long_view == 1)).sum())

        anomalies["duration_nonpositive"] += int((duration <= 0).fillna(False).sum())
        anomalies["play_exceeds_duration"] += int(
            ((play > duration) & play.notna() & duration.notna()).sum()
        )
        anomalies["play_negative"] += int((play < 0).fillna(False).sum())

        expected = _long_view_expected(play, duration)
        comparable = expected.notna() & long_view.notna()
        long_rule["comparable_rows"] += int(comparable.sum())
        long_rule["matches"] += int((expected[comparable] == long_view[comparable]).sum())
        long_rule["mismatches"] += int((expected[comparable] != long_view[comparable]).sum())
        long_rule["excluded_nonpositive_or_missing_duration"] += int((~expected.notna()).sum())

        for bucket, part_index in buckets.groupby(buckets).groups.items():
            c = click.loc[part_index]
            l = long_view.loc[part_index]
            acc = by_duration[str(bucket)]
            acc["rows"] += len(part_index)
            acc["click_observed"] += int(c.notna().sum())
            acc["click_positive"] += int((c == 1).sum())
            acc["long_observed"] += int(l.notna().sum())
            acc["long_positive"] += int((l == 1).sum())

        for tab_value, part_index in chunk.groupby("tab", dropna=False).groups.items():
            c = click.loc[part_index]
            l = long_view.loc[part_index]
            key = str(tab_value)
            acc = by_tab[key]
            acc["rows"] += len(part_index)
            acc["click_observed"] += int(c.notna().sum())
            acc["click_positive"] += int((c == 1).sum())
            acc["long_observed"] += int(l.notna().sum())
            acc["long_positive"] += int((l == 1).sum())

    duration_payload: dict[str, Any] = {}
    ordered = list(DURATION_BUCKET_ORDER) + ["missing"]
    for bucket in ordered:
        if bucket not in by_duration:
            continue
        acc = by_duration[bucket]
        duration_payload[bucket] = {
            "rows": acc["rows"],
            "row_share": float(acc["rows"] / rows) if rows else 0.0,
            "is_click_prevalence": _safe_rate(acc["click_positive"], acc["click_observed"]),
            "long_view_prevalence": _safe_rate(acc["long_positive"], acc["long_observed"]),
        }

    tab_payload: dict[str, Any] = {}
    for tab, acc in sorted(by_tab.items(), key=lambda item: item[0]):
        tab_payload[tab] = {
            "rows": acc["rows"],
            "row_share": float(acc["rows"] / rows) if rows else 0.0,
            "is_click_prevalence": _safe_rate(acc["click_positive"], acc["click_observed"]),
            "long_view_prevalence": _safe_rate(acc["long_positive"], acc["long_observed"]),
        }

    p_long_given_click = _safe_rate(joint["click_and_long"], joint["click_positive"])
    p_click_given_long = _safe_rate(joint["click_and_long"], joint["long_positive"])
    comparable = long_rule["comparable_rows"]

    return {
        "name": path.name,
        "rows": rows,
        "target_prevalence": {
            "is_click": _safe_rate(joint["click_positive"], joint["click_observed"]),
            "long_view": _safe_rate(joint["long_positive"], joint["long_observed"]),
        },
        "target_relationship": {
            "p_long_view_given_click": p_long_given_click,
            "p_click_given_long_view": p_click_given_long,
            "click_and_long_view_rows": joint["click_and_long"],
        },
        "duration_buckets": duration_payload,
        "by_tab": tab_payload,
        "duration_anomalies": dict(sorted(anomalies.items())),
        "long_view_rule_check": {
            "comparable_rows": comparable,
            "matches": long_rule["matches"],
            "mismatches": long_rule["mismatches"],
            "agreement_rate": _safe_rate(long_rule["matches"], comparable),
            "excluded_nonpositive_or_missing_duration": long_rule[
                "excluded_nonpositive_or_missing_duration"
            ],
            "rule": (
                "For positive duration: long_view=1 when play_time_ms >= duration_ms for "
                "duration <= 18,000 ms, else when play_time_ms >= 18,000 ms."
            ),
        },
    }


def build_target_audit(data_dir: str | Path, *, chunksize: int = 250_000) -> dict[str, Any]:
    """Audit target semantics and duration dependence without freezing M1 targets."""
    root = Path(data_dir)
    logs = {
        label: _summarize_log(root / filename, chunksize=chunksize)
        for label, filename in LOGS
    }
    return {
        "dataset": "KuaiRand-Pure",
        "status": "M0_TARGET_SEMANTICS_AUDIT_ONLY",
        "official_semantics": {
            "is_click": (
                "Binary feedback whose meaning depends on UI: click in two-column UI; "
                "valid_play in single-column UI. Because the public documentation does not "
                "map numeric tab IDs to UI type, RankLab does not mechanically reconstruct "
                "is_click from play/duration in this audit."
            ),
            "long_view": (
                "Binary feedback: long_view=1 when play_time_ms >= duration_ms for videos "
                "<=18s, or play_time_ms >=18s for videos >18s."
            ),
            "play_time_ms": "User view time in milliseconds.",
            "duration_ms": "Video duration in milliseconds.",
        },
        "logs": logs,
        "interpretation": (
            "Descriptive target-quality audit only. This artifact does not freeze is_click "
            "or long_view as primary targets and does not assign semantic UI names to tab IDs."
        ),
    }
