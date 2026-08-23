from __future__ import annotations

from pathlib import Path

import pandas as pd

from ranklab.data.regime_audit import build_regime_audit, summarize_log, summarize_overlap


def _write_log(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(user: int, video: int, *, click: int, long_view: int, is_rand: int, tab: int) -> dict[str, object]:
    return {
        "user_id": user,
        "video_id": video,
        "date": 20220422,
        "is_click": click,
        "long_view": long_view,
        "play_time_ms": 1000,
        "duration_ms": 2000,
        "is_rand": is_rand,
        "tab": tab,
    }


def test_summarize_log_counts_targets_and_sanity(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    _write_log(
        path,
        [
            _row(1, 10, click=1, long_view=0, is_rand=0, tab=1),
            _row(1, 11, click=0, long_view=1, is_rand=0, tab=1),
            _row(2, 10, click=1, long_view=1, is_rand=0, tab=2),
        ],
    )
    result = summarize_log(path, chunksize=2)
    assert result.rows == 3
    assert result.unique_users == 2
    assert result.unique_videos == 2
    assert result.target_prevalence["is_click"] == 2 / 3
    assert result.target_prevalence["long_view"] == 2 / 3
    assert result.duration_sanity["play_exceeds_duration"] == 0


def test_summarize_overlap_reports_shared_support(tmp_path: Path) -> None:
    standard = tmp_path / "standard.csv"
    randomized = tmp_path / "random.csv"
    _write_log(standard, [_row(1, 10, click=1, long_view=1, is_rand=0, tab=1), _row(2, 11, click=0, long_view=0, is_rand=0, tab=1)])
    _write_log(randomized, [_row(2, 11, click=1, long_view=1, is_rand=1, tab=1), _row(3, 12, click=0, long_view=0, is_rand=1, tab=1)])
    result = summarize_overlap(standard, randomized, chunksize=1)
    assert result.shared_users == 1
    assert result.shared_videos == 1
    assert result.user_jaccard == 1 / 3
    assert result.video_jaccard == 1 / 3


def test_build_regime_audit_is_descriptive_only(tmp_path: Path) -> None:
    names = [
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "log_random_4_22_to_5_08_pure.csv",
    ]
    for index, name in enumerate(names):
        _write_log(tmp_path / name, [_row(1, 10 + index, click=1, long_view=1, is_rand=int("random" in name), tab=1)])
    payload = build_regime_audit(tmp_path, chunksize=1)
    assert payload["status"] == "M0_DESCRIPTIVE_AUDIT_ONLY"
    assert len(payload["logs"]) == 3
