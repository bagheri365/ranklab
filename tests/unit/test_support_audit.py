from __future__ import annotations

from pathlib import Path

import pandas as pd

from ranklab.data.support_audit import build_support_audit


def _write(path: Path, rows: list[dict[str, int]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(user: int, video: int, tab: int, click: int, long_view: int, *, rand: int) -> dict[str, int]:
    return {
        "user_id": user,
        "video_id": video,
        "date": 20220422,
        "is_click": click,
        "long_view": long_view,
        "play_time_ms": 1000,
        "duration_ms": 2000,
        "is_rand": rand,
        "tab": tab,
    }


def test_support_audit_restricts_to_common_support(tmp_path: Path) -> None:
    _write(
        tmp_path / "log_standard_4_22_to_5_08_pure.csv",
        [_row(1, 10, 1, 1, 1, rand=0), _row(2, 20, 4, 0, 0, rand=0)],
    )
    _write(
        tmp_path / "log_random_4_22_to_5_08_pure.csv",
        [_row(1, 10, 1, 0, 0, rand=1), _row(3, 30, 2, 0, 0, rand=1)],
    )

    payload = build_support_audit(tmp_path, chunksize=1)

    assert payload["common_support"]["shared_users"] == 1
    assert payload["common_support"]["shared_videos"] == 1
    assert payload["common_support"]["shared_tabs"] == ["1"]
    assert payload["standard"][1]["rows"] == 1
    assert payload["randomized"][1]["rows"] == 1
    assert payload["tab_1_slice"] is not None
    assert payload["status"] == "M0_SUPPORT_SCENARIO_AUDIT_ONLY"
