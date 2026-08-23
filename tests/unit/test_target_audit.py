from __future__ import annotations

from pathlib import Path

import pandas as pd

from ranklab.data.target_audit import build_target_audit


def _write_log(path: Path) -> None:
    pd.DataFrame(
        {
            "is_click": [1, 0, 1, 0, 1],
            "long_view": [1, 0, 1, 0, 1],
            "play_time_ms": [5_000, 2_000, 18_000, 17_999, 30_000],
            "duration_ms": [5_000, 5_000, 20_000, 20_000, 30_000],
            "tab": [1, 1, 2, 2, 1],
        }
    ).to_csv(path, index=False)


def test_target_audit_reconstructs_long_view_rule(tmp_path: Path) -> None:
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "log_random_4_22_to_5_08_pure.csv",
    ):
        _write_log(tmp_path / filename)

    payload = build_target_audit(tmp_path, chunksize=2)
    check = payload["logs"]["standard_eval"]["long_view_rule_check"]
    assert check["comparable_rows"] == 5
    assert check["mismatches"] == 0
    assert check["agreement_rate"] == 1.0


def test_target_audit_reports_relationships_and_buckets(tmp_path: Path) -> None:
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "log_random_4_22_to_5_08_pure.csv",
    ):
        _write_log(tmp_path / filename)

    payload = build_target_audit(tmp_path)
    summary = payload["logs"]["randomized_eval"]
    assert summary["target_relationship"]["p_long_view_given_click"] == 1.0
    assert "0_7s" in summary["duration_buckets"]
    assert "18_30s" in summary["duration_buckets"]
    assert payload["status"] == "M0_TARGET_SEMANTICS_AUDIT_ONLY"
