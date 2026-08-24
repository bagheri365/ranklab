from pathlib import Path

import pandas as pd

from ranklab.evaluation.candidate_audit import build_candidate_structure_audit


def _write(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_candidate_audit_has_no_model_outputs(tmp_path: Path) -> None:
    standard = [
        {"user_id": 1, "video_id": 10, "tab": 1, "is_click": 1, "long_view": 0},
        {"user_id": 1, "video_id": 20, "tab": 2, "is_click": 0, "long_view": 0},
        {"user_id": 2, "video_id": 20, "tab": 1, "is_click": 0, "long_view": 1},
    ]
    randomized = [
        {"user_id": 1, "video_id": 20, "tab": 1, "is_click": 0, "long_view": 0},
        {"user_id": 2, "video_id": 10, "tab": 2, "is_click": 1, "long_view": 1},
    ]
    _write(tmp_path / "log_standard_4_22_to_5_08_pure.csv", standard)
    _write(tmp_path / "log_random_4_22_to_5_08_pure.csv", randomized)

    payload = build_candidate_structure_audit(tmp_path)

    assert payload["status"] == "M0_CANDIDATE_STRUCTURE_AUDIT_ONLY"
    assert len(payload["summaries"]) == 6
    assert "model" not in payload
    assert "ndcg" not in str(payload).lower()
