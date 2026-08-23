from pathlib import Path

import pandas as pd

from ranklab.data.training_contract_audit import build_training_contract_audit


def test_training_contract_collapses_pairs_and_uses_same_user_negatives_with_replacement(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    # 6 dates allow all prespecified 3/4/5-day validation windows.
    rows.extend([
        {"user_id": 1, "video_id": 10, "date": "20220409", "is_click": 0},
        {"user_id": 1, "video_id": 10, "date": "20220409", "is_click": 1},
        {"user_id": 1, "video_id": 11, "date": "20220409", "is_click": 0},
        {"user_id": 2, "video_id": 12, "date": "20220409", "is_click": 1},
        {"user_id": 2, "video_id": 13, "date": "20220409", "is_click": 0},
        {"user_id": 3, "video_id": 14, "date": "20220409", "is_click": 1},
    ])
    for date in ["20220410", "20220411", "20220412", "20220413", "20220414"]:
        rows.extend([
            {"user_id": 1, "video_id": 20, "date": date, "is_click": 1},
            {"user_id": 1, "video_id": 21, "date": date, "is_click": 0},
        ])
    pd.DataFrame(rows).to_csv(data / "log_standard_4_08_to_4_21_pure.csv", index=False)

    payload = build_training_contract_audit(data)
    assert payload["status"] == "M0_TRAINING_CONTRACT_FREEZE_CANDIDATE_ONLY"
    assert "with replacement" in payload["proposed_contract"]["negative_sampling"]
    assert set(payload["validation_window_candidates"]) == {"3", "4", "5"}

    five_day_train = payload["validation_window_candidates"]["5"]["train"]
    assert five_day_train["positive_pairs"] == 3
    assert five_day_train["negative_only_pairs"] == 2
    assert five_day_train["users_eligible_for_same_user_logged_negative_sampling"] == 2
    assert five_day_train["users_excluded_no_logged_negative"] == 1
    assert five_day_train["positive_user_retention_for_pairwise_training"] == 2 / 3


def test_validation_window_audit_reports_generic_depth_fields(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    for day in range(9, 15):
        date = f"202204{day:02d}"
        rows.extend([
            {"user_id": 1, "video_id": day * 10, "date": date, "is_click": 1},
            {"user_id": 1, "video_id": day * 10 + 1, "date": date, "is_click": 0},
        ])
    pd.DataFrame(rows).to_csv(data / "log_standard_4_08_to_4_21_pure.csv", index=False)

    payload = build_training_contract_audit(data)
    stats = payload["validation_window_candidates"]["3"]["validation"]
    assert "users_with_positive_and_2_plus_candidates" in stats
    assert "users_with_10_plus_candidates" in stats
    assert not any(key.startswith("validation_users_") for key in stats)
