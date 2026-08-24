from pathlib import Path

import pandas as pd

from ranklab.evaluation.post_eligibility_candidate_audit import (
    build_post_eligibility_candidate_audit,
)


def _write(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_post_eligibility_audit_filters_unseen_entities_without_model_outputs(
    tmp_path: Path,
) -> None:
    training = [
        {"user_id": 1, "video_id": 10, "date": 20220409, "is_click": 1},
        {"user_id": 1, "video_id": 20, "date": 20220410, "is_click": 0},
        {"user_id": 2, "video_id": 10, "date": 20220409, "is_click": 1},
        {"user_id": 2, "video_id": 20, "date": 20220410, "is_click": 0},
    ]
    standard = [
        {"user_id": 1, "video_id": 10, "tab": 1, "is_click": 1, "long_view": 0},
        {"user_id": 2, "video_id": 20, "tab": 1, "is_click": 0, "long_view": 0},
        {"user_id": 3, "video_id": 10, "tab": 1, "is_click": 1, "long_view": 1},
    ]
    randomized = [
        {"user_id": 1, "video_id": 20, "tab": 1, "is_click": 0, "long_view": 0},
        {"user_id": 2, "video_id": 10, "tab": 1, "is_click": 1, "long_view": 1},
        {"user_id": 3, "video_id": 20, "tab": 1, "is_click": 0, "long_view": 0},
    ]

    train_path = tmp_path / "train.csv"
    _write(train_path, training)
    _write(tmp_path / "log_standard_4_22_to_5_08_pure.csv", standard)
    _write(tmp_path / "log_random_4_22_to_5_08_pure.csv", randomized)

    payload = build_post_eligibility_candidate_audit(
        training_path=train_path,
        data_dir=tmp_path,
    )

    assert payload["status"] == "M0_POST_ELIGIBILITY_CANDIDATE_AUDIT_ONLY"
    primary = [
        row for row in payload["summaries"]
        if row["support"] == "primary_shared_users_and_videos_training_seen"
    ]
    assert all(row["users"] == 2 for row in primary)
    assert "model score" in payload["guardrail"]
