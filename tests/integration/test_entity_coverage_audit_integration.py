from pathlib import Path

import pandas as pd

from ranklab.evaluation.entity_coverage_audit import build_entity_coverage_audit


def _write(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_entity_coverage_audit_uses_training_contract_without_model_outputs(
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
    ]
    randomized = [
        {"user_id": 1, "video_id": 20, "tab": 1, "is_click": 0, "long_view": 0},
        {"user_id": 2, "video_id": 10, "tab": 1, "is_click": 1, "long_view": 1},
    ]

    train_path = tmp_path / "train.csv"
    _write(train_path, training)
    _write(tmp_path / "log_standard_4_22_to_5_08_pure.csv", standard)
    _write(tmp_path / "log_random_4_22_to_5_08_pure.csv", randomized)

    payload = build_entity_coverage_audit(
        training_path=train_path,
        data_dir=tmp_path,
    )

    assert payload["status"] == "M0_ENTITY_COVERAGE_AUDIT_ONLY"
    assert payload["training_index_universe"] == {
        "users": 2,
        "items": 2,
        "source": "frozen_pairwise_eligible_training_population",
    }
    assert all(
        row["fully_scorable_pair_fraction"] == 1.0
        for row in payload["summaries"]
    )
    assert "ndcg" not in str(payload).lower()
