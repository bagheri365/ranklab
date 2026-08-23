from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ranklab.data.training_audit import build_training_audit


def _write_fixture(path: Path) -> None:
    rows = []
    for day in range(9, 15):
        for user in (1, 2):
            rows.append(
                {
                    "user_id": user,
                    "video_id": 100 + day + user,
                    "date": f"202204{day:02d}",
                    "is_like": int(user == 1 and day % 2 == 0),
                    "is_follow": 0,
                    "is_comment": int(user == 2 and day == 14),
                    "is_forward": 0,
                    "is_hate": 0,
                    "is_profile_enter": int(user == 1),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_training_audit_keeps_last_dates_for_validation(tmp_path: Path) -> None:
    _write_fixture(tmp_path / "log_standard_4_08_to_4_21_pure.csv")
    payload = build_training_audit(tmp_path, validation_days=2)

    split = payload["presumptive_temporal_split"]
    assert split["train_dates"] == ["20220409", "20220410", "20220411", "20220412"]
    assert split["validation_dates"] == ["20220413", "20220414"]
    assert payload["status"] == "M0_TRAINING_CONTRACT_AUDIT_ONLY"
    assert split["validation"]["rows"] == 4
    assert payload["candidate_training_interactions"]["any_positive_social"]["positive_rows"] == 4
    assert payload["candidate_training_interactions"]["downstream_engagement"]["positive_rows"] == 7
    assert split["validation"]["candidate_training_interactions"]["downstream_engagement"]["positive_rows"] == 3


def test_training_audit_requires_enough_dates(tmp_path: Path) -> None:
    _write_fixture(tmp_path / "log_standard_4_08_to_4_21_pure.csv")
    with pytest.raises(ValueError):
        build_training_audit(tmp_path, validation_days=6)
