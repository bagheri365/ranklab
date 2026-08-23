from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from ranklab.cli import app


def test_audit_training_cli_writes_artifact(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    for day in range(9, 14):
        rows.append(
            {
                "user_id": 1,
                "video_id": day,
                "date": f"202204{day:02d}",
                "is_like": int(day == 10),
                "is_follow": 0,
                "is_comment": 0,
                "is_forward": 0,
                "is_hate": 0,
                "is_profile_enter": 0,
            }
        )
    pd.DataFrame(rows).to_csv(data / "log_standard_4_08_to_4_21_pure.csv", index=False)
    out = tmp_path / "training.json"

    result = CliRunner().invoke(
        app,
        ["audit-training", "--data-dir", str(data), "--output", str(out), "--validation-days", "2"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["status"] == "M0_TRAINING_CONTRACT_AUDIT_ONLY"
    assert payload["presumptive_temporal_split"]["validation_dates"] == ["20220412", "20220413"]
