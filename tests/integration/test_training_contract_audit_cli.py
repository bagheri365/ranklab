from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from ranklab.cli import app


def test_audit_training_contract_cli_writes_artifact(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    for day in range(9, 15):
        rows.extend([
            {"user_id": 1, "video_id": day, "date": f"202204{day:02d}", "is_click": 1},
            {"user_id": 1, "video_id": 100 + day, "date": f"202204{day:02d}", "is_click": 0},
        ])
    pd.DataFrame(rows).to_csv(data / "log_standard_4_08_to_4_21_pure.csv", index=False)
    out = tmp_path / "contract.json"

    result = CliRunner().invoke(
        app,
        ["audit-training-contract", "--data-dir", str(data), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["status"] == "M0_TRAINING_CONTRACT_FREEZE_CANDIDATE_ONLY"
    assert payload["proposed_contract"]["primary_k"] == 10
