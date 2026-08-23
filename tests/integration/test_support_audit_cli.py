from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from ranklab.cli import app

runner = CliRunner()


def test_audit_support_cli_writes_output(tmp_path: Path) -> None:
    columns = {
        "user_id": [1],
        "video_id": [10],
        "date": [20220422],
        "is_click": [1],
        "long_view": [1],
        "play_time_ms": [1000],
        "duration_ms": [2000],
        "is_rand": [0],
        "tab": [1],
    }
    pd.DataFrame(columns).to_csv(tmp_path / "log_standard_4_22_to_5_08_pure.csv", index=False)
    columns["is_rand"] = [1]
    pd.DataFrame(columns).to_csv(tmp_path / "log_random_4_22_to_5_08_pure.csv", index=False)

    output = tmp_path / "support.json"
    result = runner.invoke(
        app,
        ["audit-support", "--data-dir", str(tmp_path), "--output", str(output), "--chunksize", "1"],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["status"] == "M0_SUPPORT_SCENARIO_AUDIT_ONLY"
