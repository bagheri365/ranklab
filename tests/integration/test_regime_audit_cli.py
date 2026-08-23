from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from ranklab.cli import app


def test_audit_regimes_writes_json(tmp_path: Path) -> None:
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
    names = [
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "log_random_4_22_to_5_08_pure.csv",
    ]
    for name in names:
        frame = pd.DataFrame(columns)
        if "random" in name:
            frame["is_rand"] = 1
        frame.to_csv(tmp_path / name, index=False)

    output = tmp_path / "audit.json"
    result = CliRunner().invoke(app, ["audit-regimes", "--data-dir", str(tmp_path), "--output", str(output), "--chunksize", "1"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(output.read_text())
    assert payload["status"] == "M0_DESCRIPTIVE_AUDIT_ONLY"
