from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from ranklab.cli import app


def test_audit_targets_cli_writes_artifact(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "is_click": [1, 0],
            "long_view": [1, 0],
            "play_time_ms": [5_000, 1_000],
            "duration_ms": [5_000, 5_000],
            "tab": [1, 1],
        }
    )
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "log_random_4_22_to_5_08_pure.csv",
    ):
        frame.to_csv(tmp_path / filename, index=False)

    output = tmp_path / "target_audit.json"
    result = CliRunner().invoke(
        app,
        ["audit-targets", "--data-dir", str(tmp_path), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["status"] == "M0_TARGET_SEMANTICS_AUDIT_ONLY"
