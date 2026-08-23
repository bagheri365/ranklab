from __future__ import annotations

import csv
from pathlib import Path

from typer.testing import CliRunner

from ranklab.cli import app
from ranklab.data.provenance import KUAIRAND_PURE_EXPECTED_FILES
from ranklab.data.validation import LOG_REQUIRED_COLUMNS


def test_audit_data_writes_json_artifact(tmp_path: Path) -> None:
    for name in KUAIRAND_PURE_EXPECTED_FILES:
        path = tmp_path / name
        if name.startswith("log_"):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(sorted(LOG_REQUIRED_COLUMNS))
                writer.writerow([0] * len(LOG_REQUIRED_COLUMNS))
        else:
            path.write_text("id\n0\n", encoding="utf-8")

    output = tmp_path / "audit.json"
    result = CliRunner().invoke(
        app,
        ["audit-data", "--data-dir", str(tmp_path), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert '"status": "M0_PROVENANCE_AUDIT_ONLY"' in output.read_text()
