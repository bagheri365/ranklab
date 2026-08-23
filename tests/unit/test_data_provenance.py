from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ranklab.data.provenance import (
    KUAIRAND_PURE_EXPECTED_FILES,
    inventory_data_dir,
)
from ranklab.data.validation import LOG_REQUIRED_COLUMNS, audit_log_header


def _write_csv(path: Path, columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow([0] * len(columns))


def test_inventory_requires_all_expected_files(tmp_path: Path) -> None:
    for name in KUAIRAND_PURE_EXPECTED_FILES[:-1]:
        (tmp_path / name).write_text("x\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=KUAIRAND_PURE_EXPECTED_FILES[-1]):
        inventory_data_dir(tmp_path)


def test_inventory_is_deterministic_and_hashes_files(tmp_path: Path) -> None:
    for name in KUAIRAND_PURE_EXPECTED_FILES:
        (tmp_path / name).write_text(f"{name}\n", encoding="utf-8")

    rows = inventory_data_dir(tmp_path)
    assert [row.name for row in rows] == list(KUAIRAND_PURE_EXPECTED_FILES)
    assert all(len(row.sha256) == 64 for row in rows)
    assert all(row.bytes > 0 for row in rows)


def test_log_header_audit_accepts_documented_minimum(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    _write_csv(path, sorted(LOG_REQUIRED_COLUMNS))

    result = audit_log_header(path)
    assert result.ok
    assert result.missing_required == ()


def test_log_header_audit_reports_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    columns = sorted(LOG_REQUIRED_COLUMNS - {"long_view"})
    _write_csv(path, columns)

    result = audit_log_header(path)
    assert not result.ok
    assert result.missing_required == ("long_view",)
