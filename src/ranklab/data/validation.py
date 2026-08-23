from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

LOG_REQUIRED_COLUMNS = {
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "is_click",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "is_rand",
    "tab",
}


@dataclass(frozen=True)
class CsvHeaderAudit:
    path: Path
    columns: tuple[str, ...]
    missing_required: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_required


def read_csv_header(path: str | Path) -> tuple[str, ...]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {csv_path}") from exc


def audit_log_header(path: str | Path) -> CsvHeaderAudit:
    columns = read_csv_header(path)
    missing = tuple(sorted(LOG_REQUIRED_COLUMNS.difference(columns)))
    return CsvHeaderAudit(path=Path(path), columns=columns, missing_required=missing)
