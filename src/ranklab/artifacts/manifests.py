from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a stable, human-readable experiment manifest."""
    body = dict(payload)
    body.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
