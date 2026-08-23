from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ProtocolNotFrozenError(RuntimeError):
    pass


def load_protocol(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("protocol must be a YAML mapping")
    return payload


def require_frozen_protocol(path: str | Path) -> dict[str, Any]:
    payload = load_protocol(path)
    if payload.get("status") != "FROZEN":
        raise ProtocolNotFrozenError(f"protocol is not frozen: {path}")
    return payload
