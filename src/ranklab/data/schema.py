from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExposureRow(BaseModel):
    """Canonical exposure row after M0 maps verified KuaiRand fields."""

    model_config = ConfigDict(extra="forbid")

    user_id: int
    item_id: int
    timestamp: int | float | str
    logging_regime: str
    is_click: int | None = None
    long_view: int | None = None
