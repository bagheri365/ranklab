from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Load a KuaiRand CSV without imposing M0 semantics."""
    return pd.read_csv(Path(path), nrows=nrows)
