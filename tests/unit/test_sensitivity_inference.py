from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from ranklab.evaluation import sensitivity_inference


def test_verify_raw_rejects_drift(tmp_path, monkeypatch):
    path = tmp_path / "raw.csv.gz"
    path.write_bytes(b"abc")
    monkeypatch.setitem(
        sensitivity_inference.EXPECTED_RAW_SHA256,
        "tab1",
        "0" * 64,
    )
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        sensitivity_inference.verify_raw(path, "tab1")


def test_load_raw_requires_matching_sensitivity_label(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {
            "sensitivity": ["wrong"],
            "regime": ["standard"],
            "target": ["is_click"],
            "model": ["popularity"],
            "seed": [None],
            "user_id": [1],
            "ndcg_at_10": [1.0],
            "included_in_macro": [1],
        }
    )
    path = tmp_path / "raw.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(
        sensitivity_inference.EXPECTED_RAW_SHA256,
        "tab1",
        digest,
    )
    with pytest.raises(ValueError, match="label mismatch"):
        sensitivity_inference.load_raw(path, "tab1")
