from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ranklab.evaluation import primary_sensitivity


class DummyScorer:
    def score_items(self, user_id, item_ids):
        return np.asarray(item_ids, dtype=float)


def test_sensitivity_runner_writes_raw_artifact(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "video_id": [10, 11, 10, 12],
            "tab": [1, 1, 1, 1],
            "is_click": [1, 0, 1, 0],
            "long_view": [1, 0, 0, 0],
        }
    )

    monkeypatch.setattr(
        primary_sensitivity.primary,
        "verify_protocol",
        lambda root: "protocol",
    )
    monkeypatch.setattr(
        primary_sensitivity.primary,
        "load_frozen_scorers",
        lambda **kwargs: (
            [("popularity", "deterministic", DummyScorer())],
            {"bpr": {}, "lightgcn": {}},
        ),
    )
    monkeypatch.setattr(
        primary_sensitivity,
        "build_sensitivity_candidates",
        lambda **kwargs: {"standard": frame.copy(), "randomized": frame.copy()},
    )

    out = tmp_path / "out"
    manifest = primary_sensitivity.run_one_sensitivity(
        name="tab1",
        training_path="train.csv",
        data_dir=tmp_path,
        primary_dir=tmp_path,
        output_dir=out,
    )
    assert manifest["status"] == "M1_SUPPORT_SENSITIVITY_RAW_EVALUATION"
    assert manifest["sensitivity"] == "tab1"
    assert manifest["candidate_population"]["standard"]["users"] == 2
    assert (out / "per_user_ndcg.csv.gz").exists()
    assert (out / "manifest.json").exists()
