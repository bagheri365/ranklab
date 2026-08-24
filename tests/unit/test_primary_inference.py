from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ranklab.evaluation.primary_inference import (
    aggregate_seed_mean_per_user,
    paired_user_bootstrap_margin,
)


def test_seed_mean_aggregation_keeps_one_row_per_model_user():
    rows = []
    for seed in range(5):
        rows.append(
            {
                "regime": "standard",
                "target": "is_click",
                "model": "bpr",
                "seed": seed,
                "user_id": 1,
                "ndcg_at_10": 0.1 + seed,
                "included_in_macro": 1,
            }
        )
    rows.append(
        {
            "regime": "standard",
            "target": "is_click",
            "model": "popularity",
            "seed": np.nan,
            "user_id": 1,
            "ndcg_at_10": 0.5,
            "included_in_macro": 1,
        }
    )
    out = aggregate_seed_mean_per_user(pd.DataFrame(rows))
    bpr = out[out.model == "bpr"].iloc[0]
    pop = out[out.model == "popularity"].iloc[0]
    assert bpr.ndcg_at_10 == pytest.approx(2.1)
    assert pop.ndcg_at_10 == pytest.approx(0.5)


def test_seed_mean_rejects_missing_seed_rows():
    rows = [
        {
            "regime": "standard",
            "target": "is_click",
            "model": "bpr",
            "seed": seed,
            "user_id": 1,
            "ndcg_at_10": 0.1,
            "included_in_macro": 1,
        }
        for seed in range(4)
    ]
    with pytest.raises(ValueError, match="five stochastic seed rows"):
        aggregate_seed_mean_per_user(pd.DataFrame(rows))


def test_paired_bootstrap_margin_uses_same_users_and_expected_sign():
    a = pd.DataFrame({"user_id": [1, 2, 3], "ndcg_at_10": [0.9, 0.8, 0.7]})
    b = pd.DataFrame({"user_id": [1, 2, 3], "ndcg_at_10": [0.5, 0.4, 0.3]})
    result = paired_user_bootstrap_margin(
        a, b, replicates=500, rng_seed=123
    )
    assert result["users"] == 3
    assert result["margin"] == pytest.approx(0.4)
    assert result["ci95"]["lower"] > 0
    assert result["simultaneous_ci"]["lower"] > 0
