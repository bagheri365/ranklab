from __future__ import annotations

import pandas as pd
import pytest

from ranklab.evaluation.primary_contrasts import bootstrap_cross_cell


def _wide(users, pop, bpr, lgcn):
    return pd.DataFrame(
        {
            "popularity": pop,
            "bpr": bpr,
            "lightgcn": lgcn,
        },
        index=pd.Index(users, name="user_id"),
    )


def test_native_and_matched_are_equal_when_user_sets_match():
    left = _wide(
        [1, 2, 3],
        [0.8, 0.8, 0.8],
        [0.6, 0.6, 0.6],
        [0.5, 0.5, 0.5],
    )
    right = _wide(
        [1, 2, 3],
        [0.9, 0.9, 0.9],
        [0.6, 0.6, 0.6],
        [0.5, 0.5, 0.5],
    )
    result = bootstrap_cross_cell(
        left,
        right,
        direction="right_minus_left",
        replicates=300,
        rng_seed=1,
        chunk_size=16,
    )
    value = result["popularity_minus_bpr"]
    assert value["native"]["point"] == pytest.approx(0.1)
    assert value["matched"]["point"] == pytest.approx(0.1)
    assert value["native"]["ci95"]["lower"] == pytest.approx(0.1)
    assert value["matched"]["ci95"]["upper"] == pytest.approx(0.1)


def test_native_and_matched_can_differ_with_cell_specific_eligibility():
    left = _wide(
        [1, 2],
        [0.9, 0.7],
        [0.5, 0.5],
        [0.4, 0.4],
    )
    right = _wide(
        [2, 3],
        [0.8, 1.0],
        [0.5, 0.5],
        [0.4, 0.4],
    )
    import warnings
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        result = bootstrap_cross_cell(
            left,
            right,
            direction="right_minus_left",
            replicates=300,
            rng_seed=2,
            chunk_size=16,
        )
    assert recorded == []
    value = result["popularity_minus_bpr"]
    # Native: right margin mean .4 minus left margin mean .3 = .1.
    assert value["native"]["point"] == pytest.approx(0.1)
    # Matched user 2: .3 - .2 = .1.
    assert value["matched"]["point"] == pytest.approx(0.1)
    assert value["native"]["union_users"] == 3
    assert value["matched"]["users"] == 1


def test_direction_matches_frozen_g_and_t_sign():
    left = _wide([1, 2], [0.8, 0.8], [0.6, 0.6], [0.5, 0.5])
    right = _wide([1, 2], [0.7, 0.7], [0.6, 0.6], [0.5, 0.5])
    result = bootstrap_cross_cell(
        left,
        right,
        direction="right_minus_left",
        replicates=100,
        rng_seed=3,
    )
    assert result["popularity_minus_bpr"]["native"]["point"] == pytest.approx(-0.1)
