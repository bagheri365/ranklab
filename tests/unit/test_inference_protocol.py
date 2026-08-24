import math

import pytest

from ranklab.evaluation.inference_protocol import (
    BONFERRONI_INDIVIDUAL_CONFIDENCE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    Interval,
    behavioral_target_contrast,
    decisive_winner,
    descriptive_order,
    logging_regime_contrast,
    pairwise_margin,
    seed_mean,
    seed_order_agreement,
    selection_change,
)


def test_frozen_inference_constants():
    assert BOOTSTRAP_REPLICATES == 20_000
    assert BOOTSTRAP_SEED == 365
    assert CONFIDENCE_LEVEL == 0.95
    assert math.isclose(
        BONFERRONI_INDIVIDUAL_CONFIDENCE,
        1.0 - 0.05 / 12,
    )


def test_seed_mean_requires_exact_frozen_seed_set():
    assert seed_mean({0: 1, 1: 2, 2: 3, 3: 4, 4: 5}) == 3.0
    with pytest.raises(ValueError):
        seed_mean({0: 1, 1: 2})


def test_contrast_sign_conventions():
    assert pairwise_margin(0.7, 0.6) == pytest.approx(0.1)
    assert logging_regime_contrast(0.2, 0.1) == pytest.approx(0.1)
    assert behavioral_target_contrast(-0.1, 0.2) == pytest.approx(-0.3)


def test_descriptive_order_is_deterministic_on_exact_tie():
    assert descriptive_order({"bpr": 0.5, "lightgcn": 0.5, "popularity": 0.4}) == (
        "bpr",
        "lightgcn",
        "popularity",
    )


def test_decisive_winner_requires_all_pairwise_intervals_above_zero():
    scores = {"a": 0.7, "b": 0.6, "c": 0.5}
    assert decisive_winner(
        scores=scores,
        pair_intervals={
            ("a", "b"): Interval(0.01, 0.10),
            ("a", "c"): Interval(0.05, 0.20),
        },
    ) == "a"
    assert decisive_winner(
        scores=scores,
        pair_intervals={
            ("a", "b"): Interval(-0.01, 0.10),
            ("a", "c"): Interval(0.05, 0.20),
        },
    ) is None


def test_selection_change_does_not_force_claim_when_indeterminate():
    assert selection_change("bpr", "bpr") == "stable"
    assert selection_change("bpr", "lightgcn") == "decisive_change"
    assert selection_change("bpr", None) == "indeterminate"


def test_seed_order_agreement_is_descriptive():
    same = [("bpr", "lightgcn", "popularity")] * 5
    mixed = same[:-1] + [("lightgcn", "bpr", "popularity")]
    assert seed_order_agreement(same) is True
    assert seed_order_agreement(mixed) is False
