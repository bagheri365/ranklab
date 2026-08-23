import math

import pytest

from ranklab.evaluation.metrics import dcg_at_k, identity_gain, log2_discount, ndcg_at_k


def test_dcg_perfect_binary_fixture() -> None:
    value = dcg_at_k([1, 1, 0], 3, gain_fn=identity_gain, discount_fn=log2_discount)
    expected = 1.0 + 1.0 / math.log2(3.0)
    assert value == pytest.approx(expected)


def test_ndcg_perfect_fixture_is_one() -> None:
    assert ndcg_at_k(
        [1, 1, 0],
        3,
        gain_fn=identity_gain,
        discount_fn=log2_discount,
        zero_relevance_value=None,
    ) == pytest.approx(1.0)


def test_zero_relevance_requires_explicit_policy() -> None:
    with pytest.raises(ValueError, match="zero-relevance handling is not frozen"):
        ndcg_at_k(
            [0, 0, 0],
            3,
            gain_fn=identity_gain,
            discount_fn=log2_discount,
            zero_relevance_value=None,
        )
