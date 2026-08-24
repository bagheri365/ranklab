import math

from ranklab.evaluation.semantics import (
    MIN_CANDIDATES,
    PRIMARY_K,
    evaluate_user_ndcg,
    macro_average_ndcg,
    target_any_positive,
)


def test_frozen_constants():
    assert PRIMARY_K == 10
    assert MIN_CANDIDATES == 2


def test_any_positive_pair_relevance():
    assert target_any_positive([0, 0, 1, 0]) == 1
    assert target_any_positive([0, 0]) == 0


def test_user_with_one_candidate_is_excluded():
    result = evaluate_user_ndcg(
        item_ids=[10],
        scores=[1.0],
        relevance=[1],
    )
    assert result.ndcg is None
    assert result.included_in_macro is False
    assert result.exclusion_reason == "fewer_than_2_candidates"


def test_zero_relevance_user_is_excluded_not_scored_zero():
    result = evaluate_user_ndcg(
        item_ids=[10, 20],
        scores=[2.0, 1.0],
        relevance=[0, 0],
    )
    assert result.ndcg is None
    assert result.included_in_macro is False
    assert result.exclusion_reason == "zero_relevance"


def test_less_than_k_uses_all_available_candidates_without_padding():
    # relevant item ranked second among only two candidates
    result = evaluate_user_ndcg(
        item_ids=[10, 20],
        scores=[2.0, 1.0],
        relevance=[0, 1],
        k=10,
    )
    expected = 1.0 / math.log2(3.0)
    assert result.included_in_macro is True
    assert result.ndcg == expected


def test_exact_score_ties_use_video_id_ascending():
    result = evaluate_user_ndcg(
        item_ids=[20, 10],
        scores=[1.0, 1.0],
        relevance=[0, 1],
    )
    assert result.ndcg == 1.0


def test_macro_is_equal_user_weight():
    a = evaluate_user_ndcg(
        item_ids=[1, 2],
        scores=[2.0, 1.0],
        relevance=[1, 0],
    )
    b = evaluate_user_ndcg(
        item_ids=[1, 2],
        scores=[2.0, 1.0],
        relevance=[0, 1],
    )
    excluded = evaluate_user_ndcg(
        item_ids=[1, 2],
        scores=[2.0, 1.0],
        relevance=[0, 0],
    )
    assert macro_average_ndcg([a, b, excluded]) == (a.ndcg + b.ndcg) / 2
