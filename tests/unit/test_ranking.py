import pytest

from ranklab.evaluation.ranking import (
    deterministic_rank_indices,
    deterministic_rank_item_ids,
)


def test_ranking_is_score_desc_then_item_id_asc() -> None:
    item_ids = [40, 10, 30, 20]
    scores = [2.0, 3.0, 2.0, 2.0]

    assert deterministic_rank_item_ids(item_ids, scores) == [10, 20, 30, 40]
    assert deterministic_rank_indices(item_ids, scores) == [1, 3, 2, 0]


def test_ties_do_not_depend_on_candidate_input_order() -> None:
    a_items = [99, 7, 42, 3]
    b_items = [3, 42, 99, 7]
    scores = [0.0, 0.0, 0.0, 0.0]

    assert deterministic_rank_item_ids(a_items, scores) == [3, 7, 42, 99]
    assert deterministic_rank_item_ids(b_items, scores) == [3, 7, 42, 99]


def test_ranking_rejects_invalid_candidate_vectors() -> None:
    with pytest.raises(ValueError, match="same length"):
        deterministic_rank_item_ids([1, 2], [0.5])

    with pytest.raises(ValueError, match="unique"):
        deterministic_rank_item_ids([1, 1], [0.5, 0.2])
