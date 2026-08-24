import pandas as pd

from ranklab.evaluation.ranking import deterministic_rank_item_ids
from ranklab.models.popularity import PopularityRanker


def test_popularity_ties_use_frozen_item_id_secondary_order() -> None:
    pairs = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "video_id": [10, 20, 30],
            "is_positive": [True, True, False],
        }
    )
    model = PopularityRanker()
    model.fit(pairs)

    candidates = [99, 20, 30, 10]
    scores = model.score(123, candidates)

    # 10 and 20 tie at popularity 1 -> lower video_id first.
    # 30 and 99 tie at zero -> lower video_id first.
    assert deterministic_rank_item_ids(candidates, scores) == [10, 20, 30, 99]
