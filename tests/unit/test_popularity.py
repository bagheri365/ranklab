import pandas as pd
import pytest

from ranklab.models.popularity import PopularityRanker


def test_popularity_counts_distinct_positive_users_and_scores_unknown_zero() -> None:
    pairs = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "video_id": [10, 10, 11, 11, 12],
            "is_positive": [True, True, True, False, False],
        }
    )
    model = PopularityRanker()
    model.fit(pairs)
    assert list(model.score(999, [10, 11, 12, 13])) == [2.0, 1.0, 0.0, 0.0]


def test_popularity_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        PopularityRanker().score(1, [10])
