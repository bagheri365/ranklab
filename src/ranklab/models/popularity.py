from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ranklab.models.base import Ranker


class PopularityRanker(Ranker):
    """Non-personalized baseline ranked by frozen clicked-pair popularity.

    The score for an item is the number of unique positive user-item click pairs
    in the frozen standard-policy training slice. Unknown items score zero.
    """

    def __init__(self) -> None:
        self._scores: dict[int, float] = {}
        self._fitted = False

    def fit(self, interactions: object) -> None:
        if not isinstance(interactions, pd.DataFrame):
            raise TypeError("PopularityRanker expects a pandas DataFrame pair table")
        required = {"user_id", "video_id", "is_positive"}
        if not required.issubset(interactions.columns):
            raise ValueError(f"interactions must contain {sorted(required)}")

        positive = interactions.loc[interactions["is_positive"].astype(bool)]
        counts = positive.groupby("video_id", sort=True)["user_id"].nunique()
        self._scores = {int(item): float(count) for item, count in counts.items()}
        self._fitted = True

    def score(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        del user_id  # Deliberately non-personalized.
        if not self._fitted:
            raise RuntimeError("PopularityRanker must be fitted before scoring")
        return [self._scores.get(int(item_id), 0.0) for item_id in item_ids]
