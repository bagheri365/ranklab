from __future__ import annotations

from collections.abc import Sequence

from ranklab.models.base import Ranker


class PopularityRanker(Ranker):
    """Placeholder for Model A; scoring semantics are frozen in M0."""

    def fit(self, interactions: object) -> None:
        raise NotImplementedError("M0 must freeze the popularity/engagement score definition")

    def score(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        raise NotImplementedError("Model A is not implemented yet")
