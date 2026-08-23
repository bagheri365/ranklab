from __future__ import annotations

from collections.abc import Sequence

from ranklab.models.base import Ranker


class BPRRanker(Ranker):
    """Model B implementation placeholder."""

    def fit(self, interactions: object) -> None:
        raise NotImplementedError("Implement after M0 freezes training supervision")

    def score(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        raise NotImplementedError("BPR is not implemented yet")
