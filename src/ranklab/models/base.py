from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Ranker(ABC):
    """Minimal scoring interface shared by all RankLab primary models."""

    @abstractmethod
    def fit(self, interactions: object) -> None:
        """Fit using the frozen standard-history training contract."""

    @abstractmethod
    def score(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        """Score exactly the supplied logged candidates."""
