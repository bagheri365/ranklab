"""Deterministic candidate ranking utilities.

RankLab's frozen ordering rule is:

1. model score descending;
2. video_id ascending on an exact score tie.

The secondary key prevents candidate input order from changing metrics,
especially for the non-personalized popularity baseline where ties are common.
"""

from __future__ import annotations

from collections.abc import Sequence


def deterministic_rank_item_ids(
    item_ids: Sequence[int],
    scores: Sequence[float],
) -> list[int]:
    """Return item IDs ordered by frozen RankLab score/tie semantics."""
    if len(item_ids) != len(scores):
        raise ValueError("item_ids and scores must have the same length")
    if len(set(int(item_id) for item_id in item_ids)) != len(item_ids):
        raise ValueError("candidate item_ids must be unique")

    pairs = [
        (int(item_id), float(score))
        for item_id, score in zip(item_ids, scores)
    ]
    return [
        item_id
        for item_id, _ in sorted(
            pairs,
            key=lambda pair: (-pair[1], pair[0]),
        )
    ]


def deterministic_rank_indices(
    item_ids: Sequence[int],
    scores: Sequence[float],
) -> list[int]:
    """Return original candidate indices in frozen RankLab rank order."""
    if len(item_ids) != len(scores):
        raise ValueError("item_ids and scores must have the same length")
    if len(set(int(item_id) for item_id in item_ids)) != len(item_ids):
        raise ValueError("candidate item_ids must be unique")

    return sorted(
        range(len(item_ids)),
        key=lambda idx: (-float(scores[idx]), int(item_ids[idx])),
    )
