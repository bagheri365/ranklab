import pandas as pd

from ranklab.evaluation.scoring_eligibility import (
    TrainingIndexUniverse,
    restrict_to_training_indexed_entities,
)


def test_scoring_eligibility_requires_seen_user_and_seen_item() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "video_id": [10, 30, 20, 10],
        }
    )
    universe = TrainingIndexUniverse(
        users=frozenset({1, 2}),
        items=frozenset({10, 20}),
    )

    restricted = restrict_to_training_indexed_entities(frame, universe)

    assert list(zip(restricted.user_id, restricted.video_id)) == [
        (1, 10),
        (2, 20),
    ]
