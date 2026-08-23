import pandas as pd

from ranklab.data.training_pairs import collapse_click_pairs, summarize_training_pairs


def test_pair_collapse_promotes_any_click_and_never_overlaps_negative() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "video_id": [10, 10, 11, 10, 12],
            "date": [20220409] * 5,
            "is_click": [0, 1, 0, 0, 1],
        }
    )
    pairs = collapse_click_pairs(frame)
    labels = {(row.user_id, row.video_id): row.is_positive for row in pairs.itertuples()}
    assert labels == {(1, 10): True, (1, 11): False, (2, 10): False, (2, 12): True}

    summary = summarize_training_pairs(pairs)
    assert summary.positive_pairs == 2
    assert summary.negative_pairs == 2
    assert summary.users_with_both == 2
