import pandas as pd

from ranklab.evaluation.entity_coverage_audit import summarize_entity_coverage


def test_entity_coverage_counts_unseen_users_items_and_pairs() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "video_id": [10, 20, 10, 30],
            "tab": [1, 1, 1, 1],
            "is_click": [1, 0, 0, 0],
            "long_view": [0, 0, 0, 0],
        }
    )
    summary = summarize_entity_coverage(
        frame,
        training_users={1, 2},
        training_items={10, 20},
        regime="standard",
        support_name="primary",
    )

    assert summary.users == 3
    assert summary.unseen_users == 1
    assert summary.items == 3
    assert summary.unseen_items == 1
    assert summary.pairs_seen_user_and_item == 3
    assert summary.pairs_any_unseen_entity == 1
    assert summary.fully_scorable_pair_fraction == 0.75
