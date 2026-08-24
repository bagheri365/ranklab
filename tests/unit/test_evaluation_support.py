import pandas as pd

from ranklab.evaluation.support import (
    derive_evaluation_support,
    restrict_primary_support,
    restrict_shared_tab_sensitivity,
    restrict_tab1_sensitivity,
)


def test_primary_support_aligns_entities_but_preserves_regime_specific_pairs() -> None:
    standard = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 9],
            "video_id": [10, 20, 20, 99],
            "tab": [1, 2, 2, 4],
        }
    )
    randomized = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "video_id": [20, 10, 30],
            "tab": [1, 2, 3],
        }
    )

    support = derive_evaluation_support(standard, randomized)
    assert support.shared_users == frozenset({1, 2})
    assert support.shared_videos == frozenset({10, 20})
    assert support.shared_tabs == frozenset({1, 2})

    std_primary = restrict_primary_support(standard, support)
    rnd_primary = restrict_primary_support(randomized, support)

    # (1, 10) exists only under standard exposure and (2, 10) only under random.
    # They are retained because M0.15 aligns entities, not exposed pairs.
    assert set(zip(std_primary.user_id, std_primary.video_id)) == {
        (1, 10),
        (1, 20),
        (2, 20),
    }
    assert set(zip(rnd_primary.user_id, rnd_primary.video_id)) == {
        (1, 20),
        (2, 10),
    }


def test_tab_restrictions_are_sensitivities_after_primary_support() -> None:
    standard = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "video_id": [10, 20, 20],
            "tab": [1, 2, 7],
        }
    )
    randomized = pd.DataFrame(
        {
            "user_id": [1, 2, 2],
            "video_id": [20, 10, 20],
            "tab": [1, 2, 8],
        }
    )
    support = derive_evaluation_support(standard, randomized)

    shared_tabs = restrict_shared_tab_sensitivity(standard, support)
    tab1 = restrict_tab1_sensitivity(standard, support)

    assert set(shared_tabs["tab"]) == {1, 2}
    assert set(tab1["tab"]) == {1}
