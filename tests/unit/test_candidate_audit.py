import pandas as pd

from ranklab.evaluation.candidate_audit import (
    collapse_logged_candidates,
    summarize_candidate_structure,
)


def test_candidate_collapse_uses_target_specific_any_positive() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "video_id": [10, 10, 20, 30],
            "tab": [1, 1, 2, 1],
            "is_click": [0, 1, 0, 0],
            "long_view": [1, 0, 0, 0],
        }
    )
    collapsed = collapse_logged_candidates(frame)

    pair = collapsed.loc[
        (collapsed["user_id"] == 1) & (collapsed["video_id"] == 10)
    ].iloc[0]
    assert pair["is_click"] == 1
    assert pair["long_view"] == 1
    assert len(collapsed) == 3


def test_summary_reports_zero_relevance_and_candidate_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3, 3, 3],
            "video_id": [10, 20, 30, 10, 20, 30],
            "tab": [1, 1, 1, 1, 1, 1],
            "is_click": [1, 0, 0, 0, 0, 0],
            "long_view": [0, 0, 0, 1, 0, 0],
        }
    )
    summary = summarize_candidate_structure(
        frame, regime="standard", support_name="primary"
    )

    assert summary.users == 3
    assert summary.users_ge2_candidates == 2
    assert summary.users_ge10_candidates == 0

    click = next(t for t in summary.targets if t.target == "is_click")
    long_view = next(t for t in summary.targets if t.target == "long_view")

    assert click.users_with_relevance == 1
    assert click.users_zero_relevance == 2
    assert click.users_with_relevance_and_ge2_candidates == 1
    assert long_view.users_with_relevance == 1
    assert long_view.users_zero_relevance == 2
