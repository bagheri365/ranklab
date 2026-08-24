from __future__ import annotations

import pandas as pd

from ranklab.evaluation import primary_sensitivity


def test_sensitivity_registry_is_frozen():
    assert primary_sensitivity.SENSITIVITIES["shared_tabs"]["tabs"] == (1, 2, 11, 14)
    assert primary_sensitivity.SENSITIVITIES["tab1"]["tabs"] == (1,)
    assert primary_sensitivity.SENSITIVITIES["tab1"]["role"] == "descriptive_support_sensitivity"


def test_tab_filter_is_applied_before_candidate_collapse(monkeypatch):
    standard = pd.DataFrame(
        {
            "user_id": [1, 1],
            "video_id": [10, 10],
            "tab": [1, 2],
            "is_click": [0, 1],
            "long_view": [0, 1],
        }
    )
    randomized = standard.copy()

    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: standard.copy())
    monkeypatch.setattr(
        primary_sensitivity.primary,
        "derive_evaluation_support",
        lambda a, b: object(),
    )
    monkeypatch.setattr(
        primary_sensitivity.primary,
        "derive_training_index_universe",
        lambda p: object(),
    )
    monkeypatch.setattr(
        primary_sensitivity.primary,
        "restrict_primary_support",
        lambda frame, support: frame.copy(),
    )
    monkeypatch.setattr(
        primary_sensitivity.primary,
        "restrict_to_training_indexed_entities",
        lambda frame, universe: frame.copy(),
    )

    def collapse(frame):
        return (
            frame.groupby(["user_id", "video_id"], as_index=False)
            .agg(
                tab=("tab", "first"),
                is_click=("is_click", "max"),
                long_view=("long_view", "max"),
            )
        )

    monkeypatch.setattr(primary_sensitivity.primary, "collapse_logged_candidates", collapse)

    out = primary_sensitivity.build_sensitivity_candidates(
        training_path="train.csv",
        standard_path="standard.csv",
        randomized_path="random.csv",
        tabs=(1,),
    )
    assert len(out["standard"]) == 1
    assert int(out["standard"].iloc[0]["is_click"]) == 0
