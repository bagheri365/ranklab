from pathlib import Path


def test_results_discussion_respects_frozen_reporting_contract():
    text = Path(
        "research/reporting/M2.4_RESULTS_DISCUSSION.md"
    ).read_text(encoding="utf-8")

    assert "Winner identity is stable, but comparative margins are not." in text
    assert "Popularity is the decisive winner" in text
    assert "BPR and LightGCN are not decisively separated" in text
    assert "should not be interpreted as a causal effect" in text
    assert "`tab=1` analysis is descriptive only" in text
    assert "introduces no new scoring, tuning, bootstrap inference" in text


def test_results_discussion_contains_primary_scores():
    text = Path(
        "research/reporting/M2.4_RESULTS_DISCUSSION.md"
    ).read_text(encoding="utf-8")

    for value in (
        "0.716524",
        "0.648404",
        "0.432459",
        "0.370164",
        "0.0395",
        "0.0889",
        "0.0467",
        "0.1051",
    ):
        assert value in text
