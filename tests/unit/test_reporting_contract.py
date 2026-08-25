from ranklab.reporting.contract import (
    FINAL_M1_SHA256,
    HEADLINE,
    ALLOWED_PRIMARY_CLAIMS,
    REQUIRED_LIMITATIONS,
    FORBIDDEN_REPORTING_MOVES,
    PLANNED_PUBLIC_OUTPUTS,
)


def test_reporting_contract_is_frozen_and_complete():
    assert FINAL_M1_SHA256 == {
        "manifest": "ba9334455134d86a14d2310c809c34152c0ce60f135bb62a765cdab9a1737c2f",
        "summary": "8735976eb8d5562de9728ddc98179f86ec0e5740f4ee09d8b8f04c0b1995ed33",
        "markdown": "20ccd9083e71945581f7f2d9970404aa9b9287a30ba62a205dedcc083cf799bf",
    }
    assert HEADLINE == "Winner identity is stable, but comparative margins are not."
    assert len(ALLOWED_PRIMARY_CLAIMS) == 4
    assert len(REQUIRED_LIMITATIONS) == 6
    assert len(FORBIDDEN_REPORTING_MOVES) == 6
    assert "primary_scores_figure" in PLANNED_PUBLIC_OUTPUTS
    assert "paper_style_results_discussion" in PLANNED_PUBLIC_OUTPUTS
