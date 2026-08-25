"""Frozen M2 reporting contract.

M2 may transform and present frozen M1 results, but it may not alter model
selection, statistical decisions, estimands, or inference.
"""

from __future__ import annotations

FINAL_M1_SHA256 = {
    "manifest": "ba9334455134d86a14d2310c809c34152c0ce60f135bb62a765cdab9a1737c2f",
    "summary": "8735976eb8d5562de9728ddc98179f86ec0e5740f4ee09d8b8f04c0b1995ed33",
    "markdown": "20ccd9083e71945581f7f2d9970404aa9b9287a30ba62a205dedcc083cf799bf",
}

HEADLINE = "Winner identity is stable, but comparative margins are not."

ALLOWED_PRIMARY_CLAIMS = (
    "Popularity is the decisive winner in all four primary regime-target cells.",
    "BPR and LightGCN are not decisively separated in any primary cell.",
    "Popularity's measured advantage is substantially larger under randomized exposure than under standard logging.",
    "Target definition changes Popularity's measured margin, especially under randomized exposure.",
)

ALLOWED_SENSITIVITY_CLAIMS = (
    "The pre-specified shared-tabs sensitivity preserves the qualitative primary pattern.",
    "The tab1 sensitivity also preserves the qualitative pattern but remains descriptive only.",
)

REQUIRED_LIMITATIONS = (
    "Offline logging-policy contrasts are not causal treatment-effect estimates.",
    "Popularity is deterministic while BPR and LightGCN use five frozen seeds.",
    "BPR and LightGCN are not decisively separated under the frozen primary simultaneous intervals.",
    "tab1 must not be assigned an undocumented semantic UI meaning.",
    "Native and matched-user target contrasts differ, so target-specific eligible-user composition matters.",
    "LightGCN tuning fixed regularization while BPR searched regularization.",
)

FORBIDDEN_REPORTING_MOVES = (
    "Do not rerun scoring, tuning, bootstrapping, or model selection in M2.",
    "Do not invent a practical-significance threshold.",
    "Do not promote sensitivity analyses into the primary multiplicity family.",
    "Do not describe logging-policy contrasts as causal effects.",
    "Do not claim BPR or LightGCN superiority over the other.",
    "Do not attach semantic meaning to tab ID 1.",
)

PLANNED_PUBLIC_OUTPUTS = (
    "primary_scores_table",
    "primary_margin_table",
    "support_sensitivity_table",
    "primary_scores_figure",
    "regime_margin_figure",
    "readme_results_section",
    "reproducibility_section",
    "paper_style_results_discussion",
)
