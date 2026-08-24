"""Post-M0.18 candidate-structure audit.

This re-runs the M0.16 structural counts after both:
1. frozen M0.15 regime support;
2. frozen M0.18 training-index scoring eligibility.

No recommender checkpoint or model score is used.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ranklab.evaluation.candidate_audit import summarize_candidate_structure
from ranklab.evaluation.scoring_eligibility import (
    derive_training_index_universe,
    restrict_to_training_indexed_entities,
)
from ranklab.evaluation.support import (
    derive_evaluation_support,
    restrict_primary_support,
    restrict_shared_tab_sensitivity,
    restrict_tab1_sensitivity,
)


EVAL_COLUMNS = ("user_id", "video_id", "tab", "is_click", "long_view")


def build_post_eligibility_candidate_audit(
    *,
    training_path: str | Path,
    data_dir: str | Path,
) -> dict[str, Any]:
    root = Path(data_dir)
    standard = pd.read_csv(
        root / "log_standard_4_22_to_5_08_pure.csv",
        usecols=list(EVAL_COLUMNS),
    )
    randomized = pd.read_csv(
        root / "log_random_4_22_to_5_08_pure.csv",
        usecols=list(EVAL_COLUMNS),
    )

    support = derive_evaluation_support(standard, randomized)
    universe = derive_training_index_universe(training_path)

    variants = [
        ("primary_shared_users_and_videos_training_seen", restrict_primary_support),
        (
            "sensitivity_shared_users_videos_and_tabs_training_seen",
            restrict_shared_tab_sensitivity,
        ),
        (
            "sensitivity_shared_users_videos_tab_1_training_seen",
            restrict_tab1_sensitivity,
        ),
    ]

    summaries = []
    for support_name, support_fn in variants:
        for regime, frame in (("standard", standard), ("randomized", randomized)):
            restricted = support_fn(frame, support)
            eligible = restrict_to_training_indexed_entities(restricted, universe)
            summaries.append(
                summarize_candidate_structure(
                    eligible,
                    regime=regime,
                    support_name=support_name,
                )
            )

    return {
        "status": "M0_POST_ELIGIBILITY_CANDIDATE_AUDIT_ONLY",
        "guardrail": (
            "No recommender checkpoint, model score, model ranking, or ranking "
            "metric is loaded or computed."
        ),
        "training_index_universe": {
            "users": len(universe.users),
            "items": len(universe.items),
        },
        "summaries": [
            {
                **{
                    k: v for k, v in asdict(summary).items()
                    if k != "targets"
                },
                "targets": [asdict(target) for target in summary.targets],
            }
            for summary in summaries
        ],
        "interpretation": (
            "Descriptive candidate/relevance structure after frozen scoring "
            "eligibility. Final minimum-candidate and zero-relevance rules remain "
            "unfrozen until these counts are reviewed."
        ),
    }
