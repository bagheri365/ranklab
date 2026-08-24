"""M0.17 training-to-evaluation entity coverage audit.

The audit derives the exact BPR/LightGCN indexed training user/item universe
from the frozen pairwise training contract, then measures coverage of the
already-frozen M0.15 evaluation supports.

No recommender checkpoint is loaded and no model score or ranking metric is
computed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ranklab.evaluation.candidate_audit import collapse_logged_candidates
from ranklab.evaluation.support import (
    derive_evaluation_support,
    restrict_primary_support,
    restrict_shared_tab_sensitivity,
    restrict_tab1_sensitivity,
)
from ranklab.training.primary_multiseed import load_primary_training_data


EVAL_COLUMNS = ("user_id", "video_id", "tab", "is_click", "long_view")


@dataclass(frozen=True)
class EntityCoverageSummary:
    regime: str
    support: str
    rows: int
    candidate_pairs: int
    users: int
    items: int
    seen_users: int
    unseen_users: int
    seen_items: int
    unseen_items: int
    pairs_seen_user_and_item: int
    pairs_unseen_user: int
    pairs_unseen_item: int
    pairs_any_unseen_entity: int
    seen_user_fraction: float
    seen_item_fraction: float
    fully_scorable_pair_fraction: float


def _ratio(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def summarize_entity_coverage(
    frame: pd.DataFrame,
    *,
    training_users: set[int],
    training_items: set[int],
    regime: str,
    support_name: str,
) -> EntityCoverageSummary:
    collapsed = collapse_logged_candidates(frame)

    users = set(collapsed["user_id"].astype(int).tolist())
    items = set(collapsed["video_id"].astype(int).tolist())
    seen_users = users & training_users
    seen_items = items & training_items

    user_seen_mask = collapsed["user_id"].astype(int).isin(training_users)
    item_seen_mask = collapsed["video_id"].astype(int).isin(training_items)
    fully_seen = user_seen_mask & item_seen_mask
    any_unseen = ~fully_seen

    return EntityCoverageSummary(
        regime=regime,
        support=support_name,
        rows=int(len(frame)),
        candidate_pairs=int(len(collapsed)),
        users=len(users),
        items=len(items),
        seen_users=len(seen_users),
        unseen_users=len(users - training_users),
        seen_items=len(seen_items),
        unseen_items=len(items - training_items),
        pairs_seen_user_and_item=int(fully_seen.sum()),
        pairs_unseen_user=int((~user_seen_mask).sum()),
        pairs_unseen_item=int((~item_seen_mask).sum()),
        pairs_any_unseen_entity=int(any_unseen.sum()),
        seen_user_fraction=_ratio(len(seen_users), len(users)),
        seen_item_fraction=_ratio(len(seen_items), len(items)),
        fully_scorable_pair_fraction=_ratio(int(fully_seen.sum()), len(collapsed)),
    )


def build_entity_coverage_audit(
    *,
    training_path: str | Path,
    data_dir: str | Path,
) -> dict[str, Any]:
    training = load_primary_training_data(training_path)
    training_users = set(int(v) for v in training.eligible_users)
    training_items = set(int(v) for v in training.training_items)

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

    variants = [
        ("primary_shared_users_and_videos", restrict_primary_support),
        (
            "sensitivity_shared_users_videos_and_tabs",
            restrict_shared_tab_sensitivity,
        ),
        (
            "sensitivity_shared_users_videos_tab_1",
            restrict_tab1_sensitivity,
        ),
    ]

    summaries = []
    for support_name, restrict_fn in variants:
        for regime, frame in (("standard", standard), ("randomized", randomized)):
            restricted = restrict_fn(frame, support)
            summaries.append(
                summarize_entity_coverage(
                    restricted,
                    training_users=training_users,
                    training_items=training_items,
                    regime=regime,
                    support_name=support_name,
                )
            )

    return {
        "status": "M0_ENTITY_COVERAGE_AUDIT_ONLY",
        "guardrail": (
            "No checkpoint, model score, ranking, ranking metric, or model "
            "comparison is loaded or computed."
        ),
        "training_index_universe": {
            "users": len(training_users),
            "items": len(training_items),
            "source": "frozen_pairwise_eligible_training_population",
        },
        "evaluation_common_support": {
            "shared_users": len(support.shared_users),
            "shared_videos": len(support.shared_videos),
            "shared_tabs": sorted(support.shared_tabs),
        },
        "summaries": [asdict(summary) for summary in summaries],
        "interpretation": (
            "Descriptive entity-coverage audit only. The final unseen-entity "
            "evaluation rule remains unfrozen until these counts are reviewed."
        ),
    }
