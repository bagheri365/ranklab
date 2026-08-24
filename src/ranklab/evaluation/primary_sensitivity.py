"""M1.5 frozen support-sensitivity raw evaluation.

Reuses the frozen M1.1 scorers/checkpoints and M0 evaluation semantics, changing
only the pre-specified exposure-support restriction:
  A) shared users/videos + shared tabs {1,2,11,14}
  B) shared users/videos + tab=1 (descriptive)
"""

from __future__ import annotations

import argparse
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from ranklab.evaluation import primary_runner as primary


SENSITIVITIES = {
    "shared_tabs": {
        "tabs": (1, 2, 11, 14),
        "role": "pre_specified_support_sensitivity",
    },
    "tab1": {
        "tabs": (1,),
        "role": "descriptive_support_sensitivity",
    },
}


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_sensitivity_candidates(
    *,
    training_path: str | Path,
    standard_path: str | Path,
    randomized_path: str | Path,
    tabs: tuple[int, ...],
) -> dict[str, pd.DataFrame]:
    """Apply frozen shared-user/video support, then tab restriction, then M0.18."""
    standard_raw = pd.read_csv(standard_path, usecols=list(primary.EVAL_COLUMNS))
    randomized_raw = pd.read_csv(randomized_path, usecols=list(primary.EVAL_COLUMNS))

    support = primary.derive_evaluation_support(standard_raw, randomized_raw)
    training_universe = primary.derive_training_index_universe(training_path)

    result: dict[str, pd.DataFrame] = {}
    for regime, frame in (
        ("standard", standard_raw),
        ("randomized", randomized_raw),
    ):
        supported = primary.restrict_primary_support(frame, support)
        supported = supported.loc[supported["tab"].isin(tabs)].copy()
        supported = primary.restrict_to_training_indexed_entities(
            supported,
            training_universe,
        )
        collapsed = primary.collapse_logged_candidates(supported)
        result[regime] = collapsed.sort_values(
            ["user_id", "video_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    return result


def run_one_sensitivity(
    *,
    name: str,
    training_path: str | Path,
    data_dir: str | Path,
    primary_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    if name not in SENSITIVITIES:
        raise ValueError(f"unknown sensitivity: {name}")
    spec = SENSITIVITIES[name]

    repo_root = Path.cwd()
    protocol_sha = primary.verify_protocol(repo_root)

    data_dir = Path(data_dir)
    standard_path = data_dir / "log_standard_4_22_to_5_08_pure.csv"
    randomized_path = data_dir / "log_random_4_22_to_5_08_pure.csv"

    scorers, checkpoint_hashes = primary.load_frozen_scorers(
        training_path=training_path,
        primary_dir=primary_dir,
    )
    candidates = build_sensitivity_candidates(
        training_path=training_path,
        standard_path=standard_path,
        randomized_path=randomized_path,
        tabs=tuple(spec["tabs"]),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "per_user_ndcg.csv.gz"

    fields = [
        "sensitivity",
        "regime",
        "target",
        "model",
        "seed",
        "user_id",
        "ndcg_at_10",
        "candidates",
        "relevant",
        "included_in_macro",
        "exclusion_reason",
    ]

    summaries: dict[str, dict] = {}

    with gzip.open(raw_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for regime in ("standard", "randomized"):
            frame = candidates[regime]
            grouped = list(frame.groupby("user_id", sort=True))

            for scorer_spec in scorers:
                model = scorer_spec["model"]
                seed = scorer_spec["seed"]
                scorer = scorer_spec["scorer"]

                for user_id, user_frame in grouped:
                    item_ids = user_frame["video_id"].to_numpy()
                    scores = scorer.score_items(int(user_id), item_ids)

                    for target in ("is_click", "long_view"):
                        relevance = user_frame[target].to_numpy(dtype=int)
                        evaluation = primary.evaluate_user_ndcg(
                            scores=scores,
                            relevance=relevance,
                            item_ids=item_ids,
                            k=10,
                        )
                        writer.writerow(
                            {
                                "sensitivity": name,
                                "regime": regime,
                                "target": target,
                                "model": model,
                                "seed": seed,
                                "user_id": int(user_id),
                                "ndcg_at_10": "" if evaluation.ndcg is None else repr(float(evaluation.ndcg)),
                                "candidates": evaluation.candidates,
                                "relevant": evaluation.relevant,
                                "included_in_macro": int(
                                    evaluation.included_in_macro
                                ),
                                "exclusion_reason": (
                                    evaluation.exclusion_reason or ""
                                ),
                            }
                        )

                        key = f"{regime}|{target}|{model}|{seed}"
                        bucket = summaries.setdefault(
                            key,
                            {
                                "rows": 0,
                                "included_users": 0,
                                "fewer_than_2_candidates": 0,
                                "zero_relevance": 0,
                                "_sum_ndcg": 0.0,
                            },
                        )
                        bucket["rows"] += 1
                        if evaluation.included_in_macro:
                            bucket["included_users"] += 1
                            bucket["_sum_ndcg"] += float(evaluation.ndcg)
                        elif evaluation.exclusion_reason == "fewer_than_2_candidates":
                            bucket["fewer_than_2_candidates"] += 1
                        elif evaluation.exclusion_reason == "zero_relevance":
                            bucket["zero_relevance"] += 1

    for bucket in summaries.values():
        included = bucket["included_users"]
        bucket["macro_ndcg_at_10"] = (
            bucket.pop("_sum_ndcg") / included if included else None
        )

    raw_sha = _sha256_file(raw_path)
    manifest = {
        "status": "M1_SUPPORT_SENSITIVITY_RAW_EVALUATION",
        "sensitivity": name,
        "role": spec["role"],
        "tabs": list(spec["tabs"]),
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": checkpoint_hashes,
        "candidate_population": {
            regime: {
                "users": int(frame["user_id"].nunique()),
                "pairs": int(len(frame)),
            }
            for regime, frame in candidates.items()
        },
        "cell_seed_summaries": summaries,
        "outputs": {
            "per_user_ndcg": raw_path.name,
            "per_user_ndcg_sha256": raw_sha,
        },
        "guardrails": [
            "Uses the same frozen M0 checkpoints as the primary analysis.",
            "Changes only the pre-specified tab support restriction.",
            "Applies tab restriction before user-video candidate collapse.",
            "Keeps M0.18 training-index scoring eligibility and M0.19 NDCG semantics.",
            "No retuning, bootstrap inference, or primary-result modification is performed.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen M1 support-sensitivity raw scoring."
    )
    parser.add_argument("--sensitivity", choices=tuple(SENSITIVITIES), required=True)
    parser.add_argument(
        "--training-data",
        default="data/raw/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv",
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw/KuaiRand-Pure/data",
    )
    parser.add_argument(
        "--primary-dir",
        default="runs/m0/primary_multiseed",
    )
    parser.add_argument(
        "--output-root",
        default="runs/m1/sensitivity_raw",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_root) / args.sensitivity
    manifest = run_one_sensitivity(
        name=args.sensitivity,
        training_path=args.training_data,
        data_dir=args.data_dir,
        primary_dir=args.primary_dir,
        output_dir=output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
