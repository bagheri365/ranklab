"""M1.1 frozen primary raw evaluation runner.

This module executes the already-frozen M0 protocol without statistical
interpretation. It verifies protocol/checkpoint identities, constructs the
frozen primary evaluation candidates, and writes per-user NDCG records.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
import pandas as pd
import yaml

from ranklab.evaluation.candidate_audit import collapse_logged_candidates
from ranklab.evaluation.scoring_eligibility import (
    derive_training_index_universe,
    restrict_to_training_indexed_entities,
)
from ranklab.evaluation.semantics import PRIMARY_K, evaluate_user_ndcg
from ranklab.evaluation.support import derive_evaluation_support, restrict_primary_support
from ranklab.models.bpr_mf import BPRMatrixFactorization
from ranklab.models.lightgcn_core import LightGCNModel
from ranklab.models.popularity import PopularityRanker
from ranklab.training.primary_multiseed import load_primary_training_data
from ranklab.training.primary_seed_policy import PRIMARY_SEEDS, bpr_output_name, lightgcn_output_name


STATUS = "M1_PRIMARY_RAW_EVALUATION"
EXPECTED_PROTOCOL_SHA256 = "9bd72ae2d3042dc9511734563b90b8668519b401b847ce218f706a465cc41a32"

EXPECTED_CHECKPOINT_SHA256 = {
    "bpr": {
        0: "45360222adaf4f640de8cc1e459f27bcb72fd4685adae210aace69870c585997",
        1: "72580981edafb1db8b335ef1d4043eda91c26532037a6f39270fa9d046d48ed7",
        2: "46de29a40b0f86a3cbca9433bd0d732d894da9d54e6f3b4faddff66bd13e2247",
        3: "496e4a2db5754557754f63bc2b2793bd24f41582eeb1b60d5752b100b68cbfdf",
        4: "6a92f36b1396dfbedc7057e19a74312aa09b00be65737926fcf2562e7a1143ee",
    },
    "lightgcn": {
        0: "50438670f62a938731eba465fef44302a3e5623572f2e3eb46f0d7824a7cc189",
        1: "3ee3fffc4f9e69b08a403984338321f7ff389adf25de26047dfbcdcf5b183d9e",
        2: "408a89a1728d35a1777c2d335b7d26e2988abae7fa0771dacca6ce50a91a5536",
        3: "c3ba90b8f18b00d2691a9e42f1e62831f7434625ec77150e90894e2e980ae87d",
        4: "cde81a6b578b1ce7983ead8542e307f24dbac5a07a3c20efbf7643426fab12b0",
    },
}

EVAL_COLUMNS = ("user_id", "video_id", "tab", "is_click", "long_view")


class Scorer(Protocol):
    def score_items(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        ...


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_protocol(repo_root: str | Path) -> str:
    root = Path(repo_root)
    protocol_path = root / "research" / "protocol_frozen_m0.yaml"
    digest_path = root / "research" / "protocol_frozen_m0.sha256"
    m1_path = root / "configs" / "experiments" / "m1" / "primary.yaml"

    actual = _sha256_file(protocol_path)
    recorded = digest_path.read_text(encoding="utf-8").strip()
    config = yaml.safe_load(m1_path.read_text(encoding="utf-8"))

    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"protocol drift: expected {EXPECTED_PROTOCOL_SHA256}, got {actual}")
    if recorded != actual:
        raise RuntimeError("recorded protocol SHA256 does not match protocol bytes")
    if config["protocol_sha256"] != actual:
        raise RuntimeError("M1 config protocol SHA256 does not match protocol bytes")
    if config["status"] != "PROTOCOL_READY_NOT_RUN":
        raise RuntimeError(f"M1 config is not protocol-ready: {config['status']!r}")
    return actual


def verify_checkpoint(path: str | Path, *, model: str, seed: int) -> str:
    checkpoint = Path(path)
    expected = EXPECTED_CHECKPOINT_SHA256[model][seed]
    actual = _sha256_file(checkpoint)
    if actual != expected:
        raise RuntimeError(
            f"{model} seed {seed} checkpoint SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


class PopularityScorer:
    def __init__(self, training_path: str | Path) -> None:
        training = load_primary_training_data(training_path)
        pairs = pd.DataFrame(training.positive_pairs, columns=["user_id", "video_id"])
        pairs["is_positive"] = True
        self.model = PopularityRanker()
        self.model.fit(pairs)

    def score_items(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        return self.model.score(user_id, item_ids)


class BPRScorer:
    def __init__(self, model: BPRMatrixFactorization) -> None:
        self.model = model

    def score_items(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        return self.model.score_items(user_id, item_ids)


class CachedLightGCNScorer:
    """Cache propagated LightGCN embeddings once per checkpoint."""

    def __init__(self, model: LightGCNModel) -> None:
        self.user_index = model.user_index
        self.item_index = model.item_index
        self.users, self.items = model.final_embeddings()
        self.item_map = self.item_index.to_index

    def score_items(self, user_id: int, item_ids: Sequence[int]) -> Sequence[float]:
        u = self.user_index.encode(user_id)
        idx = np.fromiter(
            (self.item_map[item] for item in item_ids),
            dtype=np.int64,
            count=len(item_ids),
        )
        return self.items[idx] @ self.users[u]


def load_frozen_scorers(
    *,
    training_path: str | Path,
    primary_dir: str | Path,
) -> tuple[list[tuple[str, int | None, Scorer]], dict]:
    primary_dir = Path(primary_dir)
    scorers: list[tuple[str, int | None, Scorer]] = [
        ("popularity", None, PopularityScorer(training_path))
    ]
    verified: dict[str, dict[str, str]] = {"bpr": {}, "lightgcn": {}}

    for seed in PRIMARY_SEEDS:
        bpr_path = primary_dir / "bpr" / bpr_output_name(seed)
        verify_checkpoint(bpr_path, model="bpr", seed=seed)
        bpr = BPRMatrixFactorization.load(bpr_path)
        scorers.append(("bpr", seed, BPRScorer(bpr)))
        verified["bpr"][str(seed)] = EXPECTED_CHECKPOINT_SHA256["bpr"][seed]

    for seed in PRIMARY_SEEDS:
        lightgcn_path = primary_dir / "lightgcn" / lightgcn_output_name(seed)
        verify_checkpoint(lightgcn_path, model="lightgcn", seed=seed)
        lightgcn = LightGCNModel.load(lightgcn_path)
        scorers.append(("lightgcn", seed, CachedLightGCNScorer(lightgcn)))
        verified["lightgcn"][str(seed)] = EXPECTED_CHECKPOINT_SHA256["lightgcn"][seed]

    return scorers, verified


def build_primary_candidates(
    *,
    training_path: str | Path,
    data_dir: str | Path,
) -> dict[str, pd.DataFrame]:
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

    result: dict[str, pd.DataFrame] = {}
    for regime, frame in (("standard", standard), ("randomized", randomized)):
        supported = restrict_primary_support(frame, support)
        eligible = restrict_to_training_indexed_entities(supported, universe)
        collapsed = collapse_logged_candidates(eligible)
        result[regime] = collapsed.sort_values(
            ["user_id", "video_id"], kind="mergesort"
        ).reset_index(drop=True)
    return result


def _summary_bucket() -> dict:
    return {
        "rows": 0,
        "included_users": 0,
        "sum_ndcg": 0.0,
        "fewer_than_2_candidates": 0,
        "zero_relevance": 0,
    }


def run_primary_raw_evaluation(
    *,
    training_path: str | Path,
    data_dir: str | Path,
    primary_dir: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = ".",
) -> dict:
    protocol_sha = verify_protocol(repo_root)
    scorers, verified_checkpoints = load_frozen_scorers(
        training_path=training_path,
        primary_dir=primary_dir,
    )
    candidates = build_primary_candidates(
        training_path=training_path,
        data_dir=data_dir,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "per_user_ndcg.csv.gz"
    summary: dict[str, dict] = {}

    fields = [
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

    with gzip.open(raw_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for regime in ("standard", "randomized"):
            frame = candidates[regime]
            grouped = list(frame.groupby("user_id", sort=True))
            for model_name, seed, scorer in scorers:
                for user_id, user_frame in grouped:
                    item_ids = [int(v) for v in user_frame["video_id"].tolist()]
                    scores = scorer.score_items(int(user_id), item_ids)
                    for target in ("is_click", "long_view"):
                        relevance = [int(v) for v in user_frame[target].tolist()]
                        result = evaluate_user_ndcg(
                            item_ids=item_ids,
                            scores=scores,
                            relevance=relevance,
                            k=PRIMARY_K,
                        )
                        key = f"{regime}|{target}|{model_name}|{seed if seed is not None else 'deterministic'}"
                        bucket = summary.setdefault(key, _summary_bucket())
                        bucket["rows"] += 1
                        if result.included_in_macro:
                            bucket["included_users"] += 1
                            bucket["sum_ndcg"] += float(result.ndcg)
                        elif result.exclusion_reason:
                            bucket[result.exclusion_reason] += 1

                        writer.writerow(
                            {
                                "regime": regime,
                                "target": target,
                                "model": model_name,
                                "seed": "" if seed is None else seed,
                                "user_id": int(user_id),
                                "ndcg_at_10": "" if result.ndcg is None else repr(float(result.ndcg)),
                                "candidates": result.candidates,
                                "relevant": result.relevant,
                                "included_in_macro": int(result.included_in_macro),
                                "exclusion_reason": result.exclusion_reason or "",
                            }
                        )

    for bucket in summary.values():
        included = int(bucket["included_users"])
        bucket["macro_ndcg_at_10"] = (
            float(bucket["sum_ndcg"] / included) if included else None
        )
        del bucket["sum_ndcg"]

    manifest = {
        "status": STATUS,
        "interpretation_status": "RAW_METRICS_ONLY_NO_WINNER_OR_CONTRAST_CLAIMS",
        "protocol_sha256": protocol_sha,
        "primary_k": PRIMARY_K,
        "primary_support": "M0.15 shared users/videos then M0.18 training-indexed entities",
        "candidate_semantics": "M0.19 unique user-video pair with target-specific any-positive relevance",
        "checkpoint_sha256": verified_checkpoints,
        "outputs": {
            "per_user_ndcg": raw_path.name,
            "per_user_ndcg_sha256": _sha256_file(raw_path),
        },
        "candidate_population": {
            regime: {
                "users": int(frame["user_id"].nunique()),
                "pairs": int(len(frame)),
            }
            for regime, frame in candidates.items()
        },
        "cell_seed_summaries": summary,
        "guardrails": [
            "M0 protocol SHA256 verified before scoring.",
            "Every stochastic checkpoint SHA256 verified before loading.",
            "No checkpoint is trained or retuned.",
            "No M1 bootstrap, winner label, regime contrast, or target contrast is computed.",
            "Per-user records are retained for later frozen M0.20 inference.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run M1.1 frozen primary raw evaluation."
    )
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--primary-dir", default="runs/m0/primary")
    parser.add_argument("--output-dir", default="runs/m1/primary_raw")
    parser.add_argument("--repo-root", default=".")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_primary_raw_evaluation(
        training_path=args.training_data,
        data_dir=args.data_dir,
        primary_dir=args.primary_dir,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
