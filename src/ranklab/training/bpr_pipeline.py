"""Real-data BPR smoke runner for the frozen RankLab training contract.

M0.8b deliberately provides a module entry point rather than modifying the
repository's public CLI surface. It proves that the tested BPR core can train
end-to-end on the frozen KuaiRand-Pure standard-policy split, produce a
deterministic best checkpoint, and record a reproducibility manifest.

Run:
    python -m ranklab.training.bpr_pipeline \
        --data data/raw/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv \
        --output-dir runs/m0/bpr_smoke \
        --epochs 3 \
        --seed 0

This is an M0 smoke/training-infrastructure runner, not an M1 experiment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np

from ranklab.models.bpr_mf import BPRMatrixFactorization
from ranklab.training.bpr_engine import (
    macro_logged_ndcg_at_k,
    sample_same_user_logged_negatives,
    train_bpr_epoch,
)


TRAIN_DATES = tuple(f"202204{day:02d}" for day in range(9, 17))
VALIDATION_DATES = tuple(f"202204{day:02d}" for day in range(17, 22))
PRIMARY_K = 10
STATUS = "M0_BPR_PIPELINE_SMOKE_ONLY"


@dataclass(frozen=True)
class PairData:
    positive_pairs: tuple[tuple[int, int], ...]
    negative_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PipelineStats:
    training_rows: int
    validation_rows: int
    training_unique_pairs: int
    training_positive_pairs: int
    training_negative_pairs: int
    training_positive_users: int
    training_pairwise_eligible_users: int
    training_excluded_positive_users_no_logged_negative: int
    validation_users_raw: int
    validation_unique_pairs_raw: int


def _normalize_date(value: str) -> str:
    """Normalize common KuaiRand date renderings to YYYYMMDD."""

    value = value.strip()
    if value.endswith(".0"):
        value = value[:-2]
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) != 8:
        raise ValueError(f"unsupported date value: {value!r}")
    return digits


def _as_binary(value: str, *, column: str) -> int:
    text = value.strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text not in {"0", "1"}:
        raise ValueError(f"{column} must be binary, got {value!r}")
    return int(text)


def _as_int_id(value: str, *, column: str) -> int:
    text = value.strip()
    if text.endswith(".0"):
        text = text[:-2]
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{column} must be integer-like, got {value!r}") from exc


def load_frozen_bpr_data(
    path: str | Path,
) -> tuple[PairData, dict[int, list[tuple[int, int]]], PipelineStats]:
    """Load and collapse the frozen standard-history train/validation slices."""

    source = Path(path)
    train_state: dict[tuple[int, int], int] = {}
    validation_state: dict[tuple[int, int], int] = {}
    training_rows = 0
    validation_rows = 0

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "video_id", "date", "is_click"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        for row in reader:
            date = _normalize_date(row["date"])
            if date not in TRAIN_DATES and date not in VALIDATION_DATES:
                continue

            user_id = _as_int_id(row["user_id"], column="user_id")
            item_id = _as_int_id(row["video_id"], column="video_id")
            clicked = _as_binary(row["is_click"], column="is_click")
            key = (user_id, item_id)

            if date in TRAIN_DATES:
                training_rows += 1
                train_state[key] = max(train_state.get(key, 0), clicked)
            else:
                validation_rows += 1
                validation_state[key] = max(validation_state.get(key, 0), clicked)

    if training_rows == 0:
        raise ValueError("no rows found in frozen training dates")
    if validation_rows == 0:
        raise ValueError("no rows found in frozen validation dates")

    positive_pairs = tuple(sorted(
        (pair for pair, clicked in train_state.items() if clicked == 1),
        key=lambda pair: (pair[0], pair[1]),
    ))
    negative_pairs = tuple(sorted(
        (pair for pair, clicked in train_state.items() if clicked == 0),
        key=lambda pair: (pair[0], pair[1]),
    ))

    validation_by_user: dict[int, list[tuple[int, int]]] = {}
    for (user_id, item_id), clicked in sorted(validation_state.items()):
        validation_by_user.setdefault(user_id, []).append((item_id, clicked))

    positive_users = {u for u, _ in positive_pairs}
    negative_users = {u for u, _ in negative_pairs}
    eligible_users = positive_users.intersection(negative_users)

    stats = PipelineStats(
        training_rows=training_rows,
        validation_rows=validation_rows,
        training_unique_pairs=len(train_state),
        training_positive_pairs=len(positive_pairs),
        training_negative_pairs=len(negative_pairs),
        training_positive_users=len(positive_users),
        training_pairwise_eligible_users=len(eligible_users),
        training_excluded_positive_users_no_logged_negative=len(
            positive_users - eligible_users
        ),
        validation_users_raw=len(validation_by_user),
        validation_unique_pairs_raw=len(validation_state),
    )
    return PairData(positive_pairs, negative_pairs), validation_by_user, stats


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_hash(repo_root: Path) -> str | None:
    path = repo_root / "research" / "protocol_frozen_m0.yaml"
    return _sha256_file(path) if path.exists() else None


def run_bpr_pipeline(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    embedding_dim: int,
    learning_rate: float,
    regularization: float,
    batch_size: int,
    epochs: int,
    seed: int,
    init_std: float = 0.01,
    k: int = PRIMARY_K,
    repo_root: str | Path = ".",
) -> dict:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if k != PRIMARY_K:
        raise ValueError(f"M0.8b primary validation k is frozen at {PRIMARY_K}")

    data_path = Path(data_path)
    output_dir = Path(output_dir)
    repo_root = Path(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs, validation_by_user, stats = load_frozen_bpr_data(data_path)

    negative_users = {u for u, _ in pairs.negative_pairs}
    eligible_positive_pairs = tuple(
        pair for pair in pairs.positive_pairs if pair[0] in negative_users
    )
    eligible_users = sorted({u for u, _ in eligible_positive_pairs})
    eligible_user_set = set(eligible_users)
    training_items = sorted(
        {i for _, i in eligible_positive_pairs}
        | {i for u, i in pairs.negative_pairs if u in eligible_user_set}
    )
    if not eligible_positive_pairs:
        raise ValueError("no pairwise-eligible positive pairs")
    if len(training_items) < 2:
        raise ValueError("fewer than two training items")

    model = BPRMatrixFactorization.initialize(
        eligible_users,
        training_items,
        embedding_dim=embedding_dim,
        seed=seed,
        init_std=init_std,
    )

    history: list[dict] = []
    best_ndcg = float("-inf")
    best_epoch: int | None = None
    best_checkpoint = output_dir / "best_bpr.npz"

    for epoch_index in range(epochs):
        triplets, sampling = sample_same_user_logged_negatives(
            eligible_positive_pairs,
            pairs.negative_pairs,
            seed=seed,
            epoch=epoch_index,
        )
        loss = train_bpr_epoch(
            model,
            triplets,
            learning_rate=learning_rate,
            regularization=regularization,
            batch_size=batch_size,
            seed=seed,
            epoch=epoch_index,
        )
        ndcg = macro_logged_ndcg_at_k(model, validation_by_user, k=k)

        epoch_number = epoch_index + 1
        history.append(
            {
                "epoch": epoch_number,
                "loss": loss,
                "validation_ndcg_at_10": ndcg,
                "triplets": sampling.triplets_sampled,
            }
        )

        # Earliest epoch wins exact ties.
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_epoch = epoch_number
            model.save(best_checkpoint)

    if best_epoch is None:
        raise RuntimeError("no checkpoint selected")

    manifest = {
        "status": STATUS,
        "guardrails": [
            "Uses standard-policy history only.",
            "Uses frozen Apr 9-16 training and Apr 17-21 validation dates.",
            "Uses is_click only for primary training and validation selection.",
            "Does not inspect Apr 22-May 8 evaluation labels.",
            "This smoke runner does not freeze M1 hyperparameters or seeds.",
        ],
        "data": {
            "path": str(data_path),
            "sha256": _sha256_file(data_path),
            "train_dates": list(TRAIN_DATES),
            "validation_dates": list(VALIDATION_DATES),
        },
        "protocol_sha256": _protocol_hash(repo_root),
        "hyperparameters": {
            "embedding_dim": embedding_dim,
            "learning_rate": learning_rate,
            "regularization": regularization,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
            "init_std": init_std,
            "validation_k": k,
            "negatives_per_positive_per_epoch": 1,
            "negative_sampling": "same-user logged negatives; uniform with replacement",
        },
        "stats": asdict(stats),
        "training": {
            "eligible_positive_pairs": len(eligible_positive_pairs),
            "indexed_users": len(model.user_index.values),
            "indexed_items": len(model.item_index.values),
            "history": history,
        },
        "selection": {
            "metric": "macro_ndcg_at_10",
            "tie_break": "earliest_epoch",
            "best_epoch": best_epoch,
            "best_validation_ndcg_at_10": best_ndcg,
            "checkpoint": best_checkpoint.name,
            "checkpoint_sha256": _sha256_file(best_checkpoint),
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the M0.8b frozen-contract BPR smoke pipeline."
    )
    parser.add_argument("--data", required=True, help="Path to standard-history CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-std", type=float, default=0.01)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_bpr_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        init_std=args.init_std,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
