"""Primary multi-seed fitting for frozen BPR and LightGCN specifications.

Unlike the tuning pipelines, this module never computes validation metrics.
It reads only Apr 9-16 training rows, trains each frozen model to its fixed
endpoint, and saves that endpoint checkpoint.

Examples
--------
Fit BPR seeds 0-4:
    python -m ranklab.training.primary_multiseed \
        --data data/raw/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv \
        --output-dir runs/m0/primary_multiseed \
        --model bpr

Fit LightGCN seeds 0-4:
    python -m ranklab.training.primary_multiseed \
        --data data/raw/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv \
        --output-dir runs/m0/primary_multiseed \
        --model lightgcn

Verify seed-1 reproducibility without overwriting originals:
    python -m ranklab.training.primary_multiseed \
        --data data/raw/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv \
        --output-dir runs/m0/primary_multiseed \
        --model both \
        --verify-seed1
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Iterable, Literal

from ranklab.models.bpr_mf import BPRMatrixFactorization
from ranklab.models.lightgcn_core import LightGCNModel
from ranklab.training.bpr_engine import (
    sample_same_user_logged_negatives,
    train_bpr_epoch,
)
from ranklab.training.bpr_pipeline import TRAIN_DATES
from ranklab.training.lightgcn_large_scale import train_lightgcn_epoch_chunked
from ranklab.training.primary_seed_policy import (
    BPR_SELECTED,
    LIGHTGCN_SELECTED,
    PRIMARY_SEEDS,
    bpr_output_name,
    lightgcn_output_name,
    validate_primary_seed,
)


STATUS = "M0_PRIMARY_MULTI_SEED_FIT"
ModelName = Literal["bpr", "lightgcn"]


@dataclass(frozen=True)
class PrimaryTrainingStats:
    training_rows: int
    training_unique_pairs: int
    training_positive_pairs: int
    training_negative_pairs: int
    training_positive_users: int
    pairwise_eligible_users: int
    excluded_positive_users_no_logged_negative: int
    eligible_positive_pairs: int
    eligible_negative_pairs: int
    indexed_items: int


@dataclass(frozen=True)
class PrimaryTrainingData:
    positive_pairs: tuple[tuple[int, int], ...]
    negative_pairs: tuple[tuple[int, int], ...]
    eligible_positive_pairs: tuple[tuple[int, int], ...]
    eligible_negative_pairs: tuple[tuple[int, int], ...]
    eligible_users: tuple[int, ...]
    training_items: tuple[int, ...]
    stats: PrimaryTrainingStats


def _normalize_date(value: str) -> str:
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


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_hash(repo_root: Path) -> str | None:
    path = repo_root / "research" / "protocol_frozen_m0.yaml"
    return _sha256_file(path) if path.exists() else None


def load_primary_training_data(path: str | Path) -> PrimaryTrainingData:
    """Read only the frozen Apr 9-16 training labels.

    Rows outside TRAIN_DATES are skipped before user/item/click fields are
    parsed. Therefore Apr 17 onward labels cannot influence primary fitting.
    """

    source = Path(path)
    state: dict[tuple[int, int], int] = {}
    training_rows = 0

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "video_id", "date", "is_click"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        for row in reader:
            date = _normalize_date(row["date"])
            if date not in TRAIN_DATES:
                continue

            training_rows += 1
            user_id = _as_int_id(row["user_id"], column="user_id")
            item_id = _as_int_id(row["video_id"], column="video_id")
            clicked = _as_binary(row["is_click"], column="is_click")
            key = (user_id, item_id)
            state[key] = max(state.get(key, 0), clicked)

    if training_rows == 0:
        raise ValueError("no rows found in frozen training dates")

    positive_pairs = tuple(
        sorted(
            (pair for pair, clicked in state.items() if clicked == 1),
            key=lambda pair: (pair[0], pair[1]),
        )
    )
    negative_pairs = tuple(
        sorted(
            (pair for pair, clicked in state.items() if clicked == 0),
            key=lambda pair: (pair[0], pair[1]),
        )
    )

    positive_users = {u for u, _ in positive_pairs}
    negative_users = {u for u, _ in negative_pairs}
    eligible_users = tuple(sorted(positive_users.intersection(negative_users)))
    eligible_user_set = set(eligible_users)

    eligible_positive_pairs = tuple(
        pair for pair in positive_pairs if pair[0] in eligible_user_set
    )
    eligible_negative_pairs = tuple(
        pair for pair in negative_pairs if pair[0] in eligible_user_set
    )
    training_items = tuple(
        sorted(
            {i for _, i in eligible_positive_pairs}
            | {i for _, i in eligible_negative_pairs}
        )
    )

    if not eligible_positive_pairs:
        raise ValueError("no pairwise-eligible positive pairs")
    if len(training_items) < 2:
        raise ValueError("fewer than two training items")

    stats = PrimaryTrainingStats(
        training_rows=training_rows,
        training_unique_pairs=len(state),
        training_positive_pairs=len(positive_pairs),
        training_negative_pairs=len(negative_pairs),
        training_positive_users=len(positive_users),
        pairwise_eligible_users=len(eligible_users),
        excluded_positive_users_no_logged_negative=len(
            positive_users - set(eligible_users)
        ),
        eligible_positive_pairs=len(eligible_positive_pairs),
        eligible_negative_pairs=len(eligible_negative_pairs),
        indexed_items=len(training_items),
    )

    return PrimaryTrainingData(
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        eligible_positive_pairs=eligible_positive_pairs,
        eligible_negative_pairs=eligible_negative_pairs,
        eligible_users=eligible_users,
        training_items=training_items,
        stats=stats,
    )


def fit_bpr_primary_seed(
    training: PrimaryTrainingData,
    *,
    seed: int,
    output_dir: str | Path,
) -> dict:
    validate_primary_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = BPRMatrixFactorization.initialize(
        training.eligible_users,
        training.training_items,
        embedding_dim=BPR_SELECTED["embedding_dim"],
        seed=seed,
        init_std=BPR_SELECTED["init_std"],
    )

    history = []
    for epoch_index in range(BPR_SELECTED["fixed_epochs"]):
        triplets, sampling = sample_same_user_logged_negatives(
            training.eligible_positive_pairs,
            training.eligible_negative_pairs,
            seed=seed,
            epoch=epoch_index,
        )
        loss = train_bpr_epoch(
            model,
            triplets,
            learning_rate=BPR_SELECTED["learning_rate"],
            regularization=BPR_SELECTED["regularization"],
            batch_size=BPR_SELECTED["batch_size"],
            seed=seed,
            epoch=epoch_index,
        )
        history.append(
            {
                "epoch": epoch_index + 1,
                "loss": loss,
                "triplets": sampling.triplets_sampled,
            }
        )

    checkpoint = output_dir / bpr_output_name(seed)
    model.save(checkpoint)
    return {
        "model": "bpr",
        "seed": seed,
        "fixed_epochs": BPR_SELECTED["fixed_epochs"],
        "hyperparameters": dict(BPR_SELECTED),
        "history": history,
        "indexed_users": len(model.user_index.values),
        "indexed_items": len(model.item_index.values),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "validation_used_for_primary_fit": False,
    }


def fit_lightgcn_primary_seed(
    training: PrimaryTrainingData,
    *,
    seed: int,
    output_dir: str | Path,
) -> dict:
    validate_primary_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = LightGCNModel.initialize(
        user_ids=training.eligible_users,
        item_ids=training.training_items,
        positive_pairs=training.eligible_positive_pairs,
        embedding_dim=LIGHTGCN_SELECTED["embedding_dim"],
        num_layers=LIGHTGCN_SELECTED["num_layers"],
        seed=seed,
        init_std=LIGHTGCN_SELECTED["init_std"],
    )

    history = []
    for epoch_index in range(LIGHTGCN_SELECTED["fixed_epochs"]):
        triplets, sampling = sample_same_user_logged_negatives(
            training.eligible_positive_pairs,
            training.eligible_negative_pairs,
            seed=seed,
            epoch=epoch_index,
        )
        loss = train_lightgcn_epoch_chunked(
            model,
            triplets,
            learning_rate=LIGHTGCN_SELECTED["learning_rate"],
            regularization=LIGHTGCN_SELECTED["regularization"],
            gradient_chunk_size=LIGHTGCN_SELECTED["gradient_chunk_size"],
        )
        history.append(
            {
                "epoch": epoch_index + 1,
                "loss": loss,
                "triplets": sampling.triplets_sampled,
            }
        )

    checkpoint = output_dir / lightgcn_output_name(seed)
    model.save(checkpoint)
    return {
        "model": "lightgcn",
        "seed": seed,
        "fixed_epochs": LIGHTGCN_SELECTED["fixed_epochs"],
        "hyperparameters": dict(LIGHTGCN_SELECTED),
        "history": history,
        "indexed_users": len(model.user_index.values),
        "indexed_items": len(model.item_index.values),
        "graph_edges": len(model.graph.user_indices),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "validation_used_for_primary_fit": False,
    }


def _write_model_manifest(
    *,
    model: ModelName,
    output_dir: Path,
    data_path: Path,
    repo_root: Path,
    training: PrimaryTrainingData,
    seed_results: list[dict],
) -> dict:
    manifest = {
        "status": STATUS,
        "model": model,
        "data": {
            "path": str(data_path),
            "sha256": _sha256_file(data_path),
            "training_dates": list(TRAIN_DATES),
        },
        "protocol_sha256": _protocol_hash(repo_root),
        "primary_seed_policy": {
            "seeds": list(PRIMARY_SEEDS),
            "per_seed_validation_reselection": False,
            "per_seed_hyperparameter_retuning": False,
        },
        "training_stats": asdict(training.stats),
        "guardrails": [
            "Primary fitting reads only Apr 9-16 training labels.",
            "Apr 17-21 validation labels are not used by this runner.",
            "Apr 22-May 8 evaluation labels are not used.",
            "Randomized-policy evaluation labels are not used.",
            "Every seed uses the already-frozen model configuration.",
            "Every seed trains to the already-frozen fixed epoch endpoint.",
        ],
        "seeds": seed_results,
    }
    path = output_dir / f"{model}_primary_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def fit_primary_model(
    *,
    model: ModelName,
    data_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = ".",
) -> dict:
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    repo_root = Path(repo_root)
    training = load_primary_training_data(data_path)

    model_dir = output_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in PRIMARY_SEEDS:
        if model == "bpr":
            result = fit_bpr_primary_seed(training, seed=seed, output_dir=model_dir)
        elif model == "lightgcn":
            result = fit_lightgcn_primary_seed(
                training, seed=seed, output_dir=model_dir
            )
        else:
            raise ValueError(f"unsupported model: {model!r}")
        results.append(result)

    return _write_model_manifest(
        model=model,
        output_dir=output_dir,
        data_path=data_path,
        repo_root=repo_root,
        training=training,
        seed_results=results,
    )


def verify_seed1_reproducibility(
    *,
    model: ModelName,
    data_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Repeat seed 1 in an isolated directory and compare checkpoint bytes."""

    data_path = Path(data_path)
    output_dir = Path(output_dir)
    training = load_primary_training_data(data_path)

    original = (
        output_dir / model / bpr_output_name(1)
        if model == "bpr"
        else output_dir / model / lightgcn_output_name(1)
    )
    if not original.exists():
        raise FileNotFoundError(
            f"original primary seed-1 checkpoint does not exist: {original}"
        )

    repeat_dir = output_dir / "repro" / model
    repeat_dir.mkdir(parents=True, exist_ok=True)

    if model == "bpr":
        repeated = fit_bpr_primary_seed(training, seed=1, output_dir=repeat_dir)
        repeated_path = repeat_dir / bpr_output_name(1)
    elif model == "lightgcn":
        repeated = fit_lightgcn_primary_seed(
            training, seed=1, output_dir=repeat_dir
        )
        repeated_path = repeat_dir / lightgcn_output_name(1)
    else:
        raise ValueError(f"unsupported model: {model!r}")

    original_sha = _sha256_file(original)
    repeated_sha = _sha256_file(repeated_path)
    verified = original_sha == repeated_sha
    result = {
        "model": model,
        "seed": 1,
        "original_checkpoint": str(original),
        "original_sha256": original_sha,
        "repeat_checkpoint": str(repeated_path),
        "repeat_sha256": repeated_sha,
        "byte_identical": verified,
    }
    verification_path = output_dir / f"{model}_seed1_repro.json"
    verification_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not verified:
        raise RuntimeError(f"{model} seed-1 reproducibility check failed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit or verify frozen primary BPR/LightGCN seed checkpoints."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        choices=("bpr", "lightgcn", "both"),
        default="both",
    )
    parser.add_argument(
        "--verify-seed1",
        action="store_true",
        help="Repeat existing seed-1 checkpoint(s) and compare SHA256.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = ("bpr", "lightgcn") if args.model == "both" else (args.model,)

    results = {}
    for model in models:
        if args.verify_seed1:
            results[model] = verify_seed1_reproducibility(
                model=model,
                data_path=args.data,
                output_dir=args.output_dir,
            )
        else:
            results[model] = fit_primary_model(
                model=model,
                data_path=args.data,
                output_dir=args.output_dir,
            )

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
