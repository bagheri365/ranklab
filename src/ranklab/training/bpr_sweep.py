"""Frozen M0.8c BPR hyperparameter sweep."""

from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path
import shutil
from typing import Iterable

from ranklab.training.bpr_pipeline import run_bpr_pipeline


EMBEDDING_DIMS = (16, 32, 64)
LEARNING_RATES = (0.01, 0.05, 0.1)
REGULARIZATIONS = (1e-5, 1e-4, 1e-3)

EPOCHS = 30
BATCH_SIZE = 4096
INIT_STD = 0.01
TUNING_SEED = 0
VALIDATION_K = 10
STATUS = "M0_BPR_HYPERPARAMETER_SWEEP"


def frozen_grid() -> list[dict]:
    return [
        {
            "embedding_dim": embedding_dim,
            "learning_rate": learning_rate,
            "regularization": regularization,
        }
        for embedding_dim, learning_rate, regularization in itertools.product(
            EMBEDDING_DIMS, LEARNING_RATES, REGULARIZATIONS
        )
    ]


def configuration_id(config: dict) -> str:
    reg = f"{config['regularization']:.0e}".replace("+", "")
    lr = f"{config['learning_rate']:.2g}"
    return f"d{config['embedding_dim']}_lr{lr}_reg{reg}"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_key(result: dict) -> tuple:
    hp = result["hyperparameters"]
    selection = result["selection"]
    return (
        -float(selection["best_validation_ndcg_at_10"]),
        int(hp["embedding_dim"]),
        float(hp["regularization"]),
        float(hp["learning_rate"]),
        int(selection["best_epoch"]),
    )


def choose_winner(results: list[dict]) -> dict:
    if not results:
        raise ValueError("at least one sweep result is required")
    return min(results, key=selection_key)


def run_sweep(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = ".",
) -> dict:
    output_dir = Path(output_dir)
    repo_root = Path(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for config in frozen_grid():
        config_id = configuration_id(config)
        config_dir = output_dir / "configs" / config_id
        config_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = config_dir / "manifest.json"

        if manifest_path.exists():
            result = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            result = run_bpr_pipeline(
                data_path=data_path,
                output_dir=config_dir,
                embedding_dim=config["embedding_dim"],
                learning_rate=config["learning_rate"],
                regularization=config["regularization"],
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                seed=TUNING_SEED,
                init_std=INIT_STD,
                k=VALIDATION_K,
                repo_root=repo_root,
            )

        result = dict(result)
        result["configuration_id"] = config_id
        results.append(result)

    winner = choose_winner(results)
    winner_id = winner["configuration_id"]
    winner_checkpoint = (
        output_dir / "configs" / winner_id / winner["selection"]["checkpoint"]
    )
    selected_checkpoint = output_dir / "selected_bpr.npz"
    shutil.copyfile(winner_checkpoint, selected_checkpoint)

    summary = {
        "status": STATUS,
        "guardrails": [
            "Grid frozen before sweep execution.",
            "Uses only Apr 17-21 standard-policy click validation.",
            "No early stopping; every configuration trains 30 epochs.",
            "Tuning seed fixed at 0.",
            "No adaptive grid expansion after observing results.",
            "Apr 22-May 8 labels prohibited.",
        ],
        "grid": {
            "embedding_dim": list(EMBEDDING_DIMS),
            "learning_rate": list(LEARNING_RATES),
            "regularization": list(REGULARIZATIONS),
            "total_configurations": len(results),
        },
        "fixed": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "init_std": INIT_STD,
            "seed": TUNING_SEED,
            "validation_k": VALIDATION_K,
            "early_stopping": False,
        },
        "selection_rule": {
            "primary": "maximum configuration-level best validation macro NDCG@10",
            "within_configuration_tie": "earliest epoch",
            "exact_between_configuration_tie": [
                "lower embedding dimension",
                "lower regularization",
                "lower learning rate",
                "earlier best epoch",
            ],
        },
        "results": [
            {
                "configuration_id": result["configuration_id"],
                "hyperparameters": result["hyperparameters"],
                "best_epoch": result["selection"]["best_epoch"],
                "best_validation_ndcg_at_10": result["selection"][
                    "best_validation_ndcg_at_10"
                ],
                "checkpoint_sha256": result["selection"]["checkpoint_sha256"],
            }
            for result in results
        ],
        "winner": {
            "configuration_id": winner_id,
            "hyperparameters": winner["hyperparameters"],
            "best_epoch": winner["selection"]["best_epoch"],
            "best_validation_ndcg_at_10": winner["selection"][
                "best_validation_ndcg_at_10"
            ],
            "source_checkpoint_sha256": winner["selection"]["checkpoint_sha256"],
            "selected_checkpoint": selected_checkpoint.name,
            "selected_checkpoint_sha256": _sha256_file(selected_checkpoint),
        },
    }

    (output_dir / "sweep_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen M0.8c BPR sweep.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=".")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_sweep(
        data_path=args.data,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
