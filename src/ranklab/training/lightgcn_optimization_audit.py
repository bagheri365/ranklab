"""Pre-sweep LightGCN optimization-scale audit.

This module is deliberately *not* a hyperparameter sweep. It probes a
predeclared learning-rate set while fixing all other settings, so RankLab can
choose a defensible LightGCN learning-rate search range before freezing the
actual model-selection protocol.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Iterable, Sequence

from ranklab.training.lightgcn_pipeline import run_lightgcn_pipeline


STATUS = "M0_LIGHTGCN_OPTIMIZATION_SCALE_AUDIT_ONLY"

LEARNING_RATES = (0.1, 0.5, 1.0, 2.0, 5.0)
FIXED_EMBEDDING_DIM = 32
FIXED_NUM_LAYERS = 1
FIXED_REGULARIZATION = 1e-4
FIXED_GRADIENT_CHUNK_SIZE = 8192
FIXED_EPOCHS = 5
FIXED_SEED = 0
FIXED_INIT_STD = 0.01


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def learning_rate_slug(value: float) -> str:
    return f"lr{value:g}"


def run_optimization_scale_audit(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = ".",
    learning_rates: Sequence[float] = LEARNING_RATES,
) -> dict:
    if tuple(learning_rates) != LEARNING_RATES:
        raise ValueError(
            "M0.9c learning-rate probe is predeclared and may not be changed"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for learning_rate in learning_rates:
        run_dir = output_dir / learning_rate_slug(learning_rate)
        if run_dir.exists():
            shutil.rmtree(run_dir)

        manifest = run_lightgcn_pipeline(
            data_path=data_path,
            output_dir=run_dir,
            embedding_dim=FIXED_EMBEDDING_DIM,
            num_layers=FIXED_NUM_LAYERS,
            learning_rate=learning_rate,
            regularization=FIXED_REGULARIZATION,
            gradient_chunk_size=FIXED_GRADIENT_CHUNK_SIZE,
            epochs=FIXED_EPOCHS,
            seed=FIXED_SEED,
            init_std=FIXED_INIT_STD,
            repo_root=repo_root,
        )

        history = manifest["training"]["history"]
        validation_values = [row["validation_ndcg_at_10"] for row in history]
        loss_values = [row["loss"] for row in history]

        results.append(
            {
                "learning_rate": learning_rate,
                "best_epoch": manifest["selection"]["best_epoch"],
                "best_validation_ndcg_at_10": manifest["selection"][
                    "best_validation_ndcg_at_10"
                ],
                "checkpoint_sha256": manifest["selection"]["checkpoint_sha256"],
                "loss_epoch_1": loss_values[0],
                "loss_epoch_5": loss_values[-1],
                "loss_change_epoch_1_to_5": loss_values[-1] - loss_values[0],
                "validation_changed_across_epochs": len(set(validation_values)) > 1,
                "history": history,
            }
        )

    audit_manifest = {
        "status": STATUS,
        "purpose": (
            "Choose a defensible LightGCN learning-rate range before freezing "
            "the actual hyperparameter search protocol; do not select a final model."
        ),
        "predeclared_learning_rates": list(LEARNING_RATES),
        "fixed_settings": {
            "embedding_dim": FIXED_EMBEDDING_DIM,
            "num_layers": FIXED_NUM_LAYERS,
            "regularization": FIXED_REGULARIZATION,
            "gradient_chunk_size": FIXED_GRADIENT_CHUNK_SIZE,
            "epochs": FIXED_EPOCHS,
            "seed": FIXED_SEED,
            "init_std": FIXED_INIT_STD,
            "validation_k": 10,
        },
        "guardrails": [
            "Only learning rate varies.",
            "No Apr 22-May 8 labels may be inspected.",
            "This audit does not select the final LightGCN configuration.",
            "The final LightGCN sweep grid must be frozen in a later commit before running it.",
            "Do not adaptively add learning rates after seeing these audit results.",
        ],
        "results": results,
    }

    manifest_path = output_dir / "audit_manifest.json"
    manifest_path.write_text(
        json.dumps(audit_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the predeclared M0.9c LightGCN optimization-scale audit."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_optimization_scale_audit(
        data_path=args.data,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
