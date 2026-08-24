"""Frozen LightGCN hyperparameter sweep for RankLab M0.9d.

The grid is intentionally committed before execution.

Search axes:
    embedding_dim: 16, 32, 64
    num_layers: 1, 2, 3
    learning_rate: 0.5, 1.0, 2.0

Fixed:
    regularization: 1e-4
    gradient_chunk_size: 8192
    epochs: 30
    init_std: 0.01
    tuning seed: 0
    validation metric: macro NDCG@10

This is a 27-configuration search.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Iterable, Sequence

from ranklab.training.lightgcn_pipeline import run_lightgcn_pipeline


STATUS = "M0_LIGHTGCN_HYPERPARAMETER_SWEEP"

EMBEDDING_DIMS = (16, 32, 64)
NUM_LAYERS = (1, 2, 3)
LEARNING_RATES = (0.5, 1.0, 2.0)

FIXED_REGULARIZATION = 1e-4
FIXED_GRADIENT_CHUNK_SIZE = 8192
FIXED_EPOCHS = 30
FIXED_INIT_STD = 0.01
TUNING_SEED = 0
VALIDATION_K = 10


def config_name(embedding_dim: int, num_layers: int, learning_rate: float) -> str:
    return f"d{embedding_dim}_l{num_layers}_lr{learning_rate:g}"


def iter_grid() -> list[tuple[int, int, float]]:
    return [
        (embedding_dim, num_layers, learning_rate)
        for embedding_dim in EMBEDDING_DIMS
        for num_layers in NUM_LAYERS
        for learning_rate in LEARNING_RATES
    ]


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection_key(row: dict) -> tuple:
    """Sort so the frozen winner is first.

    Primary: strict maximum validation macro NDCG@10.
    Exact config-level ties prefer:
      1. fewer propagation layers;
      2. lower embedding dimension;
      3. lower learning rate;
      4. earlier best epoch.
    """

    return (
        -row["best_validation_ndcg_at_10"],
        row["num_layers"],
        row["embedding_dim"],
        row["learning_rate"],
        row["best_epoch"],
    )


def run_lightgcn_sweep(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = ".",
    grid: Sequence[tuple[int, int, float]] | None = None,
) -> dict:
    expected_grid = iter_grid()
    if grid is None:
        grid = expected_grid
    if list(grid) != expected_grid:
        raise ValueError("M0.9d LightGCN grid is frozen and may not be changed")

    output_dir = Path(output_dir)
    config_root = output_dir / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for embedding_dim, num_layers, learning_rate in grid:
        name = config_name(embedding_dim, num_layers, learning_rate)
        run_dir = config_root / name
        if run_dir.exists():
            shutil.rmtree(run_dir)

        manifest = run_lightgcn_pipeline(
            data_path=data_path,
            output_dir=run_dir,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            learning_rate=learning_rate,
            regularization=FIXED_REGULARIZATION,
            gradient_chunk_size=FIXED_GRADIENT_CHUNK_SIZE,
            epochs=FIXED_EPOCHS,
            seed=TUNING_SEED,
            init_std=FIXED_INIT_STD,
            k=VALIDATION_K,
            repo_root=repo_root,
        )

        selection = manifest["selection"]
        row = {
            "config": name,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "regularization": FIXED_REGULARIZATION,
            "best_epoch": selection["best_epoch"],
            "best_validation_ndcg_at_10": selection["best_validation_ndcg_at_10"],
            "checkpoint_sha256": selection["checkpoint_sha256"],
            "source_checkpoint": str(Path("configs") / name / "best_lightgcn.npz"),
        }
        rows.append(row)

    ranked = sorted(rows, key=_selection_key)
    winner = ranked[0]

    source_checkpoint = output_dir / winner["source_checkpoint"]
    selected_checkpoint = output_dir / "selected_lightgcn.npz"
    shutil.copyfile(source_checkpoint, selected_checkpoint)

    selected_sha = _sha256_file(selected_checkpoint)
    if selected_sha != winner["checkpoint_sha256"]:
        raise RuntimeError("selected LightGCN checkpoint hash mismatch")

    manifest = {
        "status": STATUS,
        "grid_frozen_before_execution": True,
        "search_space": {
            "embedding_dims": list(EMBEDDING_DIMS),
            "num_layers": list(NUM_LAYERS),
            "learning_rates": list(LEARNING_RATES),
            "regularization": FIXED_REGULARIZATION,
            "configuration_count": len(expected_grid),
        },
        "fixed_settings": {
            "gradient_chunk_size": FIXED_GRADIENT_CHUNK_SIZE,
            "epochs": FIXED_EPOCHS,
            "init_std": FIXED_INIT_STD,
            "tuning_seed": TUNING_SEED,
            "validation_k": VALIDATION_K,
        },
        "selection_rule": {
            "within_config": "strict maximum validation macro NDCG@10; exact tie -> earliest epoch",
            "between_configs": (
                "strict maximum best validation macro NDCG@10; exact tie -> "
                "fewer layers, lower embedding dimension, lower learning rate, "
                "earlier best epoch"
            ),
        },
        "guardrails": [
            "All 27 configurations run for the full 30 epochs; no early stopping.",
            "Only tuning seed 0 is used for hyperparameter selection.",
            "No adaptive grid expansion is allowed after results are observed.",
            "Apr 22-May 8 evaluation labels must not be inspected.",
            "Randomized-policy evaluation labels must not be inspected.",
            "This sweep selects a tuning-seed LightGCN configuration/checkpoint only.",
            "Primary multi-seed LightGCN fitting remains a later frozen step.",
            "Regularization is fixed at 1e-4 and is not claimed to be optimized.",
        ],
        "results": rows,
        "ranked_results": ranked,
        "selected": {
            **winner,
            "selected_checkpoint": selected_checkpoint.name,
            "selected_checkpoint_sha256": selected_sha,
        },
    }

    (output_dir / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen M0.9d LightGCN hyperparameter sweep."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_lightgcn_sweep(
        data_path=args.data,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
