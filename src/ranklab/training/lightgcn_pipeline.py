"""Real-data LightGCN runner for the frozen RankLab training contract.

M0.9b mirrors the validated BPR real-data pipeline while preserving the
LightGCN-specific graph semantics frozen in M0.9a.

Run:
    python -m ranklab.training.lightgcn_pipeline \
        --data data/raw/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv \
        --output-dir runs/m0/lightgcn_smoke \
        --embedding-dim 32 \
        --num-layers 1 \
        --epochs 3 \
        --seed 0

This remains an M0 engineering smoke runner, not an M1 experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from ranklab.models.lightgcn_core import LightGCNModel
from ranklab.training.bpr_engine import sample_same_user_logged_negatives
from ranklab.training.bpr_pipeline import (
    PRIMARY_K,
    TRAIN_DATES,
    VALIDATION_DATES,
    load_frozen_bpr_data,
)
from ranklab.training.lightgcn_engine import macro_logged_ndcg_at_k
from ranklab.training.lightgcn_large_scale import train_lightgcn_epoch_chunked


STATUS = "M0_LIGHTGCN_PIPELINE_SMOKE_ONLY"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_hash(repo_root: Path) -> str | None:
    path = repo_root / "research" / "protocol_frozen_m0.yaml"
    return _sha256_file(path) if path.exists() else None


def run_lightgcn_pipeline(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    embedding_dim: int,
    num_layers: int,
    learning_rate: float,
    regularization: float,
    gradient_chunk_size: int,
    epochs: int,
    seed: int,
    init_std: float = 0.01,
    k: int = PRIMARY_K,
    repo_root: str | Path = ".",
) -> dict:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if num_layers < 0:
        raise ValueError("num_layers must be non-negative")
    if k != PRIMARY_K:
        raise ValueError(f"M0.9b primary validation k is frozen at {PRIMARY_K}")

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

    eligible_negative_pairs = tuple(
        pair for pair in pairs.negative_pairs if pair[0] in eligible_user_set
    )
    training_items = sorted(
        {i for _, i in eligible_positive_pairs}
        | {i for _, i in eligible_negative_pairs}
    )

    if not eligible_positive_pairs:
        raise ValueError("no pairwise-eligible positive pairs")
    if len(training_items) < 2:
        raise ValueError("fewer than two training items")

    # Comparability choice: graph-only users are not admitted. The graph and
    # pairwise loss use the same pairwise-eligible user population as BPR.
    graph_positive_pairs = eligible_positive_pairs

    model = LightGCNModel.initialize(
        user_ids=eligible_users,
        item_ids=training_items,
        positive_pairs=graph_positive_pairs,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        seed=seed,
        init_std=init_std,
    )

    history: list[dict] = []
    best_ndcg = float("-inf")
    best_epoch: int | None = None
    best_checkpoint = output_dir / "best_lightgcn.npz"

    for epoch_index in range(epochs):
        triplets, sampling = sample_same_user_logged_negatives(
            eligible_positive_pairs,
            eligible_negative_pairs,
            seed=seed,
            epoch=epoch_index,
        )
        loss = train_lightgcn_epoch_chunked(
            model,
            triplets,
            learning_rate=learning_rate,
            regularization=regularization,
            gradient_chunk_size=gradient_chunk_size,
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
            "Uses the frozen same-user logged-negative rule inherited from BPR.",
            "Excludes positive users with zero logged negatives from both graph and pairwise loss.",
            "This smoke runner does not freeze LightGCN hyperparameters or primary seeds.",
        ],
        "inherited_from_training_contract": {
            "train_dates": list(TRAIN_DATES),
            "validation_dates": list(VALIDATION_DATES),
            "positive_pair_rule": "any-click collapsed user-video pair",
            "negative_sampling": "same-user logged negatives; uniform with replacement",
            "negatives_per_positive_per_epoch": 1,
            "validation_metric": "macro NDCG@10",
        },
        "lightgcn_specific": {
            "graph_edges": "positive click pairs from pairwise-eligible users only",
            "normalization": "symmetric degree normalization",
            "layer_aggregation": "mean of layer 0 through layer L",
            "feature_transforms": False,
            "nonlinear_activations": False,
            "gradient_update": "memory-bounded full-batch gradient accumulated in chunks",
        },
        "data": {
            "path": str(data_path),
            "sha256": _sha256_file(data_path),
        },
        "protocol_sha256": _protocol_hash(repo_root),
        "hyperparameters": {
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "regularization": regularization,
            "gradient_chunk_size": gradient_chunk_size,
            "epochs": epochs,
            "seed": seed,
            "init_std": init_std,
            "validation_k": k,
        },
        "stats": asdict(stats),
        "training": {
            "eligible_positive_pairs": len(eligible_positive_pairs),
            "eligible_negative_pairs": len(eligible_negative_pairs),
            "graph_edges": len(graph_positive_pairs),
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

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run M0.9b frozen-contract LightGCN smoke pipeline."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--gradient-chunk-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-std", type=float, default=0.01)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_lightgcn_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        gradient_chunk_size=args.gradient_chunk_size,
        epochs=args.epochs,
        seed=args.seed,
        init_std=args.init_std,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
