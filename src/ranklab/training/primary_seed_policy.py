"""Frozen shared primary multi-seed fitting policy for BPR and LightGCN.

This module defines the seed set and fixed final-fit epochs for both selected
models. It deliberately prohibits per-seed validation re-selection.
"""

from __future__ import annotations

PRIMARY_SEEDS = (0, 1, 2, 3, 4)

BPR_SELECTED = {
    "embedding_dim": 32,
    "learning_rate": 0.05,
    "regularization": 1e-3,
    "batch_size": 4096,
    "init_std": 0.01,
    "fixed_epochs": 1,
}

LIGHTGCN_SELECTED = {
    "embedding_dim": 32,
    "num_layers": 2,
    "learning_rate": 2.0,
    "regularization": 1e-4,
    "gradient_chunk_size": 8192,
    "init_std": 0.01,
    "fixed_epochs": 28,
}


def validate_primary_seed(seed: int) -> None:
    if seed not in PRIMARY_SEEDS:
        raise ValueError(
            f"primary seed {seed} is not in frozen seed set {PRIMARY_SEEDS}"
        )


def bpr_output_name(seed: int) -> str:
    validate_primary_seed(seed)
    return f"bpr_seed{seed}.npz"


def lightgcn_output_name(seed: int) -> str:
    validate_primary_seed(seed)
    return f"lightgcn_seed{seed}.npz"
