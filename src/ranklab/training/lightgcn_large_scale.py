"""Memory-bounded full-batch LightGCN optimization.

The mathematical update matches the transparent M0.9a full-batch engine, but
triplet gradients are accumulated in chunks so the real KuaiRand training set
does not require materializing every per-triplet gradient array at once.
"""

from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np

from ranklab.models.lightgcn_core import LightGCNModel
from ranklab.training.lightgcn_engine import (
    _backprop_through_propagation,
    mean_lightgcn_bpr_loss,
)


Id = Hashable
Triplet = tuple[Id, Id, Id]


def train_lightgcn_epoch_chunked(
    model: LightGCNModel,
    triplets: Sequence[Triplet],
    *,
    learning_rate: float,
    regularization: float,
    gradient_chunk_size: int,
) -> float:
    """Run one deterministic memory-bounded full-batch LightGCN+BPR update.

    Propagated embeddings are computed once from the epoch-start base
    embeddings. Triplet gradients are then accumulated chunk by chunk before a
    single full-batch parameter update. This is mathematically equivalent to
    the M0.9a full-batch gradient calculation apart from floating-point
    accumulation order.
    """

    if not triplets:
        raise ValueError("at least one triplet is required")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    if gradient_chunk_size <= 0:
        raise ValueError("gradient_chunk_size must be positive")

    user_map = model.user_index.to_index
    item_map = model.item_index.to_index

    u_idx = np.fromiter((user_map[u] for u, _, _ in triplets), dtype=np.int64)
    i_idx = np.fromiter((item_map[i] for _, i, _ in triplets), dtype=np.int64)
    j_idx = np.fromiter((item_map[j] for _, _, j in triplets), dtype=np.int64)

    final_users, final_items = model.final_embeddings()
    grad_final_users = np.zeros_like(final_users)
    grad_final_items = np.zeros_like(final_items)

    reg_user = np.zeros_like(model.user_embeddings)
    reg_item = np.zeros_like(model.item_embeddings)

    total = len(triplets)
    global_scale = 1.0 / total

    for start in range(0, total, gradient_chunk_size):
        stop = min(start + gradient_chunk_size, total)
        ub = u_idx[start:stop]
        ib = i_idx[start:stop]
        jb = j_idx[start:stop]

        u = final_users[ub]
        i = final_items[ib]
        j = final_items[jb]

        margin = np.sum(u * (i - j), axis=1)
        dmargin = -1.0 / (1.0 + np.exp(np.clip(margin, -60.0, 60.0)))

        grad_u = dmargin[:, None] * (i - j) * global_scale
        grad_i = dmargin[:, None] * u * global_scale
        grad_j = -dmargin[:, None] * u * global_scale

        np.add.at(grad_final_users, ub, grad_u)
        np.add.at(grad_final_items, ib, grad_i)
        np.add.at(grad_final_items, jb, grad_j)

        if regularization:
            np.add.at(
                reg_user,
                ub,
                model.user_embeddings[ub] * global_scale,
            )
            np.add.at(
                reg_item,
                ib,
                model.item_embeddings[ib] * global_scale,
            )
            np.add.at(
                reg_item,
                jb,
                model.item_embeddings[jb] * global_scale,
            )

    grad_base_users, grad_base_items = _backprop_through_propagation(
        model,
        grad_final_users,
        grad_final_items,
    )

    if regularization:
        grad_base_users += regularization * reg_user
        grad_base_items += regularization * reg_item

    model.user_embeddings -= learning_rate * grad_base_users
    model.item_embeddings -= learning_rate * grad_base_items

    return mean_lightgcn_bpr_loss(
        model,
        triplets,
        regularization=regularization,
    )
