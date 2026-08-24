"""Pairwise optimization and validation helpers for the LightGCN core."""

from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np

from ranklab.models.lightgcn_core import LightGCNModel
from ranklab.training.bpr_engine import ndcg_at_k


Id = Hashable
Triplet = tuple[Id, Id, Id]


def mean_lightgcn_bpr_loss(
    model: LightGCNModel,
    triplets: Sequence[Triplet],
    *,
    regularization: float = 0.0,
) -> float:
    if not triplets:
        raise ValueError("at least one triplet is required")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")

    user_map = model.user_index.to_index
    item_map = model.item_index.to_index
    u_idx = np.fromiter((user_map[u] for u, _, _ in triplets), dtype=np.int64)
    i_idx = np.fromiter((item_map[i] for _, i, _ in triplets), dtype=np.int64)
    j_idx = np.fromiter((item_map[j] for _, _, j in triplets), dtype=np.int64)

    final_users, final_items = model.final_embeddings()
    u = final_users[u_idx]
    i = final_items[i_idx]
    j = final_items[j_idx]

    margin = np.sum(u * (i - j), axis=1)
    loss = float(np.mean(np.logaddexp(0.0, -margin)))

    if regularization:
        # Standard LightGCN regularization is applied to base embeddings of the
        # sampled ids, not propagated embeddings.
        base_u = model.user_embeddings[u_idx]
        base_i = model.item_embeddings[i_idx]
        base_j = model.item_embeddings[j_idx]
        loss += 0.5 * regularization * float(
            np.mean(
                np.sum(base_u * base_u, axis=1)
                + np.sum(base_i * base_i, axis=1)
                + np.sum(base_j * base_j, axis=1)
            )
        )
    return loss


def _backprop_through_propagation(
    model: LightGCNModel,
    grad_final_users: np.ndarray,
    grad_final_items: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-propagate final-embedding gradients to layer-0 embeddings.

    Propagation is linear and symmetric. If E^(l+1) = A_hat E^l and the final
    representation is the mean of all layers, then each layer receives the
    direct mean gradient plus the reverse-propagated gradient from layer l+1.
    """

    scale = 1.0 / (model.num_layers + 1)
    grad_layers_users = [
        grad_final_users * scale for _ in range(model.num_layers + 1)
    ]
    grad_layers_items = [
        grad_final_items * scale for _ in range(model.num_layers + 1)
    ]

    for layer in range(model.num_layers, 0, -1):
        gu_next = grad_layers_users[layer]
        gi_next = grad_layers_items[layer]

        gu_prev = np.zeros_like(grad_final_users)
        gi_prev = np.zeros_like(grad_final_items)

        # Forward:
        # user_next[u] += w * item_prev[i]
        # item_next[i] += w * user_prev[u]
        # Reverse therefore uses the same normalized edge weights.
        weighted_gi = model.graph.edge_weights[:, None] * gi_next[model.graph.item_indices]
        weighted_gu = model.graph.edge_weights[:, None] * gu_next[model.graph.user_indices]
        np.add.at(gu_prev, model.graph.user_indices, weighted_gi)
        np.add.at(gi_prev, model.graph.item_indices, weighted_gu)

        grad_layers_users[layer - 1] += gu_prev
        grad_layers_items[layer - 1] += gi_prev

    return grad_layers_users[0], grad_layers_items[0]


def train_lightgcn_epoch_full_batch(
    model: LightGCNModel,
    triplets: Sequence[Triplet],
    *,
    learning_rate: float,
    regularization: float,
) -> float:
    """Run one deterministic full-batch LightGCN+BPR update.

    M0.9a intentionally starts with a transparent full-batch implementation.
    KuaiRand-scale batching/performance is addressed separately in M0.9b so the
    numerical semantics can be validated first.
    """

    if not triplets:
        raise ValueError("at least one triplet is required")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")

    user_map = model.user_index.to_index
    item_map = model.item_index.to_index
    u_idx = np.fromiter((user_map[u] for u, _, _ in triplets), dtype=np.int64)
    i_idx = np.fromiter((item_map[i] for _, i, _ in triplets), dtype=np.int64)
    j_idx = np.fromiter((item_map[j] for _, _, j in triplets), dtype=np.int64)

    final_users, final_items = model.final_embeddings()
    u = final_users[u_idx]
    i = final_items[i_idx]
    j = final_items[j_idx]

    margin = np.sum(u * (i - j), axis=1)
    dmargin = -1.0 / (1.0 + np.exp(np.clip(margin, -60.0, 60.0)))
    scale = 1.0 / len(triplets)

    grad_final_users = np.zeros_like(final_users)
    grad_final_items = np.zeros_like(final_items)

    grad_u = dmargin[:, None] * (i - j) * scale
    grad_i = dmargin[:, None] * u * scale
    grad_j = -dmargin[:, None] * u * scale

    np.add.at(grad_final_users, u_idx, grad_u)
    np.add.at(grad_final_items, i_idx, grad_i)
    np.add.at(grad_final_items, j_idx, grad_j)

    grad_base_users, grad_base_items = _backprop_through_propagation(
        model,
        grad_final_users,
        grad_final_items,
    )

    if regularization:
        reg_user = np.zeros_like(model.user_embeddings)
        reg_item = np.zeros_like(model.item_embeddings)
        np.add.at(reg_user, u_idx, model.user_embeddings[u_idx] * scale)
        np.add.at(reg_item, i_idx, model.item_embeddings[i_idx] * scale)
        np.add.at(reg_item, j_idx, model.item_embeddings[j_idx] * scale)
        grad_base_users += regularization * reg_user
        grad_base_items += regularization * reg_item

    model.user_embeddings -= learning_rate * grad_base_users
    model.item_embeddings -= learning_rate * grad_base_items

    return mean_lightgcn_bpr_loss(
        model,
        triplets,
        regularization=regularization,
    )


def macro_logged_ndcg_at_k(
    model: LightGCNModel,
    candidates_by_user: dict[Id, Sequence[tuple[Id, int]]],
    *,
    k: int,
) -> float:
    """Macro NDCG over known users with >=1 relevant and >=2 known candidates."""

    final_users, final_items = model.final_embeddings()
    user_map = model.user_index.to_index
    item_map = model.item_index.to_index
    values: list[float] = []

    for user_id in sorted(candidates_by_user, key=repr):
        if user_id not in user_map:
            continue

        candidates = [
            (item_id, int(relevant))
            for item_id, relevant in candidates_by_user[user_id]
            if item_id in item_map
        ]
        if len(candidates) < 2 or not any(r for _, r in candidates):
            continue

        item_indices = np.fromiter(
            (item_map[item_id] for item_id, _ in candidates),
            dtype=np.int64,
            count=len(candidates),
        )
        scores = final_items[item_indices] @ final_users[user_map[user_id]]
        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: (-float(pair[1]), repr(pair[0][0])),
        )
        values.append(ndcg_at_k([candidate[1] for candidate, _ in ranked], k=k))

    if not values:
        raise ValueError("no eligible validation users")
    return float(np.mean(values))
