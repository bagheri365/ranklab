"""Frozen-contract negative sampling, BPR optimization, and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np

from ranklab.models.bpr_mf import BPRMatrixFactorization


Id = Hashable
Pair = tuple[Id, Id]
Triplet = tuple[Id, Id, Id]


@dataclass(frozen=True)
class SamplingReport:
    positive_pairs_seen: int
    triplets_sampled: int
    positive_users_seen: int
    eligible_users: int
    excluded_users_no_logged_negative: int


def sample_same_user_logged_negatives(
    positive_pairs: Sequence[Pair],
    negative_pairs: Sequence[Pair],
    *,
    seed: int,
    epoch: int,
) -> tuple[list[Triplet], SamplingReport]:
    """Sample one same-user logged negative per positive, with replacement."""

    if epoch < 0:
        raise ValueError("epoch must be non-negative")

    negatives_by_user: dict[Id, list[Id]] = {}
    for user_id, item_id in negative_pairs:
        negatives_by_user.setdefault(user_id, []).append(item_id)
    for pool in negatives_by_user.values():
        pool.sort(key=repr)

    positive_users = {user_id for user_id, _ in positive_pairs}
    eligible_users = positive_users.intersection(negatives_by_user)

    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    triplets: list[Triplet] = []
    for user_id, positive_item_id in positive_pairs:
        pool = negatives_by_user.get(user_id)
        if not pool:
            continue
        negative_item_id = pool[int(rng.integers(0, len(pool)))]
        triplets.append((user_id, positive_item_id, negative_item_id))

    report = SamplingReport(
        positive_pairs_seen=len(positive_pairs),
        triplets_sampled=len(triplets),
        positive_users_seen=len(positive_users),
        eligible_users=len(eligible_users),
        excluded_users_no_logged_negative=len(positive_users - eligible_users),
    )
    return triplets, report


def mean_bpr_loss(
    model: BPRMatrixFactorization,
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
    users = np.fromiter((user_map[u] for u, _, _ in triplets), dtype=np.int64)
    positives = np.fromiter((item_map[i] for _, i, _ in triplets), dtype=np.int64)
    negatives = np.fromiter((item_map[j] for _, _, j in triplets), dtype=np.int64)

    u = model.user_factors[users]
    i = model.item_factors[positives]
    j = model.item_factors[negatives]
    margin = np.sum(u * (i - j), axis=1)
    loss = float(np.mean(np.logaddexp(0.0, -margin)))

    if regularization:
        loss += 0.5 * regularization * float(
            np.mean(
                np.sum(u * u, axis=1)
                + np.sum(i * i, axis=1)
                + np.sum(j * j, axis=1)
            )
        )
    return loss


def train_bpr_epoch(
    model: BPRMatrixFactorization,
    triplets: Sequence[Triplet],
    *,
    learning_rate: float,
    regularization: float,
    batch_size: int,
    seed: int,
    epoch: int,
) -> float:
    """Run one deterministic mini-batch SGD epoch."""

    if not triplets:
        raise ValueError("at least one triplet is required")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")

    user_map = model.user_index.to_index
    item_map = model.item_index.to_index
    users = np.fromiter((user_map[u] for u, _, _ in triplets), dtype=np.int64)
    positives = np.fromiter((item_map[i] for _, i, _ in triplets), dtype=np.int64)
    negatives = np.fromiter((item_map[j] for _, _, j in triplets), dtype=np.int64)

    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 1]))
    order = rng.permutation(len(triplets))

    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        u_idx = users[batch]
        i_idx = positives[batch]
        j_idx = negatives[batch]

        u = model.user_factors[u_idx].copy()
        i = model.item_factors[i_idx].copy()
        j = model.item_factors[j_idx].copy()

        margin = np.sum(u * (i - j), axis=1)
        dmargin = -1.0 / (1.0 + np.exp(np.clip(margin, -60.0, 60.0)))
        scale = 1.0 / len(batch)

        grad_u = dmargin[:, None] * (i - j) + regularization * u
        grad_i = dmargin[:, None] * u + regularization * i
        grad_j = -dmargin[:, None] * u + regularization * j

        np.add.at(model.user_factors, u_idx, -learning_rate * scale * grad_u)
        np.add.at(model.item_factors, i_idx, -learning_rate * scale * grad_i)
        np.add.at(model.item_factors, j_idx, -learning_rate * scale * grad_j)

    return mean_bpr_loss(model, triplets, regularization=regularization)


def ndcg_at_k(relevances: Sequence[int], *, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevances:
        return 0.0

    observed = np.asarray(relevances[:k], dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, 2 + len(observed), dtype=np.float64))
    dcg = float(np.sum(observed * discounts))

    ideal_count = min(int(np.sum(np.asarray(relevances) > 0)), k, len(relevances))
    if ideal_count == 0:
        return 0.0
    idcg = float(np.sum(discounts[:ideal_count]))
    return dcg / idcg


def macro_logged_ndcg_at_k(
    model: BPRMatrixFactorization,
    candidates_by_user: dict[Id, Sequence[tuple[Id, int]]],
    *,
    k: int,
) -> float:
    """Macro NDCG over known users with >=1 relevant and >=2 known candidates."""

    values: list[float] = []
    user_map = model.user_index.to_index
    item_map = model.item_index.to_index

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

        item_ids = [item_id for item_id, _ in candidates]
        item_indices = np.fromiter(
            (item_map[item_id] for item_id in item_ids),
            dtype=np.int64,
            count=len(item_ids),
        )
        scores = model.item_factors[item_indices] @ model.user_factors[user_map[user_id]]
        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: (-float(pair[1]), repr(pair[0][0])),
        )
        values.append(ndcg_at_k([c[1] for c, _ in ranked], k=k))

    if not values:
        raise ValueError("no eligible validation users")
    return float(np.mean(values))
