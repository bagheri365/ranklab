import numpy as np

from ranklab.models.bpr_mf import BPRMatrixFactorization
from ranklab.training.bpr_engine import (
    macro_logged_ndcg_at_k,
    mean_bpr_loss,
    sample_same_user_logged_negatives,
    train_bpr_epoch,
)


def test_same_user_negative_sampling_is_seeded_and_excludes_ineligible_users():
    positives = [(1, 10), (1, 11), (1, 12), (2, 20), (3, 30)]
    negatives = [(1, 90), (1, 91), (2, 92)]
    first, report = sample_same_user_logged_negatives(positives, negatives, seed=123, epoch=0)
    second, _ = sample_same_user_logged_negatives(
        positives, list(reversed(negatives)), seed=123, epoch=0
    )
    assert first == second
    assert len(first) == 4
    assert all(j in {90, 91} for u, _, j in first if u == 1)
    assert all(j == 92 for u, _, j in first if u == 2)
    assert report.positive_users_seen == 3
    assert report.eligible_users == 2
    assert report.excluded_users_no_logged_negative == 1


def test_sampling_with_replacement_supports_more_positives_than_unique_negatives():
    positives = [(1, 10), (1, 11), (1, 12), (1, 13)]
    negatives = [(1, 90)]
    triplets, report = sample_same_user_logged_negatives(positives, negatives, seed=5, epoch=0)
    assert triplets == [(1, 10, 90), (1, 11, 90), (1, 12, 90), (1, 13, 90)]
    assert report.triplets_sampled == 4


def test_training_reduces_pairwise_loss_on_toy_data():
    positives = [(1, 10), (1, 11), (2, 12), (2, 13)]
    negatives = [(1, 20), (1, 21), (2, 20), (2, 21)]
    baseline_triplets, _ = sample_same_user_logged_negatives(
        positives, negatives, seed=4, epoch=0
    )
    model = BPRMatrixFactorization.initialize(
        [1, 2], [10, 11, 12, 13, 20, 21], embedding_dim=8, seed=99, init_std=0.05
    )
    before = mean_bpr_loss(model, baseline_triplets)

    for epoch in range(80):
        triplets, _ = sample_same_user_logged_negatives(
            positives, negatives, seed=4, epoch=epoch
        )
        train_bpr_epoch(
            model,
            triplets,
            learning_rate=0.8,
            regularization=0.0,
            batch_size=4,
            seed=99,
            epoch=epoch,
        )

    after = mean_bpr_loss(model, baseline_triplets)
    assert after < before


def test_logged_validation_ndcg_prefers_relevant_item_when_score_is_higher():
    model = BPRMatrixFactorization.initialize([1], [10, 11], embedding_dim=2, seed=1)
    model.user_factors[:] = np.array([[1.0, 0.0]])
    model.item_factors[model.item_index.encode(10)] = np.array([2.0, 0.0])
    model.item_factors[model.item_index.encode(11)] = np.array([0.0, 0.0])

    assert macro_logged_ndcg_at_k(model, {1: [(10, 1), (11, 0)]}, k=10) == 1.0
