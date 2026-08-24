from ranklab.models.lightgcn_core import LightGCNModel
from ranklab.training.bpr_engine import sample_same_user_logged_negatives
from ranklab.training.lightgcn_engine import (
    mean_lightgcn_bpr_loss,
    train_lightgcn_epoch_full_batch,
)


def test_lightgcn_full_batch_training_reduces_toy_bpr_loss():
    positives = [(1, 10), (1, 11), (2, 12), (2, 13)]
    negatives = [(1, 20), (1, 21), (2, 20), (2, 21)]
    triplets, _ = sample_same_user_logged_negatives(
        positives,
        negatives,
        seed=3,
        epoch=0,
    )

    model = LightGCNModel.initialize(
        user_ids=[1, 2],
        item_ids=[10, 11, 12, 13, 20, 21],
        positive_pairs=positives,
        embedding_dim=8,
        num_layers=1,
        seed=9,
        init_std=0.05,
    )

    before = mean_lightgcn_bpr_loss(model, triplets)

    for _ in range(100):
        train_lightgcn_epoch_full_batch(
            model,
            triplets,
            learning_rate=1.0,
            regularization=0.0,
        )

    after = mean_lightgcn_bpr_loss(model, triplets)
    assert after < before
