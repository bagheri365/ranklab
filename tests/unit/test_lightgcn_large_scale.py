import numpy as np

from ranklab.models.lightgcn_core import LightGCNModel
from ranklab.training.bpr_engine import sample_same_user_logged_negatives
from ranklab.training.lightgcn_engine import train_lightgcn_epoch_full_batch
from ranklab.training.lightgcn_large_scale import train_lightgcn_epoch_chunked


def test_chunked_epoch_matches_transparent_full_batch_update():
    positives = [(1, 10), (1, 11), (2, 12), (2, 13)]
    negatives = [(1, 20), (1, 21), (2, 20), (2, 21)]
    triplets, _ = sample_same_user_logged_negatives(
        positives, negatives, seed=5, epoch=0
    )

    full = LightGCNModel.initialize(
        user_ids=[1, 2],
        item_ids=[10, 11, 12, 13, 20, 21],
        positive_pairs=positives,
        embedding_dim=4,
        num_layers=1,
        seed=7,
    )
    chunked = LightGCNModel.initialize(
        user_ids=[1, 2],
        item_ids=[10, 11, 12, 13, 20, 21],
        positive_pairs=positives,
        embedding_dim=4,
        num_layers=1,
        seed=7,
    )

    train_lightgcn_epoch_full_batch(
        full,
        triplets,
        learning_rate=0.3,
        regularization=1e-4,
    )
    train_lightgcn_epoch_chunked(
        chunked,
        triplets,
        learning_rate=0.3,
        regularization=1e-4,
        gradient_chunk_size=2,
    )

    np.testing.assert_allclose(full.user_embeddings, chunked.user_embeddings, rtol=0, atol=1e-15)
    np.testing.assert_allclose(full.item_embeddings, chunked.item_embeddings, rtol=0, atol=1e-15)
