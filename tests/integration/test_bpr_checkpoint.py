import numpy as np

from ranklab.models.bpr_mf import BPRMatrixFactorization
from ranklab.training.bpr_engine import sample_same_user_logged_negatives, train_bpr_epoch


def test_bpr_checkpoint_round_trip_preserves_scores(tmp_path):
    positives = [(1, 10), (1, 11), (2, 12)]
    negatives = [(1, 20), (2, 20)]
    model = BPRMatrixFactorization.initialize(
        [1, 2], [10, 11, 12, 20], embedding_dim=4, seed=42
    )

    for epoch in range(3):
        triplets, _ = sample_same_user_logged_negatives(
            positives, negatives, seed=42, epoch=epoch
        )
        train_bpr_epoch(
            model,
            triplets,
            learning_rate=0.2,
            regularization=1e-4,
            batch_size=2,
            seed=42,
            epoch=epoch,
        )

    before = np.array([model.score(1, 10), model.score(1, 20), model.score(2, 12)])
    path = tmp_path / "bpr.npz"
    model.save(path)
    restored = BPRMatrixFactorization.load(path)
    after = np.array([restored.score(1, 10), restored.score(1, 20), restored.score(2, 12)])
    np.testing.assert_array_equal(before, after)
