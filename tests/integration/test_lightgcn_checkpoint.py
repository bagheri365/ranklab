import numpy as np

from ranklab.models.lightgcn_core import LightGCNModel


def test_lightgcn_checkpoint_round_trip_preserves_scores(tmp_path):
    model = LightGCNModel.initialize(
        user_ids=[1, 2],
        item_ids=[10, 11, 12],
        positive_pairs=[(1, 10), (1, 11), (2, 12)],
        embedding_dim=4,
        num_layers=2,
        seed=42,
    )

    before = np.array([model.score(1, 10), model.score(1, 12), model.score(2, 12)])
    path = tmp_path / "lightgcn.npz"
    model.save(path)
    restored = LightGCNModel.load(path)
    after = np.array(
        [restored.score(1, 10), restored.score(1, 12), restored.score(2, 12)]
    )

    np.testing.assert_array_equal(before, after)
    assert restored.num_layers == 2
