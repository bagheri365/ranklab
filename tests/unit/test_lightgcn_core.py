import numpy as np

from ranklab.models.lightgcn_core import LightGCNModel


def test_lightgcn_initialization_is_deterministic_and_collapses_duplicate_edges():
    pairs = [(1, 10), (1, 10), (1, 11), (2, 11)]
    left = LightGCNModel.initialize(
        user_ids=[1, 2],
        item_ids=[10, 11],
        positive_pairs=pairs,
        embedding_dim=4,
        num_layers=2,
        seed=7,
    )
    right = LightGCNModel.initialize(
        user_ids=[2, 1],
        item_ids=[11, 10],
        positive_pairs=list(reversed(pairs)),
        embedding_dim=4,
        num_layers=2,
        seed=7,
    )

    assert len(left.graph.user_indices) == 3
    np.testing.assert_array_equal(left.user_embeddings, right.user_embeddings)
    np.testing.assert_array_equal(left.item_embeddings, right.item_embeddings)
    np.testing.assert_array_equal(left.graph.user_indices, right.graph.user_indices)
    np.testing.assert_array_equal(left.graph.item_indices, right.graph.item_indices)


def test_one_layer_propagation_matches_hand_calculation():
    model = LightGCNModel.initialize(
        user_ids=[1],
        item_ids=[10],
        positive_pairs=[(1, 10)],
        embedding_dim=2,
        num_layers=1,
        seed=1,
    )
    model.user_embeddings[:] = np.array([[2.0, 0.0]])
    model.item_embeddings[:] = np.array([[0.0, 4.0]])

    layers = model.propagated_layers()
    np.testing.assert_array_equal(layers[1][0], np.array([[0.0, 4.0]]))
    np.testing.assert_array_equal(layers[1][1], np.array([[2.0, 0.0]]))

    users, items = model.final_embeddings()
    np.testing.assert_array_equal(users, np.array([[1.0, 2.0]]))
    np.testing.assert_array_equal(items, np.array([[1.0, 2.0]]))
