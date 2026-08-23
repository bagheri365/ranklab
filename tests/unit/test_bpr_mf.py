import numpy as np

from ranklab.models.bpr_mf import BPRMatrixFactorization, IdIndex


def test_id_index_is_stable_across_input_order():
    assert IdIndex.from_values([3, 1, 2, 1]).values == IdIndex.from_values([2, 3, 1]).values


def test_bpr_initialization_is_seed_deterministic():
    left = BPRMatrixFactorization.initialize([1, 2], [10, 11, 12], embedding_dim=4, seed=7)
    right = BPRMatrixFactorization.initialize([2, 1], [12, 10, 11], embedding_dim=4, seed=7)
    assert left.user_index == right.user_index
    assert left.item_index == right.item_index
    np.testing.assert_array_equal(left.user_factors, right.user_factors)
    np.testing.assert_array_equal(left.item_factors, right.item_factors)
