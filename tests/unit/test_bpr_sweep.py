from ranklab.training.bpr_sweep import (
    choose_winner,
    configuration_id,
    frozen_grid,
)


def test_frozen_grid_has_27_unique_configurations():
    grid = frozen_grid()
    assert len(grid) == 27
    assert len({configuration_id(config) for config in grid}) == 27


def test_exact_tie_prefers_smaller_embedding_then_lower_regularization():
    larger = {
        "selection": {"best_validation_ndcg_at_10": 0.8, "best_epoch": 5},
        "hyperparameters": {
            "embedding_dim": 64,
            "learning_rate": 0.05,
            "regularization": 1e-4,
        },
        "configuration_id": "large",
    }
    smaller = {
        "selection": {"best_validation_ndcg_at_10": 0.8, "best_epoch": 5},
        "hyperparameters": {
            "embedding_dim": 16,
            "learning_rate": 0.05,
            "regularization": 1e-4,
        },
        "configuration_id": "small",
    }
    assert choose_winner([larger, smaller])["configuration_id"] == "small"
