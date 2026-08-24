import pytest

from ranklab.training.lightgcn_sweep import (
    EMBEDDING_DIMS,
    LEARNING_RATES,
    NUM_LAYERS,
    _selection_key,
    config_name,
    iter_grid,
    run_lightgcn_sweep,
)


def test_lightgcn_grid_is_exactly_27_predeclared_configs():
    grid = iter_grid()
    assert len(grid) == 27
    assert EMBEDDING_DIMS == (16, 32, 64)
    assert NUM_LAYERS == (1, 2, 3)
    assert LEARNING_RATES == (0.5, 1.0, 2.0)
    assert config_name(32, 2, 1.0) == "d32_l2_lr1"


def test_lightgcn_selection_key_prefers_simpler_exact_tie():
    rows = [
        {
            "best_validation_ndcg_at_10": 0.8,
            "num_layers": 2,
            "embedding_dim": 16,
            "learning_rate": 0.5,
            "best_epoch": 1,
        },
        {
            "best_validation_ndcg_at_10": 0.8,
            "num_layers": 1,
            "embedding_dim": 64,
            "learning_rate": 2.0,
            "best_epoch": 5,
        },
    ]
    assert sorted(rows, key=_selection_key)[0]["num_layers"] == 1


def test_lightgcn_sweep_rejects_posthoc_grid_change(tmp_path):
    with pytest.raises(ValueError, match="frozen"):
        run_lightgcn_sweep(
            data_path=tmp_path / "missing.csv",
            output_dir=tmp_path / "out",
            grid=[(32, 1, 1.0)],
        )
