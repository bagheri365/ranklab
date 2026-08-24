import pytest

from ranklab.training.lightgcn_optimization_audit import (
    LEARNING_RATES,
    learning_rate_slug,
    run_optimization_scale_audit,
)


def test_learning_rate_slug_is_stable():
    assert learning_rate_slug(0.1) == "lr0.1"
    assert learning_rate_slug(5.0) == "lr5"


def test_audit_rejects_posthoc_grid_change(tmp_path):
    with pytest.raises(ValueError, match="predeclared"):
        run_optimization_scale_audit(
            data_path=tmp_path / "missing.csv",
            output_dir=tmp_path / "out",
            learning_rates=(0.1, 1.0),
        )

    assert LEARNING_RATES == (0.1, 0.5, 1.0, 2.0, 5.0)
