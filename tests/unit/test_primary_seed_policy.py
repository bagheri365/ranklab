import pytest

from ranklab.training.primary_seed_policy import (
    PRIMARY_SEEDS,
    BPR_SELECTED,
    LIGHTGCN_SELECTED,
    bpr_output_name,
    lightgcn_output_name,
    validate_primary_seed,
)


def test_primary_seed_set_and_fixed_epochs_are_frozen():
    assert PRIMARY_SEEDS == (0, 1, 2, 3, 4)
    assert BPR_SELECTED["fixed_epochs"] == 1
    assert LIGHTGCN_SELECTED["fixed_epochs"] == 28


def test_primary_checkpoint_names_are_deterministic():
    assert bpr_output_name(3) == "bpr_seed3.npz"
    assert lightgcn_output_name(4) == "lightgcn_seed4.npz"


def test_nonprimary_seed_is_rejected():
    with pytest.raises(ValueError, match="frozen seed set"):
        validate_primary_seed(5)
