from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ranklab.evaluation.primary_runner import (
    BPRScorer,
    CachedLightGCNScorer,
    PopularityScorer,
    verify_checkpoint,
    verify_protocol,
)
from ranklab.models.bpr_mf import BPRMatrixFactorization
from ranklab.models.lightgcn_core import LightGCNModel


def test_verify_protocol_accepts_exact_frozen_bytes(tmp_path):
    protocol = b"frozen protocol\n"
    digest = hashlib.sha256(protocol).hexdigest()

    research = tmp_path / "research"
    config_dir = tmp_path / "configs/experiments/m1"
    research.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    (research / "protocol_frozen_m0.yaml").write_bytes(protocol)
    (research / "protocol_frozen_m0.sha256").write_text(digest)
    (config_dir / "primary.yaml").write_text(
        yaml.safe_dump(
            {"protocol_sha256": digest, "status": "PROTOCOL_READY_NOT_RUN"}
        )
    )

    import ranklab.evaluation.primary_runner as module
    old = module.EXPECTED_PROTOCOL_SHA256
    module.EXPECTED_PROTOCOL_SHA256 = digest
    try:
        assert verify_protocol(tmp_path) == digest
    finally:
        module.EXPECTED_PROTOCOL_SHA256 = old


def test_verify_checkpoint_rejects_drift(tmp_path):
    path = tmp_path / "checkpoint.npz"
    path.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="checkpoint SHA256 mismatch"):
        verify_checkpoint(path, model="bpr", seed=0)


def test_bpr_and_cached_lightgcn_scorers_match_core_models():
    pairs = [(1, 10), (1, 20), (2, 10)]
    bpr = BPRMatrixFactorization.initialize([1, 2], [10, 20], embedding_dim=3, seed=0)
    assert np.allclose(
        BPRScorer(bpr).score_items(1, [10, 20]),
        bpr.score_items(1, [10, 20]),
    )

    lightgcn = LightGCNModel.initialize(
        user_ids=[1, 2],
        item_ids=[10, 20],
        positive_pairs=pairs,
        embedding_dim=3,
        num_layers=2,
        seed=0,
    )
    cached = CachedLightGCNScorer(lightgcn)
    assert np.allclose(
        cached.score_items(1, [10, 20]),
        lightgcn.score_items(1, [10, 20]),
    )
