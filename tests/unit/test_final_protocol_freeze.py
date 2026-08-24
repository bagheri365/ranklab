from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


EXPECTED_PROTOCOL_SHA256 = "9bd72ae2d3042dc9511734563b90b8668519b401b847ce218f706a465cc41a32"
EXPECTED_DATASET_MANIFEST_SHA256 = "292d78feaff8af210fe5225c2db669fc279de616debcc3c10898e5f3833e10b2"


def test_final_m0_protocol_hash_and_m1_reference_are_frozen() -> None:
    root = Path(__file__).resolve().parents[2]
    protocol_path = root / "research/protocol_frozen_m0.yaml"
    digest_path = root / "research/protocol_frozen_m0.sha256"
    m1_path = root / "configs/experiments/m1/primary.yaml"

    actual = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    recorded = digest_path.read_text(encoding="utf-8").strip()
    m1 = yaml.safe_load(m1_path.read_text(encoding="utf-8"))

    assert actual == EXPECTED_PROTOCOL_SHA256
    assert recorded == EXPECTED_PROTOCOL_SHA256
    assert m1["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256
    assert m1["status"] == "PROTOCOL_READY_NOT_RUN"


def test_final_registry_contains_no_scaffold_nulls_or_unfrozen_status() -> None:
    root = Path(__file__).resolve().parents[2]
    protocol = yaml.safe_load(
        (root / "research/protocol_frozen_m0.yaml").read_text(encoding="utf-8")
    )
    benchmark = yaml.safe_load(
        (root / "configs/benchmark.yaml").read_text(encoding="utf-8")
    )

    assert protocol["status"] == "FROZEN"
    assert benchmark["status"] == "FROZEN"
    assert protocol["dataset"]["sha256"] == EXPECTED_DATASET_MANIFEST_SHA256

    protocol_text = (root / "research/protocol_frozen_m0.yaml").read_text()
    benchmark_text = (root / "configs/benchmark.yaml").read_text()

    assert "null" not in protocol_text
    assert "UNFROZEN" not in protocol_text
    assert "null" not in benchmark_text
    assert "UNFROZEN" not in benchmark_text


def test_model_configs_use_frozen_primary_seeds_and_selected_endpoints() -> None:
    root = Path(__file__).resolve().parents[2]
    bpr = yaml.safe_load((root / "configs/models/bpr.yaml").read_text())
    lightgcn = yaml.safe_load((root / "configs/models/lightgcn.yaml").read_text())

    assert bpr["training"]["seeds"] == [0, 1, 2, 3, 4]
    assert bpr["training"]["fixed_primary_epochs"] == 1
    assert bpr["training"]["selected_dimension"] == 32
    assert bpr["training"]["selected_learning_rate"] == 0.05
    assert bpr["training"]["selected_regularization"] == 0.001

    assert lightgcn["training"]["seeds"] == [0, 1, 2, 3, 4]
    assert lightgcn["training"]["fixed_primary_epochs"] == 28
    assert lightgcn["training"]["selected_dimension"] == 32
    assert lightgcn["training"]["selected_layers"] == 2
    assert lightgcn["training"]["selected_learning_rate"] == 2.0
    assert lightgcn["training"]["selected_regularization"] == 0.0001
