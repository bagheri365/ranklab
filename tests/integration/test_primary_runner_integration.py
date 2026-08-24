from __future__ import annotations

import csv
import gzip
import hashlib
from pathlib import Path

import pandas as pd
import yaml

from ranklab.evaluation import primary_runner
from ranklab.models.bpr_mf import BPRMatrixFactorization
from ranklab.models.lightgcn_core import LightGCNModel


def _write_eval(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_training(path: Path) -> None:
    rows = []
    for user in (1, 2):
        rows.extend(
            [
                {"user_id": user, "video_id": 10, "date": "20220409", "is_click": 1},
                {"user_id": user, "video_id": 20, "date": "20220410", "is_click": 0},
            ]
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_primary_raw_runner_writes_per_user_records_without_inference(tmp_path, monkeypatch):
    training = tmp_path / "training.csv"
    _write_training(training)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {"user_id": 1, "video_id": 10, "tab": 1, "is_click": 1, "long_view": 1},
        {"user_id": 1, "video_id": 20, "tab": 1, "is_click": 0, "long_view": 0},
        {"user_id": 2, "video_id": 10, "tab": 1, "is_click": 0, "long_view": 0},
        {"user_id": 2, "video_id": 20, "tab": 1, "is_click": 1, "long_view": 0},
    ]
    _write_eval(data_dir / "log_standard_4_22_to_5_08_pure.csv", rows)
    _write_eval(data_dir / "log_random_4_22_to_5_08_pure.csv", rows)

    repo = tmp_path / "repo"
    (repo / "research").mkdir(parents=True)
    (repo / "configs/experiments/m1").mkdir(parents=True)
    protocol_bytes = b"fixture frozen protocol\n"
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    (repo / "research/protocol_frozen_m0.yaml").write_bytes(protocol_bytes)
    (repo / "research/protocol_frozen_m0.sha256").write_text(protocol_sha)
    (repo / "configs/experiments/m1/primary.yaml").write_text(
        yaml.safe_dump(
            {"protocol_sha256": protocol_sha, "status": "PROTOCOL_READY_NOT_RUN"}
        )
    )
    monkeypatch.setattr(primary_runner, "EXPECTED_PROTOCOL_SHA256", protocol_sha)

    primary_dir = tmp_path / "primary"
    (primary_dir / "bpr").mkdir(parents=True)
    (primary_dir / "lightgcn").mkdir(parents=True)

    expected = {"bpr": {}, "lightgcn": {}}
    for seed in range(5):
        bpr = BPRMatrixFactorization.initialize([1, 2], [10, 20], embedding_dim=2, seed=seed)
        bpr_path = primary_dir / "bpr" / f"bpr_seed{seed}.npz"
        bpr.save(bpr_path)
        expected["bpr"][seed] = hashlib.sha256(bpr_path.read_bytes()).hexdigest()

        lgcn = LightGCNModel.initialize(
            user_ids=[1, 2],
            item_ids=[10, 20],
            positive_pairs=[(1, 10), (2, 10)],
            embedding_dim=2,
            num_layers=1,
            seed=seed,
        )
        lgcn_path = primary_dir / "lightgcn" / f"lightgcn_seed{seed}.npz"
        lgcn.save(lgcn_path)
        expected["lightgcn"][seed] = hashlib.sha256(lgcn_path.read_bytes()).hexdigest()

    monkeypatch.setattr(primary_runner, "EXPECTED_CHECKPOINT_SHA256", expected)

    out = tmp_path / "out"
    manifest = primary_runner.run_primary_raw_evaluation(
        training_path=training,
        data_dir=data_dir,
        primary_dir=primary_dir,
        output_dir=out,
        repo_root=repo,
    )

    assert manifest["status"] == "M1_PRIMARY_RAW_EVALUATION"
    assert manifest["interpretation_status"] == "RAW_METRICS_ONLY_NO_WINNER_OR_CONTRAST_CLAIMS"
    assert manifest["candidate_population"]["standard"] == {"users": 2, "pairs": 4}
    assert manifest["candidate_population"]["randomized"] == {"users": 2, "pairs": 4}

    with gzip.open(out / "per_user_ndcg.csv.gz", "rt", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))

    # 2 regimes * 2 users * 2 targets * (1 popularity + 5 BPR + 5 LightGCN)
    assert len(records) == 88
    assert {row["model"] for row in records} == {"popularity", "bpr", "lightgcn"}
    assert not any("winner" in key.lower() for key in manifest)
