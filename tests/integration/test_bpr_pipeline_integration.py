import csv
from hashlib import sha256
import json

from ranklab.training.bpr_pipeline import run_bpr_pipeline


def _write_fixture(path):
    rows = []
    # Three train users, each with one clicked and one non-clicked item.
    for user_id in (1, 2, 3):
        rows.extend(
            [
                {
                    "user_id": user_id,
                    "video_id": 10 + user_id,
                    "date": "20220409",
                    "is_click": 1,
                },
                {
                    "user_id": user_id,
                    "video_id": 20 + user_id,
                    "date": "20220410",
                    "is_click": 0,
                },
            ]
        )
        # Validation uses known training items and >=2 candidates/user.
        rows.extend(
            [
                {
                    "user_id": user_id,
                    "video_id": 10 + user_id,
                    "date": "20220417",
                    "is_click": 1,
                },
                {
                    "user_id": user_id,
                    "video_id": 20 + user_id,
                    "date": "20220418",
                    "is_click": 0,
                },
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["user_id", "video_id", "date", "is_click"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def test_real_data_runner_is_reproducible_on_synthetic_csv(tmp_path):
    data = tmp_path / "standard.csv"
    _write_fixture(data)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = run_bpr_pipeline(
        data_path=data,
        output_dir=first_dir,
        embedding_dim=4,
        learning_rate=0.2,
        regularization=1e-4,
        batch_size=2,
        epochs=3,
        seed=17,
        repo_root=tmp_path,
    )
    second = run_bpr_pipeline(
        data_path=data,
        output_dir=second_dir,
        embedding_dim=4,
        learning_rate=0.2,
        regularization=1e-4,
        batch_size=2,
        epochs=3,
        seed=17,
        repo_root=tmp_path,
    )

    assert first["training"]["history"] == second["training"]["history"]
    assert first["selection"]["best_epoch"] == second["selection"]["best_epoch"]
    assert first["selection"]["best_validation_ndcg_at_10"] == second["selection"]["best_validation_ndcg_at_10"]
    assert _file_hash(first_dir / "best_bpr.npz") == _file_hash(second_dir / "best_bpr.npz")

    stored = json.loads((first_dir / "manifest.json").read_text())
    assert stored["status"] == "M0_BPR_PIPELINE_SMOKE_ONLY"
    assert stored["selection"]["checkpoint_sha256"] == _file_hash(first_dir / "best_bpr.npz")
    assert stored["stats"]["training_pairwise_eligible_users"] == 3
