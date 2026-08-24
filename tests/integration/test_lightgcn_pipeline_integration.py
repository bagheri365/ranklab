import csv
from hashlib import sha256

from ranklab.training.lightgcn_pipeline import run_lightgcn_pipeline


def _write_fixture(path):
    rows = []
    for user_id in (1, 2, 3):
        rows.extend(
            [
                {"user_id": user_id, "video_id": 10 + user_id, "date": "20220409", "is_click": 1},
                {"user_id": user_id, "video_id": 20 + user_id, "date": "20220410", "is_click": 0},
                {"user_id": user_id, "video_id": 10 + user_id, "date": "20220417", "is_click": 1},
                {"user_id": user_id, "video_id": 20 + user_id, "date": "20220418", "is_click": 0},
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_id", "video_id", "date", "is_click"])
        writer.writeheader()
        writer.writerows(rows)


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def test_lightgcn_pipeline_is_reproducible_on_synthetic_csv(tmp_path):
    data = tmp_path / "standard.csv"
    _write_fixture(data)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    kwargs = dict(
        data_path=data,
        embedding_dim=4,
        num_layers=1,
        learning_rate=0.5,
        regularization=1e-4,
        gradient_chunk_size=2,
        epochs=3,
        seed=17,
        repo_root=tmp_path,
    )

    first = run_lightgcn_pipeline(output_dir=first_dir, **kwargs)
    second = run_lightgcn_pipeline(output_dir=second_dir, **kwargs)

    assert first["training"]["history"] == second["training"]["history"]
    assert first["selection"]["best_epoch"] == second["selection"]["best_epoch"]
    assert first["selection"]["best_validation_ndcg_at_10"] == second["selection"]["best_validation_ndcg_at_10"]
    assert _hash(first_dir / "best_lightgcn.npz") == _hash(second_dir / "best_lightgcn.npz")
    assert first["training"]["graph_edges"] == 3
    assert first["training"]["indexed_users"] == 3
