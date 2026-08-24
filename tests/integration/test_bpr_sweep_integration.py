import json

import ranklab.training.bpr_sweep as sweep


def test_sweep_manifest_selects_best_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "EMBEDDING_DIMS", (16, 32))
    monkeypatch.setattr(sweep, "LEARNING_RATES", (0.05,))
    monkeypatch.setattr(sweep, "REGULARIZATIONS", (1e-4,))
    monkeypatch.setattr(sweep, "EPOCHS", 2)

    def fake_run_bpr_pipeline(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        score = 0.7 if kwargs["embedding_dim"] == 16 else 0.8
        checkpoint = output_dir / "best_bpr.npz"
        checkpoint.write_bytes(f"checkpoint-{kwargs['embedding_dim']}".encode())
        result = {
            "hyperparameters": {
                "embedding_dim": kwargs["embedding_dim"],
                "learning_rate": kwargs["learning_rate"],
                "regularization": kwargs["regularization"],
                "epochs": kwargs["epochs"],
                "seed": kwargs["seed"],
            },
            "selection": {
                "best_epoch": 2,
                "best_validation_ndcg_at_10": score,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sweep._sha256_file(checkpoint),
            },
        }
        (output_dir / "manifest.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(sweep, "run_bpr_pipeline", fake_run_bpr_pipeline)

    summary = sweep.run_sweep(
        data_path=tmp_path / "unused.csv",
        output_dir=tmp_path / "sweep",
        repo_root=tmp_path,
    )

    assert summary["grid"]["total_configurations"] == 2
    assert summary["winner"]["configuration_id"].startswith("d32_")
    assert (tmp_path / "sweep" / "selected_bpr.npz").exists()
    assert (tmp_path / "sweep" / "sweep_manifest.json").exists()
