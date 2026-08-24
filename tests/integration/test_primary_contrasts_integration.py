from __future__ import annotations

import hashlib

import pandas as pd

from ranklab.evaluation import primary_contrasts


def test_cross_cell_runner_writes_all_frozen_contrast_families(tmp_path, monkeypatch):
    rows = []
    eligible = {
        ("standard", "is_click"): [1, 2, 3],
        ("standard", "long_view"): [1, 2],
        ("randomized", "is_click"): [2, 3, 4],
        ("randomized", "long_view"): [2, 4],
    }
    for (regime, target), users in eligible.items():
        for user in users:
            for model, value in (
                ("popularity", 0.8),
                ("bpr", 0.6),
                ("lightgcn", 0.5),
            ):
                rows.append(
                    {
                        "regime": regime,
                        "target": target,
                        "model": model,
                        "user_id": user,
                        "ndcg_at_10": value,
                    }
                )

    path = tmp_path / "seed_mean.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(
        primary_contrasts,
        "EXPECTED_SEED_MEAN_SHA256",
        digest,
    )
    monkeypatch.setattr(
        primary_contrasts,
        "load_seed_mean_artifact",
        lambda p: pd.read_csv(p),
    )

    out = tmp_path / "out"
    manifest = primary_contrasts.run_cross_cell_contrasts(
        seed_mean_path=path,
        output_dir=out,
        replicates=100,
        rng_seed=7,
    )

    assert manifest["status"] == "M1_PRIMARY_CROSS_CELL_CONTRASTS"
    assert set(manifest["regime_contrasts_G_AB"]) == {"is_click", "long_view"}
    assert set(manifest["target_contrasts_T_AB"]) == {"standard", "randomized"}
    assert set(
        manifest["regime_contrasts_G_AB"]["is_click"]
    ) == {
        "popularity_minus_bpr",
        "popularity_minus_lightgcn",
        "bpr_minus_lightgcn",
    }
    assert (out / "manifest.json").exists()
