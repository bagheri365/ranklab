from __future__ import annotations

import hashlib

import pandas as pd

from ranklab.evaluation import primary_inference


def test_primary_inference_reads_raw_only_and_writes_manifest(tmp_path, monkeypatch):
    rows = []
    for regime in ("standard", "randomized"):
        for target in ("is_click", "long_view"):
            for user in (1, 2, 3):
                rows.append({
                    "regime": regime, "target": target, "model": "popularity",
                    "seed": None, "user_id": user, "ndcg_at_10": 0.8,
                    "included_in_macro": 1,
                })
                for model, base in (("bpr", 0.6), ("lightgcn", 0.5)):
                    for seed in range(5):
                        rows.append({
                            "regime": regime, "target": target, "model": model,
                            "seed": seed, "user_id": user,
                            "ndcg_at_10": base + seed * 0.001,
                            "included_in_macro": 1,
                        })

    raw = tmp_path / "raw.csv.gz"
    pd.DataFrame(rows).to_csv(raw, index=False, compression="gzip")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    monkeypatch.setattr(primary_inference, "EXPECTED_RAW_SHA256", digest)

    # This fixture intentionally has fewer rows than the real artifact.
    monkeypatch.setattr(primary_inference, "load_validated_raw", lambda path: pd.read_csv(path))

    out = tmp_path / "out"
    manifest = primary_inference.run_primary_inference(
        raw_path=raw,
        output_dir=out,
        replicates=200,
        rng_seed=7,
    )
    assert manifest["status"] == "M1_PRIMARY_WITHIN_CELL_INFERENCE"
    assert set(manifest["cells"]) == {
        "standard|is_click",
        "standard|long_view",
        "randomized|is_click",
        "randomized|long_view",
    }
    assert all(
        cell["decisive_winner"] == "popularity"
        for cell in manifest["cells"].values()
    )
    assert (out / "manifest.json").exists()
    assert (out / "per_user_seed_mean_ndcg.csv.gz").exists()
