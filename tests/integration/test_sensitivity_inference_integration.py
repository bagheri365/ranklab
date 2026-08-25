from __future__ import annotations

import hashlib

import pandas as pd

from ranklab.evaluation import sensitivity_inference


def _raw(sensitivity):
    rows = []
    for regime in ("standard", "randomized"):
        for target in ("is_click", "long_view"):
            for user in (1, 2, 3):
                rows.append(
                    {
                        "sensitivity": sensitivity,
                        "regime": regime,
                        "target": target,
                        "model": "popularity",
                        "seed": None,
                        "user_id": user,
                        "ndcg_at_10": 0.8,
                        "included_in_macro": 1,
                    }
                )
                for model, base in (("bpr", 0.6), ("lightgcn", 0.5)):
                    for seed in range(5):
                        rows.append(
                            {
                                "sensitivity": sensitivity,
                                "regime": regime,
                                "target": target,
                                "model": model,
                                "seed": seed,
                                "user_id": user,
                                "ndcg_at_10": base + seed * 0.001,
                                "included_in_macro": 1,
                            }
                        )
    return pd.DataFrame(rows)


def test_sensitivity_inference_writes_both_robustness_analyses(tmp_path, monkeypatch):
    paths = {}
    for name in ("shared_tabs", "tab1"):
        path = tmp_path / f"{name}.csv.gz"
        _raw(name).to_csv(path, index=False, compression="gzip")
        paths[name] = path
        monkeypatch.setitem(
            sensitivity_inference.EXPECTED_RAW_SHA256,
            name,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    out = tmp_path / "out"
    manifest = sensitivity_inference.run_sensitivity_inference(
        shared_tabs_raw=paths["shared_tabs"],
        tab1_raw=paths["tab1"],
        output_dir=out,
        replicates=100,
        rng_seed=7,
    )

    assert manifest["status"] == "M1_SUPPORT_SENSITIVITY_INFERENCE"
    assert set(manifest["sensitivities"]) == {"shared_tabs", "tab1"}
    assert manifest["sensitivities"]["shared_tabs"]["role"] == (
        "pre_specified_support_sensitivity"
    )
    assert manifest["sensitivities"]["tab1"]["role"] == (
        "descriptive_support_sensitivity"
    )
    assert (out / "manifest.json").exists()
