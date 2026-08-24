from __future__ import annotations

import hashlib
import json

from ranklab.evaluation import primary_results


def _write(path, data):
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_consolidation_verifies_chain_and_writes_tables(tmp_path, monkeypatch):
    raw = tmp_path / "raw.gz"
    raw.write_bytes(b"raw")
    seed = tmp_path / "seed.gz"
    seed.write_bytes(b"seed")

    cells = {}
    for regime in ("standard", "randomized"):
        for target in ("is_click", "long_view"):
            cells[f"{regime}|{target}"] = {
                "decisive_winner": "popularity",
                "descriptive_order": ["popularity", "bpr", "lightgcn"],
                "scores": {
                    "popularity": 0.7,
                    "bpr": 0.6,
                    "lightgcn": 0.59,
                },
                "pairwise": {
                    pair: {
                        "users": 5,
                        "margin": 0.1 if pair != "bpr_minus_lightgcn" else 0.01,
                        "ci95": {"lower": -0.01, "upper": 0.12},
                        "simultaneous_ci": {"lower": -0.02, "upper": 0.13},
                    }
                    for pair in (
                        "popularity_minus_bpr",
                        "popularity_minus_lightgcn",
                        "bpr_minus_lightgcn",
                    )
                },
            }

    within = tmp_path / "within.json"
    within_data = {
        "status": "M1_PRIMARY_WITHIN_CELL_INFERENCE",
        "source_raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "cells": cells,
    }
    within_digest = _write(within, within_data)

    pair_result = {
        pair: {
            "native": {
                "point": 0.1,
                "ci95": {"lower": 0.01, "upper": 0.2},
                "left_users": 5,
                "right_users": 5,
                "union_users": 6,
            },
            "matched": {
                "point": 0.09,
                "ci95": {"lower": 0.01, "upper": 0.18},
                "users": 4,
            },
        }
        for pair in (
            "popularity_minus_bpr",
            "popularity_minus_lightgcn",
            "bpr_minus_lightgcn",
        )
    }
    cross = tmp_path / "cross.json"
    cross_data = {
        "status": "M1_PRIMARY_CROSS_CELL_CONTRASTS",
        "source_seed_mean_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
        "regime_contrasts_G_AB": {
            "is_click": pair_result,
            "long_view": pair_result,
        },
        "target_contrasts_T_AB": {
            "standard": pair_result,
            "randomized": pair_result,
        },
    }
    cross_digest = _write(cross, cross_data)

    monkeypatch.setattr(
        primary_results,
        "EXPECTED_SHA256",
        {
            "m1_raw": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "m1_seed_mean": hashlib.sha256(seed.read_bytes()).hexdigest(),
            "m1_within_manifest": within_digest,
            "m1_cross_manifest": cross_digest,
        },
    )

    out = tmp_path / "out"
    manifest = primary_results.run_consolidation(
        raw_path=raw,
        seed_mean_path=seed,
        within_manifest_path=within,
        cross_manifest_path=cross,
        output_dir=out,
    )

    assert manifest["status"] == "M1_PRIMARY_RESULTS_CONSOLIDATED"
    assert (out / "primary_cell_scores.csv").exists()
    assert (out / "within_cell_pairwise.csv").exists()
    assert (out / "cross_regime_G_AB.csv").exists()
    assert (out / "cross_target_T_AB.csv").exists()
    assert (out / "summary.json").exists()
    assert (out / "PRIMARY_RESULTS.md").exists()
