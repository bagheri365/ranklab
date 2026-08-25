from __future__ import annotations

import hashlib
import json

from ranklab.evaluation import final_results


def _write(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sensitivity(role):
    cells = {}
    for regime in ("standard", "randomized"):
        for target in ("is_click", "long_view"):
            cells[f"{regime}|{target}"] = {
                "descriptive_order": ["popularity", "bpr", "lightgcn"],
                "pairwise_95pct_only": {
                    "bpr_minus_lightgcn": {
                        "margin": 0.0,
                        "ci95": {"lower": -0.1, "upper": 0.1},
                    }
                },
            }
    pair = {
        "popularity_minus_bpr": {
            "native": {"point": 0.1, "ci95": {"lower": 0.05, "upper": 0.15}},
            "matched": {"point": 0.09, "ci95": {"lower": 0.04, "upper": 0.14}},
        }
    }
    return {
        "role": role,
        "cells": cells,
        "regime_contrasts_G_AB": {"is_click": pair, "long_view": pair},
        "target_contrasts_T_AB": {"standard": pair, "randomized": pair},
    }


def test_final_consolidation_writes_documentary_outputs(tmp_path, monkeypatch):
    primary_summary = tmp_path / "primary_summary.json"
    primary_manifest = tmp_path / "primary_manifest.json"
    sensitivity_manifest = tmp_path / "sensitivity_manifest.json"

    primary = {
        "status": "M1_PRIMARY_RESULTS_CONSOLIDATED",
        "winner_identity_stable": True,
        "stable_decisive_winner": "popularity",
        "bpr_vs_lightgcn": {},
        "cross_regime_G_AB": {},
        "cross_target_T_AB": {},
    }
    primary_digest = _write(primary_summary, primary)

    primary_manifest_payload = {
        "status": "M1_PRIMARY_RESULTS_CONSOLIDATED",
    }
    primary_manifest_digest = _write(primary_manifest, primary_manifest_payload)

    sensitivity_payload = {
        "status": "M1_SUPPORT_SENSITIVITY_INFERENCE",
        "sensitivities": {
            "shared_tabs": _sensitivity("pre_specified_support_sensitivity"),
            "tab1": _sensitivity("descriptive_support_sensitivity"),
        },
    }
    sensitivity_digest = _write(sensitivity_manifest, sensitivity_payload)

    monkeypatch.setattr(
        final_results,
        "EXPECTED_SHA256",
        {
            "primary_summary": primary_digest,
            "primary_manifest": primary_manifest_digest,
            "sensitivity_manifest": sensitivity_digest,
        },
    )

    out = tmp_path / "out"
    manifest = final_results.run_final_consolidation(
        primary_summary_path=primary_summary,
        primary_manifest_path=primary_manifest,
        sensitivity_manifest_path=sensitivity_manifest,
        output_dir=out,
    )

    assert manifest["status"] == "M1_FINAL_RESULTS_CONSOLIDATED"
    assert (out / "summary.json").exists()
    assert (out / "FINAL_RESULTS.md").exists()
    assert (out / "manifest.json").exists()
