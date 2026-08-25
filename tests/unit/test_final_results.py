from __future__ import annotations

from ranklab.evaluation.final_results import summarize_sensitivity


def test_sensitivity_summary_detects_common_descriptive_first_model():
    sens = {
        "role": "pre_specified_support_sensitivity",
        "cells": {
            f"{regime}|{target}": {
                "descriptive_order": ["popularity", "bpr", "lightgcn"],
                "pairwise_95pct_only": {
                    "bpr_minus_lightgcn": {
                        "margin": 0.0,
                        "ci95": {"lower": -0.1, "upper": 0.1},
                    }
                },
            }
            for regime in ("standard", "randomized")
            for target in ("is_click", "long_view")
        },
        "regime_contrasts_G_AB": {
            target: {
                "popularity_minus_bpr": {
                    "native": {"point": 0.1, "ci95": {"lower": 0.05, "upper": 0.15}},
                    "matched": {"point": 0.09, "ci95": {"lower": 0.04, "upper": 0.14}},
                }
            }
            for target in ("is_click", "long_view")
        },
        "target_contrasts_T_AB": {
            regime: {
                "popularity_minus_bpr": {
                    "native": {"point": 0.01, "ci95": {"lower": 0.0, "upper": 0.02}},
                    "matched": {"point": 0.005, "ci95": {"lower": 0.0, "upper": 0.01}},
                }
            }
            for regime in ("standard", "randomized")
        },
    }
    out = summarize_sensitivity(sens)
    assert out["all_cells_same_descriptive_first_model"] is True
    assert all(
        not x["ci95_excludes_zero"]
        for x in out["bpr_vs_lightgcn_within_cell"].values()
    )
