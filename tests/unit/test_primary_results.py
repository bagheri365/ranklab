from __future__ import annotations

from ranklab.evaluation.primary_results import (
    build_cell_score_rows,
    build_summary,
)


def _within():
    cells = {}
    for regime in ("standard", "randomized"):
        for target in ("is_click", "long_view"):
            cells[f"{regime}|{target}"] = {
                "decisive_winner": "popularity",
                "descriptive_order": ["popularity", "bpr", "lightgcn"],
                "scores": {
                    "popularity": 0.7,
                    "bpr": 0.6,
                    "lightgcn": 0.599,
                },
                "pairwise": {
                    "popularity_minus_bpr": {
                        "users": 10,
                        "margin": 0.1,
                        "ci95": {"lower": 0.08, "upper": 0.12},
                        "simultaneous_ci": {"lower": 0.07, "upper": 0.13},
                    },
                    "popularity_minus_lightgcn": {
                        "users": 10,
                        "margin": 0.101,
                        "ci95": {"lower": 0.08, "upper": 0.12},
                        "simultaneous_ci": {"lower": 0.07, "upper": 0.13},
                    },
                    "bpr_minus_lightgcn": {
                        "users": 10,
                        "margin": 0.001,
                        "ci95": {"lower": -0.01, "upper": 0.01},
                        "simultaneous_ci": {"lower": -0.02, "upper": 0.02},
                    },
                },
            }
    return {"cells": cells}


def _cross():
    return {
        "regime_contrasts_G_AB": {"is_click": {}, "long_view": {}},
        "target_contrasts_T_AB": {"standard": {}, "randomized": {}},
    }


def test_cell_score_rows_have_twelve_rows_and_ranks():
    rows = build_cell_score_rows(_within())
    assert len(rows) == 12
    first = rows[0]
    assert first["model"] == "popularity"
    assert first["descriptive_rank"] == 1


def test_summary_reports_stable_winner_and_unresolved_bpr_lightgcn():
    summary = build_summary(_within(), _cross())
    assert summary["stable_decisive_winner"] == "popularity"
    assert summary["winner_identity_stable"] is True
    assert all(
        not item["decisive_pair_separation"]
        for item in summary["bpr_vs_lightgcn"].values()
    )
