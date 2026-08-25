"""M1.6 inference for frozen support sensitivities.

This is a robustness layer only. It consumes the two hash-pinned M1.5 raw
artifacts, reuses frozen seed aggregation and user-bootstrap machinery, and
does not expand the primary M0.20 12-comparison family.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Any

import numpy as np
import pandas as pd

from ranklab.evaluation.primary_inference import (
    aggregate_seed_mean_per_user,
    paired_user_bootstrap_margin,
)
from ranklab.evaluation.primary_contrasts import bootstrap_cross_cell
from ranklab.evaluation.inference_protocol import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
)

EXPECTED_RAW_SHA256 = {
    "shared_tabs": "8700954a26b2c40b7abdc2a768c9e095f3ccbf360589fbd4e1f169b0ea0e85c7",
    "tab1": "e832abb6f15a1876ad695bcb2bdcd6ed01e8f67ebbe91db5024972afb2e465dd",
}

SENSITIVITY_ROLE = {
    "shared_tabs": "pre_specified_support_sensitivity",
    "tab1": "descriptive_support_sensitivity",
}

MODELS = ("popularity", "bpr", "lightgcn")
PAIRS = (
    ("popularity", "bpr"),
    ("popularity", "lightgcn"),
    ("bpr", "lightgcn"),
)


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_raw(path: str | Path, sensitivity: str) -> str:
    expected = EXPECTED_RAW_SHA256[sensitivity]
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{sensitivity} raw SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_raw(path: str | Path, sensitivity: str) -> pd.DataFrame:
    verify_raw(path, sensitivity)
    frame = pd.read_csv(path)
    required = {
        "sensitivity",
        "regime",
        "target",
        "model",
        "seed",
        "user_id",
        "ndcg_at_10",
        "included_in_macro",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing sensitivity raw columns: {sorted(missing)}")
    if set(frame["sensitivity"].dropna().unique()) != {sensitivity}:
        raise ValueError("raw artifact sensitivity label mismatch")
    return frame


def _cell_wide(seed_mean: pd.DataFrame, regime: str, target: str) -> pd.DataFrame:
    cell = seed_mean.loc[
        (seed_mean["regime"] == regime) & (seed_mean["target"] == target),
        ["user_id", "model", "ndcg_at_10"],
    ]
    wide = cell.pivot(index="user_id", columns="model", values="ndcg_at_10")
    if set(MODELS) - set(wide.columns):
        raise ValueError(f"{regime}|{target} missing model rows")
    if wide[list(MODELS)].isna().any().any():
        raise ValueError(f"{regime}|{target} has model-specific missingness")
    return wide.loc[:, list(MODELS)].sort_index()


def analyze_one(
    *,
    sensitivity: str,
    raw_path: str | Path,
    replicates: int = BOOTSTRAP_REPLICATES,
    rng_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    raw = load_raw(raw_path, sensitivity)
    seed_mean = aggregate_seed_mean_per_user(raw)

    cells: dict[str, Any] = {}
    wides: dict[tuple[str, str], pd.DataFrame] = {}

    for regime in ("standard", "randomized"):
        for target in ("is_click", "long_view"):
            cell = seed_mean.loc[
                (seed_mean["regime"] == regime) &
                (seed_mean["target"] == target)
            ]
            scores = {
                model: float(
                    cell.loc[cell["model"] == model, "ndcg_at_10"].mean()
                )
                for model in MODELS
            }
            order = sorted(MODELS, key=lambda m: (-scores[m], m))
            pairwise: dict[str, Any] = {}
            for a, b in PAIRS:
                result = paired_user_bootstrap_margin(
                    cell.loc[cell["model"] == a],
                    cell.loc[cell["model"] == b],
                    replicates=replicates,
                    rng_seed=rng_seed,
                )
                pairwise[f"{a}_minus_{b}"] = {
                    "users": result["users"],
                    "margin": result["margin"],
                    "ci95": result["ci95"],
                }

            cells[f"{regime}|{target}"] = {
                "scores": scores,
                "descriptive_order": order,
                "pairwise_95pct_only": pairwise,
            }
            wides[(regime, target)] = _cell_wide(seed_mean, regime, target)

    regime_contrasts = {}
    for target in ("is_click", "long_view"):
        regime_contrasts[target] = bootstrap_cross_cell(
            wides[("standard", target)],
            wides[("randomized", target)],
            direction="right_minus_left",
            replicates=replicates,
            rng_seed=rng_seed,
        )

    target_contrasts = {}
    for regime in ("standard", "randomized"):
        target_contrasts[regime] = bootstrap_cross_cell(
            wides[(regime, "is_click")],
            wides[(regime, "long_view")],
            direction="right_minus_left",
            replicates=replicates,
            rng_seed=rng_seed,
        )

    return {
        "role": SENSITIVITY_ROLE[sensitivity],
        "source_raw_sha256": verify_raw(raw_path, sensitivity),
        "cells": cells,
        "regime_contrasts_G_AB": regime_contrasts,
        "target_contrasts_T_AB": target_contrasts,
    }


def run_sensitivity_inference(
    *,
    shared_tabs_raw: str | Path,
    tab1_raw: str | Path,
    output_dir: str | Path,
    replicates: int = BOOTSTRAP_REPLICATES,
    rng_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    results = {
        "shared_tabs": analyze_one(
            sensitivity="shared_tabs",
            raw_path=shared_tabs_raw,
            replicates=replicates,
            rng_seed=rng_seed,
        ),
        "tab1": analyze_one(
            sensitivity="tab1",
            raw_path=tab1_raw,
            replicates=replicates,
            rng_seed=rng_seed,
        ),
    }

    manifest = {
        "status": "M1_SUPPORT_SENSITIVITY_INFERENCE",
        "bootstrap": {
            "replicates": int(replicates),
            "rng_seed": int(rng_seed),
            "confidence": CONFIDENCE_LEVEL,
            "ci_method": "nonparametric_percentile",
        },
        "multiplicity_policy": (
            "Sensitivity analyses do not expand or redefine the frozen primary "
            "12-comparison decisive-winner family. Within-cell sensitivity "
            "comparisons are reported with descriptive order and 95% paired "
            "bootstrap intervals only; no new decisive-winner label is assigned."
        ),
        "sensitivities": results,
        "guardrails": [
            "No model checkpoint is loaded.",
            "No candidate is rescored.",
            "Stochastic seeds are averaged inside user before bootstrap.",
            "shared_tabs is pre-specified; tab1 is descriptive only.",
            "No new primary hypothesis family or post-hoc practical threshold is created.",
        ],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen M1 support-sensitivity inference."
    )
    parser.add_argument(
        "--shared-tabs-raw",
        default="runs/m1/sensitivity_raw/shared_tabs/per_user_ndcg.csv.gz",
    )
    parser.add_argument(
        "--tab1-raw",
        default="runs/m1/sensitivity_raw/tab1/per_user_ndcg.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/m1/sensitivity_inference",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_sensitivity_inference(
        shared_tabs_raw=args.shared_tabs_raw,
        tab1_raw=args.tab1_raw,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
