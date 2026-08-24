"""M1.2 frozen primary inference over validated M1.1 per-user metrics.

This module does not score models. It consumes the immutable M1.1 raw artifact,
verifies its SHA256, applies the frozen M0.20 seed aggregation, and computes
within-cell pairwise margins plus paired user-bootstrap intervals.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ranklab.evaluation.inference_protocol import (
    BONFERRONI_INDIVIDUAL_CONFIDENCE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    Interval,
    decisive_winner,
    descriptive_order,
    pairwise_margin,
)

EXPECTED_RAW_SHA256 = "25f41d6380964e0eaf0f6746a8e4df221a3c62faa105ce33a3893d92f1d4301a"
MODELS = ("popularity", "bpr", "lightgcn")
REGIMES = ("standard", "randomized")
TARGETS = ("is_click", "long_view")


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_raw_artifact(path: str | Path) -> str:
    actual = _sha256_file(path)
    if actual != EXPECTED_RAW_SHA256:
        raise RuntimeError(
            f"M1.1 raw artifact SHA256 mismatch: expected {EXPECTED_RAW_SHA256}, got {actual}"
        )
    return actual


def load_validated_raw(path: str | Path) -> pd.DataFrame:
    verify_raw_artifact(path)
    frame = pd.read_csv(path)

    required = {
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
        raise ValueError(f"raw artifact missing required columns: {sorted(missing)}")

    expected_rows = 1_006_808
    if len(frame) != expected_rows:
        raise ValueError(f"expected {expected_rows} raw rows, found {len(frame)}")

    return frame


def aggregate_seed_mean_per_user(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one NDCG row per regime/target/model/user.

    Popularity is deterministic. BPR/LightGCN require exactly five included seed
    observations per eligible user and are averaged inside user before bootstrap.
    """
    included = frame.loc[frame["included_in_macro"] == 1].copy()
    included["user_id"] = included["user_id"].astype(int)

    pieces: list[pd.DataFrame] = []

    pop = included.loc[included["model"] == "popularity", [
        "regime", "target", "model", "user_id", "ndcg_at_10"
    ]].copy()
    if pop.duplicated(["regime", "target", "model", "user_id"]).any():
        raise ValueError("duplicate deterministic popularity user rows")
    pieces.append(pop)

    stochastic = included.loc[included["model"].isin(["bpr", "lightgcn"])].copy()
    counts = (
        stochastic.groupby(["regime", "target", "model", "user_id"], sort=False)
        .size()
    )
    if not (counts == 5).all():
        bad = counts[counts != 5].head().to_dict()
        raise ValueError(f"expected exactly five stochastic seed rows per user: {bad}")

    agg = (
        stochastic.groupby(
            ["regime", "target", "model", "user_id"],
            as_index=False,
            sort=True,
        )["ndcg_at_10"]
        .mean()
    )
    pieces.append(agg)

    result = pd.concat(pieces, ignore_index=True)
    return result.sort_values(
        ["regime", "target", "model", "user_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _percentile_interval(samples: np.ndarray, confidence: float) -> Interval:
    alpha = 1.0 - float(confidence)
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Interval(lower=float(lo), upper=float(hi))


def paired_user_bootstrap_margin(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    rng_seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired bootstrap for A-B on the same eligible users."""
    left = a[["user_id", "ndcg_at_10"]].rename(columns={"ndcg_at_10": "a"})
    right = b[["user_id", "ndcg_at_10"]].rename(columns={"ndcg_at_10": "b"})
    merged = left.merge(right, on="user_id", how="inner", validate="one_to_one")

    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("within-cell model comparison must use identical eligible users")

    diffs = (merged["a"] - merged["b"]).to_numpy(dtype=np.float64)
    point = float(diffs.mean())
    rng = np.random.default_rng(rng_seed)

    # Memory bounded: accumulate one mean per replicate, not an R x N matrix.
    draws = np.empty(replicates, dtype=np.float64)
    n = len(diffs)
    for r in range(replicates):
        idx = rng.integers(0, n, size=n)
        draws[r] = float(diffs[idx].mean())

    interval95 = _percentile_interval(draws, CONFIDENCE_LEVEL)
    simultaneous = _percentile_interval(draws, BONFERRONI_INDIVIDUAL_CONFIDENCE)
    return {
        "users": n,
        "margin": point,
        "ci95": asdict(interval95),
        "simultaneous_ci": asdict(simultaneous),
    }


def run_primary_inference(
    *,
    raw_path: str | Path,
    output_dir: str | Path,
    replicates: int = BOOTSTRAP_REPLICATES,
    rng_seed: int = BOOTSTRAP_SEED,
) -> dict:
    frame = load_validated_raw(raw_path)
    aggregated = aggregate_seed_mean_per_user(frame)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cells: dict[str, dict] = {}
    pairs = (("popularity", "bpr"), ("popularity", "lightgcn"), ("bpr", "lightgcn"))

    for regime in REGIMES:
        for target in TARGETS:
            cell = aggregated.loc[
                (aggregated["regime"] == regime) &
                (aggregated["target"] == target)
            ].copy()

            scores = {
                model: float(
                    cell.loc[cell["model"] == model, "ndcg_at_10"].mean()
                )
                for model in MODELS
            }

            pair_results: dict[str, dict] = {}
            decisive_intervals: dict[tuple[str, str], Interval] = {}

            for a_name, b_name in pairs:
                a = cell.loc[cell["model"] == a_name]
                b = cell.loc[cell["model"] == b_name]
                result = paired_user_bootstrap_margin(
                    a,
                    b,
                    replicates=replicates,
                    rng_seed=rng_seed,
                )
                # Point estimate should agree with the frozen model-level difference.
                expected = pairwise_margin(scores[a_name], scores[b_name])
                if not np.isclose(result["margin"], expected, rtol=0.0, atol=1e-12):
                    raise RuntimeError("paired margin disagrees with model point estimates")

                key = f"{a_name}_minus_{b_name}"
                pair_results[key] = result
                ci = result["simultaneous_ci"]
                decisive_intervals[(a_name, b_name)] = Interval(
                    lower=float(ci["lower"]),
                    upper=float(ci["upper"]),
                )

            winner = decisive_winner(
                scores=scores,
                pair_intervals=decisive_intervals,
            )

            cells[f"{regime}|{target}"] = {
                "scores": scores,
                "descriptive_order": list(descriptive_order(scores)),
                "pairwise": pair_results,
                "decisive_winner": winner,
            }

    manifest = {
        "status": "M1_PRIMARY_WITHIN_CELL_INFERENCE",
        "source_raw_sha256": verify_raw_artifact(raw_path),
        "seed_aggregation": "mean_inside_user_over_frozen_seeds_0_to_4",
        "bootstrap": {
            "replicates": int(replicates),
            "rng_seed": int(rng_seed),
            "ci_method": "nonparametric_percentile",
            "base_confidence": CONFIDENCE_LEVEL,
            "decisive_confidence": BONFERRONI_INDIVIDUAL_CONFIDENCE,
            "multiplicity_family": 12,
        },
        "cells": cells,
        "guardrails": [
            "Reads M1.1 per-user NDCG only; no model checkpoint is loaded.",
            "No model is trained, retuned, or rescored.",
            "Stochastic seeds are averaged inside user before bootstrap.",
            "Within-cell model comparisons use paired eligible users.",
            "Cross-regime G_AB and cross-target T_AB are not computed in M1.2.",
        ],
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    aggregated.to_csv(output_dir / "per_user_seed_mean_ndcg.csv.gz", index=False)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen M1.2 within-cell seed aggregation and inference."
    )
    parser.add_argument(
        "--raw",
        default="runs/m1/primary_raw/per_user_ndcg.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/m1/primary_inference",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_primary_inference(
        raw_path=args.raw,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
