"""M1.3 frozen cross-regime and cross-target contrasts.

Consumes the immutable M1.2 per-user seed-mean NDCG artifact and applies the
cross-cell procedures frozen in M0.20. No model is loaded or rescored.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ranklab.evaluation.inference_protocol import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
)


EXPECTED_SEED_MEAN_SHA256 = (
    "4ca8393b25e7fc6107f7ebebd38846ed64f5b53c1191e6e58776a59ffb0f9ff8"
)

MODELS = ("popularity", "bpr", "lightgcn")
MODEL_PAIRS = (
    ("popularity", "bpr"),
    ("popularity", "lightgcn"),
    ("bpr", "lightgcn"),
)
EXPECTED_ROWS = 217_503


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seed_mean_artifact(path: str | Path) -> str:
    actual = _sha256_file(path)
    if actual != EXPECTED_SEED_MEAN_SHA256:
        raise RuntimeError(
            "M1.2 seed-mean artifact SHA256 mismatch: "
            f"expected {EXPECTED_SEED_MEAN_SHA256}, got {actual}"
        )
    return actual


def load_seed_mean_artifact(path: str | Path) -> pd.DataFrame:
    verify_seed_mean_artifact(path)
    frame = pd.read_csv(path)

    required = {"regime", "target", "model", "user_id", "ndcg_at_10"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"seed-mean artifact missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(frame)}")
    if frame.duplicated(["regime", "target", "model", "user_id"]).any():
        raise ValueError("duplicate regime/target/model/user rows in seed-mean artifact")
    return frame


def _cell_wide(
    frame: pd.DataFrame,
    *,
    regime: str,
    target: str,
) -> pd.DataFrame:
    cell = frame.loc[
        (frame["regime"] == regime) & (frame["target"] == target),
        ["user_id", "model", "ndcg_at_10"],
    ]
    wide = cell.pivot(index="user_id", columns="model", values="ndcg_at_10")
    missing_models = set(MODELS) - set(wide.columns)
    if missing_models:
        raise ValueError(
            f"{regime}|{target} missing models: {sorted(missing_models)}"
        )
    if wide[list(MODELS)].isna().any().any():
        raise ValueError(
            f"{regime}|{target} has model-specific eligibility/missing NDCG"
        )
    return wide.loc[:, list(MODELS)].sort_index()


def _pair_margin_arrays(wide: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        f"{a}_minus_{b}": (
            wide[a].to_numpy(dtype=np.float64)
            - wide[b].to_numpy(dtype=np.float64)
        )
        for a, b in MODEL_PAIRS
    }


def _percentile_interval(samples: np.ndarray) -> dict[str, float]:
    alpha = 1.0 - CONFIDENCE_LEVEL
    lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"lower": float(lower), "upper": float(upper)}


def bootstrap_cross_cell(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    direction: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    rng_seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 64,
) -> dict[str, dict]:
    """Compute native and matched contrasts for all three model pairs together.

    direction controls the reported subtraction:
      - "right_minus_left": right D_AB - left D_AB
      - "left_minus_right": left D_AB - right D_AB

    Native bootstrap samples the union of user IDs and preserves each user's
    available cell observations. Matched bootstrap uses users eligible in both
    cells. Both procedures use the same resampled union-user IDs within a
    replicate, as frozen in M0.20.
    """
    if direction not in {"right_minus_left", "left_minus_right"}:
        raise ValueError("unsupported contrast direction")

    union = left.index.union(right.index).sort_values()
    common = left.index.intersection(right.index).sort_values()
    if len(common) == 0:
        raise ValueError("no matched users across cells")

    left_aligned = left.reindex(union)
    right_aligned = right.reindex(union)

    left_margin = _pair_margin_arrays(left_aligned)
    right_margin = _pair_margin_arrays(right_aligned)

    availability_left = ~left_aligned[MODELS[0]].isna().to_numpy()
    availability_right = ~right_aligned[MODELS[0]].isna().to_numpy()
    common_mask = availability_left & availability_right

    if int(common_mask.sum()) != len(common):
        raise RuntimeError("matched-user accounting mismatch")

    sign = 1.0 if direction == "right_minus_left" else -1.0

    results: dict[str, dict] = {}
    native_draws = {
        key: np.empty(replicates, dtype=np.float64)
        for key in left_margin
    }
    matched_draws = {
        key: np.empty(replicates, dtype=np.float64)
        for key in left_margin
    }

    rng_native = np.random.default_rng(rng_seed)
    rng_matched = np.random.default_rng(rng_seed)
    n_union = len(union)
    n_common = len(common)

    # Native procedure: resample the union population. A replicate with no
    # observations from either native cell is undefined, so redraw only that
    # replicate. This is vanishingly unlikely in the real primary populations
    # but keeps the implementation mathematically total for small fixtures.
    completed = 0
    while completed < replicates:
        size = min(chunk_size, replicates - completed)
        idx = rng_native.integers(0, n_union, size=(size, n_union))
        left_avail = availability_left[idx]
        right_avail = availability_right[idx]
        valid = (left_avail.sum(axis=1) > 0) & (right_avail.sum(axis=1) > 0)
        if not valid.any():
            continue

        idx = idx[valid]
        left_avail = left_avail[valid]
        right_avail = right_avail[valid]
        take = min(len(idx), replicates - completed)
        idx = idx[:take]
        left_avail = left_avail[:take]
        right_avail = right_avail[:take]

        for key in left_margin:
            l = left_margin[key][idx]
            r = right_margin[key][idx]
            l_native = np.nansum(l, axis=1) / left_avail.sum(axis=1)
            r_native = np.nansum(r, axis=1) / right_avail.sum(axis=1)
            native_draws[key][completed:completed + take] = sign * (
                r_native - l_native
            )

        completed += take

    # Matched sensitivity: freeze to the intersection first, then perform a
    # paired user bootstrap directly on that matched population.
    left_common_margins = _pair_margin_arrays(left.loc[common])
    right_common_margins = _pair_margin_arrays(right.loc[common])

    offset = 0
    while offset < replicates:
        size = min(chunk_size, replicates - offset)
        idx = rng_matched.integers(0, n_common, size=(size, n_common))
        for key in left_margin:
            diff = sign * (
                right_common_margins[key] - left_common_margins[key]
            )
            matched_draws[key][offset:offset + size] = diff[idx].mean(axis=1)
        offset += size

    left_native = _pair_margin_arrays(left)
    right_native = _pair_margin_arrays(right)

    for key in left_margin:
        left_point = float(np.mean(left_native[key]))
        right_point = float(np.mean(right_native[key]))
        native_point = sign * (right_point - left_point)

        matched_point = sign * float(
            np.mean(
                right_common_margins[key] - left_common_margins[key]
            )
        )

        results[key] = {
            "native": {
                "left_users": int(len(left)),
                "right_users": int(len(right)),
                "union_users": int(len(union)),
                "point": native_point,
                "ci95": _percentile_interval(native_draws[key]),
            },
            "matched": {
                "users": int(len(common)),
                "point": matched_point,
                "ci95": _percentile_interval(matched_draws[key]),
            },
        }

    return results


def run_cross_cell_contrasts(
    *,
    seed_mean_path: str | Path,
    output_dir: str | Path,
    replicates: int = BOOTSTRAP_REPLICATES,
    rng_seed: int = BOOTSTRAP_SEED,
) -> dict:
    frame = load_seed_mean_artifact(seed_mean_path)

    cells = {
        ("standard", "is_click"): _cell_wide(
            frame, regime="standard", target="is_click"
        ),
        ("standard", "long_view"): _cell_wide(
            frame, regime="standard", target="long_view"
        ),
        ("randomized", "is_click"): _cell_wide(
            frame, regime="randomized", target="is_click"
        ),
        ("randomized", "long_view"): _cell_wide(
            frame, regime="randomized", target="long_view"
        ),
    }

    regime_contrasts: dict[str, dict] = {}
    for target in ("is_click", "long_view"):
        standard = cells[("standard", target)]
        randomized = cells[("randomized", target)]
        regime_contrasts[target] = bootstrap_cross_cell(
            standard,
            randomized,
            direction="right_minus_left",
            replicates=replicates,
            rng_seed=rng_seed,
        )

    target_contrasts: dict[str, dict] = {}
    for regime in ("standard", "randomized"):
        click = cells[(regime, "is_click")]
        long_view = cells[(regime, "long_view")]
        target_contrasts[regime] = bootstrap_cross_cell(
            click,
            long_view,
            direction="right_minus_left",
            replicates=replicates,
            rng_seed=rng_seed,
        )

    manifest = {
        "status": "M1_PRIMARY_CROSS_CELL_CONTRASTS",
        "source_seed_mean_sha256": verify_seed_mean_artifact(seed_mean_path),
        "definitions": {
            "D_AB": "S_A_minus_S_B",
            "G_AB": "D_AB_randomized_minus_D_AB_standard",
            "T_AB": "D_AB_long_view_minus_D_AB_is_click",
        },
        "bootstrap": {
            "replicates": int(replicates),
            "rng_seed": int(rng_seed),
            "confidence": CONFIDENCE_LEVEL,
            "ci_method": "nonparametric_percentile",
        },
        "regime_contrasts_G_AB": regime_contrasts,
        "target_contrasts_T_AB": target_contrasts,
        "guardrails": [
            "Reads the immutable M1.2 per-user seed-mean artifact only.",
            "No checkpoint is loaded and no candidate is rescored.",
            "Native contrasts resample the union of user IDs while preserving each user's available cell observations.",
            "Matched sensitivities restrict to users eligible in both cells.",
            "G_AB and T_AB are effect-size estimates with 95% intervals; no additional significance label is attached.",
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
        description="Run frozen M1.3 cross-regime and cross-target contrasts."
    )
    parser.add_argument(
        "--seed-mean",
        default="runs/m1/primary_inference/per_user_seed_mean_ndcg.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/m1/cross_cell_contrasts",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_cross_cell_contrasts(
        seed_mean_path=args.seed_mean,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
