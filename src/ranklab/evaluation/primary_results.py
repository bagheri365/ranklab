"""M1.4 primary-results consolidation.

This module performs no scoring, bootstrapping, or model selection. It verifies
the immutable M1.1-M1.3 artifact chain and renders normalized tables and a
concise primary-results summary from already-computed outputs.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping, Any


EXPECTED_SHA256 = {
    "m1_raw": "25f41d6380964e0eaf0f6746a8e4df221a3c62faa105ce33a3893d92f1d4301a",
    "m1_seed_mean": "4ca8393b25e7fc6107f7ebebd38846ed64f5b53c1191e6e58776a59ffb0f9ff8",
    "m1_within_manifest": "62a3efa8b3155b18da145a41bdbb195886d7a14ada613e3e8bc1e845ad7d5d9e",
    "m1_cross_manifest": "739c8d20aacae655263409d3bc4e5b8a175c41e9268522c7d43896102940bfaf",
}

CELL_ORDER = (
    ("standard", "is_click"),
    ("standard", "long_view"),
    ("randomized", "is_click"),
    ("randomized", "long_view"),
)
MODEL_ORDER = ("popularity", "bpr", "lightgcn")
PAIR_ORDER = (
    "popularity_minus_bpr",
    "popularity_minus_lightgcn",
    "bpr_minus_lightgcn",
)


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(path: str | Path, expected: str, label: str) -> str:
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def verify_chain(
    *,
    raw_path: str | Path,
    seed_mean_path: str | Path,
    within_manifest_path: str | Path,
    cross_manifest_path: str | Path,
) -> dict[str, str]:
    return {
        "m1_raw": verify_artifact(
            raw_path, EXPECTED_SHA256["m1_raw"], "M1.1 raw"
        ),
        "m1_seed_mean": verify_artifact(
            seed_mean_path, EXPECTED_SHA256["m1_seed_mean"], "M1.2 seed mean"
        ),
        "m1_within_manifest": verify_artifact(
            within_manifest_path,
            EXPECTED_SHA256["m1_within_manifest"],
            "M1.2 manifest",
        ),
        "m1_cross_manifest": verify_artifact(
            cross_manifest_path,
            EXPECTED_SHA256["m1_cross_manifest"],
            "M1.3 manifest",
        ),
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_cell_score_rows(within: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime, target in CELL_ORDER:
        cell_key = f"{regime}|{target}"
        cell = within["cells"][cell_key]
        order = list(cell["descriptive_order"])
        for model in MODEL_ORDER:
            rows.append(
                {
                    "regime": regime,
                    "target": target,
                    "model": model,
                    "macro_ndcg_at_10": float(cell["scores"][model]),
                    "descriptive_rank": order.index(model) + 1,
                    "decisive_winner": cell["decisive_winner"],
                }
            )
    return rows


def build_within_pair_rows(within: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime, target in CELL_ORDER:
        cell = within["cells"][f"{regime}|{target}"]
        for pair in PAIR_ORDER:
            result = cell["pairwise"][pair]
            rows.append(
                {
                    "regime": regime,
                    "target": target,
                    "pair": pair,
                    "users": int(result["users"]),
                    "margin": float(result["margin"]),
                    "ci95_lower": float(result["ci95"]["lower"]),
                    "ci95_upper": float(result["ci95"]["upper"]),
                    "simultaneous_ci_lower": float(
                        result["simultaneous_ci"]["lower"]
                    ),
                    "simultaneous_ci_upper": float(
                        result["simultaneous_ci"]["upper"]
                    ),
                }
            )
    return rows


def _cross_rows(
    section: Mapping[str, Any],
    *,
    axis_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for axis_value, pair_map in section.items():
        for pair in PAIR_ORDER:
            result = pair_map[pair]
            native = result["native"]
            matched = result["matched"]
            rows.append(
                {
                    axis_name: axis_value,
                    "pair": pair,
                    "native_point": float(native["point"]),
                    "native_ci95_lower": float(native["ci95"]["lower"]),
                    "native_ci95_upper": float(native["ci95"]["upper"]),
                    "native_left_users": int(native["left_users"]),
                    "native_right_users": int(native["right_users"]),
                    "native_union_users": int(native["union_users"]),
                    "matched_point": float(matched["point"]),
                    "matched_ci95_lower": float(matched["ci95"]["lower"]),
                    "matched_ci95_upper": float(matched["ci95"]["upper"]),
                    "matched_users": int(matched["users"]),
                }
            )
    return rows


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def build_summary(
    within: Mapping[str, Any],
    cross: Mapping[str, Any],
) -> dict[str, Any]:
    winners = {
        cell_key: cell["decisive_winner"]
        for cell_key, cell in within["cells"].items()
    }
    unique_winners = sorted(set(winners.values()))
    stable_winner = unique_winners[0] if len(unique_winners) == 1 else None

    bpr_lgcn = {}
    for cell_key, cell in within["cells"].items():
        result = cell["pairwise"]["bpr_minus_lightgcn"]
        lo = float(result["simultaneous_ci"]["lower"])
        hi = float(result["simultaneous_ci"]["upper"])
        bpr_lgcn[cell_key] = {
            "margin": float(result["margin"]),
            "simultaneous_ci": {"lower": lo, "upper": hi},
            "decisive_pair_separation": bool(lo > 0.0 or hi < 0.0),
        }

    return {
        "status": "M1_PRIMARY_RESULTS_CONSOLIDATED",
        "decisive_winners": winners,
        "stable_decisive_winner": stable_winner,
        "winner_identity_stable": stable_winner is not None,
        "bpr_vs_lightgcn": bpr_lgcn,
        "cross_regime_G_AB": cross["regime_contrasts_G_AB"],
        "cross_target_T_AB": cross["target_contrasts_T_AB"],
        "interpretation_guardrail": (
            "G_AB and T_AB are reported as frozen effect-size estimates with "
            "95% percentile intervals; M1.4 adds no new significance labels."
        ),
    }


def render_markdown(
    summary: Mapping[str, Any],
    cell_rows: list[dict[str, Any]],
) -> str:
    winner = summary["stable_decisive_winner"]
    lines = [
        "# M1 Primary Results",
        "",
        "This file is generated from hash-verified M1.1-M1.3 artifacts.",
        "No scoring, tuning, bootstrap resampling, or new inference is performed here.",
        "",
        "## Primary cell scores",
        "",
        "| Regime | Target | Popularity | BPR | LightGCN | Decisive winner |",
        "|---|---|---:|---:|---:|---|",
    ]

    by_cell: dict[tuple[str, str], dict[str, float]] = {}
    winners: dict[tuple[str, str], str] = {}
    for row in cell_rows:
        key = (str(row["regime"]), str(row["target"]))
        by_cell.setdefault(key, {})[str(row["model"])] = float(
            row["macro_ndcg_at_10"]
        )
        winners[key] = str(row["decisive_winner"])

    for regime, target in CELL_ORDER:
        scores = by_cell[(regime, target)]
        lines.append(
            f"| {regime} | {target} | {_fmt(scores['popularity'])} | "
            f"{_fmt(scores['bpr'])} | {_fmt(scores['lightgcn'])} | "
            f"{winners[(regime, target)]} |"
        )

    lines.extend(
        [
            "",
            "## Consolidated primary statement",
            "",
        ]
    )
    if winner is not None:
        lines.append(
            f"- Decisive winner identity is stable across all four primary cells: `{winner}`."
        )
    else:
        lines.append("- Decisive winner identity changes across primary cells.")

    all_unresolved = all(
        not value["decisive_pair_separation"]
        for value in summary["bpr_vs_lightgcn"].values()
    )
    if all_unresolved:
        lines.append(
            "- BPR and LightGCN are not decisively separated in any primary cell "
            "under the frozen simultaneous intervals."
        )

    lines.extend(
        [
            "- Cross-regime `G_AB` and cross-target `T_AB` values are retained as "
            "effect-size estimates with their frozen 95% percentile intervals.",
            "- M1.4 does not create any additional significance classification.",
            "",
            "Detailed normalized tables are emitted alongside this summary.",
            "",
        ]
    )
    return "\n".join(lines)


def run_consolidation(
    *,
    raw_path: str | Path,
    seed_mean_path: str | Path,
    within_manifest_path: str | Path,
    cross_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    verified = verify_chain(
        raw_path=raw_path,
        seed_mean_path=seed_mean_path,
        within_manifest_path=within_manifest_path,
        cross_manifest_path=cross_manifest_path,
    )
    within = _load_json(within_manifest_path)
    cross = _load_json(cross_manifest_path)

    if within.get("status") != "M1_PRIMARY_WITHIN_CELL_INFERENCE":
        raise ValueError("unexpected M1.2 manifest status")
    if cross.get("status") != "M1_PRIMARY_CROSS_CELL_CONTRASTS":
        raise ValueError("unexpected M1.3 manifest status")
    if within.get("source_raw_sha256") != EXPECTED_SHA256["m1_raw"]:
        raise ValueError("M1.2 manifest does not point to frozen M1.1 raw artifact")
    if cross.get("source_seed_mean_sha256") != EXPECTED_SHA256["m1_seed_mean"]:
        raise ValueError("M1.3 manifest does not point to frozen M1.2 seed-mean artifact")

    cell_rows = build_cell_score_rows(within)
    within_rows = build_within_pair_rows(within)
    regime_rows = _cross_rows(
        cross["regime_contrasts_G_AB"], axis_name="target"
    )
    target_rows = _cross_rows(
        cross["target_contrasts_T_AB"], axis_name="regime"
    )
    summary = build_summary(within, cross)
    summary["verified_sha256"] = verified

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_csv(
        out / "primary_cell_scores.csv",
        cell_rows,
        [
            "regime",
            "target",
            "model",
            "macro_ndcg_at_10",
            "descriptive_rank",
            "decisive_winner",
        ],
    )
    _write_csv(
        out / "within_cell_pairwise.csv",
        within_rows,
        [
            "regime",
            "target",
            "pair",
            "users",
            "margin",
            "ci95_lower",
            "ci95_upper",
            "simultaneous_ci_lower",
            "simultaneous_ci_upper",
        ],
    )
    cross_fields = [
        "pair",
        "native_point",
        "native_ci95_lower",
        "native_ci95_upper",
        "native_left_users",
        "native_right_users",
        "native_union_users",
        "matched_point",
        "matched_ci95_lower",
        "matched_ci95_upper",
        "matched_users",
    ]
    _write_csv(
        out / "cross_regime_G_AB.csv",
        regime_rows,
        ["target", *cross_fields],
    )
    _write_csv(
        out / "cross_target_T_AB.csv",
        target_rows,
        ["regime", *cross_fields],
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "PRIMARY_RESULTS.md").write_text(
        render_markdown(summary, cell_rows),
        encoding="utf-8",
    )

    manifest = {
        "status": "M1_PRIMARY_RESULTS_CONSOLIDATED",
        "verified_sha256": verified,
        "outputs": [
            "primary_cell_scores.csv",
            "within_cell_pairwise.csv",
            "cross_regime_G_AB.csv",
            "cross_target_T_AB.csv",
            "summary.json",
            "PRIMARY_RESULTS.md",
        ],
        "guardrails": [
            "No model checkpoint is loaded.",
            "No candidate is rescored.",
            "No bootstrap is rerun.",
            "No hyperparameter or model selection decision is changed.",
            "All reported numbers are copied or normalized from hash-verified M1.2/M1.3 manifests.",
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidate hash-verified M1 primary results."
    )
    parser.add_argument(
        "--raw",
        default="runs/m1/primary_raw/per_user_ndcg.csv.gz",
    )
    parser.add_argument(
        "--seed-mean",
        default="runs/m1/primary_inference/per_user_seed_mean_ndcg.csv.gz",
    )
    parser.add_argument(
        "--within-manifest",
        default="runs/m1/primary_inference/manifest.json",
    )
    parser.add_argument(
        "--cross-manifest",
        default="runs/m1/cross_cell_contrasts/manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/m1/primary_results",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_consolidation(
        raw_path=args.raw,
        seed_mean_path=args.seed_mean,
        within_manifest_path=args.within_manifest,
        cross_manifest_path=args.cross_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
