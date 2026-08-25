"""M1.7 final results consolidation.

Verifies the frozen M1 primary-results summary and M1.6 sensitivity manifest,
then renders one final documentary package. No new scoring or inference.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Any


EXPECTED_SHA256 = {
    "primary_summary": "6d137a8c1b5f68f717d406af9f9daa2d40ca849d19b3d79f4b8a80fe31971d0f",
    "primary_manifest": "b8939f44bc336ea6bac7691652b9d88b42975fac0e064453e4bc4720bc0b2065",
    "sensitivity_manifest": "8eac4c4f487c1dbd91388ee4cc2e8b740a5d696d7cb54d47bd898f0915d1e359",
}


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: str | Path, expected: str, label: str) -> str:
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def verify_final_chain(
    *,
    primary_summary_path: str | Path,
    primary_manifest_path: str | Path,
    sensitivity_manifest_path: str | Path,
) -> dict[str, str]:
    return {
        "primary_summary": _verify(
            primary_summary_path,
            EXPECTED_SHA256["primary_summary"],
            "M1.4 primary summary",
        ),
        "primary_manifest": _verify(
            primary_manifest_path,
            EXPECTED_SHA256["primary_manifest"],
            "M1.4 primary manifest",
        ),
        "sensitivity_manifest": _verify(
            sensitivity_manifest_path,
            EXPECTED_SHA256["sensitivity_manifest"],
            "M1.6 sensitivity manifest",
        ),
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _interval_excludes_zero(ci: dict[str, float]) -> bool:
    return float(ci["lower"]) > 0.0 or float(ci["upper"]) < 0.0


def summarize_sensitivity(sensitivity: dict[str, Any]) -> dict[str, Any]:
    cells = sensitivity["cells"]
    first_models = {
        key: value["descriptive_order"][0]
        for key, value in cells.items()
    }

    bpr_lgcn_within = {
        key: {
            "margin": value["pairwise_95pct_only"]["bpr_minus_lightgcn"]["margin"],
            "ci95": value["pairwise_95pct_only"]["bpr_minus_lightgcn"]["ci95"],
            "ci95_excludes_zero": _interval_excludes_zero(
                value["pairwise_95pct_only"]["bpr_minus_lightgcn"]["ci95"]
            ),
        }
        for key, value in cells.items()
    }

    g_pop_bpr = {
        target: {
            "native_point": pair_map["popularity_minus_bpr"]["native"]["point"],
            "native_ci95": pair_map["popularity_minus_bpr"]["native"]["ci95"],
            "matched_point": pair_map["popularity_minus_bpr"]["matched"]["point"],
            "matched_ci95": pair_map["popularity_minus_bpr"]["matched"]["ci95"],
        }
        for target, pair_map in sensitivity["regime_contrasts_G_AB"].items()
    }

    t_pop_bpr = {
        regime: {
            "native_point": pair_map["popularity_minus_bpr"]["native"]["point"],
            "native_ci95": pair_map["popularity_minus_bpr"]["native"]["ci95"],
            "matched_point": pair_map["popularity_minus_bpr"]["matched"]["point"],
            "matched_ci95": pair_map["popularity_minus_bpr"]["matched"]["ci95"],
        }
        for regime, pair_map in sensitivity["target_contrasts_T_AB"].items()
    }

    return {
        "role": sensitivity["role"],
        "descriptive_first_model_by_cell": first_models,
        "all_cells_same_descriptive_first_model": (
            len(set(first_models.values())) == 1
        ),
        "bpr_vs_lightgcn_within_cell": bpr_lgcn_within,
        "G_popularity_minus_bpr": g_pop_bpr,
        "T_popularity_minus_bpr": t_pop_bpr,
    }


def build_final_summary(
    primary: dict[str, Any],
    sensitivity_manifest: dict[str, Any],
) -> dict[str, Any]:
    sensitivities = {
        name: summarize_sensitivity(value)
        for name, value in sensitivity_manifest["sensitivities"].items()
    }

    return {
        "status": "M1_FINAL_RESULTS_CONSOLIDATED",
        "primary": {
            "winner_identity_stable": primary["winner_identity_stable"],
            "stable_decisive_winner": primary["stable_decisive_winner"],
            "bpr_vs_lightgcn": primary["bpr_vs_lightgcn"],
            "cross_regime_G_AB": primary["cross_regime_G_AB"],
            "cross_target_T_AB": primary["cross_target_T_AB"],
        },
        "sensitivities": sensitivities,
        "headline": (
            "Winner identity is stable across primary cells, while pairwise "
            "model margins depend materially on logging regime and target."
        ),
        "robustness_statement": (
            "The pre-specified shared-tabs sensitivity and descriptive tab1 "
            "sensitivity preserve the same qualitative ordering pattern and "
            "the substantially larger Popularity advantage under randomized "
            "exposure."
        ),
        "limitations": [
            "Offline evaluation under different logging policies is not a causal treatment-effect analysis.",
            "Popularity is deterministic; BPR and LightGCN use five frozen seeds.",
            "BPR and LightGCN are not decisively separated under the primary simultaneous intervals.",
            "tab1 is descriptive only because public KuaiRand documentation does not map numeric tab IDs to semantic UI labels.",
            "The randomized target contrast differs between native and matched-user analyses, indicating sensitivity to target-specific eligible-user composition.",
            "LightGCN tuning used a fixed regularization value while BPR searched regularization.",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    primary = summary["primary"]
    lines = [
        "# RankLab Final M1 Results",
        "",
        "This document is generated only from hash-verified frozen M1 artifacts.",
        "No scoring, tuning, bootstrap resampling, or new statistical decision is performed here.",
        "",
        "## Primary conclusion",
        "",
        f"- Stable decisive winner: `{primary['stable_decisive_winner']}`.",
        "- Winner identity is stable across all four primary regime/target cells.",
        "- BPR and LightGCN are not decisively separated in any primary cell under the frozen simultaneous intervals.",
        "- Pairwise margins are not stable across logging regimes: Popularity's advantage is substantially larger under randomized exposure.",
        "- Target definition also changes Popularity's margin, especially under randomized exposure.",
        "",
        "## Support robustness",
        "",
    ]

    for name in ("shared_tabs", "tab1"):
        sens = summary["sensitivities"][name]
        role = sens["role"]
        first = next(iter(sens["descriptive_first_model_by_cell"].values()))
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Role: `{role}`.",
                f"- Descriptive first model is `{first}` in all four sensitivity cells.",
                "- BPR-LightGCN 95% within-cell intervals cross zero in all four cells.",
                "- The larger randomized-exposure Popularity margin is preserved.",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "> Winner identity is stable, but comparative margins are not.",
            "",
            "Under the frozen RankLab protocol, Popularity remains the leading model across logging regimes, targets, and both support sensitivities. However, the measured size of its advantage over BPR and LightGCN changes substantially with the logging policy and also changes with the behavioral target. The evidence therefore supports stability of top-model selection in this benchmark, but not stability of comparative offline effect sizes.",
            "",
            "## Limitations",
            "",
        ]
    )
    for item in summary["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def run_final_consolidation(
    *,
    primary_summary_path: str | Path,
    primary_manifest_path: str | Path,
    sensitivity_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    verified = verify_final_chain(
        primary_summary_path=primary_summary_path,
        primary_manifest_path=primary_manifest_path,
        sensitivity_manifest_path=sensitivity_manifest_path,
    )

    primary = _load_json(primary_summary_path)
    primary_manifest = _load_json(primary_manifest_path)
    sensitivity = _load_json(sensitivity_manifest_path)

    if primary.get("status") != "M1_PRIMARY_RESULTS_CONSOLIDATED":
        raise ValueError("unexpected primary summary status")
    if primary_manifest.get("status") != "M1_PRIMARY_RESULTS_CONSOLIDATED":
        raise ValueError("unexpected primary manifest status")
    if sensitivity.get("status") != "M1_SUPPORT_SENSITIVITY_INFERENCE":
        raise ValueError("unexpected sensitivity manifest status")

    summary = build_final_summary(primary, sensitivity)
    summary["verified_sha256"] = verified

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "FINAL_RESULTS.md").write_text(
        render_markdown(summary),
        encoding="utf-8",
    )

    manifest = {
        "status": "M1_FINAL_RESULTS_CONSOLIDATED",
        "verified_sha256": verified,
        "outputs": ["summary.json", "FINAL_RESULTS.md"],
        "guardrails": [
            "No model checkpoint is loaded.",
            "No candidate is rescored.",
            "No bootstrap is rerun.",
            "No new significance or practical-effect threshold is introduced.",
            "Primary and sensitivity roles remain distinct.",
        ],
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidate final frozen M1 primary and sensitivity results."
    )
    parser.add_argument(
        "--primary-summary",
        default="runs/m1/primary_results/summary.json",
    )
    parser.add_argument(
        "--primary-manifest",
        default="runs/m1/primary_results/manifest.json",
    )
    parser.add_argument(
        "--sensitivity-manifest",
        default="runs/m1/sensitivity_inference/manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/m1/final_results",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_final_consolidation(
        primary_summary_path=args.primary_summary,
        primary_manifest_path=args.primary_manifest,
        sensitivity_manifest_path=args.sensitivity_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
