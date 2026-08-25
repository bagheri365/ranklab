"""M2.2 publication tables and figures from the frozen M1 endpoint.

The numeric snapshot below is a reporting representation of already-frozen M1
results. Generation is blocked unless all three authoritative final-M1 files
match the M2.1 reporting contract hashes.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from ranklab.reporting.contract import FINAL_M1_SHA256, HEADLINE


PRIMARY_SCORES = (
    ("standard", "is_click", "popularity", 0.7165242249),
    ("standard", "is_click", "bpr", 0.6769846930),
    ("standard", "is_click", "lightgcn", 0.6768624000),
    ("standard", "long_view", "popularity", 0.6484044642),
    ("standard", "long_view", "bpr", 0.6017462345),
    ("standard", "long_view", "lightgcn", 0.6015901250),
    ("randomized", "is_click", "popularity", 0.4324591836),
    ("randomized", "is_click", "bpr", 0.3435221069),
    ("randomized", "is_click", "lightgcn", 0.3435274146),
    ("randomized", "long_view", "popularity", 0.3701638914),
    ("randomized", "long_view", "bpr", 0.2650476947),
    ("randomized", "long_view", "lightgcn", 0.2653141801),
)

PRIMARY_POPULARITY_MARGINS = (
    ("standard", "is_click", "popularity_minus_bpr", 0.0395395320),
    ("standard", "is_click", "popularity_minus_lightgcn", 0.0396618250),
    ("standard", "long_view", "popularity_minus_bpr", 0.0466582300),
    ("standard", "long_view", "popularity_minus_lightgcn", 0.0468143390),
    ("randomized", "is_click", "popularity_minus_bpr", 0.0889370766),
    ("randomized", "is_click", "popularity_minus_lightgcn", 0.0889317690),
    ("randomized", "long_view", "popularity_minus_bpr", 0.1051161966),
    ("randomized", "long_view", "popularity_minus_lightgcn", 0.1048497113),
)

SUPPORT_SENSITIVITY = (
    ("shared_tabs", "standard", "is_click", 0.7350651723, 0.7212552697, 0.7210106700),
    ("shared_tabs", "standard", "long_view", 0.6639512902, 0.6440699560, 0.6438208954),
    ("shared_tabs", "randomized", "is_click", 0.4324591836, 0.3435221069, 0.3435274146),
    ("shared_tabs", "randomized", "long_view", 0.3701638914, 0.2650476947, 0.2653141801),
    ("tab1", "standard", "is_click", 0.7368566516, 0.7234848885, 0.7233161909),
    ("tab1", "standard", "long_view", 0.6659222100, 0.6459193189, 0.6458240089),
    ("tab1", "randomized", "is_click", 0.4325400313, 0.3435548195, 0.3435738406),
    ("tab1", "randomized", "long_view", 0.3703211693, 0.2651190206, 0.2654016265),
)

REGIME_G_POP_BPR = (
    ("primary", "is_click", 0.0493975447, 0.0519719454),
    ("primary", "long_view", 0.0584579669, 0.0600588953),
    ("shared_tabs", "is_click", 0.0751271739, 0.0783872980),
    ("shared_tabs", "long_view", 0.0852348624, 0.0890308780),
    ("tab1", "is_click", 0.0756134486, 0.0789328289),
    ("tab1", "long_view", 0.0851992576, 0.0894424231),
)


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_final_m1(
    *,
    manifest_path: str | Path,
    summary_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, str]:
    paths = {
        "manifest": manifest_path,
        "summary": summary_path,
        "markdown": markdown_path,
    }
    verified = {}
    for key, path in paths.items():
        actual = _sha256_file(path)
        expected = FINAL_M1_SHA256[key]
        if actual != expected:
            raise RuntimeError(
                f"final M1 {key} SHA256 mismatch: expected {expected}, got {actual}"
            )
        verified[key] = actual
    return verified


def _write_csv(path: Path, header: tuple[str, ...], rows: tuple[tuple, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _svg_primary_scores() -> str:
    width, height = 980, 560
    left, top = 90, 85
    plot_w, plot_h = 820, 360
    ymax = 0.8
    cells = [
        ("standard", "is_click"),
        ("standard", "long_view"),
        ("randomized", "is_click"),
        ("randomized", "long_view"),
    ]
    model_order = ("popularity", "bpr", "lightgcn")
    score_map = {
        (r, t, m): v for r, t, m, v in PRIMARY_SCORES
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="490" y="32" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="600">Primary macro NDCG@10</text>',
        '<text x="490" y="56" text-anchor="middle" font-family="sans-serif" font-size="13">Frozen M1 primary cells</text>',
    ]
    for tick in range(0, 9, 1):
        value = tick / 10
        y = top + plot_h - (value / ymax) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>')
    cell_w = plot_w / len(cells)
    bar_w = 42
    offsets = (-50, 0, 50)
    for i, (regime, target) in enumerate(cells):
        cx = left + cell_w * (i + 0.5)
        label = f"{regime}\\n{target}"
        for j, model in enumerate(model_order):
            v = score_map[(regime, target, model)]
            h = (v / ymax) * plot_h
            x = cx + offsets[j] - bar_w / 2
            y = top + plot_h - h
            shade = ("#333333", "#777777", "#aaaaaa")[j]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{shade}"/>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="10">{v:.3f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{top+plot_h+28}" text-anchor="middle" font-family="sans-serif" font-size="12">{_escape(regime)}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{top+plot_h+45}" text-anchor="middle" font-family="sans-serif" font-size="12">{_escape(target)}</text>')
    legend_y = 520
    for j, model in enumerate(model_order):
        x = 285 + j * 150
        shade = ("#333333", "#777777", "#aaaaaa")[j]
        parts.append(f'<rect x="{x}" y="{legend_y-12}" width="18" height="12" fill="{shade}"/>')
        parts.append(f'<text x="{x+26}" y="{legend_y}" font-family="sans-serif" font-size="12">{model}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _svg_regime_margins() -> str:
    width, height = 900, 500
    left, top = 100, 80
    plot_w, plot_h = 700, 300
    xmax = 0.10
    rows = list(REGIME_G_POP_BPR)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="32" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="600">Randomized-minus-standard change in Popularity advantage</text>',
        '<text x="450" y="56" text-anchor="middle" font-family="sans-serif" font-size="13">G_AB for Popularity − BPR; native and matched-user estimates</text>',
    ]
    for tick in range(0, 11, 2):
        v = tick / 100
        x = left + (v / xmax) * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#e0e0e0" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+25}" text-anchor="middle" font-family="sans-serif" font-size="11">{v:.02f}</text>')
    row_h = plot_h / len(rows)
    for i, (support, target, native, matched) in enumerate(rows):
        y = top + row_h * (i + 0.5)
        xn = left + (native / xmax) * plot_w
        xm = left + (matched / xmax) * plot_w
        parts.append(f'<text x="{left-14}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{support} · {target}</text>')
        parts.append(f'<line x1="{xn:.1f}" y1="{y:.1f}" x2="{xm:.1f}" y2="{y:.1f}" stroke="#777777" stroke-width="2"/>')
        parts.append(f'<circle cx="{xn:.1f}" cy="{y:.1f}" r="5" fill="#333333"/>')
        parts.append(f'<rect x="{xm-5:.1f}" y="{y-5:.1f}" width="10" height="10" fill="#999999"/>')
    parts.extend([
        '<circle cx="315" cy="445" r="5" fill="#333333"/>',
        '<text x="329" y="449" font-family="sans-serif" font-size="12">native</text>',
        '<rect x="430" y="440" width="10" height="10" fill="#999999"/>',
        '<text x="448" y="449" font-family="sans-serif" font-size="12">matched-user</text>',
        '<text x="450" y="480" text-anchor="middle" font-family="sans-serif" font-size="11">Positive values mean the Popularity−BPR margin is larger under randomized exposure.</text>',
        "</svg>",
    ])
    return "\n".join(parts) + "\n"


def run_publication_assets(
    *,
    final_manifest: str | Path,
    final_summary: str | Path,
    final_markdown: str | Path,
    output_dir: str | Path,
) -> dict:
    verified = verify_final_m1(
        manifest_path=final_manifest,
        summary_path=final_summary,
        markdown_path=final_markdown,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        output_dir / "primary_scores.csv",
        ("regime", "target", "model", "macro_ndcg_at_10"),
        PRIMARY_SCORES,
    )
    _write_csv(
        output_dir / "primary_popularity_margins.csv",
        ("regime", "target", "comparison", "margin"),
        PRIMARY_POPULARITY_MARGINS,
    )
    _write_csv(
        output_dir / "support_sensitivity_scores.csv",
        (
            "support",
            "regime",
            "target",
            "popularity",
            "bpr",
            "lightgcn",
        ),
        SUPPORT_SENSITIVITY,
    )
    _write_csv(
        output_dir / "regime_G_popularity_minus_bpr.csv",
        ("support", "target", "native_G", "matched_user_G"),
        REGIME_G_POP_BPR,
    )

    (output_dir / "primary_scores.svg").write_text(
        _svg_primary_scores(), encoding="utf-8"
    )
    (output_dir / "regime_margin.svg").write_text(
        _svg_regime_margins(), encoding="utf-8"
    )

    output_names = (
        "primary_scores.csv",
        "primary_popularity_margins.csv",
        "support_sensitivity_scores.csv",
        "regime_G_popularity_minus_bpr.csv",
        "primary_scores.svg",
        "regime_margin.svg",
    )
    output_sha256 = {
        name: _sha256_file(output_dir / name)
        for name in output_names
    }

    manifest = {
        "status": "M2_PUBLICATION_ASSETS",
        "headline": HEADLINE,
        "verified_final_m1_sha256": verified,
        "outputs_sha256": output_sha256,
        "guardrails": [
            "Values are presentation snapshots of already-frozen M1 results.",
            "No model scoring, tuning, bootstrap, or statistical decision is run.",
            "Sensitivity outputs remain labeled by their frozen reporting roles.",
            "Figures visualize point estimates only and do not imply causal effects.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate hash-verified M2 publication tables and figures."
    )
    parser.add_argument(
        "--final-manifest",
        default="runs/m1/final_results/manifest.json",
    )
    parser.add_argument(
        "--final-summary",
        default="runs/m1/final_results/summary.json",
    )
    parser.add_argument(
        "--final-markdown",
        default="runs/m1/final_results/FINAL_RESULTS.md",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/m2/publication",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_publication_assets(
        final_manifest=args.final_manifest,
        final_summary=args.final_summary,
        final_markdown=args.final_markdown,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
