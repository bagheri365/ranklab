from __future__ import annotations

import hashlib

from ranklab.reporting import publication


def test_publication_assets_write_tables_figures_and_manifest(tmp_path, monkeypatch):
    sources = {}
    expected = {}
    for key, name in (
        ("manifest", "manifest.json"),
        ("summary", "summary.json"),
        ("markdown", "FINAL_RESULTS.md"),
    ):
        path = tmp_path / name
        path.write_text(key, encoding="utf-8")
        sources[key] = path
        expected[key] = hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(publication, "FINAL_M1_SHA256", expected)

    out = tmp_path / "out"
    manifest = publication.run_publication_assets(
        final_manifest=sources["manifest"],
        final_summary=sources["summary"],
        final_markdown=sources["markdown"],
        output_dir=out,
    )

    assert manifest["status"] == "M2_PUBLICATION_ASSETS"
    for name in (
        "primary_scores.csv",
        "primary_popularity_margins.csv",
        "support_sensitivity_scores.csv",
        "regime_G_popularity_minus_bpr.csv",
        "primary_scores.svg",
        "regime_margin.svg",
        "manifest.json",
    ):
        assert (out / name).exists()
    assert "<svg" in (out / "primary_scores.svg").read_text(encoding="utf-8")
