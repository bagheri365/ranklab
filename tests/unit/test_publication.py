from pathlib import Path

import pytest

from ranklab.reporting import publication


def test_publication_snapshot_has_four_primary_cells():
    cells = {(r, t) for r, t, _, _ in publication.PRIMARY_SCORES}
    assert cells == {
        ("standard", "is_click"),
        ("standard", "long_view"),
        ("randomized", "is_click"),
        ("randomized", "long_view"),
    }
    for cell in cells:
        models = {
            m for r, t, m, _ in publication.PRIMARY_SCORES
            if (r, t) == cell
        }
        assert models == {"popularity", "bpr", "lightgcn"}


def test_verify_final_m1_rejects_hash_drift(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    summary = tmp_path / "summary.json"
    markdown = tmp_path / "FINAL_RESULTS.md"
    for p in (manifest, summary, markdown):
        p.write_text("drift", encoding="utf-8")
    monkeypatch.setattr(
        publication,
        "FINAL_M1_SHA256",
        {"manifest": "0" * 64, "summary": "0" * 64, "markdown": "0" * 64},
    )
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        publication.verify_final_m1(
            manifest_path=manifest,
            summary_path=summary,
            markdown_path=markdown,
        )
