from ranklab.reporting.readme_update import rewrite_readme


def test_rewrite_readme_replaces_status_by_headings():
    source = """# RankLab

intro

## Status

old status with arbitrary text

## Primary study

body
"""
    out = rewrite_readme(source)
    assert "Winner identity is stable, but comparative margins are not." in out
    assert "old status with arbitrary text" not in out
    assert "## Primary study" in out


def test_rewrite_readme_rejects_missing_anchor():
    try:
        rewrite_readme("# RankLab\n\n## Status\nold\n")
    except RuntimeError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
