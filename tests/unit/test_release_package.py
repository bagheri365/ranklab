from pathlib import Path


def test_v1_release_notes_pin_frozen_conclusion_and_hashes():
    text = Path(
        "research/release/RELEASE_NOTES_v1.0.0.md"
    ).read_text(encoding="utf-8")

    assert "Winner identity is stable, but comparative margins are not." in text
    assert "9bd72ae2d3042dc9511734563b90b8668519b401b847ce218f706a465cc41a32" in text
    assert "ba9334455134d86a14d2310c809c34152c0ce60f135bb62a765cdab9a1737c2f" in text
    assert "1905dc489908fdb746775b0b5b33e2666adc5225897f548979f3e5d69ce1ae29" in text
    assert "not causal treatment-effect estimates" in text


def test_v1_release_checklist_uses_annotated_tag():
    text = Path(
        "research/release/RELEASE_CHECKLIST_v1.0.0.md"
    ).read_text(encoding="utf-8")

    assert "git tag -a v1.0.0" in text
    assert "git push origin v1.0.0" in text
    assert "RELEASE_NOTES_v1.0.0.md" in text
