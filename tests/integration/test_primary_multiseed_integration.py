import csv
from hashlib import sha256

from ranklab.training.primary_multiseed import (
    fit_primary_model,
    verify_seed1_reproducibility,
)


def _write_fixture(path):
    rows = []
    for user_id in (1, 2, 3):
        rows.extend(
            [
                {"user_id": user_id, "video_id": 10 + user_id, "date": "20220409", "is_click": 1},
                {"user_id": user_id, "video_id": 20 + user_id, "date": "20220410", "is_click": 0},
                # Deliberately invalid post-training labels: primary fit must ignore them.
                {"user_id": user_id, "video_id": 30 + user_id, "date": "20220417", "is_click": "VALIDATION_NOT_READ"},
                {"user_id": user_id, "video_id": 40 + user_id, "date": "20220422", "is_click": "EVAL_NOT_READ"},
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["user_id", "video_id", "date", "is_click"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_bpr_primary_fit_and_seed1_repro_are_validation_free(tmp_path):
    data = tmp_path / "standard.csv"
    _write_fixture(data)

    out = tmp_path / "primary"
    manifest = fit_primary_model(
        model="bpr",
        data_path=data,
        output_dir=out,
        repo_root=tmp_path,
    )

    assert len(manifest["seeds"]) == 5
    assert all(
        row["validation_used_for_primary_fit"] is False
        for row in manifest["seeds"]
    )
    assert all(row["fixed_epochs"] == 1 for row in manifest["seeds"])

    repro = verify_seed1_reproducibility(
        model="bpr",
        data_path=data,
        output_dir=out,
    )
    assert repro["byte_identical"] is True
