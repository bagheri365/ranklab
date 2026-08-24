import csv

from ranklab.training.lightgcn_optimization_audit import (
    LEARNING_RATES,
    run_optimization_scale_audit,
)


def _write_fixture(path):
    rows = []
    for user_id in (1, 2, 3):
        rows.extend(
            [
                {"user_id": user_id, "video_id": 10 + user_id, "date": "20220409", "is_click": 1},
                {"user_id": user_id, "video_id": 20 + user_id, "date": "20220410", "is_click": 0},
                {"user_id": user_id, "video_id": 10 + user_id, "date": "20220417", "is_click": 1},
                {"user_id": user_id, "video_id": 20 + user_id, "date": "20220418", "is_click": 0},
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["user_id", "video_id", "date", "is_click"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_optimization_scale_audit_runs_predeclared_grid(tmp_path):
    data = tmp_path / "standard.csv"
    _write_fixture(data)

    manifest = run_optimization_scale_audit(
        data_path=data,
        output_dir=tmp_path / "audit",
        repo_root=tmp_path,
    )

    assert [row["learning_rate"] for row in manifest["results"]] == list(LEARNING_RATES)
    assert all(len(row["history"]) == 5 for row in manifest["results"])
    assert (tmp_path / "audit" / "audit_manifest.json").exists()
