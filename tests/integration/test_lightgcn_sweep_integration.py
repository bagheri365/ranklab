import csv
from hashlib import sha256

from ranklab.training.lightgcn_sweep import run_lightgcn_sweep


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


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def test_lightgcn_sweep_runs_full_frozen_grid_and_copies_selected_checkpoint(tmp_path):
    data = tmp_path / "standard.csv"
    _write_fixture(data)

    out = tmp_path / "sweep"
    manifest = run_lightgcn_sweep(
        data_path=data,
        output_dir=out,
        repo_root=tmp_path,
    )

    assert len(manifest["results"]) == 27
    assert manifest["search_space"]["configuration_count"] == 27
    assert (out / "sweep_manifest.json").exists()
    assert (out / "selected_lightgcn.npz").exists()

    selected = manifest["selected"]
    source = out / selected["source_checkpoint"]
    copied = out / "selected_lightgcn.npz"
    assert _hash(source) == _hash(copied)
    assert _hash(copied) == selected["selected_checkpoint_sha256"]
