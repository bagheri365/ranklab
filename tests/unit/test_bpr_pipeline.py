import csv

from ranklab.training.bpr_pipeline import (
    TRAIN_DATES,
    VALIDATION_DATES,
    load_frozen_bpr_data,
)


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["user_id", "video_id", "date", "is_click"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_frozen_loader_collapses_pairs_and_uses_any_click(tmp_path):
    path = tmp_path / "standard.csv"
    _write_rows(
        path,
        [
            {"user_id": 1, "video_id": 10, "date": "20220409", "is_click": 0},
            {"user_id": 1, "video_id": 10, "date": "20220410", "is_click": 1},
            {"user_id": 1, "video_id": 20, "date": "20220411", "is_click": 0},
            {"user_id": 2, "video_id": 11, "date": "20220412", "is_click": 1},
            {"user_id": 2, "video_id": 21, "date": "20220413", "is_click": 0},
            {"user_id": 1, "video_id": 10, "date": "20220417", "is_click": 0},
            {"user_id": 1, "video_id": 10, "date": "20220418", "is_click": 1},
            {"user_id": 1, "video_id": 20, "date": "20220419", "is_click": 0},
        ],
    )

    pairs, validation, stats = load_frozen_bpr_data(path)

    assert pairs.positive_pairs == ((1, 10), (2, 11))
    assert pairs.negative_pairs == ((1, 20), (2, 21))
    assert validation == {1: [(10, 1), (20, 0)]}
    assert stats.training_rows == 5
    assert stats.training_unique_pairs == 4
    assert stats.training_pairwise_eligible_users == 2
    assert TRAIN_DATES[0] == "20220409"
    assert TRAIN_DATES[-1] == "20220416"
    assert VALIDATION_DATES == (
        "20220417",
        "20220418",
        "20220419",
        "20220420",
        "20220421",
    )
