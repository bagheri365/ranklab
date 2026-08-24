import csv

from ranklab.training.primary_multiseed import load_primary_training_data


def test_primary_loader_ignores_post_training_labels(tmp_path):
    path = tmp_path / "data.csv"
    rows = [
        {"user_id": 1, "video_id": 10, "date": "20220409", "is_click": 1},
        {"user_id": 1, "video_id": 20, "date": "20220410", "is_click": 0},
        {"user_id": 1, "video_id": 99, "date": "20220417", "is_click": "NOT_BINARY"},
        {"user_id": 1, "video_id": 98, "date": "20220422", "is_click": "NOT_BINARY"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["user_id", "video_id", "date", "is_click"],
        )
        writer.writeheader()
        writer.writerows(rows)

    training = load_primary_training_data(path)

    assert training.positive_pairs == ((1, 10),)
    assert training.negative_pairs == ((1, 20),)
    assert training.eligible_users == (1,)
