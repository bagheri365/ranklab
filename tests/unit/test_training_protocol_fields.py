from pathlib import Path

import yaml


def test_training_subsection_and_final_overall_protocol_are_frozen() -> None:
    payload = yaml.safe_load(Path("research/protocol_frozen_m0.yaml").read_text())
    assert payload["status"] == "FROZEN"
    assert payload["training"]["status"] == "FROZEN"
    assert payload["splits"]["train_window"] == "2022-04-09/2022-04-16"
    assert payload["splits"]["validation_window"] == "2022-04-17/2022-04-21"
    sampling = payload["training"]["negative_sampling_policy"]
    assert sampling["scope"] == "same_user_logged_negative_pool"
    assert sampling["replacement"] is True
