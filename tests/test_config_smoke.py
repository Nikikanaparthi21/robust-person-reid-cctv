from pathlib import Path

import yaml


def test_baseline_config_loads():
    path = Path(__file__).parents[1] / "configs" / "baseline.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["project"] == "robust-person-reid-cctv"
    assert config["privacy"]["raw_cctv_in_repository"] is False
