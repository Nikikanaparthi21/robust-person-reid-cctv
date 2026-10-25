from pathlib import Path

import yaml


def test_baseline_config_loads():
    path = Path(__file__).parents[1] / "configs" / "baseline.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["project"] == "robust-person-reid-cctv"
    assert config["privacy"]["raw_cctv_in_repository"] is False

def test_dataset_config_loads():
    path = Path(__file__).parents[1] / "configs" / "dataset.yaml"
    config = yaml.safe_load(path.read_text())

    assert config["dataset_name"] == "MARS"
    assert config["train_directory"] == "bbox_train"
    assert config["test_directory"] == "bbox_test"
    assert config["selected_split"] in ["train", "test"]
    assert config["random_seed"] == 255
    assert "processed_root" in config
    assert "tracklet_root" in config
    assert "embedding_root" in config
