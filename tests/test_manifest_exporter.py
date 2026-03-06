import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_data(tmp_path):
    """Town01 (2 spawn points, train) + Town02 (1 spawn point, val)."""
    for town, sp, step in [("Town01", 1, 0), ("Town01", 2, 0), ("Town02", 1, 0)]:
        for sensor in ["nuscenes", "sphere"]:
            d = (
                tmp_path
                / town
                / "ClearNoon"
                / "vehicle.mini.cooper_s"
                / f"spawn_point_{sp}"
                / f"step_{step}"
                / "ego_vehicle"
                / sensor
                / "transforms"
            )
            d.mkdir(parents=True)
            tf = {"fl_x": 100, "fl_y": 100, "cx": 32, "cy": 24, "w": 64, "h": 48, "frames": []}
            (d / "transforms_ego.json").write_text(json.dumps(tf))
    return tmp_path


def test_export_produces_correct_counts(fake_data, tmp_path):
    from webui.backend.services.manifest_exporter import export_manifests

    recipe = {
        "data_dir": str(fake_data),
        "output_dir": str(tmp_path / "out"),
        "global": {
            "vehicles": ["vehicle.mini.cooper_s"],
            "weathers": ["ClearNoon"],
            "input_sensor": "nuscenes",
            "target_sensor": "sphere",
        },
        "splits": {
            "train": {"towns": ["Town01"], "spawn_points": "all", "steps": "all"},
            "val": {"towns": ["Town02"], "spawn_points": "all", "steps": "all"},
        },
    }
    result = export_manifests(recipe)
    assert result["scene_counts"]["train"] == 2
    assert result["scene_counts"]["val"] == 1
    train_lines = Path(result["train"]).read_text().strip().splitlines()
    assert len(train_lines) == 2
    entry = json.loads(train_lines[0])
    assert "input" in entry and "target" in entry
    assert entry["town"] == "Town01"
    assert "nuscenes" in entry["input"]
    assert "sphere" in entry["target"]


def test_export_spawn_point_filter(fake_data, tmp_path):
    from webui.backend.services.manifest_exporter import export_manifests

    recipe = {
        "data_dir": str(fake_data),
        "output_dir": str(tmp_path / "out2"),
        "global": {
            "vehicles": ["vehicle.mini.cooper_s"],
            "weathers": ["ClearNoon"],
            "input_sensor": "nuscenes",
            "target_sensor": "sphere",
        },
        "splits": {
            "train": {"towns": ["Town01"], "spawn_points": [1], "steps": "all"},
        },
    }
    result = export_manifests(recipe)
    assert result["scene_counts"]["train"] == 1  # only spawn_point_1


def test_export_skips_missing_sensors(fake_data, tmp_path):
    from webui.backend.services.manifest_exporter import export_manifests

    recipe = {
        "data_dir": str(fake_data),
        "output_dir": str(tmp_path / "out3"),
        "global": {
            "vehicles": ["vehicle.mini.cooper_s"],
            "weathers": ["ClearNoon"],
            "input_sensor": "nuscenes",
            "target_sensor": "sphere_invisible",
        },  # doesn't exist
        "splits": {"train": {"towns": ["Town01"], "spawn_points": "all", "steps": "all"}},
    }
    result = export_manifests(recipe)
    assert result["scene_counts"]["train"] == 0  # sensor missing → skip
