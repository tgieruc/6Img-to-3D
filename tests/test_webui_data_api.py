import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def fake_data_dir(tmp_path):
    """Minimal seed4d-style directory tree."""
    for town, sp, step in [("Town02", 1, 0), ("Town05", 1, 0)]:
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
            tf = {"frames": [], "fl_x": 100, "fl_y": 100, "cx": 32, "cy": 24, "w": 64, "h": 48}
            (d / "transforms_ego.json").write_text(json.dumps(tf))
    return tmp_path


def test_scan_returns_hierarchy(fake_data_dir, monkeypatch):
    import webui.backend.api.data as data_mod

    monkeypatch.setattr(data_mod, "DATA_DIR", fake_data_dir)
    from webui.backend.main import app

    client = TestClient(app)
    resp = client.get("/api/data/scan")
    assert resp.status_code == 200
    result = resp.json()
    assert set(result["towns"]) == {"Town02", "Town05"}
    assert "ClearNoon" in result["weathers"]
    assert "nuscenes" in result["sensors"]
    assert "sphere" in result["sensors"]
    assert len(result["scenes"]) == 2


def test_options_returns_filter_values(fake_data_dir, monkeypatch):
    import webui.backend.api.data as data_mod

    monkeypatch.setattr(data_mod, "DATA_DIR", fake_data_dir)
    from webui.backend.main import app

    client = TestClient(app)
    resp = client.get("/api/data/options")
    assert resp.status_code == 200
    opts = resp.json()
    assert "Town02" in opts["towns"]
    assert "vehicle.mini.cooper_s" in opts["vehicles"]
    assert "scenes" not in opts
