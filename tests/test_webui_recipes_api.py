import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point DB at a temp file so tests are isolated
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import webui.backend.database as db_mod
    import webui.backend.main as main_mod

    test_url = f"sqlite:///{tmp_path}/test.db"
    test_engine = create_engine(test_url, connect_args={"check_same_thread": False})
    monkeypatch.setattr(db_mod, "engine", test_engine)
    monkeypatch.setattr(db_mod, "SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=test_engine))
    # main.py captured engine at import time; patch that reference too so lifespan uses test engine
    monkeypatch.setattr(main_mod, "engine", test_engine)
    from webui.backend.database import Base

    Base.metadata.create_all(bind=test_engine)
    from webui.backend.main import app

    return TestClient(app)


RECIPE = {
    "name": "test-split",
    "data_dir": "/tmp/data",
    "output_dir": "/tmp/out",
    "global_filters": {
        "vehicles": ["v"],
        "weathers": ["ClearNoon"],
        "input_sensor": "nuscenes",
        "target_sensor": "sphere",
    },
    "splits": {
        "train": {"towns": ["Town01"], "spawn_points": "all", "steps": "all"},
        "val": {"towns": ["Town02"], "spawn_points": "all", "steps": "all"},
        "test": {"towns": ["Town05"], "spawn_points": "all", "steps": "all"},
    },
}


def test_create_and_list_recipe(client):
    resp = client.post("/api/recipes", json=RECIPE)
    assert resp.status_code == 201, resp.text
    created = resp.json()
    assert created["name"] == "test-split"
    assert "id" in created

    resp2 = client.get("/api/recipes")
    assert resp2.status_code == 200
    assert len(resp2.json()) == 1


def test_get_recipe(client):
    resp = client.post("/api/recipes", json=RECIPE)
    rid = resp.json()["id"]
    resp2 = client.get(f"/api/recipes/{rid}")
    assert resp2.status_code == 200
    assert resp2.json()["name"] == "test-split"
    assert "yaml_content" in resp2.json()


def test_overlap_fails(client):
    bad = dict(RECIPE)
    bad["splits"] = {
        "train": {"towns": ["Town02"], "spawn_points": "all", "steps": "all"},
        "val": {"towns": ["Town02"], "spawn_points": "all", "steps": "all"},
    }
    resp = client.post("/api/recipes", json=bad)
    assert resp.status_code == 422, resp.text
    assert "overlap" in resp.json()["detail"].lower()


def test_get_nonexistent_returns_404(client):
    resp = client.get("/api/recipes/nonexistent-id")
    assert resp.status_code == 404


def test_export_recipe(client, tmp_path):
    import json
    from pathlib import Path

    # create minimal data structure
    for town, sp in [("Town01", 1), ("Town02", 1)]:
        for sensor in ["nuscenes", "sphere"]:
            d = (
                tmp_path
                / "data"
                / town
                / "ClearNoon"
                / "vehicle.mini.cooper_s"
                / f"spawn_point_{sp}"
                / "step_0"
                / "ego_vehicle"
                / sensor
                / "transforms"
            )
            d.mkdir(parents=True)
            tf = {"frames": [], "fl_x": 100, "fl_y": 100, "cx": 32, "cy": 24, "w": 64, "h": 48}
            (d / "transforms_ego.json").write_text(json.dumps(tf))

    recipe = {
        "name": "export-test",
        "data_dir": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "out"),
        "global_filters": {
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
    # Create recipe
    resp = client.post("/api/recipes", json=recipe)
    assert resp.status_code == 201
    rid = resp.json()["id"]

    # Export
    resp2 = client.post(f"/api/recipes/{rid}/export")
    assert resp2.status_code == 200
    result = resp2.json()
    assert result["scene_counts"]["train"] == 1
    assert result["scene_counts"]["val"] == 1
    assert Path(result["train"]).exists()
