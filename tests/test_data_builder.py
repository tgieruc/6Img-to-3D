import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np


def _make_scene(base: Path, sensor: str):
    d = base / sensor / "transforms"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "frame_0.png"), np.zeros((48, 64, 3), dtype=np.uint8))
    tf = {
        "fl_x": 100.0,
        "fl_y": 100.0,
        "cx": 32.0,
        "cy": 24.0,
        "w": 64,
        "h": 48,
        "frames": [{"file_path": "frame_0.png", "transform_matrix": np.eye(4).tolist()}],
    }
    (d / "transforms_ego.json").write_text(json.dumps(tf))
    (d / "transforms_ego_test.json").write_text(json.dumps(tf))
    return d / "transforms_ego.json"


def _make_manifest(tmp_path: Path, name: str) -> str:
    scene = tmp_path / name.replace(".jsonl", "")
    input_tf = _make_scene(scene / "input", "nuscenes")
    target_tf = _make_scene(scene / "target", "sphere")
    entry = {
        "input": str(input_tf),
        "target": str(target_tf),
        "town": "Town02",
        "weather": "ClearNoon",
        "vehicle": "v",
        "spawn_point": 1,
        "step": 0,
    }
    m = tmp_path / name
    m.write_text(json.dumps(entry) + "\n")
    return str(m)


def test_build_from_manifests_returns_dataloaders(tmp_path):
    from torch.utils.data import DataLoader

    from builder.data_builder import build_from_manifests

    train_m = _make_manifest(tmp_path, "train.jsonl")
    val_m = _make_manifest(tmp_path, "val.jsonl")

    cfg = MagicMock()
    cfg.decoder.whiteout = False
    train_cfg = MagicMock()
    train_cfg.depth = False
    train_cfg.phase = "train"
    train_cfg.batch_size = 1
    train_cfg.factor = 1.0
    train_cfg.num_workers = 0
    val_cfg = MagicMock()
    val_cfg.depth = False
    val_cfg.phase = "val"
    val_cfg.batch_size = 1
    val_cfg.factor = 1.0
    val_cfg.num_workers = 0

    train_loader, val_loader = build_from_manifests(
        train_manifest=train_m,
        val_manifest=val_m,
        config=cfg,
        train_dataset_config=train_cfg,
        val_dataset_config=val_cfg,
    )
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert len(train_loader.dataset) == 1
    assert len(val_loader.dataset) == 1
