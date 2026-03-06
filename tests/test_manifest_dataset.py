"""Integration tests for ManifestDataset and generate_manifest.py.

Skipped automatically when seed4d data / manifests are not present.
"""

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Mock heavy deps before any dataloader import.
# Note: dataloader.rays_dataset is NOT mocked here — the integration tests that
# use real seed4d data require the real RaysDataset, and the unit test
# (test_manifest_dataset_returns_dataloader) clears its own module cache.
for _mod in ["mmcv", "mmcv.image", "mmcv.image.io", "dataloader.transform_3d"]:
    if _mod not in sys.modules:
        _m = types.ModuleType(_mod)
        if _mod == "mmcv.image.io":
            _m.imread = lambda *a, **kw: None
        if _mod == "dataloader.transform_3d":
            _m.NormalizeMultiviewImage = lambda **kw: (lambda imgs: imgs)
        sys.modules[_mod] = _m

SEED4D_ROOT = Path("/home/bdw/Documents/seed4d")
TRAIN_MANIFEST = SEED4D_ROOT / "train.jsonl"
VAL_MANIFEST = SEED4D_ROOT / "val.jsonl"
DATA_DIR = SEED4D_ROOT / "data"

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason=f"seed4d data not found at {DATA_DIR}",
)


# ── manifest generator ────────────────────────────────────────────────────────


def test_generate_manifest(tmp_path):
    """generate_manifest.py writes valid JSONL with expected fields and town split."""
    result = subprocess.run(
        [
            sys.executable,
            "utils/generate_manifest.py",
            "--data-dir",
            str(DATA_DIR),
            "--output-dir",
            str(tmp_path),
            "--input-sensor",
            "nuscenes",
            "--target-sensor",
            "sphere",
            "--val-towns",
            "Town02",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    train_lines = list((tmp_path / "train.jsonl").open())
    val_lines = list((tmp_path / "val.jsonl").open())
    assert len(train_lines) > 0
    assert len(val_lines) > 0

    for line in train_lines + val_lines:
        entry = json.loads(line)
        assert "input" in entry and "target" in entry
        assert Path(entry["input"]).exists(), f"Missing input: {entry['input']}"
        assert Path(entry["target"]).exists(), f"Missing target: {entry['target']}"

    val_towns = {json.loads(l)["town"] for l in val_lines}
    assert val_towns == {"Town02"}
    train_towns = {json.loads(l)["town"] for l in train_lines}
    assert "Town02" not in train_towns


def test_generate_manifest_all_val(tmp_path):
    """With no --val-towns, all scenes go to train and val is empty."""
    result = subprocess.run(
        [
            sys.executable,
            "utils/generate_manifest.py",
            "--data-dir",
            str(DATA_DIR),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    train_lines = list((tmp_path / "train.jsonl").open())
    val_lines = list((tmp_path / "val.jsonl").open())
    assert len(train_lines) > 0
    assert len(val_lines) == 0


def test_generate_manifest_invisible_sensors(tmp_path):
    """Generator works with nuscenes_invisible / sphere_invisible."""
    result = subprocess.run(
        [
            sys.executable,
            "utils/generate_manifest.py",
            "--data-dir",
            str(DATA_DIR),
            "--output-dir",
            str(tmp_path),
            "--input-sensor",
            "nuscenes_invisible",
            "--target-sensor",
            "sphere_invisible",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    all_lines = list((tmp_path / "train.jsonl").open()) + list((tmp_path / "val.jsonl").open())
    assert len(all_lines) > 0
    for line in all_lines:
        entry = json.loads(line)
        assert "nuscenes_invisible" in entry["input"]
        assert "sphere_invisible" in entry["target"]
        assert Path(entry["input"]).exists()
        assert Path(entry["target"]).exists()


# ── ManifestDataset ───────────────────────────────────────────────────────────


@pytest.mark.skipif(not TRAIN_MANIFEST.exists(), reason="train.jsonl not generated yet — run generate_manifest.py")
def test_manifest_dataset_len():
    """ManifestDataset reports correct length from manifest."""
    from unittest.mock import MagicMock

    from dataloader.manifest_dataset import ManifestDataset

    cfg = MagicMock()
    cfg.decoder.whiteout = False
    dataset_cfg = MagicMock()
    dataset_cfg.depth = False
    dataset_cfg.phase = "val"
    dataset_cfg.factor = 1.0
    dataset_cfg.batch_size = 1

    ds = ManifestDataset(TRAIN_MANIFEST, config=cfg, dataset_config=dataset_cfg)
    expected = sum(1 for _ in TRAIN_MANIFEST.open())
    assert len(ds) == expected


@pytest.fixture()
def real_rays_dataset_module():
    """Temporarily replace any stub in sys.modules with the real rays_dataset module.

    Other test files install a no-arg stub at collection time.  Integration
    tests that exercise ManifestDataset.__getitem__ need the real RaysDataset.
    """
    import importlib
    import sys

    real_mod = importlib.import_module("dataloader.rays_dataset")
    # Force reload so we get the genuine class even if a stub was cached first.
    real_mod = importlib.reload(real_mod)
    old = sys.modules.get("dataloader.rays_dataset")
    sys.modules["dataloader.rays_dataset"] = real_mod
    yield real_mod
    if old is None:
        sys.modules.pop("dataloader.rays_dataset", None)
    else:
        sys.modules["dataloader.rays_dataset"] = old


@pytest.mark.skipif(not TRAIN_MANIFEST.exists(), reason="train.jsonl not generated yet — run generate_manifest.py")
def test_manifest_dataset_getitem(real_rays_dataset_module):
    """ManifestDataset returns images, img_meta, and a DataLoader."""
    from unittest.mock import MagicMock

    from torch.utils.data import DataLoader

    from dataloader.manifest_dataset import ManifestDataset

    cfg = MagicMock()
    cfg.decoder.whiteout = False
    dataset_cfg = MagicMock()
    dataset_cfg.depth = False
    # Use "full" phase so RaysDataset reads transforms_ego.json (present in real data).
    # seed4d data does not have transforms_ego_test.json, only transforms_ego.json.
    dataset_cfg.phase = "full"
    dataset_cfg.factor = 1.0
    dataset_cfg.batch_size = 1

    ds = ManifestDataset(TRAIN_MANIFEST, config=cfg, dataset_config=dataset_cfg)
    imgs, meta, loader = ds[0]

    # imgs is returned as a stacked numpy array (N, H, W, C) for collate compatibility
    assert isinstance(imgs, np.ndarray) and imgs.ndim == 4
    assert imgs.dtype == np.float32

    for key in ("K", "c2w", "img_shape", "pose_intrinsics", "num_cams"):
        assert key in meta, f"Missing meta key: {key}"

    n = meta["num_cams"]
    assert meta["K"].shape == (n, 3, 4)
    assert meta["pose_intrinsics"].shape == (n, 20)
    assert imgs.shape[0] == n
    assert isinstance(loader, DataLoader), f"Expected DataLoader, got {type(loader)}"


@pytest.mark.skipif(not TRAIN_MANIFEST.exists(), reason="train.jsonl not generated yet — run generate_manifest.py")
def test_manifest_dataset_all_items_loadable(real_rays_dataset_module):
    """First entry in train.jsonl loads without error and returns a DataLoader."""
    from unittest.mock import MagicMock

    from torch.utils.data import DataLoader

    from dataloader.manifest_dataset import ManifestDataset

    cfg = MagicMock()
    cfg.decoder.whiteout = False
    dataset_cfg = MagicMock()
    dataset_cfg.depth = False
    # Use "full" phase so RaysDataset reads transforms_ego.json (present in real data).
    dataset_cfg.phase = "full"
    dataset_cfg.factor = 1.0
    dataset_cfg.batch_size = 1

    ds = ManifestDataset(TRAIN_MANIFEST, config=cfg, dataset_config=dataset_cfg)
    imgs, meta, loader = ds[0]
    assert len(imgs) == meta["num_cams"]
    assert meta["num_cams"] > 0
    assert isinstance(loader, DataLoader)


# ── Unit test: DataLoader wiring (no seed4d data needed) ──────────────────────


def _make_fake_transforms(tmp_path: Path) -> Path:
    sensor_dir = tmp_path / "transforms"
    sensor_dir.mkdir(parents=True)
    import cv2

    cv2.imwrite(str(sensor_dir / "frame_0.png"), np.zeros((48, 64, 3), dtype=np.uint8))
    tf = {
        "fl_x": 100.0,
        "fl_y": 100.0,
        "cx": 32.0,
        "cy": 24.0,
        "w": 64,
        "h": 48,
        "frames": [{"file_path": "frame_0.png", "transform_matrix": np.eye(4).tolist()}],
    }
    tf_path = sensor_dir / "transforms_ego.json"
    tf_path.write_text(json.dumps(tf))
    # RaysDataset with mode="test" looks for transforms_ego_test.json
    (sensor_dir / "transforms_ego_test.json").write_text(json.dumps(tf))
    return tf_path


def _make_manifest(tmp_path: Path, input_tf: Path, target_tf: Path) -> Path:
    entry = {
        "input": str(input_tf),
        "target": str(target_tf),
        "town": "Town02",
        "weather": "ClearNoon",
        "vehicle": "vehicle.mini.cooper_s",
        "spawn_point": 1,
        "step": 0,
    }
    manifest = tmp_path / "val.jsonl"
    manifest.write_text(json.dumps(entry) + "\n")
    return manifest


def test_manifest_dataset_returns_dataloader(tmp_path):
    """ManifestDataset.__getitem__ must return a DataLoader as the third element."""
    import sys

    # Remove any cached mock of rays_dataset so the real one is importable
    for mod in list(sys.modules.keys()):
        if "rays_dataset" in mod or "manifest_dataset" in mod:
            del sys.modules[mod]

    from unittest.mock import MagicMock

    from torch.utils.data import DataLoader

    input_tf = _make_fake_transforms(tmp_path / "input")
    target_tf = _make_fake_transforms(tmp_path / "target")
    manifest = _make_manifest(tmp_path, input_tf, target_tf)

    cfg = MagicMock()
    cfg.decoder.whiteout = False
    dataset_cfg = MagicMock()
    dataset_cfg.depth = False
    dataset_cfg.phase = "val"
    dataset_cfg.factor = 1.0
    dataset_cfg.batch_size = 1

    from dataloader.manifest_dataset import ManifestDataset

    ds = ManifestDataset(manifest, config=cfg, dataset_config=dataset_cfg)
    assert len(ds) == 1

    imgs, meta, loader = ds[0]

    assert isinstance(loader, DataLoader), f"Expected DataLoader, got {type(loader)}"
    assert imgs.shape[0] == 1
    assert "K" in meta
    assert "num_cams" in meta
