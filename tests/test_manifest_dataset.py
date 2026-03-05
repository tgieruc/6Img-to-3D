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

# Mock heavy deps before any dataloader import
for _mod in ["mmcv", "mmcv.image", "mmcv.image.io", "dataloader.transform_3d", "dataloader.rays_dataset"]:
    if _mod not in sys.modules:
        _m = types.ModuleType(_mod)
        if _mod == "mmcv.image.io":
            _m.imread = lambda *a, **kw: None
        if _mod == "dataloader.transform_3d":
            _m.NormalizeMultiviewImage = lambda **kw: (lambda imgs: imgs)
        if _mod == "dataloader.rays_dataset":
            _m.RaysDataset = type("RaysDataset", (), {})
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
    from dataloader.manifest_dataset import ManifestDataset

    ds = ManifestDataset(TRAIN_MANIFEST)
    expected = sum(1 for _ in TRAIN_MANIFEST.open())
    assert len(ds) == expected


@pytest.mark.skipif(not TRAIN_MANIFEST.exists(), reason="train.jsonl not generated yet — run generate_manifest.py")
def test_manifest_dataset_getitem():
    """ManifestDataset returns images, img_meta, and target path."""
    from dataloader.manifest_dataset import ManifestDataset

    ds = ManifestDataset(TRAIN_MANIFEST)
    imgs, meta, target_path = ds[0]

    assert isinstance(imgs, list) and len(imgs) > 0
    assert all(isinstance(img, np.ndarray) for img in imgs)
    assert all(img.dtype == np.float32 for img in imgs)

    for key in ("K", "c2w", "img_shape", "pose_intrinsics", "num_cams"):
        assert key in meta, f"Missing meta key: {key}"

    n = meta["num_cams"]
    assert meta["K"].shape == (n, 3, 4)
    assert meta["pose_intrinsics"].shape == (n, 20)
    assert len(imgs) == n
    assert Path(target_path).exists(), f"Target transforms not found: {target_path}"


@pytest.mark.skipif(not TRAIN_MANIFEST.exists(), reason="train.jsonl not generated yet — run generate_manifest.py")
def test_manifest_dataset_all_items_loadable():
    """Every entry in train.jsonl loads without error."""
    from dataloader.manifest_dataset import ManifestDataset

    ds = ManifestDataset(TRAIN_MANIFEST)
    for i in range(len(ds)):
        imgs, meta, target = ds[i]
        assert len(imgs) == meta["num_cams"]
        assert meta["num_cams"] > 0
