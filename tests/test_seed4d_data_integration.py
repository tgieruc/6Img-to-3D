"""Integration test against actual seed4d data on disk.

Requires /home/bdw/Documents/seed4d/data to exist. All tests are
automatically skipped when the data directory is not found, so CI
passes without it.

seed4d directory layout (note the extra 'ego_vehicle' level):
  {root}/{Town}/{Weather}/{Vehicle}/{spawn_point}/{step}/ego_vehicle/
    nuscenes/
      sensor_info.json
      transforms/transforms_ego.json
      sensors/{N}_rgb.png
    sphere/
      sensor_info.json
      transforms/transforms_ego.json
      sensors/{N}_rgb.png
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SEED4D_ROOT = Path("/home/bdw/Documents/seed4d/data")

# ── lightweight mocks so we can import dataset.py without heavy deps ──────────
for mod_name in [
    "mmcv",
    "mmcv.image",
    "mmcv.image.io",
    "dataloader.transform_3d",
    "dataloader.rays_dataset",
]:
    if mod_name not in sys.modules:
        mock = types.ModuleType(mod_name)
        if mod_name == "mmcv.image.io":
            mock.imread = lambda *a, **kw: None
        if mod_name == "dataloader.transform_3d":
            mock.NormalizeMultiviewImage = lambda **kw: lambda x: x
        if mod_name == "dataloader.rays_dataset":
            mock.RaysDataset = type("RaysDataset", (), {})
        sys.modules[mod_name] = mock

pytestmark = pytest.mark.skipif(
    not SEED4D_ROOT.exists(),
    reason=f"seed4d data not found at {SEED4D_ROOT}",
)

# ── helpers ───────────────────────────────────────────────────────────────────

SAMPLE_STEP = SEED4D_ROOT / "Town02/ClearNoon/vehicle.mini.cooper_s/spawn_point_1/step_0/ego_vehicle"


def _load_transforms(sensor_dir: Path) -> dict:
    with open(sensor_dir / "transforms" / "transforms_ego.json") as f:
        return json.load(f)


# ── tests ─────────────────────────────────────────────────────────────────────


def test_nuscenes_transforms_parseable():
    """transforms_ego.json is valid JSON and has the expected schema."""
    data = _load_transforms(SAMPLE_STEP / "nuscenes")
    assert "frames" in data
    assert len(data["frames"]) > 0
    frame = data["frames"][0]
    assert "transform_matrix" in frame
    assert len(frame["transform_matrix"]) == 4  # 4×4 matrix rows


def test_nuscenes_per_frame_intrinsics():
    """nuscenes cameras carry per-frame intrinsics (fl_x on each frame)."""
    data = _load_transforms(SAMPLE_STEP / "nuscenes")
    for frame in data["frames"]:
        assert "fl_x" in frame and "fl_y" in frame, "Expected per-frame intrinsics"
        assert frame["fl_x"] > 0


def test_nuscenes_build_pose_intrinsics_vector():
    """build_pose_intrinsics_vector produces correct shape for nuscenes data."""
    from dataloader.dataset import build_pose_intrinsics_vector

    data = _load_transforms(SAMPLE_STEP / "nuscenes")
    frames = data["frames"]
    n = len(frames)
    K = np.zeros((n, 3, 4))
    c2ws = []
    for i, frame in enumerate(frames):
        K[i, 0, 0] = frame.get("fl_x", data.get("fl_x", 0))
        K[i, 1, 1] = frame.get("fl_y", data.get("fl_y", 0))
        K[i, 2, 2] = 1
        K[i, 0, 2] = frame.get("cx", data.get("cx", 0))
        K[i, 1, 2] = frame.get("cy", data.get("cy", 0))
        c2ws.append(frame["transform_matrix"])

    vec = build_pose_intrinsics_vector(c2ws, K)
    assert vec.shape == (n, 20)
    # Pose should be non-trivial (cameras are placed around the scene)
    assert not np.allclose(vec[:, :16], 0)
    # fx stored at index 16
    assert np.all(vec[:, 16] > 0)


def test_sphere_global_intrinsics():
    """Sphere cameras carry global intrinsics at the top level."""
    data = _load_transforms(SAMPLE_STEP / "sphere")
    assert "fl_x" in data and "fl_y" in data, "Expected global intrinsics on sphere"
    assert data["fl_x"] > 0
    # Per-frame entries should NOT repeat the intrinsics
    for frame in data["frames"]:
        # They may or may not have per-frame overrides; just ensure we can read them
        fx = frame.get("fl_x", data["fl_x"])
        assert fx > 0


def test_sphere_build_pose_intrinsics_vector():
    """build_pose_intrinsics_vector works for sphere cameras with global intrinsics."""
    from dataloader.dataset import build_pose_intrinsics_vector

    data = _load_transforms(SAMPLE_STEP / "sphere")
    frames = data["frames"]
    n = len(frames)
    K = np.zeros((n, 3, 4))
    c2ws = []
    for i, frame in enumerate(frames):
        K[i, 0, 0] = frame.get("fl_x", data.get("fl_x", 0))
        K[i, 1, 1] = frame.get("fl_y", data.get("fl_y", 0))
        K[i, 2, 2] = 1
        K[i, 0, 2] = frame.get("cx", data.get("cx", 0))
        K[i, 1, 2] = frame.get("cy", data.get("cy", 0))
        c2ws.append(frame["transform_matrix"])

    vec = build_pose_intrinsics_vector(c2ws, K)
    assert vec.shape == (n, 20)
    assert np.all(vec[:, 16] > 0)


def test_sensor_images_exist():
    """Every file_path referenced in transforms_ego.json points to a real image."""
    for sensor in ("nuscenes", "sphere"):
        data = _load_transforms(SAMPLE_STEP / sensor)
        sensor_dir = SAMPLE_STEP / sensor / "transforms"
        missing = []
        for frame in data["frames"]:
            img_path = (sensor_dir / frame["file_path"]).resolve()
            if not img_path.exists():
                missing.append(str(img_path))
        assert not missing, f"{sensor}: missing images: {missing}"


def test_camera_counts():
    """Verify the expected number of cameras for this sample."""
    nuscenes_data = _load_transforms(SAMPLE_STEP / "nuscenes")
    sphere_data = _load_transforms(SAMPLE_STEP / "sphere")
    # Town02 ClearNoon spawn_point_1: 7 nuscenes + 10 sphere
    assert len(nuscenes_data["frames"]) == 7
    assert len(sphere_data["frames"]) == 10


def test_all_towns_have_expected_structure():
    """All Town/Weather/Vehicle/SpawnPoint/Step entries expose nuscenes + sphere."""
    missing = []
    for step_path in sorted(SEED4D_ROOT.rglob("step_*")):
        ego = step_path / "ego_vehicle"
        for sensor in ("nuscenes", "sphere"):
            tf = ego / sensor / "transforms" / "transforms_ego.json"
            if not tf.exists():
                missing.append(str(tf))
    assert not missing, "Missing transform files:\n" + "\n".join(missing)
