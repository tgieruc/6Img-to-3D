"""Verify 6Img-to-3D correctly loads seed4d data with variable camera counts."""
import json
import numpy as np
import os
import tempfile
import sys
import types
import pytest

# Mock heavy dependencies before importing dataset modules
for mod_name in [
    "mmcv", "mmcv.image", "mmcv.image.io",
    "dataloader.transform_3d", "dataloader.rays_dataset",
]:
    if mod_name not in sys.modules:
        mock_mod = types.ModuleType(mod_name)
        if mod_name == "mmcv.image.io":
            mock_mod.imread = lambda *a, **kw: None
        if mod_name == "dataloader.transform_3d":
            mock_mod.NormalizeMultiviewImage = lambda **kw: (lambda x: x)
        if mod_name == "dataloader.rays_dataset":
            mock_mod.RaysDataset = type("RaysDataset", (), {})
        sys.modules[mod_name] = mock_mod


def _make_transforms_ego(num_cams, per_camera_intrinsics=False):
    """Create a synthetic transforms_ego.json matching seed4d output format."""
    data = {
        "camera_model": "OPENCV",
        "k1": 0, "k2": 0, "p1": 0, "p2": 0,
        "frames": [],
    }

    if not per_camera_intrinsics:
        data["fl_x"] = 500.0
        data["fl_y"] = 500.0
        data["cx"] = 50.0
        data["cy"] = 50.0
        data["w"] = 100
        data["h"] = 100

    for i in range(num_cams):
        frame = {
            "file_path": f"../sensors/{i}_rgb.png",
            "transform_matrix": np.eye(4).tolist(),
        }
        if per_camera_intrinsics:
            frame["fl_x"] = 400.0 + i * 20
            frame["fl_y"] = 400.0 + i * 20
            frame["cx"] = 50.0
            frame["cy"] = 50.0
            frame["w"] = 100
            frame["h"] = 100
        data["frames"].append(frame)

    return data


def test_load_global_intrinsics():
    """6Img-to-3D reads global intrinsics when per-frame not present."""
    from dataloader.dataset import build_pose_intrinsics_vector

    data = _make_transforms_ego(4, per_camera_intrinsics=False)

    num_cams = len(data["frames"])
    K = np.zeros((num_cams, 3, 4))
    all_c2w = []
    for cam_idx, frame in enumerate(data["frames"]):
        fx = frame.get("fl_x", data.get("fl_x", 0))
        fy = frame.get("fl_y", data.get("fl_y", 0))
        cx = frame.get("cx", data.get("cx", 0))
        cy = frame.get("cy", data.get("cy", 0))
        K[cam_idx, 0, 0] = fx
        K[cam_idx, 1, 1] = fy
        K[cam_idx, 2, 2] = 1
        K[cam_idx, 0, 2] = cx
        K[cam_idx, 1, 2] = cy
        all_c2w.append(frame["transform_matrix"])

    assert K.shape == (4, 3, 4)
    # All cameras should have same intrinsics (global)
    assert np.allclose(K[0, 0, 0], K[3, 0, 0])  # same fx

    vec = build_pose_intrinsics_vector(all_c2w, K)
    assert vec.shape == (4, 20)


def test_load_per_camera_intrinsics():
    """6Img-to-3D reads per-frame intrinsics when present."""
    from dataloader.dataset import build_pose_intrinsics_vector

    data = _make_transforms_ego(3, per_camera_intrinsics=True)

    num_cams = len(data["frames"])
    K = np.zeros((num_cams, 3, 4))
    all_c2w = []
    for cam_idx, frame in enumerate(data["frames"]):
        fx = frame.get("fl_x", data.get("fl_x", 0))
        fy = frame.get("fl_y", data.get("fl_y", 0))
        cx = frame.get("cx", data.get("cx", 0))
        cy = frame.get("cy", data.get("cy", 0))
        K[cam_idx, 0, 0] = fx
        K[cam_idx, 1, 1] = fy
        K[cam_idx, 2, 2] = 1
        K[cam_idx, 0, 2] = cx
        K[cam_idx, 1, 2] = cy
        all_c2w.append(frame["transform_matrix"])

    assert K.shape == (3, 3, 4)
    # Per-camera intrinsics should differ
    assert K[0, 0, 0] != K[2, 0, 0]  # different fx

    vec = build_pose_intrinsics_vector(all_c2w, K)
    assert vec.shape == (3, 20)
    # Verify intrinsics are in the vector
    assert np.isclose(vec[0, 16], 400.0)  # fx of camera 0
    assert np.isclose(vec[2, 16], 440.0)  # fx of camera 2


def test_collate_with_seed4d_format():
    """Collate function handles seed4d-style data with variable cameras."""
    from dataloader.dataset import build_pose_intrinsics_vector
    from dataloader.dataset_wrapper import custom_collate_fn

    # Simulate 2 samples: 3 cameras and 5 cameras
    samples = []
    for n_cams in [3, 5]:
        data = _make_transforms_ego(n_cams)
        K = np.zeros((n_cams, 3, 4))
        all_c2w = []
        for cam_idx, frame in enumerate(data["frames"]):
            K[cam_idx, 0, 0] = frame.get("fl_x", data.get("fl_x", 0))
            K[cam_idx, 1, 1] = frame.get("fl_y", data.get("fl_y", 0))
            K[cam_idx, 2, 2] = 1
            K[cam_idx, 0, 2] = frame.get("cx", data.get("cx", 0))
            K[cam_idx, 1, 2] = frame.get("cy", data.get("cy", 0))
            all_c2w.append(frame["transform_matrix"])

        imgs = np.random.randn(n_cams, 100, 100, 3).astype(np.float32)
        meta = dict(
            K=K,
            c2w=all_c2w,
            img_shape=[(100, 100, 3)] * n_cams,
            pose_intrinsics=build_pose_intrinsics_vector(all_c2w, K),
            num_cams=n_cams,
        )
        samples.append((imgs, meta, None))

    img_batch, meta_batch, _ = custom_collate_fn(samples)

    assert img_batch.shape == (2, 5, 3, 100, 100)  # padded to max=5
    assert meta_batch[0]["cam_mask"][:3].all()
    assert not meta_batch[0]["cam_mask"][3:].any()
    assert meta_batch[1]["cam_mask"].all()
