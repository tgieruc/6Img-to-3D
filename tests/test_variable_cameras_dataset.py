import sys
import types

import numpy as np

# Mock heavy dependencies that may not be available in the test environment
# so we can import the pure-numpy helper functions from dataloader.dataset.
for mod_name in [
    "mmcv",
    "mmcv.image",
    "mmcv.image.io",
    "dataloader.transform_3d",
    "dataloader.rays_dataset",
]:
    if mod_name not in sys.modules:
        mock_mod = types.ModuleType(mod_name)
        if mod_name == "mmcv.image.io":
            mock_mod.imread = lambda *a, **kw: None
        if mod_name == "dataloader.transform_3d":
            mock_mod.NormalizeMultiviewImage = lambda **kw: lambda x: x
        if mod_name == "dataloader.rays_dataset":
            mock_mod.RaysDataset = type("RaysDataset", (), {})
        sys.modules[mod_name] = mock_mod


def test_build_pose_intrinsics_vector():
    """Verify pose+intrinsics vector is correctly constructed."""
    from dataloader.dataset import build_pose_intrinsics_vector

    c2w = np.eye(4).tolist()
    K = np.zeros((1, 3, 4))
    K[0, 0, 0] = 500.0  # fx
    K[0, 1, 1] = 500.0  # fy
    K[0, 0, 2] = 320.0  # cx
    K[0, 1, 2] = 240.0  # cy

    vec = build_pose_intrinsics_vector([c2w], K)
    assert vec.shape == (1, 20)
    assert np.allclose(vec[0, :16], np.eye(4).flatten())
    assert np.allclose(vec[0, 16:], [500.0, 500.0, 320.0, 240.0])


def test_build_pose_intrinsics_multiple_cameras():
    """Verify multiple cameras produce correct shape."""
    from dataloader.dataset import build_pose_intrinsics_vector

    c2ws = [np.eye(4).tolist() for _ in range(4)]
    K = np.zeros((4, 3, 4))
    for i in range(4):
        K[i, 0, 0] = 500.0 + i
        K[i, 1, 1] = 500.0 + i
        K[i, 0, 2] = 320.0
        K[i, 1, 2] = 240.0

    vec = build_pose_intrinsics_vector(c2ws, K)
    assert vec.shape == (4, 20)


def test_camera_dropout():
    """Camera dropout selects K cameras from N."""
    from dataloader.dataset import apply_camera_dropout

    n_cams = 6
    indices = apply_camera_dropout(n_cams, min_cams=1, max_cams=6)
    assert 1 <= len(indices) <= 6
    assert all(0 <= i < n_cams for i in indices)
    assert len(set(indices)) == len(indices)  # No duplicates
