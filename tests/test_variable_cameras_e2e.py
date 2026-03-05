"""
Smoke test: end-to-end forward pass with variable camera counts.

Tests:
  1. Config loads without error and exposes `max_cams` (not the old `num_cams`).
  2. collate -> dataset pipeline produces valid cam_mask for variable camera counts.
  3. (Optional, requires mmcv/mmdet) Full model forward pass with < max_cams cameras.
"""

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Test 1: Config contains max_cams, not num_cams
# ---------------------------------------------------------------------------


def test_config_has_max_cams():
    """Config must expose max_cams (not the old num_cams) after Task 1."""
    # Import only config.py — no heavy mmcv/mmdet needed
    import importlib.util
    import os

    cfg_path = os.path.join(os.path.dirname(__file__), os.pardir, "config", "config.py")
    spec = importlib.util.spec_from_file_location("config.config", os.path.abspath(cfg_path))
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)

    model_cfg = cfg_mod.model
    assert model_cfg is not None, "model dict must be defined in config.py"

    # Verify max_cams appears in the config representation
    cfg_str = str(model_cfg)
    assert "max_cams" in cfg_str, f"'max_cams' must appear in model config; got:\n{cfg_str}"

    # Verify the tpv_head section explicitly carries max_cams
    tpv_head = model_cfg.get("tpv_head", {})
    assert "max_cams" in tpv_head, "'max_cams' must be present in tpv_head config dict"


# ---------------------------------------------------------------------------
# Test 2: collate -> dataset pipeline produces valid cam_mask
# ---------------------------------------------------------------------------


def test_collate_cam_mask_pipeline():
    """
    Simulate the collate -> img_meta pipeline and verify cam_mask
    is correctly produced for a mixed-camera-count batch.

    This does NOT require mmcv/mmdet — only dataloader.dataset_wrapper.
    """
    from dataloader.dataset_wrapper import custom_collate_fn

    # Build two synthetic samples with different camera counts
    n_cams_a = 3  # fewer than max
    n_cams_b = 5

    def make_sample(n_cams, h=64, w=64):
        imgs = np.random.randn(n_cams, h, w, 3).astype(np.float32)
        c2ws = [np.eye(4).tolist() for _ in range(n_cams)]
        K = np.zeros((n_cams, 3, 4))
        for i in range(n_cams):
            K[i, 0, 0] = 500.0
            K[i, 1, 1] = 500.0
            K[i, 0, 2] = 320.0
            K[i, 1, 2] = 240.0
        pose_intrinsics = np.zeros((n_cams, 20))
        for i in range(n_cams):
            pose_intrinsics[i, :16] = np.eye(4).flatten()
            pose_intrinsics[i, 16:] = [500.0, 500.0, 320.0, 240.0]
        meta = {
            "K": K,
            "c2w": c2ws,
            "img_shape": [(h, w, 3)] * n_cams,
            "pose_intrinsics": pose_intrinsics,
            "num_cams": n_cams,
        }
        return (imgs, meta, None)

    batch = custom_collate_fn([make_sample(n_cams_a), make_sample(n_cams_b)])
    img_batch, meta_batch, _ = batch

    max_cams = max(n_cams_a, n_cams_b)

    # --- image tensor shape ---
    assert img_batch.shape[0] == 2, "Batch size must be 2"
    assert img_batch.shape[1] == max_cams, f"Camera dim must equal max_cams={max_cams}"
    assert img_batch.shape[2] == 3, "Channel dim must be 3"

    # --- sample A (3 cameras) ---
    mask_a = meta_batch[0]["cam_mask"]
    assert mask_a.shape == (max_cams,), f"cam_mask shape wrong: {mask_a.shape}"
    assert mask_a[:n_cams_a].all(), "Real cameras must be True in cam_mask"
    assert not mask_a[n_cams_a:].any(), "Padded cameras must be False in cam_mask"

    # --- sample B (5 cameras — all real) ---
    mask_b = meta_batch[1]["cam_mask"]
    assert mask_b.shape == (max_cams,), f"cam_mask shape wrong: {mask_b.shape}"
    assert mask_b.all(), "All cameras in sample B are real; mask must be all True"

    # --- pose_intrinsics padded correctly ---
    pi_a = meta_batch[0]["pose_intrinsics"]
    assert pi_a.shape == (max_cams, 20), f"pose_intrinsics shape wrong: {pi_a.shape}"
    # Padded rows must be zero
    assert np.allclose(pi_a[n_cams_a:], 0.0), "Padded pose_intrinsics must be zero"


def test_collate_cam_mask_single_sample():
    """Single-sample batch: all cameras real, cam_mask all True."""
    from dataloader.dataset_wrapper import custom_collate_fn

    n_cams = 6
    h, w = 48, 64
    imgs = np.random.randn(n_cams, h, w, 3).astype(np.float32)
    meta = {
        "K": np.zeros((n_cams, 3, 4)),
        "c2w": [np.eye(4).tolist() for _ in range(n_cams)],
        "img_shape": [(h, w, 3)] * n_cams,
        "pose_intrinsics": np.zeros((n_cams, 20)),
        "num_cams": n_cams,
    }
    batch = custom_collate_fn([(imgs, meta, None)])
    img_batch, meta_batch, _ = batch

    assert img_batch.shape == (1, n_cams, 3, h, w)
    assert meta_batch[0]["cam_mask"].all(), "All cameras should be marked real"
    assert meta_batch[0]["cam_mask"].shape == (n_cams,)


def test_collate_cam_mask_single_camera():
    """Edge case: only one camera selected (minimum dropout)."""
    from dataloader.dataset_wrapper import custom_collate_fn

    n_cams = 1
    h, w = 48, 64
    imgs = np.random.randn(n_cams, h, w, 3).astype(np.float32)
    meta = {
        "K": np.zeros((n_cams, 3, 4)),
        "c2w": [np.eye(4).tolist()],
        "img_shape": [(h, w, 3)],
        "pose_intrinsics": np.zeros((n_cams, 20)),
        "num_cams": n_cams,
    }
    batch = custom_collate_fn([(imgs, meta, None)])
    img_batch, meta_batch, _ = batch

    assert img_batch.shape == (1, 1, 3, h, w)
    assert meta_batch[0]["cam_mask"].all()
    assert meta_batch[0]["cam_mask"].shape == (1,)


# ---------------------------------------------------------------------------
# Test 3: Full model forward pass (skipped when mmcv/mmdet not installed)
# ---------------------------------------------------------------------------


def test_forward_pass_variable_cameras():
    """Verify the full encoder accepts variable camera counts without crashing.

    Skipped when mmcv is not installed (e.g., CI without CUDA/mmcv).
    """
    pytest.importorskip("mmcv")
    pytest.importorskip("mmdet")

    # These imports require mmcv/mmdet to be fully installed with CUDA ops.
    try:
        pass  # registers custom mmcv modules
    except Exception as exc:
        pytest.skip(f"mmcv/mmdet/triplane_encoder import failed: {exc}")

    try:
        from builder.model_builder import build_model
    except Exception as exc:
        pytest.skip(f"model builder import failed: {exc}")

    # Load config — this must succeed without any errors
    import importlib.util
    import os

    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "config", "config.py"))
    spec = importlib.util.spec_from_file_location("config.config", cfg_path)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    model_cfg = cfg_mod.model

    assert model_cfg is not None
    assert "max_cams" in str(model_cfg)

    # Build model — skip if construction fails (missing CUDA ops, etc.)
    try:
        model = build_model(model_cfg)
        model.eval()
    except Exception as exc:
        pytest.skip(f"Model construction failed (likely missing CUDA ops): {exc}")

    # Synthesize a mini forward pass: B=1, N=3 cameras, 100x100 images
    n_cams = 3
    B = 1
    H, W = 100, 100

    imgs = torch.randn(B, n_cams, 3, H, W)

    c2ws = [np.eye(4).tolist() for _ in range(n_cams)]
    K = np.zeros((n_cams, 3, 4))
    for i in range(n_cams):
        K[i, 0, 0] = 500.0
        K[i, 1, 1] = 500.0
        K[i, 0, 2] = 50.0
        K[i, 1, 2] = 50.0

    pose_intrinsics = np.zeros((n_cams, 20))
    for i in range(n_cams):
        pose_intrinsics[i, :16] = np.eye(4).flatten()
        pose_intrinsics[i, 16:] = [500.0, 500.0, 50.0, 50.0]

    cam_mask = np.ones(n_cams, dtype=bool)

    img_metas = [
        {
            "cam_mask": cam_mask,
            "pose_intrinsics": pose_intrinsics,
            "K": K,
            "c2w": c2ws,
            "img_shape": [(H, W, 3)] * n_cams,
            "num_cams": n_cams,
        }
    ]

    with torch.no_grad():
        try:
            output = model(img_metas=img_metas, img=imgs)
        except Exception as exc:
            pytest.fail(f"Forward pass with {n_cams} cameras raised an exception: {exc}")

    # Output is a list/tuple of triplane features; verify they're non-empty tensors
    if isinstance(output, list | tuple):
        triplane = output[0]
        if isinstance(triplane, list | tuple):
            triplane = triplane[0]
    else:
        triplane = output

    assert isinstance(triplane, torch.Tensor), f"Expected tensor output, got {type(triplane)}"
    # Shape should be (B, tpv_h*tpv_w, embed_dims) — just check it's 3-D and non-empty
    assert triplane.dim() == 3, f"Expected 3-D triplane, got shape {triplane.shape}"
    assert triplane.shape[0] == B, "Batch dimension mismatch"
