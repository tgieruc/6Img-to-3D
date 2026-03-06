#!/usr/bin/env python3
"""End-to-end smoke test using real seed4d Town05 data.

Phase 1 — Data pipeline (always runs when seed4d data is present):
  - Manifest generator writes correct JSONL
  - ManifestDataset loads nuscenes images, returns correct numpy shape
  - custom_collate_fn produces padded image batch + cam_mask
  - Sphere ray loading produces (M, 10) tensor

Phase 2 — Model forward pass + training step (requires mmcv._ext + tinycudann):
  - TPVFormer encoder forward pass
  - TriplaneDecoder update + render
  - Loss.backward() + optimizer step
  - Quick eval render

Phase 2 is automatically skipped with a clear message when the CUDA extensions
are not compiled (e.g. no nvcc). To enable it:
    uv pip install ninja mmcv==2.1.0 --index-url <mmcv-dist-url>
    uv pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

Usage:
    python smoke_test_seed4d.py [--scenes N] [--train-steps N]
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

SEED4D_ROOT = Path("/home/bdw/Documents/seed4d/data")
TOWN = "Town05"
VEHICLE = "vehicle.mini.cooper_s"

if not SEED4D_ROOT.exists():
    print(f"[skip] seed4d data not found at {SEED4D_ROOT}")
    sys.exit(0)

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).parent))

from dataloader.dataset import build_pose_intrinsics_vector, img_norm_cfg
from dataloader.dataset_wrapper import custom_collate_fn
from dataloader.manifest_dataset import ManifestDataset
from dataloader.transform_3d import NormalizeMultiviewImage
from utils.generate_manifest import generate_manifest

# ── helpers ──────────────────────────────────────────────────────────────────


def find_scenes(base: Path) -> list[Path]:
    """Return ego_vehicle dirs that have both nuscenes and sphere transforms."""
    scenes = []
    for ego in sorted(base.rglob("ego_vehicle")):
        if (ego / "nuscenes" / "transforms" / "transforms_ego.json").exists() and (
            ego / "sphere" / "transforms" / "transforms_ego.json"
        ).exists():
            scenes.append(ego)
    return scenes


def build_img_batch(nuscenes_tf_path: Path, resize_factor: float = 0.25):
    """Load nuscenes images → (imgs_np, img_meta) ready for the collate fn."""
    tf = json.loads(nuscenes_tf_path.read_text())
    frames = tf["frames"]
    n = len(frames)

    normalizer = NormalizeMultiviewImage(**img_norm_cfg)
    imgs = []
    for frame in frames:
        img_path = (nuscenes_tf_path.parent / frame["file_path"]).resolve()
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)[:, :, :3].astype(np.float32)
        if resize_factor != 1.0:
            h = int(img.shape[0] * resize_factor)
            w = int(img.shape[1] * resize_factor)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        imgs.append(img)
    imgs = normalizer(imgs)

    K = np.zeros((n, 3, 4))
    c2ws = []
    for i, frame in enumerate(frames):
        K[i, 0, 0] = frame.get("fl_x", tf.get("fl_x", 800.0)) * resize_factor
        K[i, 1, 1] = frame.get("fl_y", tf.get("fl_y", 800.0)) * resize_factor
        K[i, 2, 2] = 1.0
        K[i, 0, 2] = frame.get("cx", tf.get("cx", 800.0)) * resize_factor
        K[i, 1, 2] = frame.get("cy", tf.get("cy", 450.0)) * resize_factor
        c2ws.append(frame["transform_matrix"])

    imgs_np = np.stack(imgs)  # (N, H, W, 3)
    img_meta = dict(
        K=K,
        c2w=c2ws,
        img_shape=[img.shape for img in imgs],
        pose_intrinsics=build_pose_intrinsics_vector(c2ws, K),
        cam_mask=np.ones(n, dtype=bool),
        num_cams=n,
    )
    return imgs_np, img_meta


def build_sphere_rays(sphere_tf_path: Path, max_frames: int = 3, subsample: int = 100) -> torch.Tensor:
    """Load rays from sphere cameras; returns (M, 10) tensor.
    Columns: [origin(3), direction(3), rgb_gt(3), mask(1), depth(1)]
    """
    from triplane_decoder.intrinsics import Intrinsics
    from triplane_decoder.ray_utils import get_ray_directions, get_rays

    tf = json.loads(sphere_tf_path.read_text())
    fl_x = tf.get("fl_x", 800.0)
    fl_y = tf.get("fl_y", 800.0)
    cx = tf.get("cx", 800.0)
    cy = tf.get("cy", 450.0)
    w = tf.get("w", 1600)
    h = tf.get("h", 900)

    intrinsics = Intrinsics(w, h, fl_x, fl_y, cx, cy)
    directions = get_ray_directions(intrinsics)
    to_tensor = T.ToTensor()

    rays_all = []
    for frame in tf["frames"][:max_frames]:
        c2w = torch.tensor(frame["transform_matrix"])[:3, :4]
        rays_o, rays_d = get_rays(directions.clone(), c2w)

        img_path = (sphere_tf_path.parent / frame["file_path"]).resolve()
        img = Image.open(img_path).resize((w, h), Image.LANCZOS)
        img_t = to_tensor(img).view(-1, h * w).permute(1, 0)
        if img_t.size(1) == 4:
            img_t = img_t[:, :3] * img_t[:, -1:] + (1 - img_t[:, -1:])
        img_t = img_t[:, :3]

        mask = torch.ones(rays_o.size(0), 1)
        depth = torch.zeros(rays_o.size(0), 1)
        ray_data = torch.cat([rays_o, rays_d, img_t, mask, depth], dim=-1)  # (H*W, 10)

        n_keep = max(1, ray_data.size(0) // subsample)
        idx = torch.randperm(ray_data.size(0))[:n_keep]
        rays_all.append(ray_data[idx])

    return torch.cat(rays_all, dim=0)


def check_model_deps() -> tuple[bool, str]:
    """Return (can_run_model, reason)."""
    try:
        import mmcv  # noqa: F401
        import mmcv._ext  # noqa: F401 — compiled CUDA extension

        assert hasattr(mmcv, "__version__"), "mmcv import incomplete"
    except Exception as e:
        return False, f"mmcv CUDA extension not available (needs CUDA build): {e}"
    try:
        import tinycudann  # noqa: F401
    except Exception as e:
        return False, f"tinycudann not available (needs CUDA build): {e}"
    return True, "ok"


# ── Phase 1: Data pipeline ────────────────────────────────────────────────────


def run_phase1(scenes: list[Path], tmpdir: Path) -> bool:
    print("\n=== Phase 1: Data pipeline ===")
    ok = True

    # 1a. Manifest generator
    print("[1a] Manifest generator...")
    generate_manifest(
        data_dir=SEED4D_ROOT / TOWN / "ClearNoon" / VEHICLE,
        output_dir=tmpdir,
        input_sensor="nuscenes",
        target_sensor="sphere",
        val_towns=[],
    )
    manifest_path = tmpdir / "train.jsonl"
    assert manifest_path.exists(), "train.jsonl not created"
    lines = manifest_path.read_text().strip().splitlines()
    assert len(lines) > 0, "Empty manifest"
    entry = json.loads(lines[0])
    for key in ("input", "target", "town", "weather", "vehicle", "spawn_point", "step"):
        assert key in entry, f"Missing key '{key}' in manifest entry"
    print(f"    OK — {len(lines)} entries, keys: {list(entry.keys())}")

    # 1b. ManifestDataset
    print("[1b] ManifestDataset loading...")
    ds = ManifestDataset(manifest_path)
    assert len(ds) > 0
    imgs_np, meta, target_path = ds[0]
    assert imgs_np.ndim == 4, f"Expected (N,H,W,C), got {imgs_np.shape}"
    assert meta["num_cams"] > 0
    assert "pose_intrinsics" in meta
    assert meta["pose_intrinsics"].shape == (meta["num_cams"], 20)
    print(f"    OK — dataset size={len(ds)}, sample shape={imgs_np.shape}, cams={meta['num_cams']}")

    # 1c. custom_collate_fn with real data
    print("[1c] custom_collate_fn with real seed4d data...")
    imgs0, meta0, _ = ds[0]
    sample = (imgs0, meta0, None)
    img_batch, meta_batch, _ = custom_collate_fn([sample])
    n = meta0["num_cams"]
    assert img_batch.shape == (1, n, 3, imgs0.shape[1], imgs0.shape[2]), f"Bad shape: {img_batch.shape}"
    assert meta_batch[0]["cam_mask"].all(), "cam_mask should be all True for single scene"
    assert meta_batch[0]["cam_mask"].shape == (n,)
    print(f"    OK — img_batch={img_batch.shape}, cam_mask all-true for {n} cameras")

    # 1d. build_img_batch (for use in model forward)
    print("[1d] build_img_batch (image load + normalize + K/c2w extraction)...")
    ego = scenes[0]
    nuscenes_tf = ego / "nuscenes" / "transforms" / "transforms_ego.json"
    imgs_np2, img_meta2 = build_img_batch(nuscenes_tf, resize_factor=0.25)
    assert imgs_np2.ndim == 4
    assert img_meta2["K"].shape[1:] == (3, 4)
    print(f"    OK — imgs shape={imgs_np2.shape}, n_cams={img_meta2['num_cams']}")

    # 1e. Sphere ray loading
    print("[1e] Sphere ray loading (3 frames, 1/100 subsampled)...")
    sphere_tf = ego / "sphere" / "transforms" / "transforms_ego.json"
    rays = build_sphere_rays(sphere_tf, max_frames=3, subsample=100)
    assert rays.ndim == 2 and rays.shape[1] in (10, 11), f"Expected (M, 10|11), got {rays.shape}"
    assert rays[:, 9].all(), "All mask values should be 1.0"
    print(f"    OK — rays shape={rays.shape}")

    print("\nPhase 1 PASSED")
    return ok


# ── Phase 2: Model forward + training ─────────────────────────────────────────


def run_phase2(scenes: list[Path], args) -> bool:
    print("\n=== Phase 2: Model forward pass + training ===")

    # Apply mmcv compat patches and import model.
    # force_pytorch_deform_attn() is only needed without mmcv._ext compiled;
    # with full mmcv, use the CUDA kernel directly.
    from utils.mmcv_compat import apply_patches

    apply_patches()

    from mmengine.config import Config

    from builder import model_builder
    from triplane_decoder.decoder import TriplaneDecoder
    from triplane_decoder.losses import compute_tv_loss
    from triplane_decoder.rendering import render_rays

    cfg = Config.fromfile(str(Path(__file__).parent / "config" / "config.py"))
    cfg.pif = False
    cfg.optimizer.lpips_loss_weight = 0.0
    cfg.optimizer.depth_loss_weight = 0.0

    # Reduce triplane resolution so the smoke test fits in an 11 GB GPU.
    # Production uses 200×200×16 (40 k+ queries) which needs > 12 GB activation RAM.
    # 64×64×8 gives 4 096+512+512 = 5 120 queries — comfortably fits in 11 GB.
    def _patch_tpv_res(obj, h=64, w=64, z=8):
        """Recursively replace tpv_h / tpv_w / tpv_z in a ConfigDict tree."""
        if hasattr(obj, "items"):
            for k, v in obj.items():
                if k == "tpv_h":
                    obj[k] = h
                elif k == "tpv_w":
                    obj[k] = w
                elif k == "tpv_z":
                    obj[k] = z
                elif k in ("h", "w", "z") and isinstance(v, int) and v in (200, 16):
                    # CustomPositionalEncoding uses h= / w= / z= directly
                    obj[k] = h if k == "h" else (w if k == "w" else z)
                else:
                    _patch_tpv_res(v, h, w, z)
        elif isinstance(obj, list):
            for item in obj:
                _patch_tpv_res(item, h, w, z)

    _patch_tpv_res(cfg.model)
    cfg.N_h_ = 64
    cfg.N_w_ = 64
    cfg.N_z_ = 8

    print("[2a] Building encoder + decoder...")
    triplane_encoder = model_builder.build(cfg.model).cuda()
    triplane_decoder = TriplaneDecoder(cfg).cuda()
    print(
        f"    Encoder params: {sum(p.numel() for p in triplane_encoder.parameters()):,}\n"
        f"    Decoder params: {sum(p.numel() for p in triplane_decoder.parameters()):,}"
    )

    optimizer = torch.optim.AdamW(
        list(triplane_encoder.parameters()) + list(triplane_decoder.parameters()),
        lr=cfg.optimizer.lr,
    )
    mse_loss_fn = torch.nn.MSELoss()

    for scene_idx, ego_dir in enumerate(scenes):
        nuscenes_tf = ego_dir / "nuscenes" / "transforms" / "transforms_ego.json"
        sphere_tf = ego_dir / "sphere" / "transforms" / "transforms_ego.json"
        print(f"\n[Scene {scene_idx + 1}/{len(scenes)}] {ego_dir.relative_to(SEED4D_ROOT)}")

        imgs_np, img_meta = build_img_batch(nuscenes_tf, resize_factor=args.img_scale)
        imgs_tensor = torch.from_numpy(imgs_np[None]).permute(0, 1, 4, 2, 3).cuda()
        print(f"    Input: {imgs_tensor.shape}  ({img_meta['num_cams']} cams)")

        batch = build_sphere_rays(sphere_tf, max_frames=args.max_sphere_frames, subsample=args.sphere_subsample)
        print(f"    Sphere rays: {batch.shape[0]}")
        batch_cuda = batch.cuda()

        # Training steps
        triplane_encoder.train()
        triplane_decoder.train()
        scaler = torch.cuda.amp.GradScaler()
        print(f"    Training ({args.train_steps} steps)...")
        for step in range(args.train_steps):
            with torch.cuda.amp.autocast():
                triplane, _ = triplane_encoder(img=imgs_tensor, img_metas=[img_meta])
            triplane_decoder.update_planes(triplane)

            ray_origins = batch_cuda[:, :3]
            ray_directions = batch_cuda[:, 3:6]
            rgb_gt = batch_cuda[:, 6:9]

            rendered, dist_loss, _ = render_rays(
                triplane_decoder, ray_origins, ray_directions, cfg, pif=None, training=True
            )

            loss = mse_loss_fn(rendered, rgb_gt)
            if cfg.optimizer.dist_loss_weight > 0:
                loss = loss + cfg.optimizer.dist_loss_weight * dist_loss
            if cfg.optimizer.tv_loss_weight > 0:
                loss = loss + cfg.optimizer.tv_loss_weight * compute_tv_loss(triplane_decoder)

            if loss.isnan():
                print(f"      step {step}: NaN loss")
                return False

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if cfg.optimizer.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(triplane_encoder.parameters()) + list(triplane_decoder.parameters()),
                    cfg.optimizer.clip_grad_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            print(f"      step {step}: loss={loss.item():.4f}")

        # Eval render
        print("    Eval render (16-row strip from sphere view 0)...")
        triplane_encoder.eval()
        triplane_decoder.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            triplane, _ = triplane_encoder(img=imgs_tensor, img_metas=[img_meta])
            triplane_decoder.update_planes(triplane)

            from triplane_decoder.intrinsics import Intrinsics
            from triplane_decoder.ray_utils import get_ray_directions, get_rays

            tf = json.loads(sphere_tf.read_text())
            w, h = tf.get("w", 1600), tf.get("h", 900)
            intrinsics = Intrinsics(w, h, tf["fl_x"], tf["fl_y"], tf["cx"], tf["cy"])
            directions = get_ray_directions(intrinsics)
            c2w = torch.tensor(tf["frames"][0]["transform_matrix"])[:3, :4]
            rays_o, rays_d = get_rays(directions.clone(), c2w)

            STRIP_H = 16
            rays_o_ = rays_o[: STRIP_H * w].cuda()
            rays_d_ = rays_d[: STRIP_H * w].cuda()

            chunks = []
            for i in range(0, rays_o_.size(0), cfg.decoder.testing_batch_size):
                ro = rays_o_[i : i + cfg.decoder.testing_batch_size]
                rd = rays_d_[i : i + cfg.decoder.testing_batch_size]
                chunk, _, _ = render_rays(triplane_decoder, ro, rd, cfg, pif=None, training=False)
                chunks.append(chunk)

            img_strip = torch.cat(chunks).reshape(STRIP_H, w, 3).clamp(0, 1)
            print(
                f"    Rendered {img_strip.shape}: "
                f"min={img_strip.min():.3f} max={img_strip.max():.3f} mean={img_strip.mean():.3f}"
            )

    print("\nPhase 2 PASSED")
    return True


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=3)
    # Defaults tuned for an 11 GB GPU (RTX 4070 Ti) with production ResNet-101 encoder.
    # TPV resolution is already reduced to 64×64×8 inside run_phase2.
    parser.add_argument("--max-sphere-frames", type=int, default=1)
    parser.add_argument("--sphere-subsample", type=int, default=2000)
    parser.add_argument("--img-scale", type=float, default=0.25)
    parser.add_argument("--phase1-only", action="store_true", help="Skip model (Phase 2)")
    args = parser.parse_args()

    base = SEED4D_ROOT / TOWN / "ClearNoon" / VEHICLE
    scenes = find_scenes(base)[: args.scenes]
    if not scenes:
        print(f"[error] No valid scenes in {base}")
        sys.exit(1)
    print(f"Found {len(scenes)} scene(s) in {base.relative_to(SEED4D_ROOT)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        p1_ok = run_phase1(scenes, Path(tmpdir))

    if args.phase1_only:
        print("\n[phase2 skipped: --phase1-only]")
        sys.exit(0 if p1_ok else 1)

    can_run, reason = check_model_deps()
    if not can_run:
        print(f"\n[phase2 skipped] {reason}")
        print("  To enable: build mmcv + tinycudann with nvcc (see Dockerfile)")
        sys.exit(0 if p1_ok else 1)

    try:
        p2_ok = run_phase2(scenes, args)
    except Exception as e:
        import traceback

        print(f"\nPhase 2 FAILED: {e}")
        traceback.print_exc()
        p2_ok = False

    if p1_ok and p2_ok:
        print("\n✓ All phases passed!")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
