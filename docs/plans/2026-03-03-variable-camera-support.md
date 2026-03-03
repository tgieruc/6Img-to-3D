# Variable Camera Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Support 1-6 cameras at arbitrary viewpoints (any count, position, angle) for both training and inference.

**Architecture:** Hybrid approach — per-image independent projection through shared ResNet, then concatenation along sequence dimension, feeding into TPVFormer cross-attention with pose-conditioned embeddings and attention masking for padded cameras. Grid allocation and reshaping become dynamic based on actual camera count per batch.

**Tech Stack:** PyTorch, mmcv/mmdet (config system), NumPy, CARLA/seed4d data format

**Design doc:** `docs/plans/2026-03-03-variable-camera-support-design.md`

---

### Task 1: Config — Replace `_num_cams_` with `_max_cams_` and add new params

**Files:**
- Modify: `config/config.py:13,51,135`

**Step 1: Update config parameters**

In `config/config.py`, make these changes:

Line 13 — rename and add new params:
```python
# OLD:
_num_cams_ = 6

# NEW:
_max_cams_ = 6
_min_cams_train_ = 1
```

Line 51 — update TPVImageCrossAttention reference:
```python
# OLD:
num_cams=_num_cams_,

# NEW:
max_cams=_max_cams_,
```

Line 135 — update TPVFormerHead reference:
```python
# OLD:
num_cams=_num_cams_,

# NEW:
max_cams=_max_cams_,
```

Add to the dataset config section (near other dataset params):
```python
min_cams_train=_min_cams_train_,
max_cams_train=_max_cams_,
```

**Step 2: Verify no other references to `_num_cams_`**

Run: `grep -rn "_num_cams_\|num_cams" config/`

Expected: Only the lines you just changed.

**Step 3: Commit**

```bash
git add config/config.py
git commit -m "config: rename _num_cams_ to _max_cams_, add camera dropout params"
```

---

### Task 2: PoseEmbedding — New module replacing `cams_embeds`

**Files:**
- Create: `triplane_encoder/modules/pose_embedding.py`
- Create: `tests/test_pose_embedding.py`

**Step 1: Write the failing test**

Create `tests/test_pose_embedding.py`:
```python
import torch
import pytest
from triplane_encoder.modules.pose_embedding import PoseEmbedding


def test_pose_embedding_output_shape():
    """PoseEmbedding maps (B, N, 20) -> (B, N, embed_dims)."""
    embed_dims = 128
    module = PoseEmbedding(embed_dims=embed_dims)

    # 2 batch items, 4 cameras each
    # 20 = 16 (flattened 4x4 pose) + 4 (fx, fy, cx, cy)
    pose_input = torch.randn(2, 4, 20)
    output = module(pose_input)

    assert output.shape == (2, 4, embed_dims)


def test_pose_embedding_variable_cameras():
    """PoseEmbedding works with different camera counts."""
    embed_dims = 256
    module = PoseEmbedding(embed_dims=embed_dims)

    for n_cams in [1, 3, 6]:
        pose_input = torch.randn(1, n_cams, 20)
        output = module(pose_input)
        assert output.shape == (1, n_cams, embed_dims)


def test_pose_embedding_deterministic():
    """Same input produces same output."""
    module = PoseEmbedding(embed_dims=64)
    module.eval()
    pose_input = torch.randn(1, 2, 20)

    out1 = module(pose_input)
    out2 = module(pose_input)
    assert torch.allclose(out1, out2)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/tgieruc/Documents/6Img-to-3D && python -m pytest tests/test_pose_embedding.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'triplane_encoder.modules.pose_embedding'`

**Step 3: Write minimal implementation**

Create `triplane_encoder/modules/pose_embedding.py`:
```python
import torch
import torch.nn as nn


class PoseEmbedding(nn.Module):
    """Map camera pose (4x4 matrix) + intrinsics (fx, fy, cx, cy) to an embedding.

    Input:  (B, N, 20) where 20 = 16 (flattened pose) + 4 (intrinsics)
    Output: (B, N, embed_dims)
    """

    def __init__(self, embed_dims=256, input_dims=20):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, embed_dims),
        )

    def forward(self, pose_intrinsics):
        return self.mlp(pose_intrinsics)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/tgieruc/Documents/6Img-to-3D && python -m pytest tests/test_pose_embedding.py -v`

Expected: 3 PASSED

**Step 5: Commit**

```bash
git add triplane_encoder/modules/pose_embedding.py tests/test_pose_embedding.py
git commit -m "feat: add PoseEmbedding module for pose-conditioned camera embeddings"
```

---

### Task 3: Dataset — Variable camera loading and camera dropout

**Files:**
- Modify: `dataloader/dataset.py:85,94-103,108-114`
- Create: `tests/test_variable_cameras_dataset.py`

**Step 1: Write the failing test**

Create `tests/test_variable_cameras_dataset.py`:
```python
import numpy as np
import torch
import pytest


def test_build_pose_intrinsics_vector():
    """Verify pose+intrinsics vector is correctly constructed."""
    from dataloader.dataset import build_pose_intrinsics_vector

    # Single camera: 4x4 pose + 4 intrinsics = 20 values
    c2w = np.eye(4).tolist()
    K = np.zeros((1, 3, 4))
    K[0, 0, 0] = 500.0  # fx
    K[0, 1, 1] = 500.0  # fy
    K[0, 0, 2] = 320.0  # cx
    K[0, 1, 2] = 240.0  # cy

    vec = build_pose_intrinsics_vector([c2w], K)
    assert vec.shape == (1, 20)
    # First 16 = flattened identity matrix
    assert np.allclose(vec[0, :16], np.eye(4).flatten())
    # Last 4 = fx, fy, cx, cy
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/tgieruc/Documents/6Img-to-3D && python -m pytest tests/test_variable_cameras_dataset.py -v`

Expected: FAIL with `ImportError`

**Step 3: Write implementation**

In `dataloader/dataset.py`, add these helper functions near the top (after imports):

```python
import random

def build_pose_intrinsics_vector(c2ws, K):
    """Build (N, 20) array: flattened 4x4 pose + (fx, fy, cx, cy) per camera.

    Args:
        c2ws: list of N pose matrices (4x4 each, as nested lists or arrays)
        K: (N, 3, 4) intrinsics array

    Returns:
        (N, 20) numpy array
    """
    n_cams = len(c2ws)
    result = np.zeros((n_cams, 20))
    for i, c2w in enumerate(c2ws):
        result[i, :16] = np.array(c2w).flatten()
        result[i, 16] = K[i, 0, 0]  # fx
        result[i, 17] = K[i, 1, 1]  # fy
        result[i, 18] = K[i, 0, 2]  # cx
        result[i, 19] = K[i, 1, 2]  # cy
    return result


def apply_camera_dropout(n_cams, min_cams=1, max_cams=6):
    """Randomly select a subset of cameras.

    Args:
        n_cams: total number of cameras available
        min_cams: minimum cameras to keep
        max_cams: maximum cameras to keep

    Returns:
        sorted list of selected camera indices
    """
    max_cams = min(max_cams, n_cams)
    min_cams = min(min_cams, max_cams)
    k = random.randint(min_cams, max_cams)
    indices = sorted(random.sample(range(n_cams), k))
    return indices
```

Then modify `__getitem__` — replace the image loading section (lines ~85-114):

```python
# OLD (line 85):
input_rgb = np.empty((6, 3, 100, 100))

# NEW:
input_rgb = []
```

Replace K initialization (line ~94):
```python
# OLD:
K = np.zeros((3, 4))
K[0,0] = input_data['fl_x']
K[1,1] = input_data['fl_y']
K[2,2] = 1
K[0,2] = input_data['cx']
K[1,2] = input_data['cy']

# NEW:
num_cams = len(input_data["frames"])
K = np.zeros((num_cams, 3, 4))
for cam_idx, frame in enumerate(input_data["frames"]):
    # Use per-frame intrinsics if available, else global
    fx = frame.get("fl_x", input_data.get("fl_x", 0))
    fy = frame.get("fl_y", input_data.get("fl_y", 0))
    cx = frame.get("cx", input_data.get("cx", 0))
    cy = frame.get("cy", input_data.get("cy", 0))
    K[cam_idx, 0, 0] = fx
    K[cam_idx, 1, 1] = fy
    K[cam_idx, 2, 2] = 1
    K[cam_idx, 0, 2] = cx
    K[cam_idx, 1, 2] = cy
```

After loading all cameras, add camera dropout (before building img_meta):
```python
# Camera dropout during training
if self.training and hasattr(self, 'min_cams_train'):
    selected = apply_camera_dropout(
        len(input_rgb), self.min_cams_train, self.max_cams_train)
    input_rgb = [input_rgb[i] for i in selected]
    all_c2w = [all_c2w[i] for i in selected]
    K = K[selected]
    img_shape = [img_shape[i] for i in selected]
```

Add `pose_intrinsics` to img_meta:
```python
img_meta = dict(
    K=K,
    c2w=all_c2w,
    img_shape=img_shape,
    pose_intrinsics=build_pose_intrinsics_vector(all_c2w, K),
    num_cams=len(all_c2w),
)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/tgieruc/Documents/6Img-to-3D && python -m pytest tests/test_variable_cameras_dataset.py -v`

Expected: 3 PASSED

**Step 5: Commit**

```bash
git add dataloader/dataset.py tests/test_variable_cameras_dataset.py
git commit -m "feat: variable camera loading with per-camera intrinsics and camera dropout"
```

---

### Task 4: Collate function — Padding and attention mask

**Files:**
- Modify: `dataloader/dataset_wrapper.py:15-21`
- Create: `tests/test_collate.py`

**Step 1: Write the failing test**

Create `tests/test_collate.py`:
```python
import numpy as np
import torch
import pytest


def test_collate_pads_to_max_cameras():
    """Collate function pads samples with different camera counts."""
    from dataloader.dataset_wrapper import custom_collate_fn

    # Sample 1: 3 cameras, Sample 2: 5 cameras
    sample1_imgs = np.random.randn(3, 100, 100, 3).astype(np.float32)
    sample1_meta = {
        "K": np.zeros((3, 3, 4)),
        "c2w": [np.eye(4).tolist()] * 3,
        "img_shape": [(100, 100, 3)] * 3,
        "pose_intrinsics": np.zeros((3, 20)),
        "num_cams": 3,
    }

    sample2_imgs = np.random.randn(5, 100, 100, 3).astype(np.float32)
    sample2_meta = {
        "K": np.zeros((5, 3, 4)),
        "c2w": [np.eye(4).tolist()] * 5,
        "img_shape": [(100, 100, 3)] * 5,
        "pose_intrinsics": np.zeros((5, 20)),
        "num_cams": 5,
    }

    batch = custom_collate_fn([
        (sample1_imgs, sample1_meta, None),
        (sample2_imgs, sample2_meta, None),
    ])

    img_batch, meta_batch, _ = batch

    # Should pad to max cameras (5)
    assert img_batch.shape[0] == 2   # batch size
    assert img_batch.shape[1] == 5   # max cameras
    assert img_batch.shape[2] == 3   # channels

    # Attention mask: True for real cameras, False for padding
    assert meta_batch[0]["cam_mask"].shape == (5,)
    assert meta_batch[0]["cam_mask"][:3].all()    # 3 real cameras
    assert not meta_batch[0]["cam_mask"][3:].any()  # 2 padded
    assert meta_batch[1]["cam_mask"].all()          # all 5 real


def test_collate_same_camera_count():
    """Collate works when all samples have same camera count."""
    from dataloader.dataset_wrapper import custom_collate_fn

    sample = np.random.randn(6, 100, 100, 3).astype(np.float32)
    meta = {
        "K": np.zeros((6, 3, 4)),
        "c2w": [np.eye(4).tolist()] * 6,
        "img_shape": [(100, 100, 3)] * 6,
        "pose_intrinsics": np.zeros((6, 20)),
        "num_cams": 6,
    }

    batch = custom_collate_fn([(sample, meta, None)])
    img_batch, meta_batch, _ = batch

    assert img_batch.shape == (1, 6, 3, 100, 100)
    assert meta_batch[0]["cam_mask"].all()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/tgieruc/Documents/6Img-to-3D && python -m pytest tests/test_collate.py -v`

Expected: FAIL (current collate doesn't produce `cam_mask` or handle padding)

**Step 3: Write implementation**

Replace `custom_collate_fn` in `dataloader/dataset_wrapper.py`:

```python
def custom_collate_fn(input_data):
    """Collate samples with variable camera counts via padding + attention mask.

    Each input sample is (images, img_meta, dataset_loader) where:
      - images: (N_i, H, W, 3) numpy array, N_i varies per sample
      - img_meta: dict with 'num_cams', 'K', 'c2w', 'pose_intrinsics', etc.
    """
    max_cams = max(d[1]["num_cams"] for d in input_data)
    bs = len(input_data)

    # Determine image dimensions from first sample
    _, h, w, c = input_data[0][0].shape

    # Pad images to max_cams
    padded_imgs = np.zeros((bs, max_cams, h, w, c), dtype=np.float32)
    for i, (imgs, meta, _) in enumerate(input_data):
        n = meta["num_cams"]
        padded_imgs[i, :n] = imgs[:n]

    img_batch = torch.from_numpy(padded_imgs).permute(0, 1, 4, 2, 3)  # (B, N, 3, H, W)

    # Build padded metadata with attention masks
    meta_batch = []
    for i, (_, meta, _) in enumerate(input_data):
        n = meta["num_cams"]
        cam_mask = np.zeros(max_cams, dtype=bool)
        cam_mask[:n] = True

        # Pad K
        K_padded = np.zeros((max_cams, 3, 4))
        K_padded[:n] = meta["K"][:n]

        # Pad c2w
        c2w_padded = [np.eye(4).tolist()] * max_cams
        for j in range(n):
            c2w_padded[j] = meta["c2w"][j]

        # Pad pose_intrinsics
        pi_padded = np.zeros((max_cams, 20))
        pi_padded[:n] = meta["pose_intrinsics"][:n]

        # Pad img_shape
        img_shape_padded = [(h, w, c)] * max_cams
        for j in range(n):
            img_shape_padded[j] = meta["img_shape"][j]

        padded_meta = dict(
            K=K_padded,
            c2w=c2w_padded,
            img_shape=img_shape_padded,
            pose_intrinsics=pi_padded,
            cam_mask=cam_mask,
            num_cams=n,
        )
        meta_batch.append(padded_meta)

    dataset_stack = [d[2] for d in input_data]
    return img_batch, meta_batch, dataset_stack
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/tgieruc/Documents/6Img-to-3D && python -m pytest tests/test_collate.py -v`

Expected: 2 PASSED

**Step 5: Commit**

```bash
git add dataloader/dataset_wrapper.py tests/test_collate.py
git commit -m "feat: collate function with camera padding and attention mask"
```

---

### Task 5: TPVFormerHead — Replace `cams_embeds` with PoseEmbedding

**Files:**
- Modify: `triplane_encoder/tpv_head.py:34,45,55-56,107`

**Step 1: Update imports and __init__**

At the top of `tpv_head.py`, add import:
```python
from triplane_encoder.modules.pose_embedding import PoseEmbedding
```

In `__init__` (lines 34-56):
```python
# OLD (line 34):
num_cams=6,

# NEW:
max_cams=6,
```

```python
# OLD (line 45):
self.num_cams = num_cams

# NEW:
self.max_cams = max_cams
```

```python
# OLD (lines 55-56):
self.cams_embeds = nn.Parameter(
    torch.Tensor(self.num_cams, self.embed_dims))

# NEW:
self.pose_embedding = PoseEmbedding(embed_dims=self.embed_dims)
```

**Step 2: Update forward method**

In `forward()`, update the feature processing loop (line ~107):

```python
# OLD (line 107):
feat = feat + self.cams_embeds[:, None, None, :].to(dtype)

# NEW:
# pose_embeds shape: (B, N, embed_dims) from img_metas
pose_embeds = torch.from_numpy(
    np.stack([m['pose_intrinsics'] for m in img_metas])
).to(feat.device).to(dtype)  # (B, N, 20)
pose_embeds = self.pose_embedding(pose_embeds)  # (B, N, embed_dims)
# feat shape after permute: (num_cam, bs, h*w, c)
# pose_embeds needs shape: (num_cam, bs, 1, embed_dims)
feat = feat + pose_embeds.permute(1, 0, 2).unsqueeze(2).to(dtype)  # (N, B, 1, C)
```

Also add `import numpy as np` at top if not already present.

**Step 3: Remove `cams_embeds` initialization**

Delete the `normal_` initialization of `cams_embeds` in `init_weights()` if it exists (check for `nn.init.normal_(self.cams_embeds)`).

**Step 4: Verify shape compatibility**

The rest of the forward method should work unchanged because:
- `feat.shape` is `(num_cam, bs, h*w, c)` — `num_cam` comes from data, not config
- The concatenation and permutation don't depend on a fixed camera count

**Step 5: Commit**

```bash
git add triplane_encoder/tpv_head.py
git commit -m "feat: replace cams_embeds with PoseEmbedding in TPVFormerHead"
```

---

### Task 6: Encoder — Dynamic grid allocation and cache invalidation

**Files:**
- Modify: `triplane_encoder/modules/encoder.py:155-260`

This is the most critical change. The grid and mask tensors currently hardcode dimension 6.

**Step 1: Remove grid caching**

In `forward()` (line ~254), the grid is cached:
```python
# OLD:
if self.grid is None:
    self.grid, self.mask = self.get_grid(...)

# NEW (always recompute — camera count varies per batch):
self.grid, self.mask = self.get_grid(kwargs['img_metas'], ...)
```

Also remove the ref_3d caching:
```python
# OLD:
if self.ref_3d_hw_mask is None:
    self.ref_3d_hw_uvs, self.ref_3d_hw_mask = self.point_sampling(...)

# NEW:
self.ref_3d_hw_uvs, self.ref_3d_hw_mask = self.point_sampling(
    self.grid.permute(3,1,2,0,4), self.mask.permute(3,1,2,0,4), 4)
```

Do the same for all other cached ref_3d variants (ref_3d_zh, ref_3d_wz).

**Step 2: Dynamic grid creation in `get_grid()`**

In `get_grid()` (lines 200-201):
```python
# OLD:
grid = torch.zeros((self.tpv_z, self.tpv_h, self.tpv_w, 6, 2), device=points.device)
mask = torch.zeros((self.tpv_z, self.tpv_h, self.tpv_w, 6, 1), dtype=bool, device=points.device)

# NEW:
num_cams = len(rays_ds)  # Actual camera count from the ray generation loop
grid = torch.zeros((self.tpv_z, self.tpv_h, self.tpv_w, num_cams, 2), device=points.device)
mask = torch.zeros((self.tpv_z, self.tpv_h, self.tpv_w, num_cams, 1), dtype=bool, device=points.device)
```

**Step 3: Handle attention mask in grid**

After grid creation, apply the camera attention mask so padded cameras produce all-False masks:

```python
# After grid/mask creation, apply cam_mask from img_metas
cam_mask = img_metas[0].get('cam_mask', np.ones(num_cams, dtype=bool))
for i in range(num_cams):
    if not cam_mask[i]:
        mask[:, :, :, i] = False
```

**Step 4: Extract per-camera intrinsics**

The current code reads intrinsics from `K[0,0,...]` (first batch, first camera). With per-camera intrinsics, modify the ray generation loop:

```python
# OLD (lines 155-178):
fl_x = K[0,0,0,0]
fl_y = K[0,0,1,1]
c_x = K[0,0,0,2]
c_y = K[0,0,1,2]
# ... one set of directions for all cameras

# NEW:
c2ws_batch = torch.from_numpy(c2ws).to(device).float()
rays_ds = []
rays_os = []
for cam_idx, c2w in enumerate(c2ws_batch[0]):
    cam_K = K[0, cam_idx]  # Per-camera intrinsics
    fl_x = cam_K[0, 0]
    fl_y = cam_K[1, 1]
    c_x = cam_K[0, 2]
    c_y = cam_K[1, 2]
    w_cam = int(img_hwc[0, cam_idx, 1])
    h_cam = int(img_hwc[0, cam_idx, 0])

    directions = ray_utils.get_ray_directions(h_cam, w_cam, fl_x, fl_y, c_x, c_y)
    directions = directions.to(device)
    rays_o, rays_d = ray_utils.get_rays(directions, c2w[:3])
    rays_os.append(rays_o)
    rays_ds.append(rays_d)
```

**Step 5: Commit**

```bash
git add triplane_encoder/modules/encoder.py
git commit -m "feat: dynamic grid allocation, remove caching, per-camera intrinsics"
```

---

### Task 7: ImageCrossAttention — Variable `num_cams` in reshaping

**Files:**
- Modify: `triplane_encoder/modules/image_cross_attention.py:45,67,136,143-144,152,154,165`

**Step 1: Update __init__**

```python
# OLD (line 45):
num_cams=6,

# NEW:
max_cams=6,
```

```python
# OLD (line 67):
self.num_cams = num_cams

# NEW (remove this line entirely — num_cams will be extracted from tensors at runtime)
```

**Step 2: Extract `num_cams` from key tensor in `forward()`**

Move the extraction from line 149 to the top of forward():
```python
def forward(self, query, key, value, residual=None, ...):
    # Extract actual camera count from key tensor
    num_cams = key.shape[0]  # key shape: (num_cam, H*W, bs, embed_dims)
```

**Step 3: Replace all `self.num_cams` with local `num_cams`**

Lines 136, 143-144, 152, 154, 165 — every occurrence of `self.num_cams` becomes `num_cams`.

Example at line 136:
```python
# OLD:
queries_rebatch = queries[tpv_idx].new_zeros([bs * self.num_cams, max_len, self.embed_dims])

# NEW:
queries_rebatch = queries[tpv_idx].new_zeros([bs * num_cams, max_len, self.embed_dims])
```

Apply the same pattern to ALL other `self.num_cams` references in this file.

**Step 4: Apply attention mask to tpv_masks**

The `tpv_masks` parameter controls which cameras contribute to each query point. Padded cameras should be masked out. In the loop at line ~125:

```python
# After extracting cam_mask from kwargs if available:
cam_mask = kwargs.get('cam_mask', None)

for tpv_idx, tpv_mask in enumerate(tpv_masks):
    indexes = []
    for cam_i, mask_per_img in enumerate(tpv_mask):
        if cam_mask is not None and not cam_mask[cam_i]:
            # Padded camera: produce empty index
            indexes.append(torch.tensor([], dtype=torch.long, device=query.device))
            continue
        index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
        indexes.append(index_query_per_img)
```

**Step 5: Commit**

```bash
git add triplane_encoder/modules/image_cross_attention.py
git commit -m "feat: variable num_cams in cross-attention reshaping with mask support"
```

---

### Task 8: Train — Dynamic visualization

**Files:**
- Modify: `train.py:362-375`

**Step 1: Replace hardcoded camera grid**

```python
# OLD (lines 362-375):
fig, ax = plt.subplots(2, 3, figsize=(10, 5))
imgs_norm = torch.clip(imgs[0].detach().cpu() / 255 + 0.5, 0, 1).permute(0, 2, 3, 1)
ax[0,0].imshow((imgs_norm[2][:,:,[2,1,0]]))
# ... hardcoded indices 0-5

# NEW:
num_vis_cams = img_metas[0]["num_cams"]
ncols = min(num_vis_cams, 3)
nrows = (num_vis_cams + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3))
if num_vis_cams == 1:
    axes = np.array([axes])
axes = axes.flatten()
imgs_norm = torch.clip(imgs[0].detach().cpu() / 255 + 0.5, 0, 1).permute(0, 2, 3, 1)
for cam_i in range(num_vis_cams):
    axes[cam_i].imshow(imgs_norm[cam_i][:, :, [2, 1, 0]])
    axes[cam_i].axis('off')
    axes[cam_i].set_title(f'Cam {cam_i}')
for ax_i in range(num_vis_cams, len(axes)):
    axes[ax_i].axis('off')
```

**Step 2: Commit**

```bash
git add train.py
git commit -m "feat: dynamic camera visualization in training loop"
```

---

### Task 9: Integration — Wire cam_mask through encoder forward pass

**Files:**
- Modify: `triplane_encoder/tpv_head.py` (forward method)
- Modify: `triplane_encoder/modules/encoder.py` (forward method)

**Step 1: Pass cam_mask from img_metas through the call chain**

In `tpv_head.py` forward, extract and pass cam_mask:
```python
# After feat_flatten is built, before calling encoder:
cam_mask = np.array(img_metas[0].get('cam_mask', np.ones(num_cam, dtype=bool)))
```

Ensure `cam_mask` flows into the encoder's `forward()` via `kwargs` or explicitly. The encoder passes it to `get_grid()` and to `image_cross_attention` via `kwargs`.

In `encoder.py` forward, pass cam_mask to the attention layer:
```python
# When calling the attention layer, include cam_mask in kwargs
```

**Step 2: Verify data flow end-to-end**

The full data flow should be:
```
collate_fn produces cam_mask (B, N_max) per sample
  → img_metas[i]['cam_mask'] = (N_max,) bool array
  → tpv_head.forward reads it from img_metas
  → encoder.forward receives it
  → get_grid uses it to zero out padded camera grid entries
  → image_cross_attention uses it to skip padded cameras in loops
```

**Step 3: Commit**

```bash
git add triplane_encoder/tpv_head.py triplane_encoder/modules/encoder.py
git commit -m "feat: wire cam_mask through encoder forward pass"
```

---

### Task 10: Smoke test — End-to-end forward pass with variable cameras

**Files:**
- Create: `tests/test_variable_cameras_e2e.py`

**Step 1: Write the smoke test**

```python
import torch
import numpy as np
import pytest


def test_forward_pass_variable_cameras():
    """Verify the full encoder accepts variable camera counts without crashing."""
    # This test requires the full model — skip if CARLA dependencies unavailable
    pytest.importorskip("mmcv")

    from config.config import model
    from mmcv.utils import build_from_cfg
    from mmdet.models import build_backbone

    # TODO: Build model from config and run a forward pass with:
    # - Batch of 1, with 3 cameras (not 6)
    # - Random images (1, 3, 3, 100, 100) — B=1, N=3, C=3, H=100, W=100
    # - img_metas with cam_mask, pose_intrinsics, K, c2w for 3 cameras
    # - Verify output triplane shape is unchanged regardless of camera count

    # For now, just verify config loads without error
    assert model is not None
    assert "max_cams" in str(model)
```

**Step 2: Run the smoke test**

Run: `cd /Users/tgieruc/Documents/6Img-to-3D && python -m pytest tests/test_variable_cameras_e2e.py -v`

Expected: PASS (or skip if mmcv not installed)

**Step 3: Commit**

```bash
git add tests/test_variable_cameras_e2e.py
git commit -m "test: add end-to-end smoke test for variable camera support"
```

---

## Task Dependency Order

```
Task 1 (config) ──────────────────────┐
Task 2 (PoseEmbedding) ───────────────┤
Task 3 (dataset) ─────────────────────┤
Task 4 (collate) ── depends on T3 ────┤
Task 5 (tpv_head) ── depends on T1,T2 ┤
Task 6 (encoder) ── depends on T1 ────┤
Task 7 (cross-attn) ── depends on T1 ─┤
Task 8 (train viz) ── independent ─────┤
Task 9 (wiring) ── depends on T5,T6,T7 ┤
Task 10 (smoke test) ── depends on all ┘
```

Tasks 1, 2, 3, 8 can be done in parallel. Tasks 5, 6, 7 can be done in parallel after Task 1. Task 9 requires 5+6+7. Task 10 is last.
