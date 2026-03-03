# Variable Camera Support Design

## Problem

6Img-to-3D assumes exactly 6 cameras at fixed nuscenes positions. We want to support 1-6 cameras at fully arbitrary viewpoints — any count, any position, any angle — both at training and inference time.

## Approach

Hybrid of per-image independent processing and joint cross-attention reasoning. Keep TPVFormer's cross-attention (the core multi-view reasoning mechanism) but make its input pipeline camera-count-agnostic.

### Architecture

```
Current:  6 images -> ResNet -> TPVFormer cross-attention (hardcoded 6) -> triplane -> decoder -> NeRF

Proposed: N images -> ResNet -> per-image projection -> N partial feature maps
          N partial feature maps -> concatenation along sequence dim
          TPVFormer cross-attention (pose-conditioned, attention-masked) -> triplane -> decoder -> NeRF
```

### What changes
- `cams_embeds` (learned per-index) replaced by pose-conditioned MLP
- Projection grid allocated dynamically for N cameras
- Cross-attention receives variable-length key/values with attention mask
- Dataloader supports variable camera count with padding and camera dropout

### What stays untouched
- ResNet backbone
- TPVFormerEncoder cross-attention layers
- Triplane decoder + volume renderer
- PIF module
- Ray sampling and rendering
- Loss computation

## Component Design

### 1. Pose-Conditioned Camera Embeddings

Replaces the fixed `cams_embeds = nn.Parameter(num_cams, embed_dims)` in `tpv_head.py`.

**New module — PoseEmbedding MLP:**
- Input: flattened 4x4 pose matrix (16 values) + intrinsics fx, fy, cx, cy (4 values) = 20 values
- Architecture: Linear(20, embed_dims) -> GELU -> Linear(embed_dims, embed_dims)
- Output: (embed_dims,) per camera
- Added to image features the same way `cams_embeds` was used at `tpv_head.py:107`

Including intrinsics allows the model to reason about different FOVs/focal lengths across cameras.

### 2. Per-Image Projection & Concatenation

**Per-image backbone**: Each of N images goes through the shared ResNet. Input shape changes from `(B*6, C, H, W)` to `(B*N, C, H, W)`.

**Per-image triplane projection**: Each image's 2D features are projected into 3D triplane space using that camera's pose and intrinsics. Same projection math as current grid computation in `encoder.py:200-201`, but done per-camera independently.

**Aggregation via concatenation**: All N cameras' projected features are stacked along the sequence dimension and passed as key/values to cross-attention. The attention mechanism attends over all N views jointly, naturally handling variable length.

### 3. Encoder Modifications

**Dynamic grid allocation** (`encoder.py:200-201`):
- Current: `grid = zeros(Z, H, W, 6, 2)` with hardcoded 6
- New: `grid = zeros(Z, H, W, N, 2)` where N = actual camera count

**Cross-attention reshaping** (`image_cross_attention.py:136,152,154`):
- Replace fixed `self.num_cams` class attribute with per-forward-call `num_cams` from actual batch
- Reshape logic `(B * N, seq_len, dims)` stays identical, N just varies

**Attention masking for padding**:
- Within a batch, samples may have different camera counts
- Pad all samples to max N in the batch
- Produce attention mask `(B, N_max)` — True for real cameras, False for padding
- Cross-attention ignores padded camera slots via this mask

### 4. Dataloader & Training

**Dataset** (`dataloader/dataset.py`):
- Remove hardcoded `np.empty((6, 3, 100, 100))` at line 85
- Each sample returns variable number of images, poses, intrinsics
- Add `num_cameras` field to sample metadata

**Custom collate function**:
- Find max N in the batch
- Pad images, poses, intrinsics to max N with zeros
- Produce attention mask `(B, N_max)` boolean tensor
- Mask flows through encoder to cross-attention

**Camera dropout augmentation**:
- During training with seed4d's fixed 6-camera data, each sample randomly selects K cameras where K ~ Uniform(1, 6)
- K cameras chosen randomly from the 6 available
- Forces model to reconstruct from incomplete coverage
- At inference, pass whatever cameras you have

**Visualization** (`train.py:363-374`):
- Replace hardcoded 2x3 grid indexing cameras 0-5
- Dynamic layout showing however many cameras are in the sample

### 5. Config

- `_num_cams_ = 6` becomes `_max_cams_ = 6` (upper bound for padding)
- Add `min_cams_train = 1` (minimum cameras during dropout)
- Add `pose_embed_dims` (defaults to same as `embed_dims`)

## Files Modified

| File | Change | Severity |
|------|--------|----------|
| `config/config.py` | `_num_cams_` -> `_max_cams_`, add dropout params | Low |
| `triplane_encoder/tpv_head.py` | Replace `cams_embeds` with PoseEmbedding MLP | Medium |
| `triplane_encoder/modules/encoder.py:200-201` | Dynamic grid allocation | High |
| `triplane_encoder/modules/encoder.py:258-260` | Dynamic point sampling | High |
| `triplane_encoder/modules/image_cross_attention.py` | Variable N in reshape, attention mask | Medium |
| `dataloader/dataset.py:85` | Remove hardcoded 6, variable camera loading | Medium |
| `dataloader/dataset.py` (new) | Custom collate function with padding + mask | Medium |
| `train.py:363-374` | Dynamic visualization | Low |

## Files Untouched

- `triplane_decoder/` — entire decoder directory
- `triplane_encoder/modules/encoder.py` (cross-attention layer logic itself)
- `dataloader/rays_dataset.py` — ray-based dataset for decoder training
- `eval.py` — evaluation pipeline
- All loss functions
