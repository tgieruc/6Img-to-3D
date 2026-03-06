# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

6Img-to-3D reconstructs large-scale outdoor driving scenes from six input images into a triplane representation for novel view synthesis. Research code from co-pace GmbH, published at IEEE IV 2025. Uses CARLA simulator data.

## Environment Setup

Requires Python >= 3.10, CUDA 12.6, PyTorch 2.6+. Uses [uv](https://docs.astral.sh/uv/) for dependency management. Key non-standard dependencies: tinycudann (NVIDIA tiny-cuda-nn), mmengine/mmcv/mmseg (OpenMMLab v2/v3 suite).

```bash
uv sync
uv pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
# mmcv must be built from source (nvcc required, mmseg needs mmcv<2.2.0):
CC=/usr/bin/gcc CXX=/usr/bin/g++ PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST="8.9" \
  uv pip install "mmcv>=2.0.0rc4,<2.2.0" --no-build-isolation
```

> **TORCH_CUDA_ARCH_LIST** — set to match your GPU: `7.5` (T4/RTX 2080), `8.0` (A100), `8.6` (RTX 3090), `8.9` (RTX 4070 Ti / 4090), `9.0` (H100).

For development: `uv sync --group dev && uv run pre-commit install`

Docker alternative: `docker build -t 6img-to-3d .` (CUDA 12.6 base).

## Commands

```bash
# Training
python train.py --py-config config/config.py --log-dir <run_name>

# Evaluation (renders images, computes PSNR/LPIPS/SSIM)
python eval.py --py-config config/config.py --resume-from <checkpoint.pth> --log-dir <run_name> --depth --img-gt

# Data preprocessing (converts dataset to pickles for faster loading)
python utils/pickles_generator.py --dataset-config config/_base_/dataset.py --py-config config/config.py
```

Linting: `uv run ruff check .` and `uv run ruff format --check .`
Pre-commit: `uv run pre-commit run --all-files`

## Architecture

**Encoder-decoder pipeline:**

```
6 Images → ResNet-101+FPN → Multi-scale Features
  → TPVFormerHead (queries + positional encoding)
  → TPVFormerEncoder (5 transformer layers: image cross-attn + cross-view self-attn)
  → 3 Triplane features (hw, zh, zw)
  → TriplaneDecoder (tinycudann MLP, 5 hidden layers)
  → Coarse+Fine ray sampling → Volume rendering → Novel view RGB
```

**Key modules:**

- `triplane_encoder/` — Image-to-triplane encoding via TPVFormer transformer. `tpvformer.py` is the entry point model; `tpv_head.py` generates triplane queries and reference points; `modules/` contains attention layers (image cross-attention, cross-view hybrid attention, deformable attention).
- `triplane_decoder/` — Triplane-to-image rendering. `decoder.py` holds three learned planes and a tinycudann MLP outputting RGB+density. `rendering.py` orchestrates volume rendering with coarse (uniform) + fine (PDF importance) sampling from `ray_samplers.py`. `losses.py` defines MSE, LPIPS, TV, distortion, and depth losses.
- `dataloader/` — `dataset.py` defines CarlaDataset/PickledCarlaDataset with hierarchical structure (Town/Weather/Vehicle/SpawnPoint/Step). `rays_dataset.py` provides ray-level sampling where each ray is an 11-dim vector [origin(3), direction(3), rgb_gt(3), mask(1), depth_gt(1)].
- `builder/` — `model_builder.py` constructs TPVFormer from config dicts (MMDetection registry pattern); `data_builder.py` constructs dataloaders.

## Configuration System

MMDetection-style Python config with base inheritance. Main config: `config/config.py` inherits from `config/_base_/{dataset,optimizer,triplane_decoder}.py`.

Key parameters in `config/config.py`: `_dim_=128` (feature dim), `N_h_/N_w_/N_z_=200/200/16` (plane resolution), `tpv_encoder_layers=5`, `pif=True` (projected image features conditioning).

## Training Details

- Losses: MSE + LPIPS (weight 0.2) + total variation + distortion + depth (weight 1.0)
- Optimizer: configurable in `config/_base_/optimizer.py`, default lr=5e-5
- Checkpoints saved by best PSNR and best LPIPS to `log_dir/`
- TensorBoard logging via tensorboardX
