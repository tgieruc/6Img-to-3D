# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

6Img-to-3D reconstructs large-scale outdoor driving scenes from six input images into a triplane representation for novel view synthesis. Research code from co-pace GmbH, published at IEEE IV 2025. Uses CARLA simulator data.

## Environment Setup

Requires Python 3.8, CUDA 11.8, PyTorch 2.0.1. Key non-standard dependencies: tinycudann (NVIDIA tiny-cuda-nn), mmdet/mmcv/mmseg/mmcls (OpenMMLab suite).

```bash
conda create -n sixtothree python=3.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install mmdet==2.20.0 mmengine==0.8.4 mmsegmentation==0.20.0 mmcls==0.25.0 mmcv-full==1.5.0
pip install -r requirements.txt
```

Docker alternative: `docker build -t 6img .` (CUDA 11.8 base, ~30 min build).

## Commands

```bash
# Training
python train.py --py-config config/config.py --log-dir <run_name>

# Evaluation (renders images, computes PSNR/LPIPS/SSIM)
python eval.py --py-config config/config.py --resume-from <checkpoint.pth> --log-dir <run_name> --depth --img-gt

# Data preprocessing (converts dataset to pickles for faster loading)
python utils/pickles_generator.py --dataset-config config/_base_/dataset.py --py-config config/config.py
```

No linter or test suite is configured.

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
