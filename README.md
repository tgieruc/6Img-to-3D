<div align="center">

# 6Img-to-3D: Few-Image Large-Scale Outdoor Novel View Synthesis
[![Paper](https://img.shields.io/badge/arXiv-2404.12378-brightgreen)](https://arxiv.org/abs/2404.12378)
[![Conference](https://img.shields.io/badge/IEEE_IV-2025-blue)](https://ieee-iv.org/2025/)
[![Project WebPage](https://img.shields.io/badge/Project-webpage-%23fc4d5d)](https://6img-to-3d.github.io/)
[![YouTube](https://img.shields.io/badge/YouTube-video-red?logo=youtube&logoColor=white)](https://www.youtube.com/@6Img-to-3D)

</div>

A PyTorch implementation of the 6Img-to-3D model for large-scale outdoor driving scene reconstruction. The model takes as input six images from a driving scene and outputs a parameterized triplane from which novel views can be rendered.

<p align="center">
  <img src="media\driving.gif" alt="Driving" style="width: 120%;" />
</p>

## 6Img-to-3D

Inward and outward-facing camera setups differ significantly in their view overlap. Outward-facing (inside-out) camera setups overlap minimally, whereas inward-facing (outside-in) setups can overlap across multiple cameras.

<p align="center">
  <img src="media\views.png" alt="Views" style="width: 50%;" />
</p>

Given six input images, we first encode them into feature maps using a pre-trained ResNet and an FPN. The scene coordinates are contracted to fit the unbounded scenes. MLPs, cross-and self-attention layers form the Image-to-Triplane Encoder of our framework. Images can be rendered from the resulting triplane using our renderer. We additionally condition the rendering process on projected image features.

<p align="center">
  <img src="media\method.png" alt="Method" style="width: 100%;" />
</p>


## Installation

Requires Python >= 3.10, CUDA 12.6, and [uv](https://docs.astral.sh/uv/).

```bash
# Install all dependencies (PyTorch + CUDA 12.6 wheels resolved automatically)
uv sync

# Install tiny-cuda-nn (requires CUDA toolkit at build time)
uv pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

For development (includes ruff, pre-commit):
```bash
uv sync --group dev
uv run pre-commit install
```

Docker alternative:
```bash
docker build -t 6img-to-3d .
```

## Train
To train the model use the [train.py](train.py) script

### Pre-pickle train dataset
So that the training runs faster, we first turn the train dataset into pickles with the [pickles_generator.py](utils/pickles_generator.py) script.

```bash
 python utils/generator_pickles.py --dataset-config config/_base_/dataset.py --py-config config/config.py
```

### Usage

To train the model, run the training script with the desired arguments specified using the command line interface. Here's how to use each argument:

- `--py-config`: Path to the Python configuration file (.py) containing model configurations. This file specifies the architecture and parameters of the model being trained.

- `--ckpt-path`: Path to a TPVFormer checkpoint file to initialize model weights from. If specified, the training will resume from this checkpoint.

- `--resume-from`: Path to a checkpoint file from which to resume training. This option allows you to continue training from a specific checkpoint.

- `--log-dir`: Directory where Tensorboard training logs and saved models will be stored. If not provided, logs will be saved in the default directory with a timestamp.

- `--num-scenes`: Specifies the number of scenes to train on. This argument allows for faster training when only a subset of scenes is required.

- `--from-epoch`: Specifies the starting epoch for training. If training is interrupted and resumed, you can specify the epoch from which to resume training.


### Running the Script

To run the train9ing script, execute the Python file `train.py` with the desired arguments specified using the command line interface. For example:

```bash
python train.py --py-config config/config.py --ckpt-path ckpts/tpvformer.pth --log-dir evaluation_results
```


## Eval
To evaluate the model, use the [eval.py](eval.py) script.

### Usage

The evaluation script can be run with different options to customize the evaluation process. Here's how to use each argument:

- `--py-config`: Path to the Python configuration file (.py) containing model configurations. This file specifies the architecture and parameters of the model being evaluated.

- `--dataset-config`: Path to the dataset configuration file (.py) containing dataset parameters. This file specifies dataset-specific settings such as image paths and scalling.

- `--resume-from`: Path to the checkpoint file from which to resume model evaluation. This argument allows you to continue evaluation from a previously saved checkpoint.

- `--log-dir`: Directory where evaluation Tensorboard logs and results will be saved. The default behavior is to create a directory with a timestamp indicating the evaluation start time.

- `--depth`: If specified, depth maps will also be saved.

- `--gif`: If specified, the script generates GIFs from the evaluated images.

- `--gif-gt`: If specified, GIFs are generated for ground truth images.

- `--img-gt`: If specified, the script saves ground truth images alongside the generated images.

- `--num-img`: Specifies the number of images to evaluate. By default, all images in the dataset are evaluated. This argument allows for faster evaluation when only a subset of images is required.

- `--time`: Compute inference time of the model and save results in `t_decode.txt`, `t_encode.txt`



### Running the Script

To run the evaluation script, execute the Python file `eval.py` with the desired arguments specified using the command line interface. For example:

```bash
python eval.py --py-config ckpts/6Img-to-3D/config.py --resume-from ckpts/6Img-to-3D/model_checkpoint.pth --log-dir evaluation_results --depth --img-gt --dataset-config config/_base_/dataset_eval.py
```

### License
Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG). All rights reserved.
This repository is licensed under the BSD-3-Clause license. See [LICENSE](./LICENSE) for the full license text.

### Bibtex
If you find this code useful, please reference in your paper:

```
@INPROCEEDINGS{11097387,
  author={Gieruc, Théo and Kästingschäfers, Marius and Bernhard, Sebastian and Salzmann, Mathieu},
  booktitle={2025 IEEE Intelligent Vehicles Symposium (IV)},
  title={6Img-to-3D: Few-Image Large-Scale Outdoor Novel View Synthesis},
  year={2025},
  volume={},
  number={},
  pages={2122-2129},
  doi={10.1109/IV64158.2025.11097387}}
```
