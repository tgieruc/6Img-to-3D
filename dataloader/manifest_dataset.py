"""Dataset that loads scenes from a JSONL manifest produced by utils/generate_manifest.py.

Each manifest line:
  {"input": "/abs/path/.../transforms_ego.json",
   "target": "/abs/path/.../transforms_ego.json",
   "town": ..., "weather": ..., "vehicle": ..., "spawn_point": ..., "step": ...}

The transforms_ego.json files follow the seed4d / NeRF-Studio convention:
  - Intrinsics may be global (top-level fl_x/fl_y/cx/cy/w/h) or per-frame.
  - file_path in each frame is relative to the transforms_ego.json file.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader

from dataloader.dataset import build_pose_intrinsics_vector, img_norm_cfg
from dataloader.transform_3d import NormalizeMultiviewImage


def _load_transforms(tf_path: Path) -> dict:
    with open(tf_path) as f:
        return json.load(f)


def _build_K_and_c2w(tf: dict) -> tuple[np.ndarray, list]:
    frames = tf["frames"]
    n = len(frames)
    K = np.zeros((n, 3, 4))
    c2ws = []
    for i, frame in enumerate(frames):
        K[i, 0, 0] = frame.get("fl_x", tf.get("fl_x", 0))
        K[i, 1, 1] = frame.get("fl_y", tf.get("fl_y", 0))
        K[i, 2, 2] = 1.0
        K[i, 0, 2] = frame.get("cx", tf.get("cx", 0))
        K[i, 1, 2] = frame.get("cy", tf.get("cy", 0))
        c2ws.append(frame["transform_matrix"])
    return K, c2ws


class ManifestDataset(data.Dataset):
    """Load scenes from a JSONL manifest file."""

    def __init__(self, manifest_path: str | Path, config, dataset_config):
        self.manifest_path = Path(manifest_path)
        self.config = config
        self.dataset_config = dataset_config
        self.transforms = NormalizeMultiviewImage(**img_norm_cfg)
        self.entries = []
        with open(self.manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        input_tf_path = Path(entry["input"])
        tf = _load_transforms(input_tf_path)

        K, c2ws = _build_K_and_c2w(tf)
        sensor_dir = input_tf_path.parent  # dir containing transforms_ego.json

        imgs = []
        for frame in tf["frames"]:
            img_path = (sensor_dir / frame["file_path"]).resolve()
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)[:, :, :3].astype(np.float32)
            imgs.append(img)

        imgs = self.transforms(imgs)
        img_shape = [img.shape for img in imgs]

        img_meta = dict(
            K=K,
            c2w=c2ws,
            img_shape=img_shape,
            pose_intrinsics=build_pose_intrinsics_vector(c2ws, K),
            num_cams=len(c2ws),
            town=entry.get("town"),
            weather=entry.get("weather"),
            vehicle=entry.get("vehicle"),
            spawn_point=entry.get("spawn_point"),
            step=entry.get("step"),
        )

        # Derive the sphere/ directory from the target transforms path:
        #   entry["target"] = ".../sphere/transforms/transforms_ego.json"
        #   target_dir      = ".../sphere/"
        target_tf_path = Path(entry["target"])
        target_dir = target_tf_path.parent.parent  # .../sphere/

        # Lazy import avoids module-level stub binding when tests mock rays_dataset.
        from dataloader.rays_dataset import RaysDataset

        mode = "test" if self.dataset_config.phase == "val" else self.dataset_config.phase
        sphere_dataset = RaysDataset(
            str(target_dir),
            config=self.config,
            dataset_config=self.dataset_config,
            mode=mode,
            factor=getattr(self.dataset_config, "factor", 1.0),
        )

        batch_size = getattr(self.dataset_config, "batch_size", 1)
        if self.dataset_config.phase == "train":
            sphere_dataloader = DataLoader(
                sphere_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )
        else:
            sphere_dataloader = DataLoader(
                sphere_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

        return np.stack(imgs), img_meta, sphere_dataloader
