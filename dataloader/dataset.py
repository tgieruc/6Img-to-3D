# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

import json
import os
import random

import cv2
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader

from dataloader.rays_dataset import RaysDataset
from dataloader.transform_3d import NormalizeMultiviewImage


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


img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)


class CarlaDataset(data.Dataset):
    def __init__(self, data_path, dataset_config, config):
        self.data_path = data_path
        self.dataset_config = dataset_config
        self.data = []
        self.config = config
        self.decoder_config = config.decoder
        self.hw = (48, 64)

        self.transforms = NormalizeMultiviewImage(**img_norm_cfg)

        if dataset_config["town"] == "all":
            towns = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]
        else:
            towns = dataset_config["town"]
        for town in towns:
            if dataset_config["weather"] == "all":
                weathers = [
                    folder
                    for folder in os.listdir(os.path.join(data_path, town))
                    if os.path.isdir(os.path.join(data_path, town, folder))
                ]
            else:
                weathers = dataset_config["weather"]
            for weather in weathers:
                if dataset_config["vehicle"] == "all":
                    vehicles = [
                        folder
                        for folder in os.listdir(os.path.join(data_path, town, weather))
                        if os.path.isdir(os.path.join(data_path, town, weather, folder))
                    ]
                else:
                    vehicles = dataset_config["vehicle"]
                for vehicle in vehicles:
                    if dataset_config["spawn_point"] == ["all"]:
                        spawn_points = [
                            folder
                            for folder in os.listdir(os.path.join(data_path, town, weather, vehicle))
                            if "spawn_point_" in folder
                        ]
                    else:
                        spawn_points = [f"spawn_point_{i}" for i in dataset_config["spawn_point"]]
                    for spawn_point in spawn_points:
                        if dataset_config["step"] == ["all"]:
                            steps = [
                                folder
                                for folder in os.listdir(os.path.join(data_path, town, weather, vehicle, spawn_point))
                                if "step_" in folder
                            ]
                            steps = sorted(steps, key=lambda x: int(x.split("_")[1]))

                        else:
                            steps = [f"step_{i}" for i in dataset_config["step"]]
                        for step in steps:
                            self.data.append(self.get_data(town, weather, vehicle, spawn_point, step))

    def get_data(self, town, weather, vehicle, spawn_point, step):
        data_path = os.path.join(self.data_path, town, weather, vehicle, spawn_point, step)
        data = dict(
            town=town,
            weather=weather,
            vehicle=vehicle,
            spawn_point=spawn_point,
            step=step,
            nuscenes=os.path.join(data_path, "nuscenes"),
            sphere=os.path.join(data_path, "sphere"),
        )

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        imgs_dict = {
            "img": [],
        }

        img_meta = None
        input_rgb = []
        sphere_dataloader = None

        if "input_images" in self.dataset_config.get("selection", ["input_images"]):
            with open(os.path.join(data["nuscenes"], "transforms", "transforms_ego.json")) as f:
                input_data = json.load(f)

            input_rgb = []
            all_c2w = []
            num_cams = len(input_data["frames"])
            K = np.zeros((num_cams, 3, 4))
            for cam_idx, frame in enumerate(input_data["frames"]):
                fx = frame.get("fl_x", input_data.get("fl_x", 0))
                fy = frame.get("fl_y", input_data.get("fl_y", 0))
                cx = frame.get("cx", input_data.get("cx", 0))
                cy = frame.get("cy", input_data.get("cy", 0))
                K[cam_idx, 0, 0] = fx
                K[cam_idx, 1, 1] = fy
                K[cam_idx, 2, 2] = 1
                K[cam_idx, 0, 2] = cx
                K[cam_idx, 1, 2] = cy

            for frame in input_data["frames"]:
                input_rgb.append(
                    cv2.imread(os.path.join(data["nuscenes"], "transforms", frame["file_path"]), cv2.IMREAD_UNCHANGED)[
                        :, :, :3
                    ].astype(np.float32)
                )
                all_c2w.append(frame["transform_matrix"])

            input_rgb = self.transforms(input_rgb)

            img_shape = [img.shape for img in input_rgb]

            # Camera dropout during training
            if self.dataset_config["phase"] == "train":
                min_cams = self.dataset_config.get("min_cams_train", 1)
                max_cams = self.dataset_config.get("max_cams_train", num_cams)
                selected = apply_camera_dropout(len(input_rgb), min_cams, max_cams)
                input_rgb = [input_rgb[i] for i in selected]
                all_c2w = [all_c2w[i] for i in selected]
                K = K[selected]
                img_shape = [img_shape[i] for i in selected]

            img_meta = dict(
                K=K,
                c2w=all_c2w,
                img_shape=img_shape,
                pose_intrinsics=build_pose_intrinsics_vector(all_c2w, K),
                num_cams=len(all_c2w),
            )

        mode_suffix = "test" if self.dataset_config["phase"] == "val" else self.dataset_config["phase"]

        if "sphere_dataset" in self.dataset_config.get("selection", ["sphere_dataset"]):
            sphere_dataset = RaysDataset(
                data["sphere"],
                config=self.config,
                dataset_config=self.dataset_config,
                mode=mode_suffix,
                factor=self.dataset_config.factor,
            )

            if self.dataset_config["phase"] == "train":
                sphere_dataloader = DataLoader(
                    sphere_dataset,
                    batch_size=self.dataset_config.get("batch_size", 1),
                    shuffle=True,
                    num_workers=12,
                    pin_memory=True,
                )
            else:
                sphere_dataloader = DataLoader(
                    sphere_dataset, batch_size=self.dataset_config.get("batch_size", 1), shuffle=False
                )

        if "path" in self.dataset_config.get("selection", ["path"]):
            path = f"{data['town']}_{data['weather']}_{data['vehicle']}_{data['spawn_point']}_{data['step']}"
        else:
            path = None

        return (
            (input_rgb, img_meta, sphere_dataloader) if path is None else (input_rgb, img_meta, sphere_dataloader, path)
        )


class PickledCarlaDataset(CarlaDataset):
    def __init__(self, data_path, dataset_config, config, part_num=0):
        super().__init__(data_path, dataset_config, config=config)

        self.part_num = part_num

    def __getitem__(self, index):
        data = self.data[index]

        input_rgb = []

        img_meta = None

        sphere_dataloader = None

        if "input_images" in self.dataset_config.get("selection", ["input_images"]):
            with open(os.path.join(data["nuscenes"], "transforms", "transforms_ego.json")) as f:
                input_data = json.load(f)

            input_rgb = []
            all_c2w = []
            num_cams = len(input_data["frames"])
            K = np.zeros((num_cams, 3, 4))
            for cam_idx, frame in enumerate(input_data["frames"]):
                fx = frame.get("fl_x", input_data.get("fl_x", 0))
                fy = frame.get("fl_y", input_data.get("fl_y", 0))
                cx = frame.get("cx", input_data.get("cx", 0))
                cy = frame.get("cy", input_data.get("cy", 0))
                K[cam_idx, 0, 0] = fx
                K[cam_idx, 1, 1] = fy
                K[cam_idx, 2, 2] = 1
                K[cam_idx, 0, 2] = cx
                K[cam_idx, 1, 2] = cy

            for frame in input_data["frames"]:
                input_rgb.append(
                    cv2.imread(os.path.join(data["nuscenes"], "transforms", frame["file_path"]), cv2.IMREAD_UNCHANGED)[
                        :, :, :3
                    ].astype(np.float32)
                )
                all_c2w.append(frame["transform_matrix"])

            input_rgb = self.transforms(input_rgb)

            img_shape = [img.shape for img in input_rgb]

            # Camera dropout during training
            if self.dataset_config["phase"] == "train":
                min_cams = self.dataset_config.get("min_cams_train", 1)
                max_cams = self.dataset_config.get("max_cams_train", num_cams)
                selected = apply_camera_dropout(len(input_rgb), min_cams, max_cams)
                input_rgb = [input_rgb[i] for i in selected]
                all_c2w = [all_c2w[i] for i in selected]
                K = K[selected]
                img_shape = [img_shape[i] for i in selected]

            img_meta = dict(
                K=K,
                c2w=all_c2w,
                img_shape=img_shape,
                pose_intrinsics=build_pose_intrinsics_vector(all_c2w, K),
                num_cams=len(all_c2w),
            )

        if "sphere_dataset" in self.dataset_config.get("selection", ["sphere_dataset"]):
            if self.dataset_config.get("whole_image", False):
                filename = "train_dataset_"
            else:
                filename = "train_dataset_shuffled_"
            sphere_dataloader = []
            if self.dataset_config.get("whole_image", False):
                view_ids = random.sample(range(80), self.dataset_config.get("num_imgs", 1))
            else:
                view_ids = np.arange(self.part_num, self.part_num + self.dataset_config.get("num_imgs", 1))
            for view_id in view_ids:
                with open(os.path.join(data["sphere"], f"{filename}{view_id}.npy"), "rb") as f:
                    sphere_dataloader.append(np.load(f))
            sphere_dataloader = np.concatenate(sphere_dataloader)

        return (input_rgb, img_meta, sphere_dataloader)
