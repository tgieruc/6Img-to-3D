# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

import numpy as np
import torch


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)



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