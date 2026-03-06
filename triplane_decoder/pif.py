# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================


import numpy as np
import torch
from torch import device
from torch.nn import functional as F


class PIF(torch.nn.Module):
    def __init__(self):
        """
        Lazy-initialised Projected Image Features module.

        Call update_proj_mat() with the camera parameters from each batch's
        img_metas before calling update_imgs() / forward().
        """
        super().__init__()

        # width/height follow the legacy convention (width=rows, height=cols).
        self.width = None
        self.height = None
        self.proj_mat = None
        self.imgs = None
        self._proj_cache_key = None

        # Zero-element buffer whose sole purpose is to track the device so that
        # proj_mat computed inside update_proj_mat() lands on the right device.
        self.register_buffer("_device_sentinel", torch.zeros(0))

    def update_proj_mat(self, K, c2w, img_hw, num_cams: int = None) -> None:
        """
        Recompute projection matrices from per-camera intrinsics and extrinsics,
        with caching: if K and c2w are identical to the previous call the
        matrix inversion is skipped.

        Args:
            K:        (N_cams, 3, 4) numpy array of per-camera intrinsic matrices.
            c2w:      list/array of N_cams (4, 4) camera-to-world matrices.
            img_hw:   (H, W) image dimensions matching the intrinsics.
            num_cams: actual number of cameras when K/c2w are zero-padded.
        """
        K_np = np.asarray(K, dtype=np.float64)
        c2w_np = np.asarray(c2w, dtype=np.float64)
        if num_cams is not None:
            K_np = K_np[:num_cams]
            c2w_np = c2w_np[:num_cams]

        key = (K_np.tobytes(), c2w_np.tobytes())
        if key == self._proj_cache_key:
            return
        self._proj_cache_key = key

        # Preserve legacy naming: self.width = rows (H), self.height = cols (W).
        self.width = img_hw[0]
        self.height = img_hw[1]

        c2w_t = torch.tensor(c2w_np, dtype=torch.float32)  # (C, 4, 4)
        w2c = torch.linalg.inv(c2w_t)  # (C, 4, 4)
        K_t = torch.tensor(K_np, dtype=torch.float32)  # (C, 3, 4)

        # proj_mat[i] = K[i] @ w2c[i]  →  (C, 3, 4)
        self.proj_mat = torch.bmm(K_t, w2c).to(self._device_sentinel.device)

    def update_imgs(self, imgs: torch.Tensor) -> None:
        """
        Update the images used for the pif
        imgs: (N,C,H,W)
        """
        self.imgs = imgs.squeeze(0)

    def get_uvs(self, UVW: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Project points to the image plane and return the corresponding UV coordinates.
        UVW: (N,C,3)

        Returns:
            uvw: (N,C,3)
            valid: (N,C)
        """

        # Unsqueeze once to avoid unnecessary memory allocations
        UVW_extended = torch.cat((UVW, torch.ones_like(UVW[..., 0:1])), dim=-1)  # (N,4)

        # UVW = self.proj_mat[None,:].matmul(UVW_extended[:,None,:,None]).squeeze(-1).permute(1,0,2) # (N,C,3)
        UVW = torch.einsum("ijk,lk->ilj", self.proj_mat, UVW_extended)  # (N,C,3)

        # Transform to normalized coordinates
        UVW = torch.cat([UVW[..., :2] / UVW[..., 2:], UVW[..., 2:]], dim=-1)

        # Remap UV coordinates to range [0, width]
        UVW = UVW.clone()
        UVW[..., 0] = -UVW[..., 0] + self.height

        # Determine valid pixel coordinates
        valid = (
            (UVW[..., 0] >= 0)
            & (UVW[..., 0] < self.height)
            & (UVW[..., 1] >= 0)
            & (UVW[..., 1] < self.width)
            & (UVW[..., 2] < 0)
        )

        return UVW, valid

    def forward(self, points_xyz: torch.Tensor, aggregate=False, num_features_to_keep=2) -> torch.Tensor:
        """
        points_xyz: (N,3)

        return:
            features: (N,C,3)
        """

        uvs, valid = self.get_uvs(points_xyz)

        uvs[..., :2] /= torch.tensor([self.height, self.width], device=self.imgs.device).view(1, 2) / 2
        uvs[..., :2] -= 1

        features = (
            F.grid_sample(self.imgs, uvs[:, None, :, :2], mode="bilinear", padding_mode="border", align_corners=False)
            .squeeze(2)
            .permute(0, 2, 1)
        )
        features *= valid[..., None].float()

        if aggregate:
            return self.aggregate(features, valid, num_features_to_keep=num_features_to_keep)

        return features

    def cuda(self, device: int | device | None = None) -> "PIF":
        if self.proj_mat is not None:
            self.proj_mat = self.proj_mat.cuda(device)
        if self.imgs is not None:
            self.imgs = self.imgs.cuda(device)
        return super().cuda(device)

    @staticmethod
    def aggregate(features, valid, num_features_to_keep=2):
        return features.gather(
            0,
            valid.int()
            .argsort(0, descending=True)[:num_features_to_keep, :]
            .unsqueeze(-1)
            .expand(-1, -1, features.size(-1)),
        )


def batch_project(UVW: torch.tensor, proj_mat, img_hw):
    """
    Project points to the image plane and return the corresponding UV coordinates.
    UVW: (B, N,3)
    proj_mat: (B, num_cam, 3,4)
    img_hw: (B, num_cam, 2)

    Returns:
        uvw: (B, num_cam, N,3)
        valid: (B, num_cam, N)
    """

    # Unsqueeze once to avoid unnecessary memory allocations
    UVW_extended = torch.cat((UVW, torch.ones_like(UVW[..., 0:1])), dim=-1)  # (B, N, 4)

    # UVW = self.proj_mat[None,:].matmul(UVW_extended[:,None,:,None]).squeeze(-1).permute(1,0,2) # (N,C,3)
    UVW = torch.einsum("bijk,blk->bilj", proj_mat, UVW_extended)  # (B, num_cam, N, 3)

    # Transform to normalized coordinates
    UVW[..., :2] /= UVW[..., 2:]  # (B,num_cam,N,3)

    # Remap UV coordinates to range [0,1]
    UVW[..., 1] *= -1
    UVW[..., 1] += img_hw[:, :, None, 1]

    UVW[..., :2] /= img_hw[:, :, None]

    # Determine valid pixel coordinates
    valid = ((UVW[..., :2] >= 0.0) & (UVW[..., :2] < 1.0)).all(-1) & (UVW[..., 2] < 1e-5)

    return UVW, valid
