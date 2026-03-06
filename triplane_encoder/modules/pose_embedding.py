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
