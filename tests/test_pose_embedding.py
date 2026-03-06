import importlib.util
import os

import torch

# Load PoseEmbedding directly from its file to avoid pulling in the
# heavy triplane_encoder package init (which requires mmcv, mmdet, etc.).
_MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "triplane_encoder",
    "modules",
    "pose_embedding.py",
)
_spec = importlib.util.spec_from_file_location(
    "triplane_encoder.modules.pose_embedding",
    os.path.abspath(_MODULE_PATH),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PoseEmbedding = _mod.PoseEmbedding


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
