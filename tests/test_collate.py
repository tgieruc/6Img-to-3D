import numpy as np


def test_collate_pads_to_max_cameras():
    """Collate function pads samples with different camera counts."""
    from dataloader.dataset_wrapper import custom_collate_fn

    # Sample 1: 3 cameras, Sample 2: 5 cameras
    sample1_imgs = np.random.randn(3, 100, 100, 3).astype(np.float32)
    sample1_meta = {
        "K": np.zeros((3, 3, 4)),
        "c2w": [np.eye(4).tolist()] * 3,
        "img_shape": [(100, 100, 3)] * 3,
        "pose_intrinsics": np.zeros((3, 20)),
        "num_cams": 3,
    }

    sample2_imgs = np.random.randn(5, 100, 100, 3).astype(np.float32)
    sample2_meta = {
        "K": np.zeros((5, 3, 4)),
        "c2w": [np.eye(4).tolist()] * 5,
        "img_shape": [(100, 100, 3)] * 5,
        "pose_intrinsics": np.zeros((5, 20)),
        "num_cams": 5,
    }

    batch = custom_collate_fn(
        [
            (sample1_imgs, sample1_meta, None),
            (sample2_imgs, sample2_meta, None),
        ]
    )

    img_batch, meta_batch, _ = batch

    # Should pad to max cameras (5)
    assert img_batch.shape[0] == 2  # batch size
    assert img_batch.shape[1] == 5  # max cameras
    assert img_batch.shape[2] == 3  # channels

    # Attention mask: True for real cameras, False for padding
    assert meta_batch[0]["cam_mask"].shape == (5,)
    assert meta_batch[0]["cam_mask"][:3].all()  # 3 real cameras
    assert not meta_batch[0]["cam_mask"][3:].any()  # 2 padded
    assert meta_batch[1]["cam_mask"].all()  # all 5 real


def test_collate_same_camera_count():
    """Collate works when all samples have same camera count."""
    from dataloader.dataset_wrapper import custom_collate_fn

    sample = np.random.randn(6, 100, 100, 3).astype(np.float32)
    meta = {
        "K": np.zeros((6, 3, 4)),
        "c2w": [np.eye(4).tolist()] * 6,
        "img_shape": [(100, 100, 3)] * 6,
        "pose_intrinsics": np.zeros((6, 20)),
        "num_cams": 6,
    }

    batch = custom_collate_fn([(sample, meta, None)])
    img_batch, meta_batch, _ = batch

    assert img_batch.shape == (1, 6, 3, 100, 100)
    assert meta_batch[0]["cam_mask"].all()
