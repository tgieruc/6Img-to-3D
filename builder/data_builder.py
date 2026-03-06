# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

import torch

from dataloader.dataset import CarlaDataset, PickledCarlaDataset
from dataloader.dataset_wrapper import custom_collate_fn


def build(config):
    data_path = config.dataset_params.data_path

    if config.dataset_params.train_data_loader.pickled:
        train_dataset = PickledCarlaDataset(
            data_path, dataset_config=config.dataset_params.train_data_loader, config=config
        )
    else:
        train_dataset = CarlaDataset(data_path, dataset_config=config.dataset_params.train_data_loader, config=config)

    if config.dataset_params.val_data_loader.pickled:
        val_dataset = PickledCarlaDataset(
            data_path, dataset_config=config.dataset_params.val_data_loader, config=config
        )
    else:
        val_dataset = CarlaDataset(data_path, dataset_config=config.dataset_params.val_data_loader, config=config)

    train_dataset_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        shuffle=config.dataset_params.train_data_loader["shuffle"],
        num_workers=config.dataset_params.train_data_loader["num_workers"],
        pin_memory=False,
    )

    val_dataset_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        shuffle=config.dataset_params.val_data_loader["shuffle"],
        num_workers=config.dataset_params.val_data_loader["num_workers"],
    )

    return train_dataset_loader, val_dataset_loader


def build_from_manifests(
    train_manifest: str,
    val_manifest: str,
    config,
    train_dataset_config,
    val_dataset_config,
):
    """Build train/val DataLoaders from JSONL manifest files."""
    from dataloader.manifest_dataset import ManifestDataset

    train_dataset = ManifestDataset(train_manifest, config=config, dataset_config=train_dataset_config)
    val_dataset = ManifestDataset(val_manifest, config=config, dataset_config=val_dataset_config)

    train_num_workers = getattr(train_dataset_config, "num_workers", 0)
    val_num_workers = getattr(val_dataset_config, "num_workers", 0)
    train_shuffle = getattr(train_dataset_config, "shuffle", True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        shuffle=train_shuffle,
        num_workers=train_num_workers,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        shuffle=False,
        num_workers=val_num_workers,
    )
    return train_loader, val_loader
