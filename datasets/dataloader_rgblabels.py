"""rgbLabels 版 DataLoader。"""

import os
import numpy as np
from torch.utils.data import DataLoader

from .woodscape_rgblabels import WoodScapeRGBLabelsDataset
from .transforms import get_train_transform, get_val_transform


def build_dataloader_rgblabels(cfg):
    """构建基于 rgbLabels 的 train/val DataLoader。"""
    img_size = cfg.get("img_size", 224)
    grid_size = cfg.get("grid_size", 8)
    tile_threshold = cfg.get("tile_threshold", 0.1)
    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 4)
    train_split = cfg.get("train_split", 0.8)

    image_dir = cfg["train_image"]
    all_files = sorted(os.listdir(image_dir))
    n_total = len(all_files)
    n_train = int(n_total * train_split)
    rng = np.random.default_rng(cfg.get("seed", 42))
    indices = rng.permutation(n_total).tolist()

    train_dataset = WoodScapeRGBLabelsDataset(
        image_dir=cfg["train_image"], mask_dir=cfg["train_mask"],
        transform=get_train_transform(img_size), grid_size=grid_size,
        tile_threshold=tile_threshold, indices=indices[:n_train],
    )
    val_dataset = WoodScapeRGBLabelsDataset(
        image_dir=cfg["val_image"], mask_dir=cfg["val_mask"],
        transform=get_val_transform(img_size), grid_size=grid_size,
        tile_threshold=tile_threshold, indices=indices[n_train:],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
