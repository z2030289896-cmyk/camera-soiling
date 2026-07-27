import os

import numpy as np
from torch.utils.data import DataLoader

from .woodscape import WoodScapeDataset
from .transforms import get_train_transform, get_val_transform


def build_dataloader(cfg):
    """构建 WoodScape gtLabels 的 train/val DataLoader。

    随机划分 80/20，train用数据增强，val无增强。
    """
    """构建训练和验证 DataLoader。

    Args:
        cfg: 数据集配置字典（来自 dataset.yaml）。

    Returns:
        train_loader, val_loader
    """
    img_size = cfg.get("img_size", 224)
    grid_size = cfg.get("grid_size", 8)
    tile_threshold = cfg.get("tile_threshold", 0.1)
    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 4)
    train_split = cfg.get("train_split", 0.8)

    # 获取所有图片并随机划分 train/val
    image_dir = cfg["train_image"]
    all_files = sorted(os.listdir(image_dir))
    n_total = len(all_files)
    n_train = int(n_total * train_split)

    # 随机打乱索引（固定种子保证可复现）
    rng = np.random.default_rng(cfg.get("seed", 42))
    indices = rng.permutation(n_total).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # 训练集
    train_dataset = WoodScapeDataset(
        image_dir=cfg["train_image"],
        mask_dir=cfg["train_mask"],
        transform=get_train_transform(img_size),
        grid_size=grid_size,
        tile_threshold=tile_threshold,
        indices=train_indices,
    )

    # 验证集
    val_dataset = WoodScapeDataset(
        image_dir=cfg["val_image"],
        mask_dir=cfg["val_mask"],
        transform=get_val_transform(img_size),
        grid_size=grid_size,
        tile_threshold=tile_threshold,
        indices=val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
