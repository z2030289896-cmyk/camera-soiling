"""合并数据集 DataLoader。

将 self-made + WoodScape rgbLabels 合并为统一数据集，
自动随机划分 train/val。
"""

import numpy as np
from torch.utils.data import DataLoader
from .combined_dataset import CombinedDataset
from .transforms import get_train_transform, get_val_transform


def build_dataloader_combined(cfg):
    """构建合并数据集的 train/val DataLoader。"""
    img_size = cfg.get("img_size", 224)
    grid_size = cfg.get("grid_size", 8)
    tile_threshold = cfg.get("tile_threshold", 0.1)
    smooth_sigma = cfg.get("smooth_sigma", 0.6)  # 从配置读取，默认0.6
    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 4)
    train_split = cfg.get("train_split", 0.8)
    datasets_cfg = cfg["datasets"]

    # 获取完整文件列表
    tmp = CombinedDataset(datasets_cfg=datasets_cfg, grid_size=grid_size,
                          tile_threshold=tile_threshold, smooth_sigma=smooth_sigma)
    pairs = tmp.pairs
    n_total = len(pairs)
    n_train = int(n_total * train_split)

    rng = np.random.default_rng(cfg.get("seed", 42))
    indices = rng.permutation(n_total).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = CombinedDataset(
        datasets_cfg=datasets_cfg,
        transform=get_train_transform(img_size), grid_size=grid_size,
        tile_threshold=tile_threshold, smooth_sigma=smooth_sigma,
        file_indices=train_indices,
    )
    val_dataset = CombinedDataset(
        datasets_cfg=datasets_cfg,
        transform=get_val_transform(img_size), grid_size=grid_size,
        tile_threshold=tile_threshold, smooth_sigma=smooth_sigma,
        file_indices=val_indices,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
