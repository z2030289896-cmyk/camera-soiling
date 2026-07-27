"""Self 数据集 DataLoader。

支持 train_split=1.0 时全部数据作为训练集（不创建验证集）。
"""

import numpy as np
from torch.utils.data import DataLoader
from .self_dataset import SelfDataset
from .transforms import get_train_transform, get_val_transform


def build_dataloader_self(cfg):
    """构建 Self 数据集的 train/val DataLoader。"""
    img_size = cfg.get("img_size", 224)
    grid_size = cfg.get("grid_size", 8)
    tile_threshold = cfg.get("tile_threshold", 0.1)
    batch_size = cfg.get("batch_size", 8)    # 数据量小，用更小的 batch
    num_workers = cfg.get("num_workers", 2)
    train_split = cfg.get("train_split", 0.8)

    # 获取完整文件列表
    tmp = SelfDataset(image_dir=cfg["train_image"], mask_dir=cfg["train_mask"],
                      grid_size=grid_size, tile_threshold=tile_threshold)
    pairs = tmp.pairs
    n_total = len(pairs)
    n_train = int(n_total * train_split)

    rng = np.random.default_rng(cfg.get("seed", 42))
    indices = rng.permutation(n_total).tolist()
    train_pairs = [pairs[i] for i in indices[:n_train]]
    val_pairs = [pairs[i] for i in indices[n_train:]]

    train_dataset = SelfDataset(
        image_dir=cfg["train_image"], mask_dir=cfg["train_mask"],
        transform=get_train_transform(img_size), grid_size=grid_size,
        tile_threshold=tile_threshold, file_list=train_pairs,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)

    # train_split=1.0 时不创建验证集
    if val_pairs:
        val_dataset = SelfDataset(
            image_dir=cfg.get("val_image", cfg["train_image"]),
            mask_dir=cfg.get("val_mask", cfg["train_mask"]),
            transform=get_val_transform(img_size), grid_size=grid_size,
            tile_threshold=tile_threshold, file_list=val_pairs,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        val_loader = None
    return train_loader, val_loader
