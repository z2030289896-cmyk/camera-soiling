"""RaindropsOnWindshield DataLoader。

支持文件级别的随机 train/val 划分（非场景级别），
确保训练集和验证集都包含所有场景的数据。
"""

import numpy as np
from torch.utils.data import DataLoader

from .raindrops_dataset import RaindropsDataset
from .transforms import get_train_transform, get_val_transform


def build_dataloader_raindrops(cfg):
    """构建 RaindropsOnWindshield 的 train/val DataLoader。

    划分策略：文件级别随机打乱（非按场景划分），
    确保 train/val 都包含所有场景（1o/2o/8o/D1/D2/D3/D4/v2）。

    Args:
        cfg: dataset_raindrops.yaml 配置字典

    Returns:
        train_loader, val_loader
    """
    # 读取配置参数
    img_size = cfg.get("img_size", 224)
    grid_size = cfg.get("grid_size", 8)
    tile_threshold = cfg.get("tile_threshold", 0.1)
    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 4)
    train_split = cfg.get("train_split", 0.8)
    exclude = cfg.get("exclude_scenes", [])

    # 第一步：创建临时 dataset 获取完整文件列表
    tmp = RaindropsDataset(
        image_dir=cfg["train_image"], mask_dir=cfg["train_mask"],
        grid_size=grid_size, tile_threshold=tile_threshold,
        exclude_scenes=exclude,
    )
    pairs = tmp.pairs  # 所有 (image_path, mask_path) 对
    n_total = len(pairs)
    n_train = int(n_total * train_split)

    # 第二步：随机打乱索引，文件级别划分
    rng = np.random.default_rng(cfg.get("seed", 42))
    indices = rng.permutation(n_total).tolist()
    train_pairs = [pairs[i] for i in indices[:n_train]]
    val_pairs = [pairs[i] for i in indices[n_train:]]

    # 第三步：分别创建 train/val dataset（使用预定义文件列表）
    train_dataset = RaindropsDataset(
        image_dir=cfg["train_image"], mask_dir=cfg["train_mask"],
        transform=get_train_transform(img_size),  # 训练用增强
        grid_size=grid_size, tile_threshold=tile_threshold,
        file_list=train_pairs,
    )
    val_dataset = RaindropsDataset(
        image_dir=cfg["val_image"], mask_dir=cfg["val_mask"],
        transform=get_val_transform(img_size),    # 验证无增强
        grid_size=grid_size, tile_threshold=tile_threshold,
        file_list=val_pairs,
    )

    # 第四步：创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
