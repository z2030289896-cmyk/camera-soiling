from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .tile_label import mask_to_tile_label


class WoodScapeDataset(Dataset):
    """WoodScape 镜头脏污数据集。

    读取 WoodScape 数据集的 RGB 图像和 gtLabels 灰度标签，
    将多类标签重映射为二元脏污标签，并计算 tile 级标签和覆盖率。

    gtLabels 原始值:
        0 = 背景/画面外 → ignore (-1)
        1 = 干净        → 0 (干净)
        2 = 透明脏污    → 1 (脏污)
        3 = 不透明脏污  → 1 (脏污)
    """

    # 标签重映射: gtLabels 原始值 -> 二元标签 (-1 表示忽略)
    LABEL_REMAP = np.array([-1, 0, 1, 1], dtype=np.int8)

    def __init__(
        self,
        image_dir: str,
        mask_dir: str = None,
        transform=None,
        grid_size: int = 8,
        tile_threshold: float = 0.1,
        indices: list = None,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.grid_size = grid_size
        self.tile_threshold = tile_threshold

        # 获取所有图片路径
        all_images = sorted(list(self.image_dir.glob("*")))

        # 支持 train/val 划分
        if indices is not None:
            self.images = [all_images[i] for i in indices]
        else:
            self.images = all_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        # 加载 RGB 图像
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.mask_dir is not None:
            mask_path = self.mask_dir / image_path.name
            mask_raw = np.array(Image.open(mask_path))

            # 污染类型权重：不透明污染 > 透明污染，用于增强 tile 标签
            # gtLabels: 0=背景 1=干净 2=透明(水滴) 3=不透明(泥/土)
            pixel_weight = np.zeros_like(mask_raw, dtype=np.float32)
            pixel_weight[mask_raw == 3] = 1.0   # 不透明污染：高权重
            pixel_weight[mask_raw == 2] = 0.5   # 透明污染：低权重

            # 标签重映射: 0→-1(ignore), 1→0(clean), 2,3→1(soiled)
            mask = self.LABEL_REMAP[mask_raw]
        else:
            mask = None

        # 数据变换
        if self.transform:
            # 同步变换 pixel_weight（用 NEAREST 插值保持权重值）
            pw_image = Image.fromarray((pixel_weight * 255).astype(np.uint8))
            pw_image = pw_image.resize((self.transform.img_size, self.transform.img_size),
                                       Image.NEAREST)
            pixel_weight = np.array(pw_image).astype(np.float32) / 255.0

            image, mask = self.transform(image, mask)

        # 计算 tile 标签和覆盖率 (仅训练/验证时使用)
        if mask is not None:
            # 将 mask 转为 numpy（transform 后是 tensor）
            if hasattr(mask, "numpy"):
                mask_np = mask.numpy()
            else:
                mask_np = mask

            # 创建 ignore mask: label == -1 的位置不参与 loss 计算
            ignore_mask = (mask_np == -1)

            # 转换为二元标签: -1→0, 0→0, 1→1
            binary_mask = np.where(mask_np == -1, 0, mask_np).astype(np.float32)

            # 增强 Tile 标签: 高斯中心加权 + 污染类型加权 + 非线性增强 + 边界平滑
            tile_label = mask_to_tile_label(
                binary_mask,
                grid_size=self.grid_size,
                threshold=self.tile_threshold,
                binary=False,
                use_gaussian_weight=True,   # 中心污染权重更高
                gamma=0.7,                  # 非线性增强小污染区域
                pixel_weight=pixel_weight,  # 不透明 > 透明
                smooth_sigma=0.6,           # tile 间平滑过渡
            )

            # Tile ignore mask: 简单面积比（不需要增强）
            tile_ignore = mask_to_tile_label(
                ignore_mask.astype(np.float32),
                grid_size=self.grid_size,
                threshold=0.5,
                binary=False,
                use_gaussian_weight=False,
                gamma=1.0,
                pixel_weight=None,
                smooth_sigma=0.0,
            ) > 0.5

            # 覆盖率: 有效区域内脏污像素占比
            valid_mask = mask_np != -1
            if valid_mask.sum() > 0:
                coverage = (mask_np[valid_mask] == 1).mean()
            else:
                coverage = 0.0

            return {
                "image": image,
                "tile_label": torch.from_numpy(tile_label).float(),
                "tile_ignore": torch.from_numpy(tile_ignore).bool(),
                "coverage": torch.tensor([coverage], dtype=torch.float32),
            }
        else:
            return {"image": image}
