"""WoodScape rgbLabels 数据集。

读取 rgbLabels 的 RGB 彩色标签，自动检测颜色→类别映射。
不需要硬编码颜色值，自动推断：
    - 低亮度(近似黑色) → 背景/忽略
    - 绿色调(高G低RB) → 干净
    - 其他非黑色 → 脏污

输出:
    - binary_mask: (H,W) float32, 1=脏污, 0=干净, 需配合 ignore_mask 排除背景
    - ignore_mask: (H,W) bool, True=忽略(背景)
    - tile_label: (G,G) 连续软标签 [0,1]
    - coverage: 整体覆盖率
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def auto_detect_color_map(image_dir: str, sample_size: int = 50):
    """自动分析 rgbLabels 中的颜色→类别映射。

    采样前N张图，统计所有独特RGB值，按亮度/色调推断：
    - 最暗→背景(bg)
    - 绿色调(G>R+B)→干净(clean)
    - 其余→脏污(dirty)
    """
    """自动分析 rgbLabels 中的颜色→类别映射。

    Args:
        image_dir: rgbLabels 目录路径
        sample_size: 采样图片数

    Returns:
        dict: {(R,G,B): class_name}
            class_name: 'bg', 'clean', 'dirty'
    """
    import os
    all_colors = set()
    files = sorted(os.listdir(image_dir))[:sample_size]
    for f in files:
        img = np.array(Image.open(os.path.join(image_dir, f)))
        colors = np.unique(img.reshape(-1, 3), axis=0)
        for c in colors:
            all_colors.add(tuple(c.tolist()))

    color_map = {}
    brightness = {}
    for c in all_colors:
        r, g, b = c
        brightness[c] = int(r) + int(g) + int(b)

    # 1. 最暗的颜色 → 背景
    bg_color = min(brightness, key=brightness.get)
    color_map[bg_color] = 'bg'

    # 2. 绿色调 (G > R+B) → 干净
    remaining = [c for c in all_colors if c != bg_color]
    for c in remaining:
        r, g, b = c
        if g > r + b:
            color_map[c] = 'clean'
        else:
            color_map[c] = 'dirty'

    # 如果没有绿色调，最高亮度的非bg → clean
    if 'clean' not in color_map.values():
        non_bg = {c: v for c, v in brightness.items() if c != bg_color}
        clean_color = max(non_bg, key=non_bg.get)
        color_map[clean_color] = 'clean'
        for c in all_colors:
            if c not in color_map:
                color_map[c] = 'dirty'

    return color_map


class WoodScapeRGBLabelsDataset(Dataset):
    """rgbLabels 数据集。

    自动检测颜色→类别映射，生成 Binary Mask（脏污=1，背景忽略）。
    使用与原始 woodscape.py 相同的增强 tile 标签生成。
    """

    def __init__(self, image_dir: str, mask_dir: str = None,
                 transform=None, grid_size: int = 8,
                 tile_threshold: float = 0.1, indices: list = None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.grid_size = grid_size
        self.tile_threshold = tile_threshold

        all_images = sorted(list(self.image_dir.glob("*")))
        self.images = [all_images[i] for i in indices] if indices else all_images

        # 自动检测颜色映射
        if self.mask_dir:
            self.color_map = auto_detect_color_map(str(self.mask_dir))
            self.bg_color = [c for c, v in self.color_map.items() if v == 'bg'][0]
            self.dirty_colors = [c for c, v in self.color_map.items() if v == 'dirty']
            print(f"Auto-detected color map: {self.color_map}")
            print(f"  Background: {self.bg_color}")
            print(f"  Dirty:      {self.dirty_colors}")

    def __len__(self):
        return len(self.images)

    def _rgb_to_binary_mask(self, rgb_img: np.ndarray) -> tuple:
        """将 RGB 标签转为 binary mask + ignore mask。

        Args:
            rgb_img: (H, W, 3) uint8

        Returns:
            binary_mask: (H, W) float32, 1=脏污, 0=干净
            ignore_mask: (H, W) bool, True=背景/忽略
        """
        h, w = rgb_img.shape[:2]
        # 打包为 int32 加速匹配
        packed = (rgb_img[:, :, 0].astype(np.int32) << 16) + \
                 (rgb_img[:, :, 1].astype(np.int32) << 8) + \
                 rgb_img[:, :, 2].astype(np.int32)

        # 检测背景像素
        bg_val = (self.bg_color[0] << 16) + (self.bg_color[1] << 8) + self.bg_color[2]

        # 检测脏污像素
        dirty_vals = [(r << 16) + (g << 8) + b for r, g, b in self.dirty_colors]

        ignore_mask = (packed == bg_val)
        binary_mask = np.zeros((h, w), dtype=np.float32)
        for dv in dirty_vals:
            binary_mask[packed == dv] = 1.0

        return binary_mask, ignore_mask

    def _save_mask_viz(self, binary_mask, ignore_mask, path):
        """保存 Binary Mask 可视化（用于验证标签是否正确）。"""
        viz = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
        viz[ignore_mask] = [0, 0, 0]        # 背景 → 黑
        viz[binary_mask == 0] = [0, 255, 0]  # 干净 → 绿
        viz[binary_mask == 1] = [255, 0, 0]  # 脏污 → 红
        Image.fromarray(viz).save(path)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.mask_dir is not None:
            mask_path = self.mask_dir / image_path.name
            rgb_mask = np.array(Image.open(mask_path).convert("RGB"))
            binary_mask, ignore_mask = self._rgb_to_binary_mask(rgb_mask)
        else:
            binary_mask, ignore_mask = None, None

        # 数据变换
        if self.transform and binary_mask is not None:
            image, mask_remapped = self.transform(image, binary_mask)
            # transform 后的 ignore_mask：resize + 与 binary_mask 同步
            ignore_img = Image.fromarray(ignore_mask)
            ignore_img = ignore_img.resize(
                (self.transform.img_size, self.transform.img_size), Image.NEAREST)
            ignore_mask = np.array(ignore_img)

        if binary_mask is not None:
            if hasattr(mask_remapped, "numpy"):
                mask_np = mask_remapped.numpy()
            else:
                mask_np = mask_remapped

            # 计算 tile 标签和覆盖率（复用增强 tile_label 逻辑）
            from .tile_label import mask_to_tile_label
            tile_label = mask_to_tile_label(
                mask_np, grid_size=self.grid_size, threshold=self.tile_threshold,
                binary=False, use_gaussian_weight=True, gamma=0.7,
                smooth_sigma=0.6,
            )
            tile_ignore = mask_to_tile_label(
                ignore_mask.astype(np.float32), grid_size=self.grid_size,
                threshold=0.5, binary=False,
                use_gaussian_weight=False, gamma=1.0, smooth_sigma=0.0,
            ) > 0.5

            valid = ~(ignore_mask.astype(bool))
            coverage = mask_np[valid].mean() if valid.sum() > 0 else 0.0

            return {
                "image": image,
                "tile_label": torch.from_numpy(tile_label).float(),
                "tile_ignore": torch.from_numpy(tile_ignore),
                "coverage": torch.tensor([coverage], dtype=torch.float32),
            }
        else:
            return {"image": image}
