"""Self 数据集。

vis/ 目录下的标签为二值 PNG：红色(255,0,0)=脏污，黑色(0,0,0)=背景。
images/ 目录下为对应 RGB 原图。
"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SelfDataset(Dataset):
    """Self 二值 mask 数据集。

    vis 标签：红色像素(255,0,0) → 脏污(1)，黑色像素(0,0,0) → 背景(ignore)。
    使用增强 tile 标签生成（高斯中心加权+gamma+平滑）。
    """

    def __init__(self, image_dir: str, mask_dir: str = None,
                 transform=None, grid_size: int = 8,
                 tile_threshold: float = 0.1,
                 file_list: list = None):
        """
        Args:
            image_dir: images/ 目录路径
            mask_dir: vis/ 目录路径
            transform: 数据变换
            grid_size: tile 网格大小
            tile_threshold: tile 二值化阈值
            file_list: 预定义文件列表 [(img_path, mask_path), ...]
        """
        self.transform = transform
        self.grid_size = grid_size
        self.tile_threshold = tile_threshold

        if file_list is not None:
            self.pairs = file_list
        else:
            # 自动匹配 images/ 和 vis/ 中同名文件
            self.pairs = []
            img_root = Path(image_dir)
            mask_root = Path(mask_dir) if mask_dir else None
            for img_file in sorted(img_root.glob("*")):
                if not img_file.is_file() or img_file.suffix not in ('.png', '.jpg'):
                    continue
                mask_file = mask_root / img_file.name if mask_root else None
                if mask_file and mask_file.exists():
                    self.pairs.append((str(img_file), str(mask_file)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # 加载 RGB 原图
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except (OSError, IOError):
            return self.__getitem__((idx + 1) % len(self))

        # 加载 vis 标签：红色(255,0,0)→脏污，黑色(0,0,0)→背景
        vis = np.array(Image.open(mask_path).convert("RGB"))
        # R 通道 == 255 且 G==0 且 B==0 → 红色 = 脏污
        is_red = (vis[:, :, 0] > 200) & (vis[:, :, 1] < 50) & (vis[:, :, 2] < 50)
        binary_mask = is_red.astype(np.float32)           # 脏污=1.0
        ignore_mask = (vis[:, :, 0] == 0) & (vis[:, :, 1] == 0) & (vis[:, :, 2] == 0)  # 纯黑=背景

        # 数据变换
        if self.transform:
            image, binary_mask = self.transform(image, binary_mask)
            ignore_img = Image.fromarray(ignore_mask)
            ignore_img = ignore_img.resize(
                (self.transform.img_size, self.transform.img_size), Image.NEAREST)
            ignore_mask = np.array(ignore_img)

        if hasattr(binary_mask, "numpy"):
            mask_np = binary_mask.numpy()
        else:
            mask_np = binary_mask

        # 增强 tile 标签
        from .tile_label import mask_to_tile_label
        tile_label = mask_to_tile_label(
            mask_np, grid_size=self.grid_size, threshold=self.tile_threshold,
            binary=False, use_gaussian_weight=True, gamma=0.7, smooth_sigma=0.6,
        )
        tile_ignore = mask_to_tile_label(
            ignore_mask.astype(np.float32), grid_size=self.grid_size,
            threshold=0.5, binary=False,
            use_gaussian_weight=False, gamma=1.0, smooth_sigma=0.0,
        ) > 0.5

        valid = ~ignore_mask
        coverage = mask_np[valid].mean() if valid.sum() > 0 else 0.0

        return {
            "image": image,
            "tile_label": torch.from_numpy(tile_label).float(),
            "tile_ignore": torch.from_numpy(tile_ignore),
            "coverage": torch.tensor([coverage], dtype=torch.float32),
        }
