"""RaindropsOnWindshield 数据集。

二值 mask：白色(255)=雨滴/脏污，黑色(0)=背景。
自动递归遍历 images/ 和 masks/ 的子目录（含嵌套），匹配同名文件。
支持损坏图片自动跳过，不影响训练。
"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RaindropsDataset(Dataset):
    """RaindropsOnWindshield 二值 mask 数据集。

    白色像素 = 脏污(1)，黑色像素 = 背景(0/ignore)。
    使用与 WoodScape 相同的增强 tile 标签生成（高斯中心加权+gamma+平滑）。

    数据目录结构：
        images/
            ├── 1o/  (500张, 有雨滴)
            ├── 2o/  (1700张)
            ├── 8o/  (1800张)
            ├── D1/  (190张)
            ├── D2/  (100张)
            ├── D3/  (嵌套子目录, 3张, 干净)
            ├── D4/  (嵌套子目录, 2张, 干净)
            └── v2/  (300张)
        masks/  （同结构）
    """

    def __init__(self, image_dir: str, mask_dir: str = None,
                 transform=None, grid_size: int = 8,
                 tile_threshold: float = 0.1,
                 exclude_scenes: list = None,
                 file_list: list = None):
        """
        Args:
            image_dir: images/ 根目录路径
            mask_dir: masks/ 根目录路径
            transform: 数据变换（TrainTransform 或 ValTransform）
            grid_size: tile 网格大小（默认 8×8=64 tiles）
            tile_threshold: tile 标签二值化阈值
            exclude_scenes: 排除的场景子目录名列表（如 ["D3", "D4"]）
            file_list: 预定义文件列表 [(img_path, mask_path), ...]，
                       用于 train/val 随机划分
        """
        self.transform = transform
        self.grid_size = grid_size
        self.tile_threshold = tile_threshold
        exclude_scenes = exclude_scenes or []

        # 如果传入了预定义列表（train/val划分时使用），直接使用
        if file_list is not None:
            self.pairs = file_list
        else:
            # 自动递归遍历所有场景子目录，匹配 image ↔ mask
            self.pairs = []
            img_root = Path(image_dir)
            mask_root = Path(mask_dir) if mask_dir else None
            for subdir in sorted(img_root.iterdir()):
                if not subdir.is_dir() or subdir.name in exclude_scenes:
                    continue
                # rglob 递归查找所有 PNG（支持 D3/D4 的嵌套结构）
                for img_file in sorted(subdir.rglob("*.png")):
                    if not img_file.is_file():
                        continue
                    # 保留相对于场景子目录的路径结构
                    rel_path = img_file.relative_to(subdir)
                    mask_file = mask_root / subdir.name / rel_path if mask_root else None
                    if mask_file and mask_file.exists():
                        self.pairs.append((str(img_file), str(mask_file)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """加载第 idx 张图片和 mask，转换为模型输入格式。

        流程：
            RGB图像(1280×1024) → transform → (3,224,224) tensor
            二值mask → 白(255)→1.0, 黑(0)→0.0 → resize → 增强tile标签

        Returns:
            dict: {"image": tensor, "tile_label": tensor, "tile_ignore": tensor, "coverage": tensor}
        """
        img_path, mask_path = self.pairs[idx]

        # 加载 RGB 图像（损坏图片自动跳过，取下一张）
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except (OSError, IOError):
            return self.__getitem__((idx + 1) % len(self))

        # 加载二值 mask：R通道>128 → 脏污(1.0)，R通道=0 → 背景(忽略)
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        binary_mask = (mask_rgb[:, :, 0] > 128).astype(np.float32)  # 脏污=1.0
        ignore_mask = (mask_rgb[:, :, 0] == 0)                      # 背景=True

        # 数据变换：image + mask 同步 resize/augment
        if self.transform:
            image, binary_mask = self.transform(image, binary_mask)
            # ignore_mask 同步 resize（NEAREST 插值保持二值）
            ignore_img = Image.fromarray(ignore_mask)
            ignore_img = ignore_img.resize(
                (self.transform.img_size, self.transform.img_size), Image.NEAREST)
            ignore_mask = np.array(ignore_img)

        # 转换为 numpy（transform 后可能是 tensor）
        if hasattr(binary_mask, "numpy"):
            mask_np = binary_mask.numpy()
        else:
            mask_np = binary_mask

        # === 生成增强 tile 标签 ===
        # 复用与 WoodScape 相同的增强策略
        from .tile_label import mask_to_tile_label
        # tile_label: 连续软标签 [0,1]，含高斯中心加权/gamma/平滑
        tile_label = mask_to_tile_label(
            mask_np, grid_size=self.grid_size, threshold=self.tile_threshold,
            binary=False, use_gaussian_weight=True, gamma=0.7, smooth_sigma=0.6,
        )
        # tile_ignore: 背景面积>50%的 tile 不参与 loss
        tile_ignore = mask_to_tile_label(
            ignore_mask.astype(np.float32), grid_size=self.grid_size,
            threshold=0.5, binary=False,
            use_gaussian_weight=False, gamma=1.0, smooth_sigma=0.0,
        ) > 0.5

        # 整体覆盖率：有效区域内脏污像素占比
        valid = ~ignore_mask
        coverage = mask_np[valid].mean() if valid.sum() > 0 else 0.0

        return {
            "image": image,
            "tile_label": torch.from_numpy(tile_label).float(),
            "tile_ignore": torch.from_numpy(tile_ignore),
            "coverage": torch.tensor([coverage], dtype=torch.float32),
        }
