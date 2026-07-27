"""合并数据集：统一处理 self-made(红黑) + woodscape rgbLabels(4色) 两种 mask 格式。

输出统一的二元 mask：异常=1, 干净=0, 背景=ignore。
"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    """合并多个子数据集，统一转为二元mask。

    自动枚举所有子数据集的 image/mask 对，每个子集可指定不同的 mask 解析方式。
    """

    def __init__(self, datasets_cfg: list, transform=None,
                 grid_size: int = 8, tile_threshold: float = 0.1,
                 smooth_sigma: float = 0.6,  # 0=不平滑, 0.3=轻度, 0.6=默认
                 file_indices: list = None):
        """
        Args:
            datasets_cfg: 子数据集配置列表，每项含 image_dir, mask_dir, mask_type
            transform: 数据变换
            grid_size: tile 网格
            tile_threshold: tile 二值化阈值
            file_indices: 预定义索引列表 [(dataset_idx, pair_idx), ...]，用于 train/val 划分
        """
        self.transform = transform
        self.grid_size = grid_size
        self.tile_threshold = tile_threshold
        self.smooth_sigma = smooth_sigma

        # 收集所有 image/mask 对
        self.pairs = []  # [(img_path, mask_path, mask_type), ...]
        for ds_cfg in datasets_cfg:
            img_dir = Path(ds_cfg["image_dir"])
            mask_dir = Path(ds_cfg["mask_dir"])
            mask_type = ds_cfg["mask_type"]
            for img_file in sorted(img_dir.glob("*.png")):
                mask_file = mask_dir / img_file.name
                if mask_file.exists():
                    self.pairs.append((str(img_file), str(mask_file), mask_type))

        # train/val 划分
        if file_indices is not None:
            self.pairs = [self.pairs[i] for i in file_indices]

    def __len__(self):
        return len(self.pairs)

    def _parse_mask(self, mask_rgb: np.ndarray, mask_type: str):
        """根据 mask 类型解析 RGB mask → (binary_mask, ignore_mask)。

        两种 mask 格式统一输出:
            binary_mask: (H,W) float32, 1=异常, 0=干净
            ignore_mask: (H,W) bool, True=背景(不参与loss)

        颜色容差: R/G/B >200 判定为"有该颜色", <50 判定为"无该颜色"。
        """
        if mask_type == "red_black":
            # self-made 风格: 红色多边形标注 → 异常, 纯黑背景 → 干净
            # 无 ignore 区域（全图都参与 loss）
            is_red = (mask_rgb[:,:,0] > 200) & (mask_rgb[:,:,1] < 50) & (mask_rgb[:,:,2] < 50)
            binary_mask = is_red.astype(np.float32)  # 红→1
            ignore_mask = np.zeros(mask_rgb.shape[:2], dtype=bool)

        elif mask_type == "rgb_4color":
            # WoodScape rgbLabels: 4色语义分割
            # 黑=背景(ignore), 绿=干净, 蓝=透明脏污(异常), 红=不透明脏污(异常)
            # 蓝色和红色统一为"异常"类别
            is_black = (mask_rgb[:,:,0] == 0) & (mask_rgb[:,:,1] == 0) & (mask_rgb[:,:,2] == 0)
            is_red = (mask_rgb[:,:,0] > 200) & (mask_rgb[:,:,1] < 50) & (mask_rgb[:,:,2] < 50)
            is_blue = (mask_rgb[:,:,2] > 200) & (mask_rgb[:,:,0] < 50) & (mask_rgb[:,:,1] < 50)

            binary_mask = (is_red | is_blue).astype(np.float32)  # 红+蓝→异常=1
            ignore_mask = is_black  # 黑→ignore=True
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")

        return binary_mask, ignore_mask

    def __getitem__(self, idx):
        img_path, mask_path, mask_type = self.pairs[idx]

        # 加载图像和mask
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except (OSError, IOError):
            return self.__getitem__((idx + 1) % len(self))

        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        binary_mask, ignore_mask = self._parse_mask(mask_rgb, mask_type)

        # 数据变换
        if self.transform:
            # resize pixel_weight (无加权，全1)
            image, binary_mask = self.transform(image, binary_mask)
            ignore_img = Image.fromarray(ignore_mask)
            ignore_img = ignore_img.resize(
                (self.transform.img_size, self.transform.img_size), Image.NEAREST)
            ignore_mask = np.array(ignore_img)

        if hasattr(binary_mask, "numpy"):
            mask_np = binary_mask.numpy()
        else:
            mask_np = binary_mask

        # tile 标签
        from .tile_label import mask_to_tile_label
        tile_label = mask_to_tile_label(
            mask_np, grid_size=self.grid_size, threshold=self.tile_threshold,
            binary=False, use_gaussian_weight=True, gamma=0.7,
            smooth_sigma=self.smooth_sigma,
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
