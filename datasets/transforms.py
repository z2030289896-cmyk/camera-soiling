import random

import numpy as np
import torch
from PIL import Image
from utils.pil_utils import functional as F


class TrainTransform:
    """训练数据增强：同步处理 image 和 mask。

    对 image 和 mask 应用一致的几何变换（Resize、Flip），
    仅在 image 上应用颜色增强。

    Mask 值域处理:
        mask 原始值为 {-1, 0, 1}（-1=背景忽略, 0=干净, 1=脏污），
        通过 +1 偏移到 {0, 1, 2} 以兼容 PIL uint8 格式，
        PIL 变换后再 -1 恢复到原始值域。
    """

    def __init__(self, img_size: int = 224):
        self.img_size = img_size

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        # numpy -> PIL
        image = Image.fromarray(image)

        # 将 mask 偏移到非负值域以兼容 PIL uint8
        # -1→0(背景), 0→1(干净), 1→2(脏污)
        mask = Image.fromarray((mask + 1).astype(np.uint8))

        # Resize: image 用双线性插值，mask 用最近邻保持标签值
        image = F.resize(image, [self.img_size, self.img_size],
                         interpolation=F.InterpolationMode.BILINEAR)
        mask = F.resize(mask, [self.img_size, self.img_size],
                        interpolation=F.InterpolationMode.NEAREST)

        # 随机水平翻转（image 和 mask 同步）
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # 颜色增强（仅 image）—— 加大范围打破亮度偏置
        image = F.adjust_brightness(image, brightness_factor=1.0 + random.uniform(-0.3, 0.3))
        image = F.adjust_contrast(image, contrast_factor=1.0 + random.uniform(-0.3, 0.3))

        # 随机边缘遮挡：迫使模型不依赖边缘/暗角特征
        if random.random() > 0.3:
            image_np = np.array(image)
            h, w = image_np.shape[:2]
            margin = random.randint(h // 6, h // 3)
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                image_np[:margin, :, :] = 0
            elif edge == 'bottom':
                image_np[-margin:, :, :] = 0
            elif edge == 'left':
                image_np[:, :margin, :] = 0
            else:
                image_np[:, -margin:, :] = 0
            image = Image.fromarray(image_np)

        # PIL -> tensor
        image = F.to_tensor(image)
        # 恢复 mask 到原始值域: 0→-1, 1→0, 2→1
        mask = torch.from_numpy(np.array(mask)).long() - 1

        return image, mask


class ValTransform:
    """验证数据变换：仅 Resize，无数据增强。

    Mask 值域处理同 TrainTransform。
    """

    def __init__(self, img_size: int = 224):
        self.img_size = img_size

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        # numpy -> PIL
        image = Image.fromarray(image)

        # 偏移 mask 到非负值域: -1→0, 0→1, 1→2
        mask = Image.fromarray((mask + 1).astype(np.uint8))

        # Resize
        image = F.resize(image, [self.img_size, self.img_size],
                         interpolation=F.InterpolationMode.BILINEAR)
        mask = F.resize(mask, [self.img_size, self.img_size],
                        interpolation=F.InterpolationMode.NEAREST)

        # PIL -> tensor
        image = F.to_tensor(image)
        # 恢复: 0→-1, 1→0, 2→1
        mask = torch.from_numpy(np.array(mask)).long() - 1

        return image, mask


def get_train_transform(img_size: int = 224) -> TrainTransform:
    return TrainTransform(img_size)


def get_val_transform(img_size: int = 224) -> ValTransform:
    return ValTransform(img_size)
