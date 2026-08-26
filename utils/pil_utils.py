"""PIL 图像工具 — 零 torchvision 依赖。

替代项目中用到的 torchvision 工具函数:
    torchvision.transforms.functional → functional 命名空间 (resize/hflip/亮度/对比度/to_tensor)
    torchvision.utils.save_image      → save_image

API 与 torchvision 完全一致，底层全部使用 PIL 原生接口。
"""

from PIL import Image, ImageEnhance
import numpy as np
import torch

# Pillow ≥10 使用新枚举，旧版回退到旧常量
try:
    _BILINEAR = Image.Resampling.BILINEAR
    _NEAREST = Image.Resampling.NEAREST
except AttributeError:  # Pillow < 9.1
    _BILINEAR = Image.BILINEAR
    _NEAREST = Image.NEAREST

try:
    _FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
except AttributeError:  # Pillow < 9.1
    _FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT


class InterpolationMode:
    """插值模式 (与 torchvision.transforms.InterpolationMode 对应)。"""
    NEAREST = _NEAREST
    BILINEAR = _BILINEAR


def resize(img: Image.Image, size, interpolation=_BILINEAR) -> Image.Image:
    """调整图像尺寸。

    Args:
        img: PIL Image
        size: int 或 [h, w] 序列 (与 torchvision 一致, 内部转为 PIL 的 (w, h))
        interpolation: 插值模式

    Returns:
        PIL Image
    """
    if isinstance(size, (list, tuple)) and len(size) == 2:
        size = (size[1], size[0])  # [h, w] → (w, h) PIL 格式
    return img.resize(size, interpolation)


def hflip(img: Image.Image) -> Image.Image:
    """水平镜像翻转 (左右互换)。"""
    return img.transpose(_FLIP_LEFT_RIGHT)


def adjust_brightness(img: Image.Image, brightness_factor: float) -> Image.Image:
    """调整亮度 (1.0=不变, <1 变暗, >1 变亮)。"""
    return ImageEnhance.Brightness(img).enhance(brightness_factor)


def adjust_contrast(img: Image.Image, contrast_factor: float) -> Image.Image:
    """调整对比度 (1.0=不变, <1 降低, >1 增强)。"""
    return ImageEnhance.Contrast(img).enhance(contrast_factor)


def to_tensor(pic: Image.Image) -> torch.Tensor:
    """PIL Image → (C, H, W) float32 tensor, 值域 [0, 1]。

    Args:
        pic: PIL RGB Image (uint8)

    Returns:
        (C, H, W) float32 tensor
    """
    arr = np.asarray(pic, dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return tensor


def save_image(tensor: torch.Tensor, path: str) -> None:
    """保存 (C,H,W) float tensor 为图片 (与 torchvision.utils.save_image 一致)。

    Args:
        tensor: (C, H, W) 或 (B, C, H, W) float tensor, 值域 [0, 1]
        path: 保存路径
    """
    if tensor.ndim == 4:
        tensor = tensor[0]  # batch 取第一张
    img = (tensor.detach().cpu().clamp(0.0, 1.0) * 255.0).permute(1, 2, 0).to(torch.uint8).numpy()
    Image.fromarray(img).save(str(path))


class _Functional:
    """functional 命名空间 — 兼容 `from torchvision.transforms import functional as F` 的用法。"""
    InterpolationMode = InterpolationMode
    resize = staticmethod(resize)
    hflip = staticmethod(hflip)
    adjust_brightness = staticmethod(adjust_brightness)
    adjust_contrast = staticmethod(adjust_contrast)
    to_tensor = staticmethod(to_tensor)


functional = _Functional()
