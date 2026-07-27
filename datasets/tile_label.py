"""Tile 标签生成模块。

将像素级 mask 转换为 grid_size × grid_size 的 tile 级监督信号。
支持 4 种增强策略以提升标签质量：
    1. 高斯中心加权 — 污染靠近 tile 中心权重大，模拟视觉影响
    2. 污染类型加权 — 不透明(泥)>透明(水滴) 的逐像素权重
    3. 非线性增强 (gamma) — 放大低值区，防止小污染被模型忽略
    4. 边界高斯平滑 — tile 间过渡自然，消除网格伪影
"""

import numpy as np


def _gaussian_blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """纯 NumPy 实现的 2D 高斯模糊（分离卷积，避免 scipy 依赖）。

    分离卷积技巧: 先对每一行做 1D 卷积，再对每一列做 1D 卷积，
    效果等价于 2D 卷积但速度快一个数量级。

    Args:
        arr: (H, W) 输入数组
        sigma: 高斯核标准差（越大越模糊）

    Returns:
        (H, W) 模糊后的数组
    """
    # 计算核大小：±3σ 范围，确保覆盖 99.7% 的质量
    size = int(2 * np.ceil(3 * sigma) + 1)
    ax = np.arange(-(size // 2), size // 2 + 1)
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel /= kernel.sum()  # 归一化确保总和为1

    # 分离卷积：先水平方向（axis=1），再垂直方向（axis=0）
    tmp = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, arr)
    result = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, tmp)
    return result


def _gaussian_kernel(tile_h: int, tile_w: int, sigma: float = 0.4) -> np.ndarray:
    """生成以 tile 中心为峰值的高斯权重核。

    中心像素权重最高 → 靠近中心的污染贡献更多 → 编码"视觉影响"先验。
    例: 28×28 tile, sigma=0.4 → 中心像素权重约为边缘的 3 倍。

    Args:
        tile_h: tile 高度（像素），如 224/8=28
        tile_w: tile 宽度（像素）
        sigma: 高斯核标准差（相对于 tile 尺寸的比例，默认 0.4）

    Returns:
        (tile_h, tile_w) 归一化权重，sum = 1.0
    """
    # 计算中心坐标
    cy, cx = (tile_h - 1) / 2.0, (tile_w - 1) / 2.0
    # σ 按 tile 尺寸缩放
    sy, sx = tile_h * sigma, tile_w * sigma

    ys = np.arange(tile_h)
    xs = np.arange(tile_w)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    # 2D 高斯: exp(-((x-cx)²/(2σx²) + (y-cy)²/(2σy²)))
    kernel = np.exp(-((xx - cx) ** 2 / (2 * sx ** 2) + (yy - cy) ** 2 / (2 * sy ** 2)))
    kernel /= kernel.sum()  # 归一化
    return kernel.astype(np.float32)


def mask_to_tile_label(
    mask: np.ndarray,
    grid_size: int = 8,
    threshold: float = 0.1,
    binary: bool = False,
    # === 增强参数 ===
    use_gaussian_weight: bool = True,
    gaussian_sigma: float = 0.4,
    gamma: float = 0.7,
    pixel_weight: np.ndarray = None,
    smooth_sigma: float = 0.6,
) -> np.ndarray:
    """将像素级 mask 转换为增强的 tile 级标签。

    算法步骤:
        1. 将 (H,W) mask 划分为 grid_size×grid_size 个 tile
        2. 对每个 tile，计算加权面积比:
           label[i,j] = sum(mask × gaussian_kernel × pixel_weight) / sum(gaussian_kernel)
        3. 应用 gamma 非线性增强（放大低值）
        4. 对 tile 网格做高斯平滑（消除突变）

    Args:
        mask: (H, W) 二元或连续 mask，值域 [0, 1] (1=脏污)
        grid_size: tile 网格大小 (默认 8×8=64 tiles)
        threshold: 二值化阈值（仅 binary=True 时使用）
        binary: True 返回二元 {0,1}，False 返回连续 [0,1]
        use_gaussian_weight: 是否使用高斯中心加权
        gaussian_sigma: 高斯核 σ（0.4 → 中心权重大约是边缘的 3 倍）
        gamma: 非线性增强指数 (<1 放大低值区，>1 抑制低值区)
        pixel_weight: (H, W) 逐像素权重 (如不透明污染=1.0, 透明污染=0.5)
        smooth_sigma: tile 网格高斯平滑 σ (0=不平滑)

    Returns:
        (grid_size, grid_size) 增强 tile 标签

    示例:
        # 简单面积比（旧版标签）
        label = mask_to_tile_label(mask, grid_size=8,
                                    use_gaussian_weight=False, gamma=1.0,
                                    pixel_weight=None, smooth_sigma=0.0)
        # 增强标签（默认）
        label = mask_to_tile_label(mask, grid_size=8)  # 使用所有默认增强
    """
    h, w = mask.shape
    tile_h = h // grid_size  # 每个 tile 的高度（像素），如 224/8=28
    tile_w = w // grid_size  # 每个 tile 的宽度（像素）

    # === 第一步：预计算高斯权重核 ===
    # 所有 tile 共用同一个核（因为所有 tile 尺寸相同）
    if use_gaussian_weight:
        gauss_kernel = _gaussian_kernel(tile_h, tile_w, sigma=gaussian_sigma)
    else:
        # 不使用权重的退化情况：等价于简单面积比
        gauss_kernel = np.ones((tile_h, tile_w), dtype=np.float32) / (tile_h * tile_w)

    labels = np.zeros((grid_size, grid_size), dtype=np.float32)

    # === 第二步：逐 tile 计算加权面积比 ===
    for i in range(grid_size):
        for j in range(grid_size):
            # 提取当前 tile 的像素区域
            tile = mask[
                i * tile_h:(i + 1) * tile_h,
                j * tile_w:(j + 1) * tile_w,
            ].astype(np.float32)

            # 提取对应的逐像素权重（如污染类型权重）
            if pixel_weight is not None:
                pw = pixel_weight[
                    i * tile_h:(i + 1) * tile_h,
                    j * tile_w:(j + 1) * tile_w,
                ].astype(np.float32)
            else:
                pw = np.ones_like(tile)

            # 加权面积比 = sum(mask × gauss × type_weight) / sum(gauss)
            # 注意分母是整个 gauss_kernel 的 sum，不是 gauss×pw 的 sum
            # 这样设计是为了：即使 pw 改变了分子，分母不变，保证数值范围 [0,1]
            weighted = tile * gauss_kernel * pw
            denominator = gauss_kernel.sum()
            labels[i, j] = weighted.sum() / max(denominator, 1e-8)

    # === 第三步：非线性增强 (gamma correction) ===
    # gamma < 1: 低值区域被"拉高" (例: 0.04→0.20, 0.5→0.61)
    # gamma > 1: 低值区域被"压低" (例: 0.04→0.001, 0.5→0.25)
    # 默认 gamma=0.7 增强小污染区域，防止模型将其忽略
    labels = np.clip(labels, 0.0, 1.0)
    labels = labels ** gamma

    # === 第四步：tile 网格空间平滑 ===
    # 对 8×8 标签做高斯模糊，使相邻 tile 过渡更自然
    # 消除因网格划分产生的硬边界伪影
    if smooth_sigma > 0:
        labels = _gaussian_blur_2d(labels, sigma=smooth_sigma)
        labels = np.clip(labels, 0.0, 1.0)

    # === 第五步（可选）：二值化 ===
    if binary:
        labels = (labels > threshold).astype(np.float32)

    return labels
