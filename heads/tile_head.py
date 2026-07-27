import torch
import torch.nn as nn


class TileHead(nn.Module):
    """Tile 级脏污预测头 — 保证空间对应关系。

    设计原则：
        先做空间池化到 grid_size，再做 1×1 分类卷积。
        这样每个输出 tile (i,j) 只依赖其对应的图像空间区域。

    流程：
        Feature (B, C, 14, 14)
            ↓  Conv3×3 + BN + ReLU
            ↓  Conv3×3 + BN + ReLU
        (B, 128, 14, 14)
            ↓  AdaptiveAvgPool2d((8, 8))  ← 空间池化：每个 bin 来自对应区域
        (B, 128, 8, 8)
            ↓  Conv2d(128 → 1, 1×1)       ← 逐 tile 分类
        (B, 1, 8, 8)
            ↓  squeeze
        (B, 8, 8)                          ← 每个位置对应固定图像 tile

    输出含义：
        output[i, j] = 图像第 i 行第 j 列 tile 的脏污 logit
        经过 Sigmoid 后为脏污概率 ∈ [0, 1]
    """

    def __init__(self, in_channels: int, grid_size: int):
        super().__init__()

        self.grid_size = grid_size

        # 特征精炼：保持空间分辨率
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 空间池化到目标 grid：每个 bin 聚合对应空间区域
        self.spatial_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # 逐 tile 分类：1×1 卷积 → 每个 tile 独立输出一个 logit
        self.tile_cls = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        # (B, C, 14, 14)
        x = self.refine(x)

        # (B, 128, 14, 14) → (B, 128, 8, 8)
        # 空间池化：每个 8×8 bin 来自对应 14×14 区域
        x = self.spatial_pool(x)

        # (B, 128, 8, 8) → (B, 1, 8, 8)
        # 逐 tile 独立分类
        x = self.tile_cls(x)

        # (B, 1, 8, 8) → (B, 8, 8)，输出 logits（BCEWithLogitsLoss 内部处理 sigmoid）
        x = x.squeeze(1)

        return x
