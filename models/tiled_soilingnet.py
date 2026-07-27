"""TiledSoilingNet — 完整的 Tile-Based 相机镜头脏污检测网络。

这是项目的核心模型文件。模型不包含单独的 RegressionHead，
coverage 直接由 tile_prob 取均值得到，减少了参数和潜在的梯度冲突。
"""

import torch
import torch.nn as nn

from backbones.builder import build_backbone
from heads.tile_head import TileHead


class TiledSoilingNet(nn.Module):
    """Tile-Based 相机镜头脏污检测网络。

    完整数据流：
        Input (B, 3, 224, 224)
            │
            ▼ Backbone (ResNet18 layer3 / MobileNetV3-Small)
        Feature Map (B, C, 14, 14)      ResNet18: C=256, MobileNetV3: C=48
            │
            ▼ TileHead (空间池化 → 逐tile分类)
        tile_logits (B, 8, 8)           每格 raw logits
            │
            ├── Sigmoid ──▶ tile_prob (B, 8, 8) ∈ [0,1]  推理/可视化用
            │
            └── mean(dim=1,2) ──▶ coverage (B,)           整体覆盖率

    输出字典:
        - "tile_logits": (B, 8, 8)   BCEWithLogitsLoss 使用
        - "tile_prob":   (B, 8, 8)   推理时直接作为置信度
        - "coverage":    (B,)        = mean(tile_prob)，整体覆盖率

    设计决策:
        - 不使用单独的 RegressionHead：减少 12K 参数，避免 tile 和 coverage
          两个任务之间的梯度冲突
        - coverage = mean(tile_prob)：如果 tile 预测准确，coverage 自然准确
        - 使用 layer3 (14×14) 而非 layer4 (7×7)：保留空间分辨率用于 tile 定位
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: model.yaml 配置字典，包含 backbone 和 tile_head 子配置
        """
        super().__init__()

        # === Backbone：特征提取 ===
        # 根据 cfg["backbone"]["name"] 选择 ResNet18 或 MobileNetV3
        backbone_cfg = cfg["backbone"]
        self.backbone = build_backbone(backbone_cfg)
        backbone_channels = backbone_cfg.get("out_channels", 256)
        # ResNet18 → 256ch@14×14, MobileNetV3 → 48ch@14×14

        # === TileHead：空间池化 + 逐tile分类 ===
        # 输出 (B, 8, 8)，每个位置对应图像中一个有空间意义的 tile
        tile_cfg = cfg["tile_head"]
        self.tile_head = TileHead(
            in_channels=tile_cfg.get("in_channels", backbone_channels),
            grid_size=tile_cfg["grid_size"],
        )

    def forward(self, x):
        """前向传播。

        Args:
            x: (B, 3, 224, 224) RGB 图像 tensor

        Returns:
            dict: {"tile_logits", "tile_prob", "coverage"}
        """
        # 1. Backbone: 提取空间特征
        feature = self.backbone(x)                       # → (B, C, 14, 14)

        # 2. TileHead: 空间池化 → 逐tile logits
        tile_logits = self.tile_head(feature)            # → (B, 8, 8)

        # 3. Sigmoid: logits → 概率 [0, 1]
        tile_prob = torch.sigmoid(tile_logits)           # → (B, 8, 8)

        # 4. Coverage: 所有 tile 概率的均值
        coverage = tile_prob.mean(dim=(1, 2))            # → (B,)

        return {
            "tile_logits": tile_logits,   # 用于 BCEWithLogitsLoss (接受logits)
            "tile_prob": tile_prob,       # 用于 SmoothL1Loss + 推理可视化
            "coverage": coverage,         # 用于 Coverage MSE (可选)
        }
