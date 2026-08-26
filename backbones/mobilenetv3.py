"""MobileNetV3-Small 轻量级 backbone。

专为边缘设备部署设计，参数量仅 ResNet18 的 1/20 (0.6M vs 12M)。
截断到 features[:9] 以输出 14×14 空间分辨率，保证 tile 定位精度。

实现: 使用内部移植的 mobilenet_v3_small_internal (backbones/mobilenetv3_internal.py)
零 torchvision 依赖，预训练权重从本地 backbones/weights/ 加载。
"""

import torch.nn as nn

from .mobilenetv3_internal import mobilenet_v3_small_internal


class MobileNetV3Backbone(nn.Module):
    """MobileNetV3-Small 轻量 backbone，适用于边缘设备部署。

    截断到 14×14 空间分辨率（16× 下采样），保留空间信息用于 tile 级定位。
    使用 InvertedResidual + Squeeze-and-Excitation (SE) 模块。

    特征图尺度（输入 224×224）:
        features[0]:   112×112 (16ch)   Conv+BN+ReLU
        features[1:3]:  56×56  (16-24ch) InvertedResidual
        features[4:6]:  28×28  (40ch)    InvertedResidual
        features[7:9]:  14×14  (48ch)    InvertedResidual+SE  ← 截断点
        features[9:13]:  7×7   (96-576ch) 丢弃（分辨率太低）

    与 ResNet18 对比:
        |            | ResNet18 | MobileNetV3 |
        |------------|----------|-------------|
        | 输出通道    | 256      | 48          |
        | 参数量      | 11.2M    | 0.6M        |
        | 模型大小    | 74 MB    | 7 MB        |
        | 适用场景    | 精度优先  | 边缘部署     |
    """

    def __init__(self, pretrained: bool = True):
        """
        Args:
            pretrained: 是否加载 ImageNet 预训练权重
        """
        super().__init__()

        # 内部实现构建完整 MobileNetV3-Small，预训练权重从本地加载
        model = mobilenet_v3_small_internal(pretrained=pretrained)

        # 截取前 9 层：输出 (B, 48, 14, 14)
        # features[9:] 是 7×7 低分辨率层，对 tile 定位帮助不大
        self.features = model.features[:9]

    def forward(self, x):
        """前向传播。

        Args:
            x: (B, 3, 224, 224) RGB 图像

        Returns:
            (B, 48, 14, 14) 特征图（通道数远小于 ResNet18 的 256）
        """
        return self.features(x)

    @property
    def out_channels(self) -> int:
        """输出通道数。

        MobileNetV3-Small features[:9] = 48ch。
        TileHead 需要根据此值设置 in_channels。
        """
        return 48
