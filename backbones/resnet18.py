import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Backbone(nn.Module):
    """ResNet18 特征提取网络。

    输出 layer3 特征图（而非 layer4），以保留足够的空间分辨率用于 tile 定位。

    特征图尺度（输入 224×224）:
        layer0: 112×112 (stride 2 + maxpool stride 2 = 4× down)
        layer1:  56×56  (stride 8× down)
        layer2:  28×28  (stride 16× down)
        layer3:  14×14  (stride 16× down, no extra stride)
        layer4:   7×7   (stride 32× down)

    选择 layer3 (14×14, 256ch) 而非 layer4 (7×7, 512ch) 的原因:
        - 14×14 → 8×8 下采样时，每个 tile 有 ~1.75×1.75 个特征像素
        - 保留真正的空间定位能力，而非将卷积通道重排为 tile
        - layer3 的语义信息足够用于脏污检测
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        model = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )

        self.layer0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

        # layer4 不使用（保留以兼容，但不参与前向）
        self.layer4 = model.layer4

    def forward(self, x):
        """前向传播，返回 layer3 特征图。

        Args:
            x: (B, 3, 224, 224)

        Returns:
            (B, 256, 14, 14) 特征图
        """
        x = self.layer0(x)   # (B, 64, 112, 112)
        x = self.layer1(x)   # (B, 64, 56, 56)
        x = self.layer2(x)   # (B, 128, 28, 28)
        x = self.layer3(x)   # (B, 256, 14, 14)
        return x

    @property
    def out_channels(self) -> int:
        """输出通道数（layer3 = 256）。"""
        return 256
