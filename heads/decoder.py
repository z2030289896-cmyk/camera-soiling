import torch.nn as nn


class Decoder(nn.Module):
    """特征图解码器。

    将 backbone 输出的 14×14 低分辨率特征图通过 5 次 2× 上采样
    逐步恢复到 224×224 输入尺寸，用于像素级可视化。
    非训练必需模块。
    """

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)
