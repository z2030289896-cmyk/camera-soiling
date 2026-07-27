import torch.nn as nn


class RegressionHead(nn.Module):
    """回归头。

    AdaptiveAvgPool2d(1) → Flatten → Linear → ReLU → Linear → Sigmoid。
    输出整体脏污覆盖率 [0,1]。
    注意：当前模型不使用此模块（coverage=mean(tile_prob)）。
    """

    def __init__(self,in_channels):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)


        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                in_channels,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                1
            ),

            nn.Sigmoid()
        )


    def forward(self,x):

        x = self.pool(x)

        return self.fc(x)