"""ResNet18 内部实现 — 零 torchvision 依赖。

从 torchvision/models/resnet.py 移植，仅保留 ResNet18 需要的 BasicBlock。
实现与 torchvision 完全一致以保证:
    1. state_dict 键名完全匹配 (预训练权重可直接加载)
    2. 前向输出数值完全一致

预训练权重: backbones/weights/resnet18-f37072fd.pth
来源: https://download.pytorch.org/models/resnet18-f37072fd.pth
"""

from typing import Callable, Optional, Union

import torch
from torch import nn, Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1,
            dilation: int = 1) -> nn.Conv2d:
    """3×3 卷积 (与 torchvision 一致, padding=1)。"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1×1 卷积 (与 torchvision 一致)。"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """ResNet18 基础残差块: conv3x3 → BN → ReLU → conv3x3 → BN → +identity → ReLU。"""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18Internal(nn.Module):
    """ResNet18 完整结构 (与 torchvision 完全一致, 不含 fc)。

    组件属性命名与 torchvision 一致:
        conv1/bn1/relu/maxpool, layer1~layer4
    """

    def __init__(self, block: Callable[..., nn.Module] = BasicBlock,
                 layers: list = (2, 2, 2, 2)) -> None:
        super().__init__()
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # stem: conv1 → bn1 → relu → maxpool
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layer1~layer4
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 权重初始化 (与 torchvision 一致)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1,
                    dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers_list = [block(self.inplanes, planes, stride, downsample,
                             self.groups, self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_list.append(block(self.inplanes, planes, groups=self.groups,
                                     base_width=self.base_width, dilation=self.dilation,
                                     norm_layer=norm_layer))

        return nn.Sequential(*layers_list)

    def forward(self, x: Tensor) -> Tensor:
        """完整前向: 返回 layer4 输出 (B, 512, 7, 7)。"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18_internal(pretrained: bool = True, weights_path: str = None) -> ResNet18Internal:
    """构建 ResNet18 (内部实现)。

    Args:
        pretrained: 是否加载 ImageNet 预训练权重
        weights_path: 预训练权重文件路径 (None 时用默认本地路径)

    Returns:
        ResNet18Internal 完整模型
    """
    model = ResNet18Internal(BasicBlock, (2, 2, 2, 2))

    if pretrained:
        from pathlib import Path
        if weights_path is None:
            weights_path = str(Path(__file__).parent / "weights" / "resnet18-f37072fd.pth")
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"预训练权重不存在: {weights_path}\n"
                f"请下载: https://download.pytorch.org/models/resnet18-f37072fd.pth")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # 过滤 fc 键 (项目仅使用卷积层部分)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        model.load_state_dict(state_dict)

    return model
