"""MobileNetV3-Small 内部实现 — 零 torchvision 依赖。

从 torchvision/models/mobilenetv3.py 移植，仅保留项目需要的组件。
实现与 torchvision 完全一致以保证:
    1. state_dict 键名完全匹配 (预训练权重可直接加载)
    2. 前向输出数值完全一致

预训练权重: backbones/weights/mobilenet_v3_small-047dcff4.pth
来源: https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth
"""

from functools import partial
from typing import Any, Callable, Optional, Union

import torch
from torch import nn, Tensor


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """确保通道数能被 divisor 整除 (来自 TensorFlow 官方实现)。"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 向下取整时最多减少 10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2dNormActivation(torch.nn.Sequential):
    """Conv2d + BatchNorm2d + Activation 封装块。

    以 Sequential 形式存储 [Conv2d, BN, Act]，保证 state_dict 键名
    与 torchvision 一致 (0.weight, 1.weight, 1.bias, ...)。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]] = 3,
        stride: Union[int, tuple[int, int]] = 1,
        padding: Optional[Union[int, tuple[int, int]]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, tuple) else len(dilation)
                kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * _conv_dim
                dilation = dilation if isinstance(dilation, tuple) else (dilation,) * _conv_dim
                padding = tuple((k - 1) // 2 * d for k, d in zip(kernel_size, dilation))

        if bias is None:
            bias = norm_layer is None

        layers: list[nn.Module] = [
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation=dilation, groups=groups, bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels


class SqueezeExcitation(torch.nn.Module):
    """Squeeze-and-Excitation 注意力模块。

    scale_activation 在 MobileNetV3 中使用 Hardsigmoid (与 torchvision 一致)。
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class InvertedResidualConfig:
    """倒残差块结构配置 (MobileNetV3 论文 Table 1/2)。"""

    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    """MobileNetV3 核心倒残差块: expand → depthwise → SE → project。"""

    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: list[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand (1×1 升维, 仅当 expanded != input 时)
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise (3×3/5×5 逐通道卷积)
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # project (1×1 降维, 无激活)
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


def _mobilenet_v3_conf(arch: str, width_mult: float = 1.0, reduced_tail: bool = False,
                       dilated: bool = False, **kwargs: Any):
    """生成 MobileNetV3-Small 的层配置 (与 torchvision 完全一致)。"""
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96, True, "HS", 2, 1),  # C4
            bneck_conf(96, 5, 576, 96, True, "HS", 1, 1),
            bneck_conf(96, 5, 576, 96, True, "HS", 1, 1),
        ]
        last_channel = adjust_channels(1024)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


class MobileNetV3Internal(nn.Module):
    """MobileNetV3 主网络 (仅 features 部分, 项目只用到 features[:9])。

    与 torchvision 的 MobileNetV3 结构完全一致以保证权重兼容。
    """

    def __init__(
        self,
        inverted_residual_setting: list,
        last_channel: int,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: list[nn.Module] = []

        # 第一层: 3→16, k=3, s=2, Hardswish
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2,
                norm_layer=norm_layer, activation_layer=nn.Hardswish,
            )
        )

        # 倒残差块 ×11
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # 最后一层: 96→576, k=1, Hardswish
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels, lastconv_output_channels,
                kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)

        # 权重初始化 (与 torchvision 一致)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


def mobilenet_v3_small_internal(pretrained: bool = True,
                                weights_path: str = None) -> MobileNetV3Internal:
    """构建 MobileNetV3-Small (内部实现)。

    Args:
        pretrained: 是否加载 ImageNet 预训练权重
        weights_path: 预训练权重文件路径 (None 时用默认本地路径)

    Returns:
        MobileNetV3Internal 完整模型 (features 含 13 层)
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small")
    model = MobileNetV3Internal(inverted_residual_setting, last_channel)

    if pretrained:
        from pathlib import Path
        if weights_path is None:
            weights_path = str(Path(__file__).parent / "weights" / "mobilenet_v3_small-047dcff4.pth")
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"预训练权重不存在: {weights_path}\n"
                f"请下载: https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # 过滤 classifier 键 (项目仅使用 features 部分)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
        model.load_state_dict(state_dict)

    return model
