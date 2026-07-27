from .resnet18 import ResNet18Backbone
from .mobilenetv3 import MobileNetV3Backbone


def build_backbone(cfg):
    """根据配置创建 backbone 特征提取网络。

    Args:
        cfg: backbone 配置字典，需包含 'name' 字段。

    Returns:
        nn.Module: backbone 网络实例。

    Raises:
        ValueError: 未知的 backbone 名称。
    """
    name = cfg["name"]

    if name == "resnet18":
        return ResNet18Backbone(
            pretrained=cfg.get("pretrained", True)
        )

    elif name == "mobilenetv3":
        return MobileNetV3Backbone(
            pretrained=cfg.get("pretrained", True)
        )

    else:
        raise ValueError(f"Unknown backbone: {name}")
