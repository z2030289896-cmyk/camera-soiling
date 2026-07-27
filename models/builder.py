"""模型工厂函数 build_model(cfg)。根据配置创建 TiledSoilingNet 实例。"""
from .tiled_soilingnet import TiledSoilingNet


def build_model(cfg):
    """根据配置创建完整的 TiledSoilingNet 模型。

    Args:
        cfg: 模型配置字典（来自 model.yaml）。

    Returns:
        TiledSoilingNet: 完整的脏污检测模型。
    """
    return TiledSoilingNet(cfg)
