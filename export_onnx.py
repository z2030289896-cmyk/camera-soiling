"""ONNX 模型导出脚本。

用法:
    python export_onnx.py

会自动加载 cfg/export.yaml 配置并导出 ONNX 模型。
"""


import sys
from pathlib import Path

import torch
import torch.onnx

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.builder import build_model
from utils.config import load_config


def export_onnx():
    # 加载配置
    export_cfg = load_config("cfg/export.yaml")
    model_cfg = load_config("cfg/model.yaml")

    # 设备
    device = torch.device("cpu")

    # 创建模型
    print("Building model...")
    model = build_model(model_cfg)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # 加载权重
    checkpoint_path = export_cfg["checkpoint"]
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", "?")
        print(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch})")
    else:
        print(f"Warning: checkpoint not found at {checkpoint_path}")

    # 输入尺寸
    input_size = tuple(export_cfg.get("input_size", [1, 3, 224, 224]))
    dummy_input = torch.randn(*input_size, device=device)

    # 输出路径
    output_path = export_cfg.get("output_path", "outputs/model.onnx")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 动态轴
    dynamic_axes = export_cfg.get("dynamic_axes", {
        "input": {0: "batch_size"},
        "tile": {0: "batch_size"},
        "coverage": {0: "batch_size"},
    })

    # 导出
    print(f"Exporting ONNX to: {output_path}")
    print(f"  Input shape:  {input_size}")
    print(f"  Opset version: {export_cfg.get('opset_version', 17)}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=export_cfg.get("opset_version", 17),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["tile", "coverage"],
        dynamic_axes=dynamic_axes,
    )

    # 验证
    print("\nVerifying exported model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified OK!")

    # 计算大小
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")
    print(f"\nDone! Exported to: {output_path}")


if __name__ == "__main__":
    export_onnx()
