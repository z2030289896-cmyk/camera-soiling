"""测试 ResNet18 Backbone 前向传播。"""

import torch
from backbones.resnet18 import ResNet18Backbone

# 创建 backbone（不加载预训练权重以加速测试）
model = ResNet18Backbone(pretrained=False)
model.eval()

# 测试: 输入 (1, 3, 224, 224)
x = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    y = model(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")  # 期望: (1, 512, 7, 7)

# 验证输出尺寸
assert y.shape == (1, 512, 7, 7), f"Expected (1, 512, 7, 7), got {y.shape}"
print("Backbone test passed!")
