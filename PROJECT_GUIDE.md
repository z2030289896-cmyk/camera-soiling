# Camera Soiling Detection — 项目文件详解

本文档介绍 `camera_soiling/` 目录下**每一个文件**的作用。

---

## 根目录 (camera_soiling/)

| 文件 | 行数 | 作用 |
|------|:---:|------|
| `.gitignore` | 13 | Git 忽略规则：排除 outputs/、data/、.pth 等大文件 |
| `README.MD` | — | 项目说明文档 |
| `requirements.txt` | 8 | Python 依赖：torch, torchvision, numpy, Pillow, PyYAML, onnx |
| `__init__.py` | 0 | Python 包标记 |
| `convert_labelme_to_mask.py` | 165 | LabelMe JSON 标注 → 多边形 mask 转换脚本 |
| `export_onnx.py` | 96 | 将训练好的 .pth 模型导出为 ONNX 格式 |
| `predict.py` | — | 推理/评估入口：`--image_dir` 推理、`--eval` 评估、`--stats` 无GT统计 |
| `train.py` | — | WoodScape gtLabels 训练入口（最基础的训练脚本） |
| `train_rgblabels.py` | — | WoodScape rgbLabels 训练入口 |
| `train_raindrops.py` | — | RaindropsOnWindshield 训练入口 |
| `train_self.py` | — | Self 数据集微调入口（支持全数据训练） |
| `train_combined.py` | — | 合并数据集从零训练入口（self-made + WoodScape） |

---

## backbones/ — 特征提取网络

| 文件 | 作用 |
|------|------|
| `__init__.py` | 包标记 |
| `builder.py` | 工厂函数 `build_backbone(cfg)`，根据配置名创建 ResNet18 或 MobileNetV3 |
| `resnet18.py` | ResNet18 Backbone：截断到 layer3，输出 (B,256,14,14)，12M 参数 |
| `resnet.py` | ResNet18 内部实现（从 torchvision 移植）：BasicBlock/conv3x3 等组件，零 torchvision 依赖，state_dict 键名与 torchvision 完全兼容 |
| `mobilenetv3.py` | MobileNetV3-Small Backbone：截断到 features[:9]，输出 (B,48,14,14)，0.6M 参数 |
| `mobilenet.py` | MobileNetV3 内部实现（从 torchvision 移植）：InvertedResidual/SE/Conv2dNormActivation 等组件，零 torchvision 依赖，state_dict 键名与 torchvision 完全兼容 |
| `weights/` | 预训练权重目录：resnet18-f37072fd.pth (45MB) + mobilenet_v3_small-047dcff4.pth (9.9MB)，本地加载无需联网 |

---

## cfg/ — 配置文件（YAML）

| 文件 | 作用 |
|------|------|
| `dataset.yaml` | WoodScape gtLabels 数据集配置（路径、grid_size、标签映射） |
| `dataset_rgblabels.yaml` | WoodScape rgbLabels 数据集配置（自动颜色检测） |
| `dataset_raindrops.yaml` | RaindropsOnWindshield 二值 mask 配置 |
| `dataset_self.yaml` | Self/self1/self-made 数据集配置（支持 train_split=1.0 全训练） |
| `dataset_combined.yaml` | 合并数据集配置（self-made + WoodScape，支持 smooth_sigma 参数） |
| `model.yaml` | ResNet18 模型结构配置（逐层描述） |
| `model_mobilenetv3.yaml` | MobileNetV3 模型结构配置（逐层描述） |
| `train.yaml` | 训练超参数：epochs、lr、batch_size、损失权重、消融实验名 |
| `export.yaml` | ONNX 导出配置：输入尺寸、opset、动态轴 |

---

## datasets/ — 数据处理管线

### 数据集类

| 文件 | 作用 |
|------|------|
| `woodscape.py` | WoodScape gtLabels 数据集：灰度标签 → 重映射为二元 mask → tile label |
| `woodscape_rgblabels.py` | WoodScape rgbLabels 数据集：自动检测颜色 → 二元 mask → tile label |
| `raindrops_dataset.py` | RaindropsOnWindshield 数据集：二值 mask（白=异常，黑=背景） |
| `self_dataset.py` | Self 系列数据集：红黑 mask（红=异常，黑=干净），支持 self/self1/self-made |
| `combined_dataset.py` | **合并数据集**：统一 self-made(红黑) + WoodScape(4色) 两种 mask 格式为二元 mask |

### 数据加载

| 文件 | 作用 |
|------|------|
| `dataloader.py` | WoodScape gtLabels DataLoader：随机 80/20 划分 |
| `dataloader_rgblabels.py` | WoodScape rgbLabels DataLoader |
| `dataloader_raindrops.py` | RaindropsOnWindshield DataLoader：文件级别随机划分 |
| `dataloader_self.py` | Self 数据集 DataLoader：支持 train_split=1.0 全训练模式 |
| `dataloader_combined.py` | 合并数据集 DataLoader：从配置读取 smooth_sigma 参数 |

### 数据处理工具

| 文件 | 作用 |
|------|------|
| `tile_label.py` | **核心标签生成**：像素级 mask → (8,8) tile 标签，支持高斯中心加权、gamma 增强、边界平滑 |
| `transforms.py` | 数据增强：Resize、Flip、亮度/对比度抖动、边缘遮挡（防 Shortcut Learning） |

---

## heads/ — 网络输出头

| 文件 | 作用 |
|------|------|
| `tile_head.py` | **TileHead**：Refine→AdaptiveAvgPool2d(8,8)→1×1 Conv，输出 (B,1,8,8) logits，保持空间对应 |
| `decoder.py` | 特征图解码器：5×上采样 14×14→224×224，用于像素级可视化（非训练必需） |
| `regression_head.py` | RegressionHead：AdaptiveAvgPool2d→FC→Sigmoid，**当前未使用**（coverage=mean(tile_prob)） |

---

## losses/ — 损失函数

| 文件 | 作用 |
|------|------|
| `total_loss.py` | **核心损失函数**：SmoothL1Loss + BCEWithLogitsLoss（消融确认最优），支持 HardNeg、FP Penalty、Coverage MSE、TV 平滑等可选组件 |
| `bce.py` | BCEWithLogitsLoss 简单封装 |
| `mse.py` | MSELoss 简单封装 |

---

## metrics/ — 评价和可视化

| 文件 | 作用 |
|------|------|
| `visualize.py` | **热力图可视化**：save_heatmap（绝对值线性颜色映射）、save_overlay、save_grid_compare 等 |
| `evaluation.py` | **评估模块**：有GT时生成全指标报告（连续指标+二值指标+混淆矩阵+best/worst cases+JSON） |
| `diagnostics.py` | 训练诊断工具：GT/Pred 直方图、FP 空间分布热力图 |
| `coverage.py` | 基础覆盖率计算 |

---

## models/ — 完整模型

| 文件 | 作用 |
|------|------|
| `tiled_soilingnet.py` | **TiledSoilingNet**：Backbone + TileHead，coverage=mean(sigmoid(tile_logits))，无需独立 RegressionHead |
| `builder.py` | 工厂函数 `build_model(cfg)` |
| `test_resnet.py` | 快速测试：验证 Backbone 输出 shape=(1,256,14,14) |

---

## utils/ — 工具函数

| 文件 | 作用 |
|------|------|
| `config.py` | YAML 配置加载 |
| `checkpoint.py` | 模型/优化器/调度器保存和加载（支持断点续训） |
| `logger.py` | 统一日志工具 |
| `seed.py` | 固定随机种子（random/numpy/torch/cudnn） |
| `pil_utils.py` | PIL 图像工具（替代 torchvision 工具函数）：save_image / resize / hflip / 亮度对比度 / to_tensor，零 torchvision 依赖 |

---

## outputs/ — 输出目录

| 子目录 | 来源 | 内容 |
|------|------|------|
| `checkpoint/` | 训练自动 | 模型权重 (best/last/epoch_*.pth) |
| `heatmap/` | 训练每epoch | GT vs Pred 对比可视化 (16图/epoch) |
| `logs/` | 训练自动 | training_log.csv |
| `diagnostics/` | 训练每25轮 | GT/Pred直方图+FP热力图 |
| `fixed_samples/` | 训练每5轮 | 固定样本 GT/Pred/Diff 对比 |
| `experiments/` | 训练结束 | **所有实验归档**（配置+日志+热力图+模型） |
| `results/visual/` | predict.py | 推理可视化（热力图/overlay/grid/original + stats.json） |
| `results/metrics/` | --eval/--stats | 评估报告（JSON + 混淆矩阵 + CSV + best/worst cases） |
| `gt_viz/` | 手动 | GT标签质量验证 |
| `rgblabels_viz/` | 手动 | rgbLabels颜色检测验证 |
