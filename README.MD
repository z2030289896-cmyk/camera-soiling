# Camera Soiling Detection

基于 Tile-Based Regression 的相机镜头脏污检测模型。

> 判断"当前镜头脏污是否影响机器人正常视觉工作"。

---

## 项目结构

```
camera_soiling/
│
├── backbones/              # 特征提取网络
│   ├── resnet18.py         # ResNet18 layer3 (14×14×256, 12M params)
│   └── mobilenetv3.py      # MobileNetV3-Small (14×14×48, 0.6M params)
│
├── cfg/                    # 配置文件
│   ├── dataset.yaml        # WoodScape gtLabels
│   ├── dataset_rgblabels.yaml  # WoodScape rgbLabels (自动颜色检测)
│   ├── dataset_self.yaml   # Self 数据集 (红黑二值mask)
│   ├── dataset_combined.yaml   # 合并数据集 (self-made + woodscape)
│   ├── model.yaml          # ResNet18 模型
│   ├── model_mobilenetv3.yaml  # MobileNetV3 模型
│   ├── train.yaml          # 训练参数/损失权重
│   └── export.yaml         # ONNX 导出
│
├── datasets/               # 数据处理
│   ├── woodscape.py        # gtLabels 数据集
│   ├── woodscape_rgblabels.py  # rgbLabels 数据集
│   ├── combined_dataset.py # 合并数据集 (self-made+WoodScape统一mask)
│   ├── self_dataset.py     # Self/self1/self-made 数据集 (红黑mask)
│   ├── raindrops_dataset.py # RaindropsOnWindshield 二值mask
│   ├── dataloader_combined.py # 合并数据集DataLoader
│   ├── dataloader_self.py  # Self数据集DataLoader (支持全训练)
│   ├── dataloader_raindrops.py # Raindrops DataLoader
│   ├── dataloader_rgblabels.py # rgbLabels DataLoader
│   ├── tile_label.py       # Mask→tile标签 (高斯+gamma+平滑)
│   └── transforms.py       # 图像增强
│
├── heads/                  # 网络输出头
│   └── tile_head.py        # 空间池化→逐tile分类
│
├── losses/
│   └── total_loss.py       # SmoothL1 + BCE (最优配置)
│
├── metrics/                # 评价和可视化
│   ├── visualize.py        # 热力图/overlay/grid (绝对值颜色映射)
│   ├── evaluation.py       # 有GT评估：二值+连续性全指标
│   └── diagnostics.py      # 训练诊断
│
├── models/
│   └── tiled_soilingnet.py # Backbone + TileHead, coverage=mean(tile_prob)
│
├── train.py                # gtLabels 训练
├── train.py                # WoodScape gtLabels 训练
├── train_rgblabels.py      # WoodScape rgbLabels 训练
├── train_raindrops.py      # RaindropsOnWindshield 训练
├── train_self.py           # Self 数据集微调
├── train_combined.py       # 合并数据集从零训练
├── predict.py              # 推理 (--image_dir) / 评估 (--eval) / 统计 (--stats)
├── export_onnx.py          # ONNX 导出
└── requirements.txt
```

---

## 网络架构

```
Input (B,3,224,224) → Backbone → (B,C,14,14) → TileHead → tile_logits (B,8,8)
                                                      ↓ sigmoid
                                                tile_prob (B,8,8)
                                                      ↓ mean
                                                coverage (B,)
```

| Backbone | 通道 | 参数 | 大小 | 场景 |
|----------|:---:|:---:|:---:|------|
| ResNet18 | 256 | 12.06M | 74MB | 精度优先 |
| MobileNetV3 | 48 | 0.60M | 7MB | 边缘部署 |

---

## 数据管线

### 标签统一化

所有数据集最终转为 **二元 mask**：异常=1, 干净=0, 背景=ignore

| 数据集 | mask格式 | 异常 | 干净 | 背景 |
|--------|---------|------|------|------|
| woodscape gtLabels | 灰度 | 值2(透明)/值3(不透明) | 值1 | 值0 |
| woodscape rgbLabels | RGB | 蓝+红 | 绿 | 黑 |
| self-made/self1 | RGB | 红 | 黑 | — |
| RaindropsOnWindshield | 二值 | 白 | — | 黑 |
| combined | 混合 | 自动检测 | 自动检测 | 自动检测 |

### Tile 标签增强

默认参数：高斯中心加权(σ=0.4) + gamma=0.7 + 边界平滑(σ=0.6)

---

## 损失函数

```
TotalLoss = 1.0 × SmoothL1Loss(tile_prob, label)   ← 连续回归
          + 0.3 × BCEWithLogitsLoss(tile_logits, label) ← 防误检
```

消融实验确认此配置最优。权重在 `cfg/train.yaml` 可调。

---

## 快速开始

### 训练

```bash
python train.py                     # WoodScape gtLabels
python train_self.py --epochs 20    # Self 数据集微调 (基于预训练)
python train_combined.py            # 合并数据集从零训练
```

### 推理

```bash
python predict.py --image_dir data/images_20260721/clean --stats
```
输出：`*_original.png` + `*_heatmap.png` + `*_overlay.png` + `*_grid.png` + `stats.json` + `results.csv`

### 评估 (需要 GT)

```bash
python predict.py --eval --checkpoint outputs/experiments/xxx/best.pth
```
输出：终端 Summary + `evaluation.json` + 混淆矩阵图 + best/worst cases

---

## 评估指标

### 有GT指标 (`--eval`)

| 指标 | 含义 | 理想值 |
|------|------|:---:|
| 干净区域预测值 | GT干净tile上模型输出的均值 | 0.00 |
| 干净区域误报率(>0.1) | GT干净tile中Pred>0.1的比例 | 0% |
| 脏污区域预测值 | GT脏污tile上模型输出的均值 | 1.00 |
| 脏污区域漏检率(<0.5) | GT脏污tile中Pred<0.5的比例 | 0% |
| 区分度 | (脏污均值-干净均值)/(两标准差和) | >2 |
| Coverage偏差 | 预测coverage-GT coverage均值 | 0 |
| F1-score | 2PR/(P+R), 阈值0.5 | 100% |

### 无GT统计 (`--stats`)

仅描述tile值分布，不判断对错：

```
低值区  [0-0.1)   — tile预测值≈0
中低区  [0.1-0.5) — tile预测值0.1-0.5
中高区  [0.5-0.7) — tile预测值0.5-0.7
高值区  [0.7-1.0] — tile预测值≈1
```

---

## 输出目录

| 目录 | 生成方式 | 内容 |
|------|----------|------|
| `outputs/checkpoint/` | 训练自动 | 模型权重 |
| `outputs/heatmap/` | 训练每epoch | GT vs Pred对比图 |
| `outputs/logs/` | 训练自动 | training_log.csv |
| `outputs/experiments/` | 训练结束 | 完整归档(配置+日志+模型) |
| `outputs/results/visual/` | `predict.py` | 推理可视化 |
| `outputs/results/metrics/` | `--eval`/`--stats` | 评估报告 |

---

## 实验记录

### 消融实验 (outputs/experiments/)

| 实验 | 说明 | 参数 | Best Val |
|------|------|:---:|:---:|
| A_SmoothL1+BCE | ResNet18 + gtLabels, 纯tile loss | 12M | 0.1409 |
| A_SmoothL1+BCE_mobilenetv3 | MobileNetV3 + gtLabels, 纯tile loss | 0.6M | 0.1493 |
| B_SmoothL1+BCE+TV | ResNet18 + gtLabels + TV平滑 | 12M | 0.1429 |
| C_SmoothL1+BCE+Cov | ResNet18 + gtLabels + Coverage | 12M | 0.1458 |
| RGBLabels_resnet18 | ResNet18 + rgbLabels | 12M | 0.1244 |
| RGBLabels_mobilenetv3 | MobileNetV3 + rgbLabels | 0.6M | 0.1407 |
| Raindrops_resnet18 | ResNet18 + RaindropsOnWindshield | 12M | 0.1407 |
| Raindrops_mobilenetv3 | MobileNetV3 + RaindropsOnWindshield | 0.6M | 0.1668 |
| Self_original | MobileNetV3, self/数据集微调(49张) | 0.6M | 0.1340 |
| Self_made_from_original | MobileNetV3, self-made微调(212张) | 0.6M | 0.1678 |
| **Combined_mobilenetv3** | **从零训练, self-made+WoodScape(5212张)** | **0.6M** | **0.1440** |
| Self1_finetuned | Combined微调, self1数据集(53张) | 0.6M | 0.1676 |

### 泛化性验证 (outputs/results/visual/)

| 目录 | 数据集 | 模型 |
|------|--------|------|
| generalization_m1760061 | m1760061 (4场景) | 5个模型对比 |
| generalization_20260716 | 20260716 (57张) | 4个模型对比 |
| generalization_20260717 | 20260717_merged (158张) | 4个模型对比 |
| generalization_20260720 | 20260720_merged (72张) | 4个模型 + Self_finetuned |
| generalization_20260721 | images_20260721 (638张) | 5个模型 + Combined + Self1 |
| generalization_2024 | 2024-07-16 (4场景) | 4个模型对比 |
| generalization_7-15 | 7-15 (58张) | 4个模型对比 |
| generalization_raindrops | RaindropsOnWindshield (8场景) | Combined |
| generalization_tmp | tmp (64张) | Self_finetuned |
| cross_Raindrops2WoodScape | WoodScape验证集 | Raindrops训练模型跨域 |

### 评估指标记录 (outputs/results/metrics/)

| 目录 | 内容 |
|------|------|
| eval_A / eval_A_mbv3 / eval_B / eval_rgblabels / eval_rgblabels_mbv3 | WoodScape验证集评估 |
| reeval_20250716 | 4模型重评估(新指标) |
| Combined_mobilenetv3 / Self_original / Self_made | 对应模型评估 |
| cross_* | 跨域泛化评估 |
| Raindrops_* | Raindrops数据集评估 |
| generalization_* | 无GT统计报告 |

### 最优模型

| 模型 | 训练集 | 参数 | F1 | 干净区预测值 | clean低值区 |
|------|--------|:---:|:---:|:---:|:---:|
| R18_gtLabels | WoodScape | 12M | 95.3% | 0.036 | 37% |
| MBV3_rgbLabels | WoodScape | 0.6M | 92.0% | 0.076 | 83% |
| **Combined_mobilenetv3** | **5212张合并** | **0.6M** | **94.7%** | **0.050** | **100%** |
| Self1_finetuned | Combined微调53张 | 0.6M | — | — | 70% |
