"""多任务损失函数模块。

TotalLoss 是项目的核心损失函数，支持 6 种 loss 组件的任意组合。
消融实验确认最优配置为 SmoothL1 + BCE (Experiment A)。
"""

import torch
import torch.nn.functional as F


class TotalLoss:
    """连续回归 + 分类 混合损失函数。

    核心组件 (默认开启):
        SmoothL1Loss — 连续回归到 GT 标签 [0,1]
        BCEWithLogitsLoss — 推向 0/1 极端，防误检

    可选组件 (默认关闭，通过参数开启):
        Hard Negative Mining / FP Penalty / Coverage MSE / TV Smooth

    最优配置 (消融实验确认):
        TotalLoss(tile_weight=1.0, bce_weight=0.3)
    """

    def __init__(self,
                 tile_weight: float = 1.0,       # SmoothL1Loss 权重
                 bce_weight: float = 0.3,          # BCEWithLogitsLoss 权重
                 hard_neg_weight: float = 0.0,     # Hard Negative (默认关闭)
                 fp_penalty_weight: float = 0.0,   # FP 惩罚 (默认关闭)
                 hard_neg_threshold: float = 0.7,  # Hard Neg 判定阈值
                 fp_threshold: float = 0.05):      # FP 判定阈值
        self.tile_weight = tile_weight
        self.bce_weight = bce_weight
        self.hard_neg_weight = hard_neg_weight
        self.fp_penalty_weight = fp_penalty_weight
        self.hard_neg_threshold = hard_neg_threshold
        self.fp_threshold = fp_threshold

    def __call__(self, outputs: dict, batch: dict):
        """计算总损失。

        Args:
            outputs: 模型输出 {"tile_logits": (B,G,G), "tile_prob": (B,G,G), "coverage": (B,)}
            batch:   数据 batch {"tile_label": (B,G,G), "tile_ignore": (B,G,G), "coverage": (B,1)}

        Returns:
            total_loss (scalar tensor), loss_dict (dict of float)

        损失公式:
            total = tile_weight * masked_mean(SmoothL1 + bce_weight*BCE) * hard_neg_factor
                  + fp_penalty_weight * mean(pred² on clean tiles)
                  + coverage_weight * MSE(coverage)
                  + smooth_weight * TV(tile_prob)
        """
        # 从 outputs 和 batch 中提取数据
        tile_logits = outputs["tile_logits"]          # (B, G, G) 原始 logits
        tile_prob = outputs["tile_prob"]              # (B, G, G) sigmoid 后概率 [0,1]
        tile_label = batch["tile_label"]              # (B, G, G) GT 连续软标签 [0,1]
        tile_ignore = batch["tile_ignore"]            # (B, G, G) bool 忽略标记

        # 有效 tile mask：1=参与计算, 0=忽略(背景>50%)
        valid_mask = (~tile_ignore).float()
        n_valid = valid_mask.sum().clamp(min=1)  # 至少为1避免除零

        # ================================================================
        # 1. SmoothL1Loss — 连续回归
        #    SmoothL1 在 |x|<1 时等价于 0.5*x² (平滑)，|x|≥1 时为 |x|-0.5 (线性)
        #    相比纯 MSE，对离群点更鲁棒
        # ================================================================
        l1_per_tile = F.smooth_l1_loss(tile_prob, tile_label, reduction="none")
        l1_per_tile = l1_per_tile * valid_mask  # 仅有效 tile 参与

        # ================================================================
        # 2. BCEWithLogitsLoss — 推向 0/1 极端
        #    对 clean tile (GT→0): 将 logits 推向 -∞, sigmoid→0
        #    对 dirty tile (GT→1): 将 logits 推向 +∞, sigmoid→1
        #    纯 SmoothL1 无法提供足够强的梯度，BCE 补充这个短板
        # ================================================================
        bce_per_tile = F.binary_cross_entropy_with_logits(
            tile_logits, tile_label, reduction="none"
        )
        bce_per_tile = bce_per_tile * valid_mask

        # ================================================================
        # 3. Hard Negative Mining — 重点惩罚困难负样本
        #    干净 tile (GT<fp_threshold) 但模型给了高置信度 (Pred>hard_neg_threshold)
        #    → 这类 tile 的 loss 乘以 hard_neg_weight (默认 2.0)
        # ================================================================
        hard_neg_mask = ((tile_label < self.fp_threshold) &
                         (tile_prob > self.hard_neg_threshold) &
                         (~tile_ignore))
        hard_neg_factor = torch.where(
            hard_neg_mask,
            self.hard_neg_weight,          # 困难负样本：加倍惩罚
            torch.ones_like(tile_prob)     # 其他：不额外惩罚
        )

        # 将 hard_neg_factor 应用到 SmoothL1 + BCE 的组合 loss 上
        tile_loss = ((l1_per_tile + self.bce_weight * bce_per_tile) *
                     hard_neg_factor).sum() / n_valid

        # ================================================================
        # 4. False Positive Penalty — 显式 L2 惩罚
        #    GT≈0 的干净 tile 上，任何非零预测都会被惩罚 (pred²)
        #    力度由 fp_penalty_weight 控制 (默认 0.5)
        # ================================================================
        fp_mask = (tile_label < self.fp_threshold) & (~tile_ignore)
        if fp_mask.sum() > 0:
            fp_penalty = (tile_prob[fp_mask] ** 2).mean()
        else:
            fp_penalty = torch.tensor(0.0, device=tile_prob.device)

        # ================================================================
        # 总损失 = tile_loss + fp_penalty (coverage/smooth 已禁用)
        # ================================================================
        total = (self.tile_weight * tile_loss +
                 self.fp_penalty_weight * fp_penalty)

        # 返回总损失 + 各组件分开的数值（用于日志记录）
        return total, {
            "tile_loss": tile_loss.item(),
            "total_loss": total.item(),
        }
