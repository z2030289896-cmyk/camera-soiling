"""多任务损失函数模块。

TotalLoss 是项目的核心损失函数，支持 6 种 loss 组件的任意组合。
消融实验确认最优配置为 SmoothL1 + BCE (Experiment A)。
"""

import torch
import torch.nn.functional as F


class TotalLoss:
    """混合损失函数，支持 6 种可独立开关的 loss 组件。

    组件:
        1. SmoothL1Loss    — 连续回归到 GT 标签 [0,1]，处理中间置信度
        2. BCEWithLogitsLoss — 推向 0/1 极端，防止干净 tile 上维持中间值
        3. Hard Negative Mining — GT≈0 但 Pred 偏高的 tile 加倍惩罚
        4. False Positive Penalty — 显式 L2 惩罚干净 tile 上的非零预测
        5. Coverage MSE — 全局覆盖率监督（默认关闭）
        6. TV Smooth — 空间平滑，鼓励相邻 tile 一致（默认关闭）

    默认配置（消融实验 A，效果最优）：
        TotalLoss(tile_weight=1.0, bce_weight=0.3,
                  coverage_weight=0.0, smooth_weight=0.0,
                  hard_neg_weight=0.0, fp_penalty_weight=0.0)
    """

    def __init__(self,
                 tile_weight: float = 1.0,       # SmoothL1Loss 权重
                 bce_weight: float = 0.3,          # BCEWithLogitsLoss 权重
                 coverage_weight: float = 0.1,     # Coverage MSE 权重 (默认关闭)
                 smooth_weight: float = 0.01,      # TV 平滑权重 (默认关闭)
                 hard_neg_weight: float = 2.0,     # Hard Negative 额外倍数
                 fp_penalty_weight: float = 0.5,   # FP 惩罚权重
                 hard_neg_threshold: float = 0.7,  # Hard Neg 判定阈值 (Pred>此值)
                 fp_threshold: float = 0.05):      # FP 判定阈值 (GT<此值)
        """
        Args:
            tile_weight: SmoothL1Loss 权重
            bce_weight: BCEWithLogitsLoss 权重 (0=纯回归, >0=加分类推力)
            coverage_weight: Coverage MSE 权重 (0=关闭)
            smooth_weight: TV 平滑权重 (0=关闭)
            hard_neg_weight: Hard Negative 额外权重倍数 (1=不额外惩罚)
            fp_penalty_weight: FP 显式惩罚权重 (0=关闭)
            hard_neg_threshold: 判定 Hard Negative 的 Pred 下限
            fp_threshold: 判定干净 tile 的 GT 上限
        """
        self.tile_weight = tile_weight
        self.bce_weight = bce_weight
        self.coverage_weight = coverage_weight
        self.smooth_weight = smooth_weight
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
        coverage_pred = outputs["coverage"]           # (B,) 模型预测覆盖率
        coverage_gt = batch["coverage"].squeeze(-1)  # (B,) GT 覆盖率

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
        # 5. Coverage MSE — 全局覆盖率监督
        #    coverage = mean(tile_prob), 直接与 GT 覆盖率比较
        #    小权重 (0.1) 即可纠正系统性偏高/偏低
        # ================================================================
        if self.coverage_weight > 0:
            cov_loss = F.mse_loss(coverage_pred, coverage_gt)
        else:
            cov_loss = torch.tensor(0.0, device=tile_prob.device)

        # ================================================================
        # 6. TV (Total Variation) Smooth — 空间平滑
        #    水平方向: |tile[i,j] - tile[i,j+1]| 的均值
        #    垂直方向: |tile[i,j] - tile[i+1,j]| 的均值
        #    仅计算相邻 tile 都有效的情况
        # ================================================================
        if self.smooth_weight > 0:
            # 水平相邻差异
            tv_h = torch.abs(tile_prob[:, :, 1:] - tile_prob[:, :, :-1])
            valid_h = valid_mask[:, :, 1:] * valid_mask[:, :, :-1]  # 两边都有效
            tv_h = (tv_h * valid_h).sum() / valid_h.sum().clamp(min=1)
            # 垂直相邻差异
            tv_v = torch.abs(tile_prob[:, 1:, :] - tile_prob[:, :-1, :])
            valid_v = valid_mask[:, 1:, :] * valid_mask[:, :-1, :]
            tv_v = (tv_v * valid_v).sum() / valid_v.sum().clamp(min=1)
            tv_loss = tv_h + tv_v
        else:
            tv_loss = torch.tensor(0.0, device=tile_prob.device)

        # ================================================================
        # 总损失 = 加权求和
        # ================================================================
        total = (self.tile_weight * tile_loss +
                 self.coverage_weight * cov_loss +
                 self.smooth_weight * tv_loss +
                 self.fp_penalty_weight * fp_penalty)

        # 返回总损失 + 各组件分开的数值（用于日志记录）
        return total, {
            "tile_loss": tile_loss.item(),
            "coverage_loss": cov_loss.item(),
            "smooth_loss": tv_loss.item(),
            "total_loss": total.item(),
        }
