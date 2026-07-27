"""自动量化评估模块。

对模型推理结果进行全面评估，包括:
- 整体 Summary（终端打印）
- Tile-Level 分类指标 (Accuracy/Precision/Recall/F1)
- Confusion Matrix (图片 + 终端)
- Coverage 误差统计
- Worst/Best Cases 自动收集
- 扩展 results.csv
- evaluation.json

不修改任何现有模型/训练/推理代码，仅作为后处理评估层。
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Evaluator:
    """模型评估器：接收预测结果和 GT，自动生成完整评估报告。

    Args:
        threshold: tile 置信度二值化阈值（默认 0.5）
        output_dir: 评估结果输出目录
        top_n: Worst/Best Cases 收集数量
    """

    def __init__(self, threshold: float = 0.5, output_dir: str = "outputs/eval",
                 top_n: int = 20):
        self.threshold = threshold
        self.output_dir = Path(output_dir)
        self.top_n = top_n

        # 累积统计
        self.all_tp = 0; self.all_tn = 0; self.all_fp = 0; self.all_fn = 0
        self.all_cov_errors = []
        self.all_tile_maes = []
        self.per_image_results = []
        self.total_tiles = 0
        self.total_valid_tiles = 0
        # 连续性指标（不依赖 0.5 阈值）
        self.all_clean_preds = []   # GT<0.05 的 tile 上模型预测值
        self.all_dirty_preds = []   # GT>0.5 的 tile 上模型预测值
        self.all_cov_preds = []
        self.all_cov_gts = []

    def add_sample(self, filename: str, tile_pred: np.ndarray, tile_label: np.ndarray,
                   tile_ignore: np.ndarray, cov_pred: float, cov_gt: float,
                   image_path: str = None):
        """添加一个样本的预测结果和 GT。

        Args:
            filename: 图片文件名
            tile_pred: (G, G) 预测概率 [0,1]
            tile_label: (G, G) GT 连续标签 [0,1]
            tile_ignore: (G, G) bool, True=忽略
            cov_pred: 预测 coverage
            cov_gt: GT coverage
            image_path: 原始图片路径（用于 worst/best cases 收集）
        """
        valid = ~tile_ignore
        pred_bin = (tile_pred > self.threshold).astype(np.int32)
        label_bin = (tile_label > self.threshold).astype(np.int32)

        # TP/FP/TN/FN (仅有效 tile)
        tp = ((pred_bin == 1) & (label_bin == 1) & valid).sum()
        fp = ((pred_bin == 1) & (label_bin == 0) & valid).sum()
        tn = ((pred_bin == 0) & (label_bin == 0) & valid).sum()
        fn = ((pred_bin == 0) & (label_bin == 1) & valid).sum()

        self.all_tp += tp; self.all_fp += fp
        self.all_tn += tn; self.all_fn += fn
        self.total_valid_tiles += valid.sum()
        self.total_tiles += tile_pred.size

        # Tile MAE
        tile_mae = np.abs(tile_pred[valid] - tile_label[valid]).mean() if valid.sum() > 0 else 0
        self.all_tile_maes.append(tile_mae)

        # 连续性指标（非二值化）
        clean_mask = (tile_label < 0.05) & valid
        dirty_mask = (tile_label > 0.5) & valid
        if clean_mask.sum() > 0:
            self.all_clean_preds.extend(tile_pred[clean_mask].tolist())
        if dirty_mask.sum() > 0:
            self.all_dirty_preds.extend(tile_pred[dirty_mask].tolist())
        self.all_cov_preds.append(float(cov_pred))
        self.all_cov_gts.append(float(cov_gt))

        # Coverage error
        cov_error = abs(cov_pred - cov_gt)
        self.all_cov_errors.append(cov_error)

        # 每图指标
        acc = (tp + tn) / max(tp + fp + tn + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        self.per_image_results.append({
            "filename": filename,
            "tile_mae": float(tile_mae),
            "cov_error": float(cov_error),
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "cov_pred": float(cov_pred), "cov_gt": float(cov_gt),
            "image_path": image_path,
        })

    def finalize(self):
        """完成评估，生成所有报告。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. 终端 Summary ---
        self._print_summary()

        # --- 2. Confusion Matrix ---
        self._save_confusion_matrix()

        # --- 3. Coverage 统计 ---
        cov_errors = np.array(self.all_cov_errors)
        coverage_stats = {
            "mae": float(cov_errors.mean()),
            "mean_error": float(np.mean(self.all_cov_errors)),
            "max_error": float(cov_errors.max()),
            "std": float(cov_errors.std()),
        }

        # --- 4. Worst/Best Cases ---
        self._save_cases("worst", reverse=True)
        self._save_cases("best", reverse=False)

        # --- 5. 扩展 CSV ---
        self._save_extended_csv()

        # --- 6. evaluation.json ---
        self._save_json(coverage_stats)

        print(f"\nEvaluation complete. Results saved to: {self.output_dir}")

    # ======================== 内部方法 ========================

    def _compute_global_metrics(self):
        tp, fp = self.all_tp, self.all_fp
        tn, fn = self.all_tn, self.all_fn
        total = tp + fp + tn + fn
        acc = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        # 连续性指标
        clean_preds = np.array(self.all_clean_preds) if self.all_clean_preds else np.array([0])
        dirty_preds = np.array(self.all_dirty_preds) if self.all_dirty_preds else np.array([0])
        cov_preds = np.array(self.all_cov_preds)
        cov_gts = np.array(self.all_cov_gts)

        # 分离度：脏污 tile 和干净 tile 预测值的距离
        separation = (dirty_preds.mean() - clean_preds.mean()) / \
                     max(dirty_preds.std() + clean_preds.std(), 1e-8)

        return {
            "images": len(self.per_image_results),
            "total_tiles": int(self.total_tiles),
            "total_valid_tiles": int(self.total_valid_tiles),
            # 二值指标
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1),
            "tile_mae": float(np.mean(self.all_tile_maes)),
            "coverage_mae": float(np.mean(self.all_cov_errors)),
            # 连续性指标（关键！不依赖 0.5 阈值）
            "clean_tile_pred_mean": float(clean_preds.mean()),
            "clean_tile_pred_median": float(np.median(clean_preds)),
            "clean_tile_gt_0.1_ratio": float((clean_preds > 0.1).mean()),
            "clean_tile_gt_0.2_ratio": float((clean_preds > 0.2).mean()),
            "clean_tile_gt_0.3_ratio": float((clean_preds > 0.3).mean()),
            "dirty_tile_pred_mean": float(dirty_preds.mean()),
            "dirty_tile_pred_median": float(np.median(dirty_preds)),
            "dirty_tile_lt_0.5_ratio": float((dirty_preds < 0.5).mean()),
            "separation_score": float(separation),   # 越高越好
            "coverage_bias": float((cov_preds - cov_gts).mean()),  # 正值=系统性高估
        }

    def _print_summary(self):
        m = self._compute_global_metrics()
        lines = [
            "", "=" * 55,
            "Evaluation Summary",
            "=" * 55,
            f"  Images evaluated       : {m['images']}",
            f"  Valid tiles            : {m['total_valid_tiles']:,}",
            "",
            "--- 二值指标 (threshold=0.5) ---",
            f"  TP / FP / TN / FN      : {m['tp']} / {m['fp']} / {m['tn']} / {m['fn']}",
            f"  Tile Accuracy          : {m['accuracy']*100:.2f}%",
            f"  Precision              : {m['precision']*100:.2f}%",
            f"  Recall                 : {m['recall']*100:.2f}%",
            f"  F1-score               : {m['f1']*100:.2f}%",
            "",
            "--- 实际表现 (原始预测值，不依赖0.5阈值) ---",
            f"  干净区域预测值         : {m['clean_tile_pred_mean']:.4f}  (越低越好，理想=0.00)",
            f"  干净区域误报率(>0.1)   : {m['clean_tile_gt_0.1_ratio']*100:.1f}% (越低越好，理想=0%)",
            f"  干净区域误报率(>0.2)   : {m['clean_tile_gt_0.2_ratio']*100:.1f}%",
            "",
            f"  脏污区域预测值         : {m['dirty_tile_pred_mean']:.4f}  (越高越好，理想=1.00)",
            f"  脏污区域漏检率(<0.5)   : {m['dirty_tile_lt_0.5_ratio']*100:.1f}% (越低越好，理想=0%)",
            "",
            f"  干净/脏污区分度        : {m['separation_score']:.2f}  (越高越好，>2=良好)",
            f"  覆盖率偏差             : {m['coverage_bias']:+.4f}  (正=偏高/虚警多，负=偏低/漏检多)",
            f"  Tile MAE               : {m['tile_mae']:.4f}",
            f"  Coverage MAE           : {m['coverage_mae']:.4f}",
            "=" * 55,
        ]
        print("\n".join(lines))

    def _save_confusion_matrix(self):
        m = self._compute_global_metrics()
        tp, fp, tn, fn = m['tp'], m['fp'], m['tn'], m['fn']

        # 终端打印
        print(f"\nConfusion Matrix (threshold={self.threshold}):")
        print(f"               Pred Clean   Pred Dirty")
        print(f"  GT Clean        {tn:6d}      {fp:6d}")
        print(f"  GT Dirty        {fn:6d}      {tp:6d}")

        # 图片保存
        img_w, img_h = 300, 280
        img = np.full((img_h, img_w, 3), 245, dtype=np.uint8)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
            font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except (IOError, OSError):
            font = font_sm = ImageFont.load_default()

        # 表格
        draw.rectangle([60, 60, 260, 220], outline=(0, 0, 0), width=2)
        draw.line([160, 60, 160, 220], fill=(0, 0, 0), width=1)
        draw.line([60, 140, 260, 140], fill=(0, 0, 0), width=1)
        # 标题
        draw.text((100, 25), f"Confusion Matrix (th={self.threshold})", fill=(0, 0, 0), font=font)
        draw.text((75, 75), "Pred Clean", fill=(0, 0, 0), font=font_sm)
        draw.text((180, 75), "Pred Dirty", fill=(0, 0, 0), font=font_sm)
        draw.text((8, 105), "GT Clean", fill=(0, 0, 0), font=font_sm)
        draw.text((8, 175), "GT Dirty", fill=(0, 0, 0), font=font_sm)
        # 数值
        draw.text((100, 110), f"TN={tn}", fill=(0, 100, 0), font=font)
        draw.text((195, 110), f"FP={fp}", fill=(200, 0, 0), font=font)
        draw.text((100, 180), f"FN={fn}", fill=(200, 0, 0), font=font)
        draw.text((195, 180), f"TP={tp}", fill=(0, 100, 0), font=font)

        pil.save(str(self.output_dir / "confusion_matrix.png"))

    def _save_cases(self, label: str, reverse: bool):
        """收集 best/worst cases。"""
        case_dir = self.output_dir / f"{label}_cases"
        case_dir.mkdir(parents=True, exist_ok=True)

        # 按 tile_mae 排序
        sorted_results = sorted(self.per_image_results,
                                key=lambda x: x['tile_mae'],
                                reverse=reverse)[:self.top_n]

        for rank, r in enumerate(sorted_results):
            src = r.get('image_path')
            if src and os.path.exists(src):
                dst = case_dir / f"{label}_{rank:02d}_{r['filename']}"
                shutil.copy2(src, dst)
            # 保存指标到文本
            info_path = case_dir / f"{label}_{rank:02d}_{Path(r['filename']).stem}.txt"
            with open(info_path, 'w') as f:
                f.write(f"File: {r['filename']}\n")
                f.write(f"Tile MAE: {r['tile_mae']:.4f}\n")
                f.write(f"Coverage Error: {r['cov_error']:.4f}\n")
                f.write(f"Accuracy: {r['accuracy']:.4f}  Precision: {r['precision']:.4f}\n")
                f.write(f"Recall: {r['recall']:.4f}  F1: {r['f1']:.4f}\n")
                f.write(f"TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}\n")

        print(f"  Saved {len(sorted_results)} {label} cases to {case_dir}")

    def _save_extended_csv(self):
        import csv
        path = self.output_dir / "results_extended.csv"
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ["filename", "cov_pred", "cov_gt", "cov_error",
                      "tile_mae", "accuracy", "precision", "recall", "f1",
                      "tp", "fp", "tn", "fn"]
            writer.writerow(header)
            for r in self.per_image_results:
                writer.writerow([
                    r['filename'], f"{r['cov_pred']:.6f}", f"{r['cov_gt']:.6f}",
                    f"{r['cov_error']:.6f}", f"{r['tile_mae']:.6f}",
                    f"{r['accuracy']:.6f}", f"{r['precision']:.6f}",
                    f"{r['recall']:.6f}", f"{r['f1']:.6f}",
                    r['tp'], r['fp'], r['tn'], r['fn'],
                ])
        print(f"  Extended CSV saved to: {path}")

    def _save_json(self, coverage_stats):
        m = self._compute_global_metrics()
        m['coverage_mae'] = coverage_stats['mae']
        m['coverage_max_error'] = coverage_stats['max_error']
        path = self.output_dir / "evaluation.json"
        with open(path, 'w') as f:
            json.dump(m, f, indent=2)
        print(f"  JSON saved to: {path}")


def evaluate_model(model, dataloader, device, threshold=0.5,
                   output_dir="outputs/eval", image_dir=None):
    """对模型在给定 dataloader 上进行全面评估。

    Args:
        model: 模型
        dataloader: 包含 GT 标签的 DataLoader
        device: torch device
        threshold: tile 二值化阈值
        output_dir: 输出目录
        image_dir: 原始图片目录（用于 worst/best cases 图片复制）

    Returns:
        Evaluator 实例，包含所有评估结果
    """
    import torch

    evaluator = Evaluator(threshold=threshold, output_dir=output_dir)
    model.eval()

    print(f"Evaluating on {len(dataloader.dataset)} images...")
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            tile_label = batch['tile_label'].numpy()
            tile_ignore = batch['tile_ignore'].numpy()
            cov_gt = batch['coverage'].squeeze(-1).numpy()

            outputs = model(images)
            tile_prob = torch.sigmoid(outputs['tile_logits']).cpu().numpy()
            cov_pred = outputs['coverage'].cpu().numpy()

            for i in range(len(images)):
                filename = f"img_{len(evaluator.per_image_results):04d}.png"
                img_path = None
                if image_dir:
                    img_path = os.path.join(image_dir, filename)

                evaluator.add_sample(
                    filename=filename,
                    tile_pred=tile_prob[i],
                    tile_label=tile_label[i],
                    tile_ignore=tile_ignore[i].astype(bool),
                    cov_pred=float(cov_pred[i]),
                    cov_gt=float(cov_gt[i]),
                    image_path=img_path,
                )

    evaluator.finalize()
    return evaluator
