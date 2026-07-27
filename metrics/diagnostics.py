"""训练诊断工具：Tile Label/Prediction 分布 + False Positive 空间分析。"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def tile_label_histogram(tile_labels, ignore_mask, path, bins=20):
    """保存 GT tile label 分布直方图到文本文件。

    统计有效tile的标签值分布，输出ASCII柱状图。
    """
    """保存 GT tile label 分布直方图。

    Args:
        tile_labels: (N, G, G) 或 flattened 的 GT 值
        ignore_mask: 同形状 bool，True=忽略
        path: 保存路径
        bins: 直方图 bin 数
    """
    valid = tile_labels[~ignore_mask]
    hist, edges = np.histogram(valid, bins=bins, range=(0, 1))

    # ASCII 直方图
    max_h = hist.max()
    lines = ["=" * 50, "GT Tile Label Histogram", "=" * 50]
    for i in range(bins):
        bar = "#" * int(hist[i] / max(max_h, 1) * 30)
        lines.append(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}] {hist[i]:5d} {bar}")
    lines.append("=" * 50)

    with open(path, "w") as f:
        f.write("\n".join(lines))

    return valid


def prediction_histogram(tile_preds, ignore_mask, path, bins=20):
    """保存模型预测值分布直方图到文本文件。

    输出每个bin的计数+统计摘要(mean/std/分位数)。
    """
    """保存模型预测分布直方图。

    Args:
        tile_preds: (N, G, G) 预测概率
        ignore_mask: 同形状 bool
        path: 保存路径
        bins: bin 数
    """
    valid = tile_preds[~ignore_mask]
    hist, edges = np.histogram(valid, bins=bins, range=(0, 1))

    max_h = hist.max()
    lines = ["=" * 50, "Prediction Histogram", "=" * 50]
    for i in range(bins):
        bar = "#" * int(hist[i] / max(max_h, 1) * 30)
        lines.append(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}] {hist[i]:5d} {bar}")

    # 统计
    lines.append(f"\n  Mean: {valid.mean():.4f}")
    lines.append(f"  Median: {np.median(valid):.4f}")
    lines.append(f"  Std: {valid.std():.4f}")
    lines.append(f"  >0.3: {(valid>0.3).mean()*100:.1f}%")
    lines.append(f"  >0.7: {(valid>0.7).mean()*100:.1f}%")
    lines.append(f"  <0.1: {(valid<0.1).mean()*100:.1f}%")
    lines.append("=" * 50)

    with open(path, "w") as f:
        f.write("\n".join(lines))


def false_positive_heatmap(preds, labels, ignore_mask, path, threshold=0.05):
    """生成 False Positive 空间分布热力图。

    统计每个tile位置平均有多少次GT≈0但Pred>0.3的误识别，
    输出红色强度图+数值标注。
    """
    """生成 False Positive 空间分布热力图。

    显示每个 tile 位置平均有多少次 GT≈0 但 Pred 偏高的误识别。

    Args:
        preds: (N, G, G) 预测
        labels: (N, G, G) GT
        ignore_mask: (N, G, G) bool
        path: 保存路径
        threshold: GT 小于此值视为"干净 tile"

    Returns:
        (G, G) FP 率矩阵
    """
    G = preds.shape[1]
    fp_map = np.zeros((G, G))
    fp_count = np.zeros((G, G))

    for i in range(preds.shape[0]):
        for r in range(G):
            for c in range(G):
                if ignore_mask[i, r, c]:
                    continue
                if labels[i, r, c] < threshold and preds[i, r, c] > 0.3:
                    fp_map[r, c] += preds[i, r, c]
                    fp_count[r, c] += 1

    # 归一化
    fp_rate = fp_count / max(fp_count.max(), 1)

    # 生成可视化
    img_size = 224
    tile_sz = img_size // G
    viz = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    for r in range(G):
        for c in range(G):
            y0, x0 = r * tile_sz, c * tile_sz
            y1, x1 = y0 + tile_sz, x0 + tile_sz
            # 红色强度 = FP 率
            intensity = int(fp_rate[r, c] * 200 + 55) if fp_count[r, c] > 0 else 20
            viz[y0:y1, x0:x1] = [intensity, 20, 20]

    # 加网格线和数值
    pil = Image.fromarray(viz)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=12)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for r in range(G):
        for c in range(G):
            y0, x0 = r * tile_sz, c * tile_sz
            y1, x1 = y0 + tile_sz, x0 + tile_sz
            draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=1)
            if fp_count[r, c] > 0:
                text = f"{fp_rate[r,c]:.2f}"
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text((x0 + (tile_sz - tw)//2, y0 + (tile_sz - th)//2),
                          text, fill=(255, 255, 255), font=font)

    pil.save(path)
    return fp_rate


def generate_diagnostics(model, dataloader, device, output_dir):
    """生成完整诊断报告。

    Args:
        model: 模型
        dataloader: 验证 DataLoader
        device: torch device
        output_dir: 输出目录路径
    """
    import torch
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []
    all_ignore = []

    with torch.no_grad():
        for batch in dataloader:
            out = model(batch['image'].to(device))
            all_preds.append(out['tile_prob'].cpu().numpy())
            all_labels.append(batch['tile_label'].numpy())
            all_ignore.append(batch['tile_ignore'].numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    ignore = np.concatenate(all_ignore, axis=0)

    # 1. GT Histogram
    valid_labels = tile_label_histogram(
        labels, ignore, str(output_dir / "gt_histogram.txt"))

    # 2. Prediction Histogram
    prediction_histogram(preds, ignore, str(output_dir / "pred_histogram.txt"))

    # 3. False Positive Heatmap
    fp_map = false_positive_heatmap(
        preds, labels, ignore, str(output_dir / "fp_heatmap.png"))

    # 4. Summary
    clean_mask = (labels < 0.05) & (~ignore)
    soiled_mask = (labels > 0.5) & (~ignore)
    clean_preds = preds[clean_mask]
    soiled_preds = preds[soiled_mask]

    summary = [
        "=" * 50,
        "Diagnostic Summary",
        "=" * 50,
        f"Total valid tiles: {clean_mask.sum() + soiled_mask.sum():,}",
        f"Clean tiles (GT<0.05): {clean_mask.sum():,} ({clean_mask.sum()/(clean_mask.sum()+soiled_mask.sum())*100:.1f}%)",
        f"Soiled tiles (GT>0.5): {soiled_mask.sum():,} ({soiled_mask.sum()/(clean_mask.sum()+soiled_mask.sum())*100:.1f}%)",
        "",
        f"Clean tile pred mean: {clean_preds.mean():.4f}",
        f"Clean tile pred median: {np.median(clean_preds):.4f}",
        f"Clean tile FP rate (>0.3): {(clean_preds>0.3).mean()*100:.1f}%",
        f"Clean tile FP rate (>0.5): {(clean_preds>0.5).mean()*100:.1f}%",
        "",
        f"Soiled tile pred mean: {soiled_preds.mean():.4f}",
        f"Soiled tile pred median: {np.median(soiled_preds):.4f}",
        f"Soiled tile miss rate (<0.3): {(soiled_preds<0.3).mean()*100:.1f}%",
        f"Soiled tile miss rate (<0.5): {(soiled_preds<0.5).mean()*100:.1f}%",
        "",
        f"Prediction overall mean: {preds[~ignore].mean():.4f}",
        f"Prediction overall std: {preds[~ignore].std():.4f}",
        "=" * 50,
    ]

    with open(str(output_dir / "summary.txt"), "w") as f:
        f.write("\n".join(summary))

    print("\n".join(summary))
    return preds, labels, ignore
