"""推理入口脚本。

用法:
    python predict.py --image path/to/image.jpg
    python predict.py --image_dir data/woodscape/rgbImages/ --output outputs/predict/

输出:
    outputs/predict/
    ├── <name>_heatmap.png       # tile 热力图
    ├── <name>_overlay.png       # 原图叠加
    └── results.csv              # 所有图像的 coverage + 8×8 tile 概率
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.transforms import get_val_transform
from metrics.visualize import save_heatmap, save_overlay, save_grid_overlay
from models.builder import build_model
from utils.config import load_config


def predict_single(model, image_path, transform, device):
    """对单张 RGB 图像进行推理，返回 8×8 tile 概率 + coverage。

    流程: 加载图片 → resize(224,224) → 模型前向 → sigmoid → 返回结果

    Returns:
        dict: {"tile": (8,8) numpy, "coverage": float, "image": (3,H,W) tensor}
    """
    # 加载图片
    image = np.array(Image.open(image_path).convert("RGB"))

    # 变换
    img_size = transform.img_size
    dummy_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    image_tensor, _ = transform(image, dummy_mask)

    # 推理
    model.eval()
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_batch)

    tile_prob = outputs["tile_prob"][0].cpu()      # (8, 8) 概率 [0,1]
    coverage = outputs["coverage"][0].item()

    return {
        "tile": tile_prob.numpy(),
        "coverage": coverage,
        "image": image_tensor,
    }


def main():
    parser = argparse.ArgumentParser(description="Camera Soiling Detection - Inference")
    parser.add_argument("--image", type=str, help="单张图像路径")
    parser.add_argument("--image_dir", type=str, help="图像目录（批量推理）")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoint/best.pth",
                        help="模型权重路径")
    parser.add_argument("--model_config", type=str, default="cfg/model.yaml",
                        help="模型配置文件")
    parser.add_argument("--output", type=str, default="outputs/results/visual/latest",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    parser.add_argument("--stats", action="store_true",
                        help="无GT统计模式：推理后自动生成Coverage分布/tile分布报告")
    parser.add_argument("--eval", action="store_true",
                        help="评估模式：对验证集运行完整评估并生成报告")
    parser.add_argument("--eval_threshold", type=float, default=0.5,
                        help="Tile 二值化阈值 (默认 0.5)")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("必须指定 --image 或 --image_dir")

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model_cfg = load_config(args.model_config)
    model = build_model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    ckpt_size_mb = Path(args.checkpoint).stat().st_size / (1024 * 1024) if Path(args.checkpoint).exists() else 0
    print(f"Model params: {n_params:,}  |  Model size: {ckpt_size_mb:.1f} MB")
    model = model.to(device)

    # 加载权重
    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", "?")
        print(f"Loaded checkpoint: {args.checkpoint} (epoch {epoch})")
    else:
        print(f"Warning: checkpoint not found at {args.checkpoint}, using random weights")

    # 变换
    img_size = model_cfg.get("img_size", 224)
    transform = get_val_transform(img_size)

    # 输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 推理
    if args.image:
        # 单张推理
        result = predict_single(model, args.image, transform, device)

        tile_prob = result["tile"]
        coverage = result["coverage"]
        image = result["image"]

        print(f"Image: {args.image}")
        print(f"Coverage: {coverage:.4f} ({coverage*100:.1f}%)")

        # 打印 tile 网格
        print("Tile Soiling Map (8x8):")
        for row in tile_prob:
            print(" ".join(f"{v:.2f}" for v in row))

        # 保存可视化（原图 + 热力图 + 叠加）
        stem = Path(args.image).stem
        save_heatmap(tile_prob, str(output_dir / f"{stem}_heatmap.png"))
        save_overlay(image, tile_prob, str(output_dir / f"{stem}_overlay.png"))
        save_grid_overlay(image, tile_prob, str(output_dir / f"{stem}_grid.png"))
        from utils.pil_utils import save_image
        save_image(image, str(output_dir / f"{stem}_original.png"))

        # 保存 coverage + tile 数据到 CSV
        _save_csv(output_dir / "results.csv", [(stem, coverage, tile_prob)])
        print(f"Results saved to: {output_dir}")

    elif args.image_dir:
        # 批量推理
        image_dir = Path(args.image_dir)
        image_paths = sorted(image_dir.glob("*"))
        print(f"Processing {len(image_paths)} images from {image_dir}...")

        results = []
        for img_path in image_paths:
            result = predict_single(model, img_path, transform, device)
            coverage = result["coverage"]
            tile = result["tile"]
            image = result["image"]
            results.append((img_path.stem, coverage, tile))

            stem = img_path.stem
            # 保存热力图
            save_heatmap(tile, str(output_dir / f"{stem}_heatmap.png"))
            save_overlay(image, tile, str(output_dir / f"{stem}_overlay.png"))
            save_grid_overlay(image, tile, str(output_dir / f"{stem}_grid.png"))
            from utils.pil_utils import save_image
            save_image(image, str(output_dir / f"{stem}_original.png"))

            if len(results) % 50 == 0:
                print(f"  Processed {len(results)}/{len(image_paths)} images...")

        # 保存 coverage + tile 数据到 CSV
        _save_csv(output_dir / "results.csv", results)

        # 汇总统计
        covs = np.array([r[1] for r in results])
        print(f"\nBatch inference complete ({len(results)} images)")
        print(f"  Mean coverage: {covs.mean():.4f} ({covs.mean()*100:.1f}%)")
        print(f"  Std coverage:  {covs.std():.4f}")
        print(f"  Min coverage:  {covs.min():.4f} ({covs.min()*100:.1f}%)")
        print(f"  Max coverage:  {covs.max():.4f} ({covs.max()*100:.1f}%)")
        print(f"  Dirty (>50%):  {(covs > 0.5).sum()} images")
        # 无GT统计报告
        if args.stats:
            _generate_stats(results, output_dir)

        print(f"Results saved to: {output_dir}")


def _generate_stats(results, output_dir):
    """生成无GT统计报告：Coverage分布 + 预测果断度。"""
    import json
    import numpy as np
    from pathlib import Path

    out = Path(output_dir)
    tile_values = np.array([v for _, _, tile in results for v in tile.flatten()])
    coverages = np.array([cov for _, cov, _ in results])

    # 四区分类（覆盖全值域 [0,1]）
    clean_pct = float(((tile_values>=0)&(tile_values<0.1)).mean())*100     # 果断干净
    unsure_clean_pct = float(((tile_values>=0.1)&(tile_values<0.5)).mean())*100  # 犹豫(偏干净)
    weak_dirty_pct = float(((tile_values>=0.5)&(tile_values<0.7)).mean())*100    # 弱脏污(过阈值但不坚定)
    strong_dirty_pct = float((tile_values>=0.7).mean())*100                # 果断脏污
    decisive_pct = clean_pct + strong_dirty_pct                             # 果断率

    stats = {
        "images": len(results),
        "coverage": {
            "mean": float(coverages.mean()), "median": float(np.median(coverages)),
            "std": float(coverages.std()),
            "min": float(coverages.min()), "max": float(coverages.max()),
            "dirty_ratio": float((coverages>0.5).mean())*100,
        },
        "tile_distribution": {
            "low": clean_pct,              # [0-0.1) 低值区
            "mid_low": unsure_clean_pct,    # [0.1-0.5) 中低区
            "mid_high": weak_dirty_pct,     # [0.5-0.7) 中高区
            "high": strong_dirty_pct,       # [0.7-1.0] 高值区
        },
    }

    with open(out / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*55}")
    print(f"无GT评估 ({len(results)} images)")
    print(f"{'='*55}")
    print(f"  Coverage: mean={coverages.mean():.3f} median={np.median(coverages):.3f} "
          f"range=[{coverages.min():.2f},{coverages.max():.2f}]")
    print(f"  Coverage>0.5(脏污): {(coverages>0.5).mean()*100:.0f}%")
    print(f"")
    print(f"  低值区  [0-0.1):  {clean_pct:.0f}%  (tile预测值≈0)")
    print(f"  中低区[0.1-0.5): {unsure_clean_pct:.0f}%  (tile预测值0.1-0.5)")
    print(f"  中高区[0.5-0.7): {weak_dirty_pct:.0f}%  (tile预测值0.5-0.7)")
    print(f"  高值区[0.7-1.0]: {strong_dirty_pct:.0f}%  (tile预测值≈1)")
    print(f"  ⚠ 无GT，仅描述分布，不判断对错。评估准确度请用 --eval")
    print(f"{'='*55}")


def _save_csv(csv_path, results):
    """保存 coverage + 8×8 tile 概率到 CSV。

    Args:
        csv_path: CSV 文件路径
        results: [(filename, coverage, tile_8x8_array), ...]
    """
    grid_size = results[0][2].shape[0]
    tile_cols = [f"tile_{i}_{j}" for i in range(grid_size) for j in range(grid_size)]
    header = ["filename", "coverage"] + tile_cols

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name, cov, tile in results:
            row = [name, f"{cov:.6f}"] + [f"{v:.6f}" for v in tile.flatten()]
            writer.writerow(row)

    print(f"  Coverage + tile data saved to: {csv_path}")


def _run_evaluation(args):
    """评估模式：对验证集进行完整量化评估。"""
    from datasets.dataloader import build_dataloader
    from metrics.evaluation import evaluate_model

    print(f"\n{'='*50}")
    print(f"Evaluation Mode (threshold={args.eval_threshold})")
    print(f"{'='*50}")

    ds_cfg = load_config("cfg/dataset.yaml")
    _, val_loader = build_dataloader(ds_cfg)

    # 加载模型
    model_cfg = load_config(args.model_config)
    model = build_model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    model = model.to(device := torch.device(args.device if torch.cuda.is_available() else "cpu"))
    print(f"Model params: {n_params:,}")

    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    evaluate_model(model, val_loader, device, threshold=args.eval_threshold,
                   output_dir=args.output)


if __name__ == "__main__":
    # 先检查是否 --eval 模式
    if "--eval" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--eval", action="store_true")
        parser.add_argument("--eval_threshold", type=float, default=0.5)
        parser.add_argument("--checkpoint", type=str, default="outputs/checkpoint/best.pth")
        parser.add_argument("--model_config", type=str, default="cfg/model.yaml")
        parser.add_argument("--output", type=str, default="outputs/results/metrics/latest")
        parser.add_argument("--device", type=str, default="cuda")
        args, _ = parser.parse_known_args()
        _run_evaluation(args)
    else:
        main()
