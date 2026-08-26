"""合并数据集训练入口 — 从零训练 MobileNetV3。

将 self-made(红黑二值mask) + WoodScape rgbLabels(4色mask) 合并为统一数据集。
两种mask格式自动归一化为：异常=1, 干净=0, 背景=ignore。

用法:
    python train_combined.py                    # 默认50 epochs, 从零训练
    python train_combined.py --epochs 100       # 自定义epochs
"""

import argparse, csv, shutil, sys
from pathlib import Path
import torch, torch.optim as optim
from torch.amp import GradScaler, autocast

# 项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataloader_combined import build_dataloader_combined
from losses.total_loss import TotalLoss
from metrics.visualize import (save_heatmap_compare, save_overlay_compare,
                               save_grid_compare)
from models.builder import build_model
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed


def _save_image(tensor, path):
    """将 (3,H,W) tensor 保存为 PNG 原图。"""
    from utils.pil_utils import save_image
    save_image(tensor, path)


def train_one_epoch(model, dataloader, loss_fn, optimizer, scaler, device, epoch, cfg, logger):
    """训练一个 epoch：遍历所有 batch，前向→loss→反向→更新权重。

    使用混合精度(AMP)加速，支持纯 CPU 模式自动降级。
    返回该 epoch 的平均 loss。
    """
    model.train()  # 训练模式
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # === 数据移到GPU ===
        images = batch["image"].to(device)
        for k in ["tile_label", "tile_ignore", "coverage"]:
            batch[k] = batch[k].to(device)

        optimizer.zero_grad()

        # === 混合精度前向+反向 ===
        if cfg.get("use_amp", True) and device.type == "cuda":
            # CUDA: 使用 AMP 加速，减少显存占用
            with autocast("cuda"):
                outputs = model(images)
                loss, loss_dict = loss_fn(outputs, batch)
            scaler.scale(loss).backward()  # 梯度缩放防止下溢
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU 或禁用 AMP：标准流程
            outputs = model(images)
            loss, loss_dict = loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()

        running_loss += loss_dict["total_loss"]

        # 定期打印训练进度
        if (batch_idx + 1) % cfg["log_interval"] == 0:
            logger.info(f"Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(dataloader):4d} "
                        f"| Loss: {running_loss/(batch_idx+1):.4f}")

    return running_loss / len(dataloader)


@torch.no_grad()  # 验证阶段关闭梯度计算，节省显存
def validate(model, dataloader, loss_fn, device, epoch, cfg, logger):
    """验证一个 epoch，并保存 GT vs Pred 对比可视化。

    从 4 个不同 batch (0/10/20/30) 各取 1 张图生成对比图。
    """
    model.eval()  # 评估模式（关闭 Dropout/BatchNorm）
    running_loss = 0.0
    running_rmse = 0.0  # 累加每个 batch 的 MSE，最后开方得 RMSE
    n_batches = 0
    viz_samples = []  # [(image, tile_prob, coverage, gt_tile), ...]

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        for k in ["tile_label", "tile_ignore", "coverage"]:
            batch[k] = batch[k].to(device)

        outputs = model(images)
        _, loss_dict = loss_fn(outputs, batch)
        running_loss += loss_dict["total_loss"]

        # 计算 Tile RMSE: sqrt(mean((pred - label)² over valid tiles))
        tile_prob = outputs["tile_prob"]          # (B, 8, 8)
        tile_label = batch["tile_label"]          # (B, 8, 8)
        tile_ignore = batch["tile_ignore"]        # (B, 8, 8) bool
        for b in range(images.shape[0]):
            valid = ~tile_ignore[b]
            if valid.sum() > 0:
                mse = ((tile_prob[b][valid] - tile_label[b][valid]) ** 2).mean()
                running_rmse += mse.item()
                n_batches += 1

        # 从不同 batch 收集样本（确保覆盖不同脏污类型）
        if len(viz_samples) < 4 and batch_idx in [0, 10, 20, 30]:
            idx = (epoch + batch_idx) % images.shape[0]  # 用 epoch 偏移选不同位置
            viz_samples.append((
                images[idx].cpu(),
                outputs["tile_prob"][idx].cpu(),      # 模型预测概率
                outputs["coverage"][idx].cpu(),        # 模型预测覆盖率
                batch["tile_label"][idx].cpu(),         # GT tile 标签
            ))

    avg = running_loss / len(dataloader)
    val_rmse = (running_rmse / max(n_batches, 1)) ** 0.5  # RMSE = sqrt(mean(MSE))
    logger.info(f"VAL   Epoch {epoch:3d} | Loss: {avg:.4f} | RMSE: {val_rmse:.4f}")

    # === 保存可视化对比图 ===
    if viz_samples:
        hd = Path(cfg.get("heatmap_dir", "outputs/heatmap"))
        hd.mkdir(parents=True, exist_ok=True)
        for si, (img, tile_prob, cov, gt_tile) in enumerate(viz_samples):
            s = f"val_epoch_{epoch:03d}_s{si}"          # 文件名前缀
            gt_np, pred_np = gt_tile.numpy(), tile_prob.numpy()
            # 原图
            _save_image(img, str(hd / f"{s}_original.png"))
            # GT(左) vs Pred(右) 并排对比
            save_heatmap_compare(img, gt_np, pred_np,
                                 str(hd / f"{s}_compare_heatmap.png"))
            save_overlay_compare(img, gt_np, pred_np,
                                 str(hd / f"{s}_compare_overlay.png"))
            save_grid_compare(img, gt_np, pred_np,
                              str(hd / f"{s}_compare_grid.png"))
        logger.info(f"  Saved {len(viz_samples)} viz samples to {hd}")

    return avg, val_rmse


def main():
    """主训练流程：加载配置 → 准备数据 → 从零训练 → 归档 → 评估。"""
    # === 命令行参数 ===
    parser = argparse.ArgumentParser(description="合并数据集从零训练")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--bce_weight", type=float, default=None, help="BCE损失权重 (覆盖配置文件)")
    args = parser.parse_args()

    # === 加载配置 ===
    dataset_cfg = load_config("cfg/dataset_combined.yaml")      # 合并数据集配置
    model_cfg = load_config("cfg/model_mobilenetv3.yaml")       # MobileNetV3 模型
    train_cfg = load_config("cfg/train.yaml")                    # 训练超参数
    train_cfg["epochs"] = args.epochs
    cfg = {**dataset_cfg, **train_cfg}  # 合并为统一字典

    # === 初始化 ===
    set_seed(cfg.get("seed", 42))  # 固定随机种子
    logger = get_logger()
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # === 数据：合并 self-made + WoodScape rgbLabels ===
    train_loader, val_loader = build_dataloader_combined(dataset_cfg)
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # === 模型：从零训练 MobileNetV3（ImageNet 预训练权重在 backbone 内部加载）===
    model = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,} (训练方式: 从零训练)")

    # === 损失函数（消融实验最优配置：SmoothL1 + BCE）===
    bce_weight = args.bce_weight if args.bce_weight is not None else cfg.get("bce_weight", 0.5)
    logger.info(f"Loss config: SmoothL1=1.0, BCE={bce_weight}")
    loss_fn = TotalLoss(tile_weight=1.0, bce_weight=bce_weight)

    # === 优化器 + 学习率调度 ===
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg.get("weight_decay", 1e-4))
    # CosineAnnealing: 学习率从 lr 余弦衰减到 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # === 混合精度梯度缩放器 ===
    scaler = GradScaler("cuda", enabled=cfg.get("use_amp", True))

    # === 输出目录 ===
    ckpt_dir = Path(cfg["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.get("log_dir", "outputs/logs")); log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "lr", "train_loss", "val_loss", "val_rmse", "best_val_loss"])

    # === 训练循环 ===
    start_epoch, best_val_loss = 1, float("inf")
    logger.info(f"Training {cfg['epochs']} epochs...")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        logger.info(f"{'='*50}\nEpoch {epoch}/{cfg['epochs']}")

        # 训练
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler,
                                     device, epoch, cfg, logger)
        scheduler.step()  # 更新学习率
        lr = scheduler.get_last_lr()[0]
        logger.info(f"LR: {lr:.2e} | Train: {train_loss:.4f}")

        # 验证 + 保存可视化
        if epoch % cfg.get("val_interval", 1) == 0:
            val_loss, val_rmse = validate(model, val_loader, loss_fn, device, epoch, cfg, logger)
            csv_writer.writerow([epoch, f"{lr:.6e}", f"{train_loss:.6f}",
                                  f"{val_loss:.6f}", f"{val_rmse:.6f}",
                                  f"{best_val_loss:.6f}"])
            csv_file.flush()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                                str(ckpt_dir / "best.pth"))
                logger.info(f"  -> Best model ({best_val_loss:.4f})")

        # 每 epoch 保存最新（用于断点续训）
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                        str(ckpt_dir / "last.pth"))

    csv_file.close()

    # === 训练完成统计 ===
    best_mb = (ckpt_dir / "best.pth").stat().st_size / (1024 * 1024)
    logger.info(f"Complete! Best: {best_val_loss:.4f} | "
                f"Params: {n_params:,} | Size: {best_mb:.1f} MB")

    # === 实验归档 ===
    exp_name = dataset_cfg.get("experiment", "Combined")
    exp_name = f"{exp_name}_BCE{bce_weight}"  # 区分不同bce_weight的实验
    exp_dir = Path("outputs/experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    # 配置文件快照
    for f in ["cfg/dataset_combined.yaml", "cfg/train.yaml", "cfg/model_mobilenetv3.yaml"]:
        src = Path(f)
        if src.exists():
            shutil.copy2(src, exp_dir / src.name)
    # 训练日志
    if Path("outputs/logs").exists():
        shutil.copytree("outputs/logs", exp_dir / "logs", dirs_exist_ok=True)
    # 验证热力图
    if Path("outputs/heatmap").exists():
        shutil.copytree("outputs/heatmap", exp_dir / "heatmap", dirs_exist_ok=True)
    # 最佳模型
    if (ckpt_dir / "best.pth").exists():
        shutil.copy2(ckpt_dir / "best.pth", exp_dir / "best.pth")
    logger.info(f"Archived to: {exp_dir}")

    # === 自动评估 ===
    logger.info("Running evaluation...")
    from metrics.evaluation import evaluate_model
    eval_dir = Path("outputs/results/metrics") / exp_name
    evaluate_model(model, val_loader, device, output_dir=str(eval_dir))


if __name__ == "__main__":
    main()
