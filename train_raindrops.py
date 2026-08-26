"""RaindropsOnWindshield 训练入口。

基于 RaindropsOnWindshield 二值 mask 数据集训练脏污检测模型。
训练完成后自动运行评估并生成完整指标报告。

用法:
    python train_raindrops.py                              # ResNet18
    python train_raindrops.py --model cfg/model_mobilenetv3.yaml  # MobileNetV3
    python train_raindrops.py --epochs 100                 # 自定义训练轮数
"""

import argparse, csv, shutil, sys
from pathlib import Path
import numpy as np
import torch, torch.optim as optim
from torch.amp import GradScaler, autocast

# 将项目根目录加入 Python 路径，确保相对导入正常工作
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataloader_raindrops import build_dataloader_raindrops
from losses.total_loss import TotalLoss
from metrics.visualize import (save_heatmap_compare, save_overlay_compare,
                               save_grid_compare)
from models.builder import build_model
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed


def _save_image(tensor, path):
    """将 (3, H, W) tensor 保存为 PNG 图片。"""
    from utils.pil_utils import save_image
    save_image(tensor, path)


def train_one_epoch(model, dataloader, loss_fn, optimizer, scaler, device, epoch, cfg, logger):
    """训练一个 epoch。

    使用混合精度（AMP）加速训练，每个 batch 后更新权重。
    返回该 epoch 的平均 loss。
    """
    model.train()  # 训练模式（启用 dropout/batchnorm）
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # 数据移到 GPU
        images = batch["image"].to(device)
        for k in ["tile_label", "tile_ignore", "coverage"]:
            batch[k] = batch[k].to(device)

        optimizer.zero_grad()

        # 混合精度前向+反向（仅在 CUDA 可用时启用）
        if cfg.get("use_amp", True) and device.type == "cuda":
            with autocast("cuda"):
                outputs = model(images)
                loss, loss_dict = loss_fn(outputs, batch)
            scaler.scale(loss).backward()  # 梯度缩放
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, loss_dict = loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()

        running_loss += loss_dict["total_loss"]

        # 定期打印训练进度
        if (batch_idx + 1) % cfg["log_interval"] == 0:
            avg = running_loss / (batch_idx + 1)
            logger.info(f"Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(dataloader):4d} | Loss: {avg:.4f}")

    return running_loss / len(dataloader)


@torch.no_grad()  # 验证阶段不计算梯度，节省显存
def validate(model, dataloader, loss_fn, device, epoch, cfg, logger):
    """验证一个 epoch，并保存 4 个样本的可视化对比图。

    从 4 个不同 batch（0/10/20/30）各取 1 张图，
    生成 GT vs Pred 对比热力图/overlay/grid。
    """
    model.eval()  # 评估模式
    running_loss = 0.0
    viz_samples = []  # 收集用于可视化的样本 [(image, tile_prob, coverage, gt_tile), ...]

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        for k in ["tile_label", "tile_ignore", "coverage"]:
            batch[k] = batch[k].to(device)

        outputs = model(images)
        _, loss_dict = loss_fn(outputs, batch)
        running_loss += loss_dict["total_loss"]

        # 从不同 batch 收集样本，确保覆盖不同脏污类型
        if len(viz_samples) < 4 and batch_idx in [0, 10, 20, 30]:
            # 用 epoch 偏移确保每轮选不同位置的图
            idx = (epoch + batch_idx) % images.shape[0]
            viz_samples.append((
                images[idx].cpu(),
                outputs["tile_prob"][idx].cpu(),       # 模型预测概率
                outputs["coverage"][idx].cpu(),         # 模型预测覆盖率
                batch["tile_label"][idx].cpu(),          # GT tile 标签
            ))

    avg = running_loss / len(dataloader)
    logger.info(f"VAL   Epoch {epoch:3d} | Loss: {avg:.4f}")

    # 保存可视化对比图
    if viz_samples:
        hd = Path(cfg.get("heatmap_dir", "outputs/heatmap"))
        hd.mkdir(parents=True, exist_ok=True)
        for si, (img, tile_prob, cov, gt_tile) in enumerate(viz_samples):
            s = f"val_epoch_{epoch:03d}_s{si}"
            gt_np, pred_np = gt_tile.numpy(), tile_prob.numpy()
            # 原图
            _save_image(img, str(hd / f"{s}_original.png"))
            # GT(左) vs Pred(右) 并排对比
            save_heatmap_compare(img, gt_np, pred_np, str(hd / f"{s}_compare_heatmap.png"))
            save_overlay_compare(img, gt_np, pred_np, str(hd / f"{s}_compare_overlay.png"))
            save_grid_compare(img, gt_np, pred_np, str(hd / f"{s}_compare_grid.png"))
        logger.info(f"  Saved {len(viz_samples)} viz samples to {hd}")

    return avg


def main():
    """主训练流程：加载配置 → 准备数据 → 训练 → 归档 → 评估。"""
    # === 命令行参数 ===
    parser = argparse.ArgumentParser(
        description="RaindropsOnWindshield 脏污检测训练")
    parser.add_argument("--model", type=str, default="cfg/model.yaml",
                        help="模型配置文件路径")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    args = parser.parse_args()

    # === 加载配置 ===
    dataset_cfg = load_config("cfg/dataset_raindrops.yaml")  # RaindropsOnWindshield 数据
    model_cfg = load_config(args.model)                       # 模型结构
    train_cfg = load_config("cfg/train.yaml")                  # 训练超参数
    train_cfg["epochs"] = args.epochs
    cfg = {**dataset_cfg, **train_cfg}  # 合并为统一字典

    # === 初始化 ===
    set_seed(cfg.get("seed", 42))  # 固定随机种子，确保可复现
    logger = get_logger()
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # === 数据 ===
    # build_dataloader_raindrops 内部做文件级随机 80/20 划分
    train_loader, val_loader = build_dataloader_raindrops(dataset_cfg)
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # === 模型 ===
    model = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    # === 损失函数（使用消融实验确认的最优配置：SmoothL1 + BCE）===
    loss_cfg = cfg.get("loss", {})
    loss_fn = TotalLoss(
        tile_weight=loss_cfg.get("tile_weight", 1.0),          # SmoothL1Loss
        bce_weight=loss_cfg.get("bce_weight", 0.3),            # BCEWithLogitsLoss
    )

    # === 优化器 + 调度器 ===
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg.get("weight_decay", 1e-4))
    # CosineAnnealing：学习率从 lr 逐步衰减到 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    # 混合精度梯度缩放器
    scaler = GradScaler("cuda", enabled=cfg.get("use_amp", True))

    # === 输出目录 ===
    ckpt_dir = Path(cfg["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.get("log_dir", "outputs/logs")); log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "lr", "train_loss", "val_loss", "best_val_loss"])

    # === 断点续训 ===
    start_epoch, best_val_loss = 1, float("inf")
    resume_path = ckpt_dir / "last.pth"
    if resume_path.exists():
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, resume_path)
        logger.info(f"Resumed from epoch {start_epoch}")

    # === 训练循环 ===
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
            val_loss = validate(model, val_loader, loss_fn, device, epoch, cfg, logger)
            csv_writer.writerow([epoch, f"{lr:.6e}", f"{train_loss:.6f}",
                                  f"{val_loss:.6f}", f"{best_val_loss:.6f}"])
            csv_file.flush()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                                str(ckpt_dir / "best.pth"))
                logger.info(f"  -> Best model ({best_val_loss:.4f})")

        # 定期保存 checkpoint
        if epoch % cfg.get("save_interval", 25) == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                            str(ckpt_dir / f"epoch_{epoch:03d}.pth"))
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                        str(ckpt_dir / "last.pth"))

    csv_file.close()

    # === 训练完成统计 ===
    best_mb = (ckpt_dir / "best.pth").stat().st_size / (1024 * 1024)
    logger.info(f"Complete! Best: {best_val_loss:.4f} | Params: {n_params:,} | Size: {best_mb:.1f} MB")

    # === 实验归档：保存配置 + 日志 + 热力图 + 模型到固定目录 ===
    bb = model_cfg.get("backbone", {}).get("name", "unknown")
    exp_dir = Path("outputs/experiments") / f"Raindrops_{bb}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    # 配置文件快照
    for f in ["cfg/dataset_raindrops.yaml", "cfg/train.yaml", args.model]:
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

    # === 自动评估：对验证集生成完整量化指标 ===
    logger.info("Running evaluation...")
    from metrics.evaluation import evaluate_model
    eval_dir = Path("outputs/results/metrics") / f"Raindrops_{bb}"
    evaluate_model(model, val_loader, device, output_dir=str(eval_dir))


if __name__ == "__main__":
    main()
