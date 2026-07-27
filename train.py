"""训练入口脚本（WoodScape gtLabels 数据集）。

完整的训练流程：加载配置 → 准备数据 → 创建模型 → 训练循环 → 归档 → 评估。

用法:
    python train.py                              # 默认 ResNet18
    python train.py --model cfg/model_mobilenetv3.yaml  # MobileNetV3
"""

import argparse, csv, shutil, sys
from pathlib import Path
import numpy as np
import torch, torch.optim as optim
from torch.amp import GradScaler, autocast

# 项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataloader import build_dataloader
from losses.total_loss import TotalLoss
from metrics.diagnostics import generate_diagnostics
from metrics.visualize import (save_heatmap_compare, save_overlay_compare,
                               save_grid_compare)
from models.builder import build_model
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed


def _save_image(tensor, path):
    """将 (3,H,W) tensor 保存为 PNG 原图。"""
    from torchvision.utils import save_image
    save_image(tensor, path)


def train_one_epoch(model, dataloader, loss_fn, optimizer, scaler, device, epoch, cfg, logger):
    """训练一个 epoch：遍历所有batch，前向→loss→反向→更新权重。

    使用混合精度(AMP)加速训练。返回各项 loss 的平均值。
    """
    model.train()  # 训练模式（启用 Dropout/BatchNorm 更新）
    running_loss = 0.0          # 累计总 loss
    running_tile_loss = 0.0     # 累计 tile 分类 loss
    running_cov_loss = 0.0      # 累计 coverage 回归 loss
    running_smooth_loss = 0.0   # 累计 TV 平滑 loss

    for batch_idx, batch in enumerate(dataloader):
        # === 数据移到 GPU ===
        images = batch["image"].to(device)
        batch["tile_label"] = batch["tile_label"].to(device)
        batch["tile_ignore"] = batch["tile_ignore"].to(device)
        batch["coverage"] = batch["coverage"].to(device)

        optimizer.zero_grad()  # 清零梯度

        # === 混合精度前向 + 反向 ===
        if cfg.get("use_amp", True) and device.type == "cuda":
            with autocast("cuda"):
                outputs = model(images)
                loss, loss_dict = loss_fn(outputs, batch)
            scaler.scale(loss).backward()  # 梯度缩放（防止 FP16 下溢）
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, loss_dict = loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()

        running_loss += loss_dict["total_loss"]
        running_tile_loss += loss_dict["tile_loss"]
        running_cov_loss += loss_dict["coverage_loss"]
        running_smooth_loss += loss_dict.get("smooth_loss", 0.0)

        # 定期打印训练进度
        if (batch_idx + 1) % cfg["log_interval"] == 0:
            avg_total = running_loss / (batch_idx + 1)
            avg_tile = running_tile_loss / (batch_idx + 1)
            avg_cov = running_cov_loss / (batch_idx + 1)
            avg_smooth = running_smooth_loss / (batch_idx + 1)
            logger.info(f"Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(dataloader):4d} | "
                        f"Total: {avg_total:.4f} | Tile: {avg_tile:.4f} | "
                        f"Cov: {avg_cov:.4f} | Smooth: {avg_smooth:.4f}")

    n = len(dataloader)
    return (running_loss/n, running_tile_loss/n, running_cov_loss/n, running_smooth_loss/n)


@torch.no_grad()  # 验证阶段不计算梯度，节省显存
def validate(model, dataloader, loss_fn, device, epoch, cfg, logger):
    """验证一个 epoch：计算 loss + logits/prob 分布 + 保存 GT vs Pred 对比图。

    从 4 个不同 batch 各取 1 张图，生成热力图/overlay/grid 对比。
    """
    model.eval()  # 评估模式
    running_loss = running_tile_loss = running_cov_loss = running_smooth_loss = 0.0
    viz_samples = []  # [(image, tile_prob, coverage, gt_tile), ...]

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        batch["tile_label"] = batch["tile_label"].to(device)
        batch["tile_ignore"] = batch["tile_ignore"].to(device)
        batch["coverage"] = batch["coverage"].to(device)

        outputs = model(images)
        _, loss_dict = loss_fn(outputs, batch)

        running_loss += loss_dict["total_loss"]
        running_tile_loss += loss_dict["tile_loss"]
        running_cov_loss += loss_dict["coverage_loss"]
        running_smooth_loss += loss_dict.get("smooth_loss", 0.0)

        # 从 4 个不同 batch 各取 1 张（覆盖不同脏污类型）
        if len(viz_samples) < 4 and batch_idx in [0, 10, 20, 30]:
            idx = (epoch + batch_idx) % images.shape[0]  # epoch偏移确保多样化
            viz_samples.append((
                images[idx].cpu(),
                outputs["tile_prob"][idx].cpu(),       # 模型预测概率
                outputs["coverage"][idx].cpu(),         # 覆盖率
                batch["tile_label"][idx].cpu(),          # GT tile 标签
            ))

    n = len(dataloader)
    avg_total, avg_tile = running_loss/n, running_tile_loss/n
    avg_cov, avg_smooth = running_cov_loss/n, running_smooth_loss/n

    # === 诊断 logits/prob 分布（检测 sigmoid 饱和）===
    all_logits, all_probs = [], []
    for batch in dataloader:
        out = model(batch['image'].to(device))
        all_logits.append(out['tile_logits'].cpu())
        all_probs.append(out['tile_prob'].cpu())
        if len(all_logits) >= 10:  # 采样 10 个 batch 足够
            break
    all_logits = torch.cat([x.flatten() for x in all_logits])
    all_probs = torch.cat([x.flatten() for x in all_probs])

    logger.info(f"VAL   Epoch {epoch:3d} | Total: {avg_total:.4f} | "
                f"Tile: {avg_tile:.4f} | Cov: {avg_cov:.4f} | Smooth: {avg_smooth:.4f}")
    logger.info(f"      logits: min={all_logits.min():+.3f} max={all_logits.max():+.3f} "
                f"mean={all_logits.mean():+.3f} std={all_logits.std():.3f} | "
                f"prob: min={all_probs.min():.3f} max={all_probs.max():.3f} "
                f"mean={all_probs.mean():.3f} std={all_probs.std():.3f}")

    # === 保存 GT vs Pred 对比可视化 ===
    if viz_samples:
        hd = Path(cfg.get("heatmap_dir", "outputs/heatmap"))
        hd.mkdir(parents=True, exist_ok=True)
        for si, (img, tile_prob, cov, gt_tile) in enumerate(viz_samples):
            s = f"val_epoch_{epoch:03d}_s{si}"
            gt_np, pred_np = gt_tile.numpy(), tile_prob.numpy()
            _save_image(img, str(hd / f"{s}_original.png"))
            save_heatmap_compare(img, gt_np, pred_np, str(hd / f"{s}_compare_heatmap.png"))
            save_overlay_compare(img, gt_np, pred_np, str(hd / f"{s}_compare_overlay.png"))
            save_grid_compare(img, gt_np, pred_np, str(hd / f"{s}_compare_grid.png"))
        logger.info(f"  Saved {len(viz_samples)} viz samples to {hd}")

    return avg_total, avg_tile, avg_cov, avg_smooth


# 固定验证样本索引（覆盖不同 coverage 级别）
FIXED_SAMPLE_INDICES = [15, 67, 123, 289]


@torch.no_grad()
def _save_fixed_samples(model, val_loader, device, epoch):
    """每隔 5 轮保存固定验证样本的 GT/Pred/Diff 热力图（Phase 3 诊断）。"""
    from metrics.visualize import save_heatmap, save_comparison

    fixed_dir = Path("outputs/fixed_samples")
    fixed_dir.mkdir(parents=True, exist_ok=True)

    # 收集验证集所有样本
    all_images, all_labels, all_ignore = [], [], []
    for batch in val_loader:
        all_images.append(batch['image'])
        all_labels.append(batch['tile_label'])
        all_ignore.append(batch['tile_ignore'])
    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0)

    for si, idx in enumerate(FIXED_SAMPLE_INDICES):
        if idx >= len(images): continue
        img = images[idx:idx+1].to(device)
        gt = labels[idx].numpy()

        out = model(img)
        pred = out['tile_prob'][0].cpu().numpy()

        s = f"epoch_{epoch:03d}_sample{si}_idx{idx}"
        save_heatmap(gt, str(fixed_dir/f"{s}_gt_heatmap.png"), auto_norm=False)
        save_heatmap(pred, str(fixed_dir/f"{s}_pred_heatmap.png"), auto_norm=False)
        save_comparison(images[idx], gt, pred, str(fixed_dir/f"{s}_compare.png"))
        diff = np.abs(pred - gt)
        save_heatmap(diff, str(fixed_dir/f"{s}_diff.png"), auto_norm=False)


def main():
    """主训练流程。"""
    # === 命令行参数 ===
    parser = argparse.ArgumentParser(description="Camera Soiling Detection - Training")
    parser.add_argument("--model", type=str, default="cfg/model.yaml",
                        help="模型配置文件")
    args = parser.parse_args()

    # === 加载配置 ===
    dataset_cfg = load_config("cfg/dataset.yaml")   # WoodScape gtLabels 数据
    model_cfg = load_config(args.model)              # 模型结构
    train_cfg = load_config("cfg/train.yaml")         # 训练超参数

    # 实验名自动包含 backbone 信息
    bb = model_cfg.get("backbone", {}).get("name", "unknown")
    exp_base = train_cfg.get("experiment", "unknown")
    train_cfg["experiment"] = f"{exp_base}_{bb}"

    cfg = {**dataset_cfg, **train_cfg}
    cfg["model"] = model_cfg

    # === 初始化 ===
    set_seed(cfg.get("seed", 42))
    logger = get_logger()
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # === 数据 ===
    train_loader, val_loader = build_dataloader(dataset_cfg)
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # === 模型（Backbone 加载 ImageNet 预训练权重）===
    model = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    # === 损失函数（权重从 train.yaml 读取）===
    loss_cfg = cfg.get("loss", {})
    loss_fn = TotalLoss(
        tile_weight=loss_cfg.get("tile_weight", 1.0),
        bce_weight=loss_cfg.get("bce_weight", 0.3),
        coverage_weight=loss_cfg.get("coverage_weight", 0.0),
        smooth_weight=loss_cfg.get("smooth_weight", 0.0),
        hard_neg_weight=loss_cfg.get("hard_neg_weight", 0.0),
        fp_penalty_weight=loss_cfg.get("fp_penalty_weight", 0.0),
    )

    # === 优化器 + 学习率调度 ===
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # === 混合精度 ===
    scaler = GradScaler("cuda", enabled=cfg.get("use_amp", True) and device.type == "cuda")

    # === 输出目录 ===
    ckpt_dir = Path(cfg.get("checkpoint_dir", "outputs/checkpoint"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.get("log_dir", "outputs/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # === 断点续训 ===
    start_epoch, best_val_loss = 1, float("inf")
    resume_path = ckpt_dir / "last.pth"
    if resume_path.exists():
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, resume_path)
        logger.info(f"Resumed from {resume_path}, epoch {start_epoch}, "
                     f"best_val_loss={best_val_loss:.4f}")

    # === 训练日志 CSV ===
    csv_path = log_dir / "training_log.csv"
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if start_epoch == 1:
        csv_writer.writerow(["epoch", "lr", "train_loss", "train_tile", "train_cov",
                              "train_smooth", "val_loss", "val_tile", "val_cov",
                              "val_smooth", "pred_coverage", "best_val_loss"])

    # === 训练循环 ===
    logger.info(f"Starting training for {cfg['epochs']} epochs...")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        logger.info(f"{'='*60}\nEpoch {epoch}/{cfg['epochs']}")

        # 训练
        train_loss, train_tile, train_cov, train_smooth = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, epoch, cfg, logger)

        scheduler.step()  # 更新学习率
        lr = scheduler.get_last_lr()[0]
        logger.info(f"LR: {lr:.2e} | Train: {train_loss:.4f} | Smooth: {train_smooth:.4f}")

        # 验证 + 保存可视化 + CSV 记录
        if epoch % cfg.get("val_interval", 1) == 0:
            val_loss, val_tile, val_cov, val_smooth = validate(
                model, val_loader, loss_fn, device, epoch, cfg, logger)
            csv_writer.writerow([epoch, f"{lr:.6e}",
                                  f"{train_loss:.6f}", f"{train_tile:.6f}",
                                  f"{train_cov:.6f}", f"{train_smooth:.6f}",
                                  f"{val_loss:.6f}", f"{val_tile:.6f}",
                                  f"{val_cov:.6f}", f"{val_smooth:.6f}",
                                  "", f"{best_val_loss:.6f}"])
            csv_file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                                str(ckpt_dir / "best.pth"))
                logger.info(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

        # 定期保存 checkpoint
        if epoch % cfg.get("save_interval", 5) == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                            str(ckpt_dir / f"epoch_{epoch:03d}.pth"))

        # 定期诊断（每 25 轮）
        if epoch % 25 == 0:
            diag_dir = Path("outputs/diagnostics")
            diag_dir.mkdir(parents=True, exist_ok=True)
            generate_diagnostics(model, val_loader, device, diag_dir / f"epoch_{epoch:03d}")
            logger.info(f"  Diagnostics saved to {diag_dir/f'epoch_{epoch:03d}'}")

        # 固定样本对比（每 5 轮）
        if epoch % 5 == 0:
            _save_fixed_samples(model, val_loader, device, epoch)

        # 始终保存最新（断点续训用）
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss,
                        str(ckpt_dir / "last.pth"))

    csv_file.close()

    # === 训练完成统计 ===
    best_mb = (ckpt_dir / "best.pth").stat().st_size / (1024 * 1024)
    logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")
    logger.info(f"Model params: {n_params:,}  |  Model size: {best_mb:.1f} MB")

    # === 实验归档 ===
    exp_name = cfg.get("experiment", "unknown")
    exp_dir = Path("outputs/experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 配置快照
    model_cfg_path = Path(args.model)
    for f in ["cfg/dataset.yaml", "cfg/train.yaml"]:
        src = Path(f)
        if src.exists(): shutil.copy2(src, exp_dir / f)
    if model_cfg_path.exists():
        shutil.copy2(model_cfg_path, exp_dir / model_cfg_path.name)

    # 拷贝输出
    for sub in ["logs", "heatmap", "diagnostics", "fixed_samples"]:
        sp = Path("outputs") / sub
        if sp.exists():
            shutil.copytree(sp, exp_dir / sub, dirs_exist_ok=True)

    for pth in ["best.pth", "last.pth"]:
        src = ckpt_dir / pth
        if src.exists(): shutil.copy2(src, exp_dir / pth)

    logger.info(f"Experiment archived to: {exp_dir}")


if __name__ == "__main__":
    main()
