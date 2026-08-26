"""rgbLabels 版训练入口。

与原始 train.py 使用相同模型，仅数据源从 gtLabels 换为 rgbLabels。
用法:
    python train_rgblabels.py                              # ResNet18
    python train_rgblabels.py --model cfg/model_mobilenetv3.yaml  # MobileNetV3
"""

import argparse, csv, shutil, sys
from pathlib import Path
import numpy as np
import torch, torch.optim as optim
from torch.amp import GradScaler, autocast

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataloader_rgblabels import build_dataloader_rgblabels
from losses.total_loss import TotalLoss
from metrics.visualize import (save_heatmap_compare, save_overlay_compare,
                               save_grid_compare)
from models.builder import build_model
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed


def _save_image(tensor, path):
    """保存 (3,H,W) tensor 为 PNG。"""
    from utils.pil_utils import save_image
    save_image(tensor, path)


def train_one_epoch(model, dataloader, loss_fn, optimizer, scaler, device, epoch, cfg, logger):
    """训练一个 epoch。"""
    model.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        for k in ["tile_label", "tile_ignore", "coverage"]:
            batch[k] = batch[k].to(device)
        optimizer.zero_grad()
        if cfg.get("use_amp", True) and device.type == "cuda":
            with autocast("cuda"):
                outputs = model(images)
                loss, loss_dict = loss_fn(outputs, batch)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            outputs = model(images)
            loss, loss_dict = loss_fn(outputs, batch)
            loss.backward(); optimizer.step()
        running_loss += loss_dict["total_loss"]
        if (batch_idx + 1) % cfg["log_interval"] == 0:
            logger.info(f"Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(dataloader):4d} | Loss: {running_loss/(batch_idx+1):.4f}")
    return running_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, epoch, cfg, logger):
    """验证 + 保存可视化（热力图/overlay/grid + GT vs Pred 对比）。"""
    model.eval()
    running_loss = 0.0
    viz_samples = []

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        for k in ["tile_label", "tile_ignore", "coverage"]:
            batch[k] = batch[k].to(device)
        outputs = model(images)
        _, loss_dict = loss_fn(outputs, batch)
        running_loss += loss_dict["total_loss"]

        # 收集4个样本用于可视化
        if len(viz_samples) < 4 and batch_idx in [0, 10, 20, 30]:
            idx = (epoch + batch_idx) % images.shape[0]
            viz_samples.append((
                images[idx].cpu(),
                outputs["tile_prob"][idx].cpu(),
                outputs["coverage"][idx].cpu(),
                batch["tile_label"][idx].cpu(),  # GT tile label
            ))

    avg = running_loss / len(dataloader)
    logger.info(f"VAL   Epoch {epoch:3d} | Loss: {avg:.4f}")

    # 保存可视化
    if viz_samples:
        hd = Path(cfg.get("heatmap_dir", "outputs/heatmap"))
        hd.mkdir(parents=True, exist_ok=True)
        for si, (img, tile_prob, coverage, gt_tile) in enumerate(viz_samples):
            s = f"val_epoch_{epoch:03d}_s{si}"
            gt_np, pred_np = gt_tile.numpy(), tile_prob.numpy()
            _save_image(img, str(hd / f"{s}_original.png"))
            save_heatmap_compare(img, gt_np, pred_np, str(hd / f"{s}_compare_heatmap.png"))
            save_overlay_compare(img, gt_np, pred_np, str(hd / f"{s}_compare_overlay.png"))
            save_grid_compare(img, gt_np, pred_np, str(hd / f"{s}_compare_grid.png"))
        logger.info(f"  Saved {len(viz_samples)} viz samples to {hd}")
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cfg/model.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    dataset_cfg = load_config("cfg/dataset_rgblabels.yaml")
    model_cfg = load_config(args.model)
    train_cfg = load_config("cfg/train.yaml")
    train_cfg["epochs"] = args.epochs
    cfg = {**dataset_cfg, **train_cfg}

    set_seed(cfg.get("seed", 42))
    logger = get_logger()
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 数据
    train_loader, val_loader = build_dataloader_rgblabels(dataset_cfg)
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # 模型
    model = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    loss_cfg = cfg.get("loss", {})
    loss_fn = TotalLoss(
        tile_weight=loss_cfg.get("tile_weight", 1.0),
        bce_weight=loss_cfg.get("bce_weight", 0.3),
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = GradScaler("cuda", enabled=cfg.get("use_amp", True))

    ckpt_dir = Path(cfg["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.get("log_dir", "outputs/logs")); log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    csv_file = open(csv_path, "a", newline=""); csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "lr", "train_loss", "val_loss", "best_val_loss"])

    start_epoch, best_val_loss = 1, float("inf")
    resume_path = ckpt_dir / "last.pth"
    if resume_path.exists():
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, resume_path)
        logger.info(f"Resumed from epoch {start_epoch}")

    logger.info(f"Training {cfg['epochs']} epochs...")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        logger.info(f"{'='*50}\nEpoch {epoch}/{cfg['epochs']}")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, epoch, cfg, logger)
        scheduler.step(); lr = scheduler.get_last_lr()[0]
        logger.info(f"LR: {lr:.2e} | Train: {train_loss:.4f}")

        if epoch % cfg.get("val_interval", 1) == 0:
            val_loss = validate(model, val_loader, loss_fn, device, epoch, cfg, logger)
            csv_writer.writerow([epoch, f"{lr:.6e}", f"{train_loss:.6f}", f"{val_loss:.6f}", f"{best_val_loss:.6f}"])
            csv_file.flush()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, str(ckpt_dir / "best.pth"))
                logger.info(f"  -> Best model ({best_val_loss:.4f})")

        if epoch % cfg.get("save_interval", 25) == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, str(ckpt_dir / f"epoch_{epoch:03d}.pth"))
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, str(ckpt_dir / "last.pth"))

    csv_file.close()
    best_mb = (ckpt_dir / "best.pth").stat().st_size / (1024*1024)
    logger.info(f"Complete! Best: {best_val_loss:.4f} | Params: {n_params:,} | Size: {best_mb:.1f} MB")

    # 归档
    bb = model_cfg.get("backbone", {}).get("name", "unknown")
    exp_dir = Path("outputs/experiments") / f"RGBLabels_{bb}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for f in ["cfg/dataset_rgblabels.yaml", "cfg/train.yaml", args.model]:
        src = Path(f)
        if src.exists(): shutil.copy2(src, exp_dir / src.name)
    if Path("outputs/logs").exists():
        shutil.copytree("outputs/logs", exp_dir / "logs", dirs_exist_ok=True)
    if Path("outputs/heatmap").exists():
        shutil.copytree("outputs/heatmap", exp_dir / "heatmap", dirs_exist_ok=True)
    if (ckpt_dir / "best.pth").exists():
        shutil.copy2(ckpt_dir / "best.pth", exp_dir / "best.pth")
    logger.info(f"Archived to: {exp_dir}")


if __name__ == "__main__":
    main()
