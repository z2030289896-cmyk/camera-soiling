"""Self 数据集微调训练。

基于预训练权重在 Self 数据集上微调。支持 train_split=1.0 全数据训练。

用法:
    python train_self.py --epochs 20
    python train_self.py --pretrained outputs/experiments/xxx/best.pth
"""

import argparse, csv, shutil, sys
from pathlib import Path
import torch, torch.optim as optim
from torch.amp import GradScaler, autocast

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataloader_self import build_dataloader_self
from losses.total_loss import TotalLoss
from metrics.visualize import (save_heatmap_compare, save_overlay_compare,
                               save_grid_compare)
from models.builder import build_model
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.config import load_config
from utils.logger import get_logger
from utils.seed import set_seed


def _save_image(tensor, path):
    from torchvision.utils import save_image
    save_image(tensor, path)


def train_one_epoch(model, dataloader, loss_fn, optimizer, scaler, device, epoch, cfg, logger):
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
        if (batch_idx + 1) % max(1, cfg.get("log_interval", 5)) == 0:
            logger.info(f"Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(dataloader):4d} | Loss: {running_loss/(batch_idx+1):.4f}")
    return running_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, epoch, cfg, logger):
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
        if not viz_samples and batch_idx == 0:
            idx = 0
            viz_samples.append((
                images[idx].cpu(), outputs["tile_prob"][idx].cpu(),
                outputs["coverage"][idx].cpu(), batch["tile_label"][idx].cpu(),
            ))
    avg = running_loss / len(dataloader)
    logger.info(f"VAL   Epoch {epoch:3d} | Loss: {avg:.4f}")
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
        logger.info(f"  Saved viz to {hd}")
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pretrained", type=str,
                        default="outputs/experiments/Self_original/best.pth")
    args = parser.parse_args()

    dataset_cfg = load_config("cfg/dataset_self.yaml")
    model_cfg = load_config("cfg/model_mobilenetv3.yaml")
    train_cfg = load_config("cfg/train.yaml")
    train_cfg["epochs"] = args.epochs
    cfg = {**dataset_cfg, **train_cfg}

    set_seed(cfg.get("seed", 42))
    logger = get_logger()
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_loader, val_loader = build_dataloader_self(dataset_cfg)
    has_val = val_loader is not None and len(val_loader.dataset) > 0
    if has_val:
        logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    else:
        logger.info(f"Train: {len(train_loader.dataset)} (全部训练，无验证集)")

    # 加载预训练模型
    model = build_model(model_cfg).to(device)
    pretrained_path = Path(args.pretrained)
    if pretrained_path.exists():
        ckpt = torch.load(pretrained_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        logger.info(f"Loaded pretrained: {pretrained_path}")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    loss_fn = TotalLoss(tile_weight=1.0, bce_weight=0.3)
    # 微调用更小的学习率
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"] * 0.5,
                           weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = GradScaler("cuda", enabled=cfg.get("use_amp", True))

    ckpt_dir = Path(cfg["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.get("log_dir", "outputs/logs")); log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    csv_file = open(csv_path, "a", newline=""); csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "lr", "train_loss", "val_loss", "best_val_loss"])

    start_epoch, best_val_loss = 1, float("inf")
    logger.info(f"Fine-tuning {cfg['epochs']} epochs (lr={cfg['learning_rate']*0.5:.1e})...")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        logger.info(f"{'='*50}\nEpoch {epoch}/{cfg['epochs']}")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device, epoch, cfg, logger)
        scheduler.step(); lr = scheduler.get_last_lr()[0]
        logger.info(f"LR: {lr:.2e} | Train: {train_loss:.4f}")
        if has_val and epoch % cfg.get("val_interval", 1) == 0:
            val_loss = validate(model, val_loader, loss_fn, device, epoch, cfg, logger)
            csv_writer.writerow([epoch, f"{lr:.6e}", f"{train_loss:.6f}", f"{val_loss:.6f}", f"{best_val_loss:.6f}"])
            csv_file.flush()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, str(ckpt_dir / "best.pth"))
                logger.info(f"  -> Best model ({best_val_loss:.4f})")
        elif not has_val:
            best_val_loss = train_loss
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, str(ckpt_dir / "best.pth"))
            csv_writer.writerow([epoch, f"{lr:.6e}", f"{train_loss:.6f}", "N/A", f"{best_val_loss:.6f}"])
            csv_file.flush()
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, str(ckpt_dir / "last.pth"))

    csv_file.close()
    best_mb = (ckpt_dir / "best.pth").stat().st_size / (1024*1024)
    logger.info(f"Complete! Best: {best_val_loss:.4f} | Params: {n_params:,} | Size: {best_mb:.1f} MB")

    # 归档（优先用 dataset 配置的 experiment 名，避免被 train.yaml 覆盖）
    exp_name = dataset_cfg.get("experiment", "Self_finetuned")
    exp_dir = Path("outputs/experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    for f in ["cfg/dataset_self.yaml", "cfg/train.yaml", "cfg/model_mobilenetv3.yaml"]:
        src = Path(f)
        if src.exists(): shutil.copy2(src, exp_dir / src.name)
    if Path("outputs/logs").exists():
        shutil.copytree("outputs/logs", exp_dir / "logs", dirs_exist_ok=True)
    if Path("outputs/heatmap").exists():
        shutil.copytree("outputs/heatmap", exp_dir / "heatmap", dirs_exist_ok=True)
    if (ckpt_dir / "best.pth").exists():
        shutil.copy2(ckpt_dir / "best.pth", exp_dir / "best.pth")
    logger.info(f"Archived to: {exp_dir}")

    # 自动评估（仅当有验证集时）
    if has_val:
        logger.info("Running evaluation...")
        from metrics.evaluation import evaluate_model
        eval_dir = Path("outputs/results/metrics") / exp_name
        evaluate_model(model, val_loader, device, output_dir=str(eval_dir))


if __name__ == "__main__":
    main()
