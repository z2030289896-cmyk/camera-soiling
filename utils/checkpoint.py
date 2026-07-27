import torch


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path):
    """保存训练 checkpoint（含模型、优化器、调度器状态）。

    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器（可为 None）
        epoch: 当前 epoch 编号
        best_val_loss: 当前最佳验证 loss
        path: 保存路径
    """
    save_dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()

    torch.save(save_dict, path)


def load_checkpoint(model, optimizer, scheduler, path):
    """加载训练 checkpoint。

    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器（可为 None）
        path: checkpoint 路径

    Returns:
        (start_epoch, best_val_loss)
        start_epoch: 下一个训练的 epoch 编号
        best_val_loss: 历史最佳验证 loss
    """
    ckpt = torch.load(path, weights_only=False)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

    return start_epoch, best_val_loss
