import os
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    accuracy,
    path,
):
    # 如果路径中包含目录且目录不存在，则自动创建
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),
        "accuracy": accuracy,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    model,
    optimizer,
    path,
    device,
):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    epoch = checkpoint.get("epoch", 0)
    accuracy = checkpoint.get("accuracy", 0.0)

    return epoch, accuracy
