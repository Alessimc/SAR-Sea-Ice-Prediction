import os
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import logging

from src.utils import init_logging

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config_dataclass,
):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": asdict(config_dataclass),
    }
    torch.save(ckpt, path)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        pred = model(x)
        loss = criterion(pred, y)

        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    use_amp: bool,
) -> float:
    model.train()
    total = 0.0
    n = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        total += float(loss.item()) * x.size(0)
        n += x.size(0)

    return total / max(n, 1)


def append_loss_line(loss_log_path: str, epoch: int, train_loss: float, val_loss: float):
    file_exists = os.path.exists(loss_log_path)
    with open(loss_log_path, "a") as f:
        if not file_exists:
            f.write("epoch,train_loss,val_loss\n")
        f.write(f"{epoch},{train_loss:.8e},{val_loss:.8e}\n")
