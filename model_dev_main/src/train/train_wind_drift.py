#!/usr/bin/env python3
import argparse
import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_dev_main.src.dataloader.DriftWindDataset import DriftWindDataset
from model_dev_main.src.train.train_utils import (
    ensure_dir,
    save_checkpoint,
    train_one_epoch,
    evaluate,
    append_loss_line,
    init_logging,
)

logger = init_logging()


@dataclass
class TrainConfig:
    # Run output
    experiment_name: str
    runs_root: str
    run_dir: str
    ckpt_dir: str
    plots_dir: str
    loss_log: str

    # Data
    train_index: str
    val_index: str
    norm_yaml: str

    # Model
    model_module: str
    model_class: str
    base_channels: int
    include_wspd: bool
    in_ch: int
    out_ch: int

    # Training
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int
    amp: bool
    seed: int
    normalize_y: bool

    # Early stopping
    patience: int
    min_delta: float


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_model(module: str, class_name: str):
    m = importlib.import_module(module)
    if not hasattr(m, class_name):
        raise AttributeError(
            f"Module '{module}' has no class '{class_name}'"
        )
    return getattr(m, class_name)



def load_yaml(path: str) -> Dict[str, Any]:
    # Prefer PyYAML; if not installed, you can swap to json easily.
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config_used(src_config_path: str, run_dir: str):
    dst = os.path.join(run_dir, "config_used.yaml")
    shutil.copy2(src_config_path, dst)
    return dst


def build_cfg(d: Dict[str, Any]) -> TrainConfig:
    experiment_name = d["experiment_name"]
    runs_root = d.get(
        "runs_root",
        "model_dev_main/runs"
    )

    run_dir = os.path.join(runs_root, experiment_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    plots_dir = os.path.join(run_dir, "plots")
    loss_log = os.path.join(run_dir, "loss.csv")

    paths = d["paths"]
    model = d["model"]
    train = d["train"]
    es = d.get("early_stopping", {})

    model_module = model["module"]
    model_class = model["class_name"]

    include_wspd = bool(model.get("include_wspd", False))
    in_ch = int(model.get("in_ch", (5 if include_wspd else 4)))
    out_ch = int(model.get("out_ch", 2))
    base_channels = int(model.get("base_channels", 32))

    return TrainConfig(
        # Run output
        experiment_name=experiment_name,
        runs_root=runs_root,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        plots_dir=plots_dir,
        loss_log=loss_log,

        # Data
        train_index=paths["train_index"],
        val_index=paths["val_index"],
        norm_yaml=paths["norm_yaml"],

        # Model
        model_module=model_module,
        model_class=model_class,
        base_channels=base_channels,
        include_wspd=include_wspd,
        in_ch=in_ch,
        out_ch=out_ch,

        # Training
        epochs=int(train.get("epochs", 50)),
        batch_size=int(train.get("batch_size", 4)),
        lr=float(train.get("lr", 1e-4)),
        weight_decay=float(train.get("weight_decay", 0.0)),
        num_workers=int(train.get("num_workers", 4)),
        amp=bool(train.get("amp", True)),
        seed=int(train.get("seed", 0)),
        normalize_y=bool(train.get("normalize_y", True)),

        # Early stopping
        patience=int(es.get("patience", 15)),
        min_delta=float(es.get("min_delta", 0.0)),
    )



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config file")
    args = p.parse_args()

    d = load_yaml(args.config)
    cfg = build_cfg(d)

    # Create run folder structure
    ensure_dir(cfg.run_dir)
    ensure_dir(cfg.ckpt_dir)
    ensure_dir(cfg.plots_dir)

    config_used_path = save_config_used(args.config, cfg.run_dir)
    logger.info(f"Saved config_used.yaml to: {config_used_path}")


    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and cfg.amp

    logger.info(f"Config file: {args.config}")
    logger.info(f"Run dir: {cfg.run_dir}")
    logger.info(f"Using device: {device}")
    logger.info(f"Using AMP: {use_amp}")

    # Load data
    CACHE_SIZE_TRAIN = 32   # maybe go bac to 16
    CACHE_SIZE_VAL   = 0    

    PREFETCH_FACTOR = 4

    train_ds = DriftWindDataset(
        cfg.train_index,
        norm_yaml_path=cfg.norm_yaml,
        normalize_y=cfg.normalize_y,
        include_wspd=cfg.include_wspd,
        return_meta=False,
        cache_size=CACHE_SIZE_TRAIN,
    )

    val_ds = DriftWindDataset(
        cfg.val_index,
        norm_yaml_path=cfg.norm_yaml, 
        normalize_y=cfg.normalize_y,
        include_wspd=cfg.include_wspd,
        return_meta=False,
        cache_size=CACHE_SIZE_VAL,
    )

    use_workers = cfg.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_workers,
        prefetch_factor=PREFETCH_FACTOR if use_workers else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_workers,
        prefetch_factor=PREFETCH_FACTOR if use_workers else None,
    )


    # Model
    ModelClass = import_model(cfg.model_module, cfg.model_class)
    model = ModelClass(
        in_channels=cfg.in_ch,
        out_channels=cfg.out_ch,
        base_channels=cfg.base_channels,
    ).to(device)
    logger.info(f"Model:\n{model}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_train = float("inf")
    patience_left = cfg.patience

    logger.info("Starting training...")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss = evaluate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs} | train={train_loss:.6e} | val={val_loss:.6e} | patience_left={patience_left}"
        )
        append_loss_line(cfg.loss_log, epoch, train_loss, val_loss)

        # save last
        save_checkpoint(
            path=os.path.join(cfg.ckpt_dir, "last.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            config_dataclass=cfg,
        )

        # save best train
        if train_loss < best_train:
            best_train = train_loss
            save_checkpoint(
                path=os.path.join(cfg.ckpt_dir, "best_train.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                config_dataclass=cfg,
            )

        # save best val + early stopping
        improved = (best_val - val_loss) > cfg.min_delta
        if improved:
            best_val = val_loss
            patience_left = cfg.patience
            save_checkpoint(
                path=os.path.join(cfg.ckpt_dir, "best_val.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                config_dataclass=cfg,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. Best val={best_val:.6e} "
                    f"(min_delta={cfg.min_delta}, patience={cfg.patience})."
                )
                break

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
