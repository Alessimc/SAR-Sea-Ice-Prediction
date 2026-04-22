#!/usr/bin/env python3
"""
Train script: residual prediction hardcoded.

Model predicts DELTA drift:
    delta = model(x)
and final prediction is:
    y_hat = past_drift + delta

We train on residual target:
    delta_target = y_future - past_drift

We validate / early-stop on FULL prediction MSE:
    MSE(y_hat, y_future)

Uses the SAME YAML config schema as your non-residual script.
No extra config keys required.

Assumptions:
- x contains past drift as channels 0 and 1: ["past_drift_u", "past_drift_v", ...]
- y is future drift with 2 channels (u,v)
- normalization is consistent: past drift in x is in the same space as y if normalize_y=True
"""

import argparse
import importlib
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset
from model_dev_main.src.train.train_utils import (
    ensure_dir,
    save_checkpoint,
    append_loss_line,
    init_logging,
)

logger = init_logging()


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    # Run output
    experiment_name: str
    runs_root: str
    run_dir: str
    ckpt_dir: str
    plots_dir: str
    loss_log: str

    # Paths
    train_index: str
    val_index: str
    norm_yaml: str

    # Data options
    include_wspd: bool
    sar_channels: List[str]
    sar_to_db: bool
    cache_size_train: int
    cache_size_val: int
    prefetch_factor: int

    # SAR postprocess options
    sar_postprocess: bool
    sar_clip_percentiles: Optional[Tuple[float, float]]  # quantiles in [0,1], e.g. (0.01, 0.99)
    sar_zero_is_nodata: bool

    # Model
    model_module: str
    model_class: str
    base_channels: int
    in_ch: Optional[int]  # allow None => infer from dataset
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
        raise AttributeError(f"Module '{module}' has no class '{class_name}'")
    return getattr(m, class_name)


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config_used(src_config_path: str, run_dir: str):
    dst = os.path.join(run_dir, "config_used.yaml")
    shutil.copy2(src_config_path, dst)
    return dst


def _parse_clip_percentiles(v: Any) -> Optional[Tuple[float, float]]:
    """
    Accepts:
      - None / missing => None
      - [0.01, 0.99] or (0.01, 0.99)
      - [1, 99] (percent form) => will be converted to (0.01, 0.99)
    Returns tuple(q_lo, q_hi) with 0 <= q_lo < q_hi <= 1
    """
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) == 2:
        a, b = float(v[0]), float(v[1])
        if a > 1.0 or b > 1.0:
            a /= 100.0
            b /= 100.0
        if not (0.0 <= a < b <= 1.0):
            raise ValueError(f"sar_clip_percentiles must be within [0,1] and lo<hi. Got: {v}")
        return (a, b)
    raise ValueError(f"sar_clip_percentiles must be a 2-list/tuple or null. Got: {type(v)} {v}")


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
    data = d.get("data", {})

    model_module = model["module"]
    model_class = model["class_name"]
    base_channels = int(model.get("base_channels", 32))
    out_ch = int(model.get("out_ch", 2))

    in_ch = model.get("in_ch", None)
    in_ch = int(in_ch) if in_ch is not None else None

    include_wspd = bool(data.get("include_wspd", model.get("include_wspd", False)))
    sar_channels = data.get("sar_channels", ["HV"])
    sar_to_db = bool(data.get("sar_to_db", True))

    cache_size_train = int(data.get("cache_size_train", 32))
    cache_size_val = int(data.get("cache_size_val", 0))
    prefetch_factor = int(data.get("prefetch_factor", 4))

    sar_postprocess = bool(data.get("sar_postprocess", True))
    sar_clip_percentiles = _parse_clip_percentiles(data.get("sar_clip_percentiles", [0.01, 0.99]))
    sar_zero_is_nodata = bool(data.get("sar_zero_is_nodata", False))

    return TrainConfig(
        experiment_name=experiment_name,
        runs_root=runs_root,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        plots_dir=plots_dir,
        loss_log=loss_log,

        train_index=paths["train_index"],
        val_index=paths["val_index"],
        norm_yaml=paths["norm_yaml"],

        include_wspd=include_wspd,
        sar_channels=[str(c) for c in sar_channels],
        sar_to_db=sar_to_db,
        cache_size_train=cache_size_train,
        cache_size_val=cache_size_val,
        prefetch_factor=prefetch_factor,

        sar_postprocess=sar_postprocess,
        sar_clip_percentiles=sar_clip_percentiles,
        sar_zero_is_nodata=sar_zero_is_nodata,

        model_module=model_module,
        model_class=model_class,
        base_channels=base_channels,
        in_ch=in_ch,
        out_ch=out_ch,

        epochs=int(train.get("epochs", 50)),
        batch_size=int(train.get("batch_size", 4)),
        lr=float(train.get("lr", 1e-4)),
        weight_decay=float(train.get("weight_decay", 0.0)),
        num_workers=int(train.get("num_workers", 4)),
        amp=bool(train.get("amp", True)),
        seed=int(train.get("seed", 0)),
        normalize_y=bool(train.get("normalize_y", True)),

        patience=int(es.get("patience", 15)),
        min_delta=float(es.get("min_delta", 0.0)),
    )


# -------------------------
# Residual train / eval
# -------------------------
def _get_past_drift_from_x(x: torch.Tensor, u_idx: int = 0, v_idx: int = 1) -> torch.Tensor:
    # x: (B, C, H, W) -> past: (B, 2, H, W)
    return x[:, [u_idx, v_idx], :, :]


def train_one_epoch_residual(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    past_u_idx: int = 0,
    past_v_idx: int = 1,
) -> float:
    model.train()
    running = 0.0
    n = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)  # future drift (B,2,H,W)

        past = _get_past_drift_from_x(x, past_u_idx, past_v_idx)
        delta_target = y - past

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            delta_pred = model(x)
            loss = criterion(delta_pred, delta_target)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = x.shape[0]
        running += loss.item() * bs
        n += bs

    return running / max(1, n)


@torch.no_grad()
def evaluate_residual_full_mse(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    past_u_idx: int = 0,
    past_v_idx: int = 1,
) -> float:
    """
    Validate using FULL prediction MSE:
        pred_full = past + model(x)
        loss = MSE(pred_full, y)
    """
    model.eval()
    running = 0.0
    n = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        past = _get_past_drift_from_x(x, past_u_idx, past_v_idx)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            delta_pred = model(x)
            pred_full = past + delta_pred
            loss = criterion(pred_full, y)

        bs = x.shape[0]
        running += loss.item() * bs
        n += bs

    return running / max(1, n)


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config file")
    args = p.parse_args()

    d = load_yaml(args.config)
    cfg = build_cfg(d)

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
    logger.info("RESIDUAL MODE: model predicts delta; val is FULL MSE on past+delta.")

    # Datasets
    train_ds = DriftWindSARDataset(
        cfg.train_index,
        norm_yaml_path=cfg.norm_yaml,
        normalize_y=cfg.normalize_y,
        include_wspd=cfg.include_wspd,
        return_meta=False,
        cache_size=cfg.cache_size_train,
        sar_channels=tuple(cfg.sar_channels),
        sar_to_db=cfg.sar_to_db,
        sar_postprocess=cfg.sar_postprocess,
        sar_clip_percentiles=cfg.sar_clip_percentiles,
        sar_zero_is_nodata=cfg.sar_zero_is_nodata,
    )

    val_ds = DriftWindSARDataset(
        cfg.val_index,
        norm_yaml_path=cfg.norm_yaml,
        normalize_y=cfg.normalize_y,
        include_wspd=cfg.include_wspd,
        return_meta=False,
        cache_size=cfg.cache_size_val,
        sar_channels=tuple(cfg.sar_channels),
        sar_to_db=cfg.sar_to_db,
        sar_postprocess=cfg.sar_postprocess,
        sar_clip_percentiles=cfg.sar_clip_percentiles,
        sar_zero_is_nodata=cfg.sar_zero_is_nodata,
    )

    inferred_in_ch = len(getattr(train_ds, "x_channels", [])) or train_ds[0]["x"].shape[0]
    if cfg.in_ch is None:
        in_ch = inferred_in_ch
        logger.info(f"Inferred in_channels={in_ch} from dataset.")
    else:
        in_ch = cfg.in_ch
        if in_ch != inferred_in_ch:
            raise ValueError(
                f"Config in_ch={in_ch} but dataset provides {inferred_in_ch} channels. "
                f"Check include_wspd / sar_channels / sar_to_db and norm stats."
            )

    if hasattr(train_ds, "x_channels"):
        logger.info(f"Input channels ({len(train_ds.x_channels)}): {train_ds.x_channels}")
    if hasattr(train_ds, "y_channels"):
        logger.info(f"Target channels ({len(train_ds.y_channels)}): {train_ds.y_channels}")

    # Sanity: ensure past drift is at indices 0,1
    if hasattr(train_ds, "x_channels"):
        if len(train_ds.x_channels) >= 2:
            if train_ds.x_channels[0] != "past_drift_u" or train_ds.x_channels[1] != "past_drift_v":
                logger.warning(
                    "Expected past drift channels at x[0]=past_drift_u and x[1]=past_drift_v, "
                    f"but got: x[0]={train_ds.x_channels[0]}, x[1]={train_ds.x_channels[1]}. "
                    "Residual training will still use indices [0,1] unless you edit the script."
                )

    use_workers = cfg.num_workers > 0
    pin = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=use_workers,
        prefetch_factor=cfg.prefetch_factor if use_workers else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=use_workers,
        prefetch_factor=cfg.prefetch_factor if use_workers else None,
    )

    # Model
    ModelClass = import_model(cfg.model_module, cfg.model_class)
    model = ModelClass(
        in_channels=in_ch,
        out_channels=cfg.out_ch,
        base_channels=cfg.base_channels,
    ).to(device)
    logger.info(f"Model:\n{model}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")     # FULL MSE best
    best_train = float("inf")   # RESIDUAL MSE best
    patience_left = cfg.patience

    logger.info("Starting training...")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch_residual(
            model, train_loader, criterion, optimizer, device, scaler, use_amp,
            past_u_idx=0, past_v_idx=1
        )
        val_loss = evaluate_residual_full_mse(
            model, val_loader, criterion, device, use_amp,
            past_u_idx=0, past_v_idx=1
        )

        # Early stopping uses FULL val loss (comparable across runs)
        improved = (best_val - val_loss) > cfg.min_delta
        if improved:
            best_val = val_loss
            patience_left = cfg.patience
        else:
            patience_left -= 1

        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs} | train_residual_mse={train_loss:.6e} | "
            f"val_full_mse={val_loss:.6e} | best_val_full_mse={best_val:.6e} | "
            f"patience_left={patience_left}"
        )

        # Log to loss.csv (keep same 3 columns for compatibility)
        append_loss_line(cfg.loss_log, epoch, train_loss, val_loss)

        # Save last
        save_checkpoint(
            path=os.path.join(cfg.ckpt_dir, "last.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,  # residual MSE
            val_loss=val_loss,      # full MSE
            config_dataclass=cfg,
        )

        # Best train (residual)
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

        # Best val (full)
        if improved:
            save_checkpoint(
                path=os.path.join(cfg.ckpt_dir, "best_val.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                config_dataclass=cfg,
            )

        if patience_left <= 0:
            logger.info(
                f"Early stopping triggered at epoch {epoch}. Best val(full)={best_val:.6e} "
                f"(min_delta={cfg.min_delta}, patience={cfg.patience})."
            )
            break

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
