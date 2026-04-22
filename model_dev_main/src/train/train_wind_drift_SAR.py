#!/usr/bin/env python3
import argparse
import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset

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
    x_groups: Optional[List[str]]

    # SAR-only postprocessing options
    sar_postprocess: bool
    sar_clip_percentiles: Optional[Tuple[float, float]]  # quantiles in [0,1], e.g. (0.01, 0.99)
    sar_zero_is_nodata: bool

    sar_clip_db: bool
    sar_clip_db_bounds: Dict[str, Tuple[float, float]]

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

    scheduler: str          # "none" | "plateau" | "cosine"
    scheduler_factor: float
    scheduler_patience: int
    scheduler_min_lr: float
    scheduler_tmax: int     # for cosine

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
        # if user gave percent scale, convert
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
    sched = train.get("scheduler", "none")
    es = d.get("early_stopping", {})
    data = d.get("data", {})

    # Model
    model_module = model["module"]
    model_class = model["class_name"]
    base_channels = int(model.get("base_channels", 32))
    out_ch = int(model.get("out_ch", 2))

    # Optional: allow user to specify in_ch, but prefer inference from dataset
    in_ch = model.get("in_ch", None)
    in_ch = int(in_ch) if in_ch is not None else None

    # Data options
    include_wspd = bool(data.get("include_wspd", model.get("include_wspd", False)))
    sar_channels = data.get("sar_channels", ["HV"])
    sar_to_db = bool(data.get("sar_to_db", True))

    cache_size_train = int(data.get("cache_size_train", 32))
    cache_size_val = int(data.get("cache_size_val", 0))
    prefetch_factor = int(data.get("prefetch_factor", 4))
    x_groups = data.get("x_groups", None)

    # NEW: SAR postprocess options (defaults chosen to be safe)
    sar_postprocess = bool(data.get("sar_postprocess", True))
    sar_clip_percentiles = _parse_clip_percentiles(data.get("sar_clip_percentiles", [0.01, 0.99]))
    sar_zero_is_nodata = bool(data.get("sar_zero_is_nodata", False))

    sar_clip_db = bool(data.get("sar_clip_db", False))
    sar_clip_db_bounds = data.get("sar_clip_db_bounds", {})
    # normalize keys to "HH"/"HV"
    sar_clip_db_bounds = {str(k).upper(): tuple(v) for k, v in sar_clip_db_bounds.items()}

    return TrainConfig(
        # Run output
        experiment_name=experiment_name,
        runs_root=runs_root,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        plots_dir=plots_dir,
        loss_log=loss_log,

        # Paths
        train_index=paths["train_index"],
        val_index=paths["val_index"],
        norm_yaml=paths["norm_yaml"],

        # Data options
        include_wspd=include_wspd,
        sar_channels=[str(c) for c in sar_channels],
        sar_to_db=sar_to_db,
        cache_size_train=cache_size_train,
        cache_size_val=cache_size_val,
        prefetch_factor=prefetch_factor,
        x_groups=[str(g) for g in x_groups] if x_groups is not None else None,

        # SAR-only postprocess options
        sar_postprocess=sar_postprocess,
        sar_clip_percentiles=sar_clip_percentiles,
        sar_zero_is_nodata=sar_zero_is_nodata,

        sar_clip_db=sar_clip_db,
        sar_clip_db_bounds=sar_clip_db_bounds,

        # Model
        model_module=model_module,
        model_class=model_class,
        base_channels=base_channels,
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

        scheduler=str(sched),
        scheduler_factor=float(train.get("scheduler_factor", 0.5)),
        scheduler_patience=int(train.get("scheduler_patience", 5)),
        scheduler_min_lr=float(train.get("scheduler_min_lr", 1e-6)),
        scheduler_tmax=int(train.get("scheduler_tmax", int(train.get("epochs", 50)))),

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

    # --------------------
    # Load datasets
    # --------------------
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
        sar_clip_percentiles=cfg.sar_clip_percentiles,  # tuple or None
        sar_zero_is_nodata=cfg.sar_zero_is_nodata,
        sar_clip_db=cfg.sar_clip_db,
        sar_clip_db_bounds=cfg.sar_clip_db_bounds,
        x_groups=cfg.x_groups,
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
        sar_clip_db=cfg.sar_clip_db,
        sar_clip_db_bounds=cfg.sar_clip_db_bounds,
        x_groups=cfg.x_groups,
    )

    # Infer in_ch from dataset if not provided (or verify if provided)
    inferred_in_ch = len(getattr(train_ds, "x_channels", [])) or train_ds[0]["x"].shape[0]
    if cfg.in_ch is None:
        in_ch = inferred_in_ch
        logger.info(f"Inferred in_channels={in_ch} from dataset.")
    else:
        in_ch = cfg.in_ch
        if in_ch != inferred_in_ch:
            raise ValueError(
                f"Config in_ch={in_ch} but dataset provides {inferred_in_ch} channels. "
                f"Check data.include_wspd / data.sar_channels / sar_to_db naming and YAML stats."
            )

    # Log channel names (super useful for reproducibility)
    if hasattr(train_ds, "x_channels"):
        logger.info(f"Input channels ({len(train_ds.x_channels)}): {train_ds.x_channels}")
    if hasattr(train_ds, "y_channels"):
        logger.info(f"Target channels ({len(train_ds.y_channels)}): {train_ds.y_channels}")
    # Log SAR postprocess details
    logger.info(
        f"SAR postprocess: {cfg.sar_postprocess} | "
        f"clip_percentiles={cfg.sar_clip_percentiles} | "
        f"zero_is_nodata={cfg.sar_zero_is_nodata}"
    )
    if hasattr(train_ds, "_sar_hh_hv_x_idx"):
        logger.info(f"SAR HH/HV channel indices in x: {train_ds._sar_hh_hv_x_idx}")

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

    # --------------------
    # Model
    # --------------------
    ModelClass = import_model(cfg.model_module, cfg.model_class)
    model = ModelClass(
        in_channels=in_ch,
        out_channels=cfg.out_ch,
        base_channels=cfg.base_channels,
    ).to(device)
    logger.info(f"Model:\n{model}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    #############
    scheduler = None
    if cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.scheduler_min_lr,
        )
    elif cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.scheduler_tmax,
            eta_min=cfg.scheduler_min_lr,
        )
    elif cfg.scheduler in (None, "none", ""):
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler '{cfg.scheduler}'")
    #############

    best_val = float("inf")
    best_train = float("inf")
    patience_left = cfg.patience

    logger.info("Starting training...")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss = evaluate(model, val_loader, criterion, device)
        ###########
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        ##########
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train={train_loss:.6e} | val={val_loss:.6e} | "
            f"lr={current_lr:.3e} | patience_left={patience_left}"
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
