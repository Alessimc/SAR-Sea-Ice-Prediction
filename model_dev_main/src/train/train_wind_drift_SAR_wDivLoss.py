#!/usr/bin/env python3
import argparse
import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset

from model_dev_main.src.train.train_utils import (
    ensure_dir,
    save_checkpoint,
    append_loss_line,
    init_logging,
)

logger = init_logging()

# ---------------------------------------------------------------------
# Hardcoded masking behavior (as requested)
# ---------------------------------------------------------------------
DIV_MASK_Q = 0.90          # keep top 10% by |div(target)|  (q=0.9)
DIV_MIN_MASK_FRAC = 0.001  # if mask gets too small, fall back to full image
DIV_MIN_FINITE = 16        # if too few finite pixels, fall back


# -----------------------------
# MSE + masked divergence loss utilities
# -----------------------------
def _make_central_diff_kernels(dx: float, dy: float, device: torch.device, dtype: torch.dtype):
    kx = torch.tensor([[-1.0, 0.0, 1.0]], device=device, dtype=dtype) / (2.0 * dx)
    ky = torch.tensor([[-1.0], [0.0], [1.0]], device=device, dtype=dtype) / (2.0 * dy)
    kx = kx.view(1, 1, 1, 3)
    ky = ky.view(1, 1, 3, 1)
    return kx, ky


def divergence_2d(u: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    """
    Compute divergence for a 2D vector field u = (u_x, u_y).
    u: (B, 2, H, W)
    returns: (B, 1, H, W)  (same spatial size via padding)
    """
    if u.ndim != 4 or u.size(1) < 2:
        raise ValueError(f"Expected u shape (B,2,H,W) (or more channels). Got {tuple(u.shape)}")

    ux = u[:, 0:1, :, :]
    uy = u[:, 1:2, :, :]

    device, dtype = u.device, u.dtype
    kx, ky = _make_central_diff_kernels(dx, dy, device, dtype)

    ux_p = F.pad(ux, (1, 1, 0, 0), mode="replicate")
    uy_p = F.pad(uy, (0, 0, 1, 1), mode="replicate")

    dux_dx = F.conv2d(ux_p, kx)  # (B,1,H,W)
    duy_dy = F.conv2d(uy_p, ky)  # (B,1,H,W)
    return dux_dx + duy_dy


class MSEPlusMaskedDivergenceLoss(nn.Module):
    """
    L = MSE(pred, target) + lam * masked_L1(div(pred), div(target))

    Masking (hardcoded):
      - Per-sample mask keeps pixels where |div(target)| is in the top 10% (quantile q=0.9)
      - Quantile computed in float32 on FINITE values to avoid AMP + NaNs
      - If mask too small, falls back to full-image divergence L1 for that sample
    """
    def __init__(self, lam: float = 0.0, dx: float = 1.0, dy: float = 1.0):
        super().__init__()
        self.lam = float(lam)
        self.dx = float(dx)
        self.dy = float(dy)
        self.mse = nn.MSELoss()

    def _masked_l1(self, div_p: torch.Tensor, div_t: torch.Tensor) -> torch.Tensor:
        """
        div_p, div_t: (B,1,H,W)
        returns: scalar tensor (mean over batch)
        """
        diff = torch.abs(div_p - div_t)  # (B,1,H,W)
        mag = torch.abs(div_t)           # (B,1,H,W)

        B = diff.size(0)
        losses = []

        for i in range(B):
            mag_i = mag[i].reshape(-1)
            diff_i = diff[i].reshape(-1)

            # Force float32 for quantile (AMP may give fp16/bf16)
            mag_i_f32 = mag_i.to(torch.float32)

            finite = torch.isfinite(mag_i_f32)
            if int(finite.sum().item()) < DIV_MIN_FINITE:
                # Too many NaNs/Infs, just use unmasked for stability
                losses.append(diff_i.mean())
                continue

            # quantile on finite values only
            thr = torch.quantile(mag_i_f32[finite], DIV_MASK_Q)

            # mask on full vector (but finite only)
            mask = (mag_i_f32 >= thr) & finite
            mask_frac = mask.float().mean().item()

            if mask_frac < DIV_MIN_MASK_FRAC:
                losses.append(diff_i.mean())
            else:
                losses.append(diff_i[mask].mean())

        return torch.stack(losses).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, return_components: bool = False):
        loss_mse = self.mse(pred, target)

        if self.lam <= 0.0:
            if return_components:
                z = torch.zeros((), device=pred.device, dtype=pred.dtype)
                return loss_mse, loss_mse.detach(), z.detach()
            return loss_mse

        div_p = divergence_2d(pred, dx=self.dx, dy=self.dy)
        div_t = divergence_2d(target, dx=self.dx, dy=self.dy)

        loss_div = self._masked_l1(div_p, div_t)
        total = loss_mse + self.lam * loss_div

        if return_components:
            return total, loss_mse.detach(), loss_div.detach()
        return total


@torch.no_grad()
def evaluate_with_components(
    model: nn.Module,
    loader: DataLoader,
    criterion: MSEPlusMaskedDivergenceLoss,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, float]:
    """
    Returns:
      total_loss_mean, mse_mean, div_mean  (div is UNWEIGHTED masked L1(div mismatch))
    """
    model.eval()
    total_sum = 0.0
    mse_sum = 0.0
    div_sum = 0.0
    n = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        with torch.autocast(device.type, enabled=use_amp):
            pred = model(x)
            total, mse_term, div_term = criterion(pred, y, return_components=True)

        bs = x.size(0)
        total_sum += float(total.item()) * bs
        mse_sum += float(mse_term.item()) * bs
        div_sum += float(div_term.item()) * bs
        n += bs

    if n == 0:
        return float("nan"), float("nan"), float("nan")
    return total_sum / n, mse_sum / n, div_sum / n


# -----------------------------
# Config / helpers
# -----------------------------
@dataclass
class TrainConfig:
    experiment_name: str
    runs_root: str
    run_dir: str
    ckpt_dir: str
    plots_dir: str
    loss_log: str

    train_index: str
    val_index: str
    norm_yaml: str

    include_wspd: bool
    sar_channels: List[str]
    sar_to_db: bool
    cache_size_train: int
    cache_size_val: int
    prefetch_factor: int
    x_groups: Optional[List[str]]

    sar_postprocess: bool
    sar_clip_percentiles: Optional[Tuple[float, float]]
    sar_zero_is_nodata: bool

    sar_clip_db: bool
    sar_clip_db_bounds: Dict[str, Tuple[float, float]]

    model_module: str
    model_class: str
    base_channels: int
    in_ch: Optional[int]
    out_ch: int

    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int
    amp: bool
    seed: int
    normalize_y: bool

    scheduler: str
    scheduler_factor: float
    scheduler_patience: int
    scheduler_min_lr: float
    scheduler_tmax: int

    patience: int
    min_delta: float

    divergence_lambda: float
    divergence_dx: float
    divergence_dy: float


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
    runs_root = d.get("runs_root", "model_dev_main/runs")

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
    x_groups = data.get("x_groups", None)

    sar_postprocess = bool(data.get("sar_postprocess", True))
    sar_clip_percentiles = _parse_clip_percentiles(data.get("sar_clip_percentiles", [0.01, 0.99]))
    sar_zero_is_nodata = bool(data.get("sar_zero_is_nodata", False))

    sar_clip_db = bool(data.get("sar_clip_db", False))
    sar_clip_db_bounds = data.get("sar_clip_db_bounds", {})
    sar_clip_db_bounds = {str(k).upper(): tuple(v) for k, v in sar_clip_db_bounds.items()}

    divergence_lambda = float(train.get("divergence_lambda", 0.0))
    divergence_dx = float(train.get("divergence_dx", 1.0))
    divergence_dy = float(train.get("divergence_dy", 1.0))

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
        x_groups=[str(g) for g in x_groups] if x_groups is not None else None,

        sar_postprocess=sar_postprocess,
        sar_clip_percentiles=sar_clip_percentiles,
        sar_zero_is_nodata=sar_zero_is_nodata,

        sar_clip_db=sar_clip_db,
        sar_clip_db_bounds=sar_clip_db_bounds,

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

        scheduler=str(sched),
        scheduler_factor=float(train.get("scheduler_factor", 0.5)),
        scheduler_patience=int(train.get("scheduler_patience", 5)),
        scheduler_min_lr=float(train.get("scheduler_min_lr", 1e-6)),
        scheduler_tmax=int(train.get("scheduler_tmax", int(train.get("epochs", 50)))),

        patience=int(es.get("patience", 15)),
        min_delta=float(es.get("min_delta", 0.0)),

        divergence_lambda=divergence_lambda,
        divergence_dx=divergence_dx,
        divergence_dy=divergence_dy,
    )


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

    if hasattr(train_ds, "x_channels"):
        logger.info(f"Input channels ({len(train_ds.x_channels)}): {train_ds.x_channels}")
    if hasattr(train_ds, "y_channels"):
        logger.info(f"Target channels ({len(train_ds.y_channels)}): {train_ds.y_channels}")

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

    ModelClass = import_model(cfg.model_module, cfg.model_class)
    model = ModelClass(
        in_channels=in_ch,
        out_channels=cfg.out_ch,
        base_channels=cfg.base_channels,
    ).to(device)
    logger.info(f"Model:\n{model}")

    if cfg.out_ch < 2 and cfg.divergence_lambda > 0:
        raise ValueError("divergence loss needs out_ch>=2 (u,v). Set divergence_lambda=0 or use out_ch>=2.")

    criterion = MSEPlusMaskedDivergenceLoss(
        lam=cfg.divergence_lambda,
        dx=cfg.divergence_dx,
        dy=cfg.divergence_dy,
    )

    logger.info(
        f"Using loss: total = MSE + λ * masked_L1(div mismatch) | "
        f"λ={cfg.divergence_lambda} dx={cfg.divergence_dx} dy={cfg.divergence_dy} "
        f"mask=top{int((1.0-DIV_MASK_Q)*100)}% |div_gt| (q={DIV_MASK_Q})"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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

    best_val = float("inf")
    best_train = float("inf")
    patience_left = cfg.patience

    logger.info("Starting training...")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_total_sum = 0.0
        n_train = 0

        first_batch_logged = False
        first_batch_mse = None
        first_batch_div = None

        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device.type, enabled=use_amp):
                pred = model(x)
                total, mse_term, div_term = criterion(pred, y, return_components=True)

            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            train_total_sum += float(total.item()) * bs
            n_train += bs

            if not first_batch_logged:
                first_batch_logged = True
                first_batch_mse = float(mse_term.item())
                first_batch_div = float(div_term.item())

        train_loss = train_total_sum / max(1, n_train)

        val_loss, val_mse, val_div = evaluate_with_components(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train={train_loss:.6e} | val={val_loss:.6e} | "
            f"lr={current_lr:.3e} | patience_left={patience_left}"
        )

        if cfg.divergence_lambda > 0 and first_batch_mse is not None and first_batch_div is not None:
            wdiv = cfg.divergence_lambda * first_batch_div
            total_fb = first_batch_mse + wdiv
            div_pct = 100.0 * wdiv / total_fb if total_fb > 0 else 0.0
            mse_pct = 100.0 - div_pct
            logger.info(
                "  Train loss breakdown (first batch): "
                f"MSE={first_batch_mse:.4e} | Div(masked)={first_batch_div:.4e} | "
                f"λ*Div={wdiv:.4e} | %MSE={mse_pct:.1f}% | %Div={div_pct:.1f}%"
            )

        if cfg.divergence_lambda > 0:
            wdiv_v = cfg.divergence_lambda * val_div
            total_v = val_mse + wdiv_v
            div_pct_v = 100.0 * wdiv_v / total_v if total_v > 0 else 0.0
            mse_pct_v = 100.0 - div_pct_v
            logger.info(
                "  Val loss breakdown (mean): "
                f"MSE={val_mse:.4e} | Div(masked)={val_div:.4e} | "
                f"λ*Div={wdiv_v:.4e} | %MSE={mse_pct_v:.1f}% | %Div={div_pct_v:.1f}%"
            )

        append_loss_line(cfg.loss_log, epoch, train_loss, val_loss)

        save_checkpoint(
            path=os.path.join(cfg.ckpt_dir, "last.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            config_dataclass=cfg,
        )

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