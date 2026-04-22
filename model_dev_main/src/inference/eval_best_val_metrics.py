#!/usr/bin/env python3
"""
Compute validation metrics for a trained drift model using the BEST saved checkpoint.

- Loads run_dir/config_used.yaml (for paths + data options)
- Loads run_dir/checkpoints/best_val.pt (or chosen ckpt)
- Uses ckpt["config"] to mirror training settings when possible
- Computes metrics in physical units (denorm) if requested:
    * rmse_u, rmse_v
    * rmse_speed, mae_speed
    * mean_abs_angle_deg (masked by true speed >= angle_eps)
    * rmse_divergence + pearson_divergence
    * rmse_divergence_top10_true + pearson_divergence_top10_true
    * skill_vs_persistence (baseline: future drift = past drift)

Divergence matches your plotting script:
- central differences on the interior ONLY
- output cropped to (H-2, W-2)

Example:
python eval_best_val_metrics.py \
  --run_dir model_dev_main/runs/EXPERIMENT_NAME \
  --ckpt best_val.pt \
  --dx 100 --dy 100 \
  --denorm
"""

import argparse
import importlib
import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset
from src.utils import init_logging

logger = init_logging()


# -------------------------
# utils
# -------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def import_model(module: str, class_name: str):
    m = importlib.import_module(module)
    if not hasattr(m, class_name):
        raise AttributeError(f"Module '{module}' has no class '{class_name}'")
    return getattr(m, class_name)


def _ensure_groups(groups: Sequence[str]) -> List[str]:
    return [g.lower() for g in groups]


def denorm_y(y_norm: torch.Tensor, ds: DriftWindSARDataset) -> torch.Tensor:
    # Only denorm if dataset has norm params AND y was normalized
    if not getattr(ds, "do_norm", False) or not getattr(ds, "normalize_y", True):
        return y_norm
    return y_norm * ds.y_std.to(y_norm.device) + ds.y_mean.to(y_norm.device)


def denorm_x(x_norm: torch.Tensor, ds: DriftWindSARDataset) -> torch.Tensor:
    if not getattr(ds, "do_norm", False):
        return x_norm
    return x_norm * ds.x_std.to(x_norm.device) + ds.x_mean.to(x_norm.device)


def divergence_centered_torch(u: torch.Tensor, v: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """
    Match your plotting function:
      du_dx = (u[:, 2:] - u[:, :-2]) / (2*dx)   -> (B,H,W-2)
      dv_dy = (v[2:, :] - v[:-2, :]) / (2*dy)   -> (B,H-2,W)
      crop both to (B,H-2,W-2) and add.
    Inputs:
      u,v: (B,H,W)
    Returns:
      div: (B,H-2,W-2)
    """
    du_dx = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dx)      # (B, H, W-2)
    dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / (2.0 * dy)      # (B, H-2, W)

    du_dx = du_dx[:, 1:-1, :]                              # (B, H-2, W-2)
    dv_dy = dv_dy[:, :, 1:-1]                              # (B, H-2, W-2)
    return du_dx + dv_dy


def pearsonr_from_sums(n: int, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float:
    if n < 2:
        return float("nan")
    num = sxy - (sx * sy / n)
    denx = sxx - (sx * sx / n)
    deny = syy - (sy * sy / n)
    denom = np.sqrt(max(denx, 1e-30) * max(deny, 1e-30))
    return float(num / denom) if denom > 0 else float("nan")


class ReservoirSampler:
    """
    Reservoir sampling for streaming quantile estimation.
    Keeps up to k samples from a stream without storing all elements.
    """
    def __init__(self, k: int, seed: int = 0):
        self.k = int(k)
        self.rng = np.random.default_rng(seed)
        self.buf = np.empty(self.k, dtype=np.float32)
        self.n_seen = 0
        self.filled = 0

    def add(self, x: np.ndarray):
        if x.size == 0:
            return
        for v in x:
            self.n_seen += 1
            if self.filled < self.k:
                self.buf[self.filled] = v
                self.filled += 1
            else:
                j = self.rng.integers(0, self.n_seen)
                if j < self.k:
                    self.buf[j] = v

    def quantile(self, q: float) -> float:
        if self.filled == 0:
            return float("nan")
        return float(np.quantile(self.buf[:self.filled], q))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Run dir containing config_used.yaml and checkpoints/")
    ap.add_argument("--ckpt", default="best_val.pt", help="Checkpoint name inside run_dir/checkpoints/")
    ap.add_argument("--dx", type=float, required=True, help="Grid spacing in x (meters). For 100m pixels use 100.")
    ap.add_argument("--dy", type=float, required=True, help="Grid spacing in y (meters). For 100m pixels use 100.")
    ap.add_argument("--denorm", action="store_true", help="Compute metrics in physical units (m/s, 1/s)")
    ap.add_argument("--angle_eps", type=float, default=0.02, help="Mask angle error where true speed < eps (m/s)")
    ap.add_argument("--batch_size", type=int, default=None, help="Override eval batch size")
    ap.add_argument("--num_workers", type=int, default=None, help="Override eval num_workers")

    # top-10% |div_true| metrics
    ap.add_argument("--topq", type=float, default=0.90, help="Quantile for |div_true| threshold (0.90 => top 10%)")
    ap.add_argument("--div_sample_per_batch", type=int, default=50000,
                    help="How many |div_true| pixels to sample per batch for quantile estimation")
    ap.add_argument("--reservoir_size", type=int, default=2_000_000,
                    help="Reservoir size for quantile estimation (memory ~ 4*size bytes)")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    run_dir = args.run_dir
    cfg_path = os.path.join(run_dir, "config_used.yaml")
    ckpt_path = os.path.join(run_dir, "checkpoints", args.ckpt)

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    cfg = load_yaml(cfg_path)
    paths = cfg["paths"]
    data_cfg = cfg.get("data", {})
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"

    # -------------------------
    # Load checkpoint (your format)
    # -------------------------
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {}) or {}

    # prefer ckpt config for settings that affect dataset/model reconstruction
    include_wspd = bool(ckpt_cfg.get("include_wspd", data_cfg.get("include_wspd", False)))
    normalize_y = bool(ckpt_cfg.get("normalize_y", train_cfg.get("normalize_y", True)))

    base_channels = int(ckpt_cfg.get("base_channels", model_cfg.get("base_channels", 16)))
    out_ch = int(ckpt_cfg.get("out_ch", model_cfg.get("out_ch", 2)))

    sar_to_db = bool(ckpt_cfg.get("sar_to_db", data_cfg.get("sar_to_db", True)))
    sar_postprocess = bool(ckpt_cfg.get("sar_postprocess", data_cfg.get("sar_postprocess", True)))
    sar_zero_is_nodata = bool(ckpt_cfg.get("sar_zero_is_nodata", data_cfg.get("sar_zero_is_nodata", False)))

    # These are in config_used.yaml under data most likely:
    sar_channels = tuple(data_cfg.get("sar_channels", ["HV"]))
    x_groups = data_cfg.get("x_groups", None)

    sar_clip_percentiles = data_cfg.get("sar_clip_percentiles", [0.01, 0.99])
    sar_clip_percentiles = tuple(sar_clip_percentiles) if sar_clip_percentiles is not None else None

    sar_clip_db = bool(data_cfg.get("sar_clip_db", False))
    sar_clip_db_bounds = {str(k).upper(): tuple(v) for k, v in data_cfg.get("sar_clip_db_bounds", {}).items()}

    cache_size_val = int(data_cfg.get("cache_size_val", 0))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 4))

    # -------------------------
    # Build val dataset for model (exact)
    # -------------------------
    val_ds_model = DriftWindSARDataset(
        paths["val_index"],
        norm_yaml_path=paths["norm_yaml"],
        normalize_y=normalize_y,
        include_wspd=include_wspd,
        return_meta=False,
        cache_size=cache_size_val,
        x_groups=x_groups,

        sar_channels=sar_channels,
        sar_to_db=sar_to_db,
        sar_postprocess=sar_postprocess,
        sar_clip_percentiles=sar_clip_percentiles,
        sar_zero_is_nodata=sar_zero_is_nodata,

        sar_clip_db=sar_clip_db,
        sar_clip_db_bounds=sar_clip_db_bounds,
    )

    # -------------------------
    # Build val dataset for persistence baseline (must include drift)
    # -------------------------
    groups = x_groups if x_groups is not None else ["drift", "wind", "sar"]
    groups = _ensure_groups(groups)
    if "drift" not in groups:
        groups = ["drift"] + groups

    val_ds_pers = DriftWindSARDataset(
        paths["val_index"],
        norm_yaml_path=paths["norm_yaml"],
        normalize_y=normalize_y,
        include_wspd=include_wspd,
        return_meta=False,
        cache_size=0,
        x_groups=groups,

        sar_channels=sar_channels,
        sar_to_db=sar_to_db,
        sar_postprocess=sar_postprocess,
        sar_clip_percentiles=sar_clip_percentiles,
        sar_zero_is_nodata=sar_zero_is_nodata,

        sar_clip_db=sar_clip_db,
        sar_clip_db_bounds=sar_clip_db_bounds,
    )

    # loaders
    bs = args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 4))
    nw = args.num_workers if args.num_workers is not None else int(train_cfg.get("num_workers", 4))

    loader_model = DataLoader(
        val_ds_model,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
        prefetch_factor=prefetch_factor if nw > 0 else None,
    )
    loader_pers = DataLoader(
        val_ds_pers,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
        prefetch_factor=prefetch_factor if nw > 0 else None,
    )

    # -------------------------
    # Build model
    # -------------------------
    inferred_in_ch = len(getattr(val_ds_model, "x_channels", [])) or val_ds_model[0]["x"].shape[0]
    in_ch = int(model_cfg.get("in_ch", inferred_in_ch)) if model_cfg.get("in_ch", None) is not None else inferred_in_ch

    ModelClass = import_model(model_cfg["module"], model_cfg["class_name"])
    model = ModelClass(
        in_channels=in_ch,
        out_channels=out_ch,
        base_channels=base_channels,
    ).to(device)

    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model_state'. Keys: {list(ckpt.keys())}")

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    use_amp = (device.type == "cuda")

    # ==========================================================
    # PASS 1: estimate threshold for global top-10% |div_true|
    # ==========================================================
    logger.info("Pass 1/2: estimating global |div_true| quantile threshold via reservoir sampling...")
    sampler = ReservoirSampler(k=args.reservoir_size, seed=args.seed)

    for batch in loader_model:
        y = batch["y"].to(device, non_blocking=True)
        if args.denorm:
            y = denorm_y(y, val_ds_model)

        u_t, v_t = y[:, 0], y[:, 1]
        div_t = divergence_centered_torch(u_t, v_t, dx=args.dx, dy=args.dy)  # (B,H-2,W-2)
        abs_div = torch.abs(div_t).reshape(-1)

        n = abs_div.numel()
        k = min(args.div_sample_per_batch, n)
        if k <= 0:
            continue

        idx = torch.randint(0, n, (k,), device=abs_div.device)
        samp = abs_div[idx].detach().float().cpu().numpy()
        samp = samp[np.isfinite(samp)]
        sampler.add(samp)

    thr = sampler.quantile(args.topq)
    if not np.isfinite(thr):
        raise RuntimeError("Could not estimate divergence threshold (no samples).")
    logger.info(f"Estimated global |div_true| quantile q={args.topq:.2f} threshold: {thr:.6e} [1/s]")

    # ==========================================================
    # PASS 2: compute metrics (global + top10 subset)
    # ==========================================================
    logger.info("Pass 2/2: computing validation metrics...")

    sum_sq_u = sum_sq_v = 0.0
    sum_sq_speed = 0.0
    sum_abs_speed = 0.0

    sum_ang = 0.0
    ang_count = 0.0

    # divergence metrics (global; computed on cropped grid)
    sum_sq_div = 0.0
    n_div_pix = 0

    # Pearson accumulators (global, exact from sums) for speed and divergence
    sp_n = 0
    sp_sx = sp_sy = sp_sxx = sp_syy = sp_sxy = 0.0

    dv_n = 0
    dv_sx = dv_sy = dv_sxx = dv_syy = dv_sxy = 0.0

    # persistence baseline MSE accum (u,v)
    sum_sq_u_pers = sum_sq_v_pers = 0.0

    # top10 divergence RMSE + Pearson (exact from sums)
    top_n = 0
    top_sum_sq_div = 0.0
    top_sx = top_sy = top_sxx = top_syy = top_sxy = 0.0

    for batch_m, batch_p in zip(loader_model, loader_pers):
        x = batch_m["x"].to(device, non_blocking=True)
        y = batch_m["y"].to(device, non_blocking=True)

        # model pred
        if use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                yhat = model(x)
        else:
            yhat = model(x)

        # persistence x (must include drift in first 2 channels of drift group)
        x_p = batch_p["x"].to(device, non_blocking=True)

        if args.denorm:
            y = denorm_y(y, val_ds_model)
            yhat = denorm_y(yhat, val_ds_model)
            x_p = denorm_x(x_p, val_ds_pers)

        u_t, v_t = y[:, 0], y[:, 1]
        u_p, v_p = yhat[:, 0], yhat[:, 1]

        # persistence prediction = past drift u,v
        u_pers = x_p[:, 0]
        v_pers = x_p[:, 1]

        # pixels for vector/speed/angle metrics
        B, H, W = u_t.shape
        n = B * H * W

        du = u_p - u_t
        dv = v_p - v_t
        sum_sq_u += float((du ** 2).sum().item())
        sum_sq_v += float((dv ** 2).sum().item())

        du0 = u_pers - u_t
        dv0 = v_pers - v_t
        sum_sq_u_pers += float((du0 ** 2).sum().item())
        sum_sq_v_pers += float((dv0 ** 2).sum().item())

        sp_t = torch.sqrt(u_t ** 2 + v_t ** 2)
        sp_p = torch.sqrt(u_p ** 2 + v_p ** 2)
        dsp = sp_p - sp_t
        sum_sq_speed += float((dsp ** 2).sum().item())
        sum_abs_speed += float(torch.abs(dsp).sum().item())

        # angle error, masked by true speed
        m_ang = (sp_t >= args.angle_eps)
        ang_t = torch.atan2(v_t, u_t)
        ang_p = torch.atan2(v_p, u_p)
        dtheta = (ang_p - ang_t + np.pi) % (2 * np.pi) - np.pi
        ang_abs = torch.abs(dtheta) * (180.0 / np.pi)
        sum_ang += float(ang_abs[m_ang].sum().item())
        ang_count += float(m_ang.sum().item())

        # Pearson on speed (exact sums)
        # x = pred, y = true
        sp_pred = sp_p.reshape(-1).double()
        sp_true = sp_t.reshape(-1).double()
        sp_n += sp_pred.numel()
        sp_sx  += float(sp_pred.sum().item())
        sp_sy  += float(sp_true.sum().item())
        sp_sxx += float((sp_pred * sp_pred).sum().item())
        sp_syy += float((sp_true * sp_true).sum().item())
        sp_sxy += float((sp_pred * sp_true).sum().item())

        # divergence on cropped grid
        div_t = divergence_centered_torch(u_t, v_t, dx=args.dx, dy=args.dy)   # (B,H-2,W-2)
        div_p = divergence_centered_torch(u_p, v_p, dx=args.dx, dy=args.dy)

        ddiv = div_p - div_t
        sum_sq_div += float((ddiv ** 2).sum().item())
        n_div_pix += ddiv.numel()

        # Pearson on divergence (exact sums)
        dv_pred = div_p.reshape(-1).double()
        dv_true = div_t.reshape(-1).double()
        dv_n += dv_pred.numel()
        dv_sx  += float(dv_pred.sum().item())
        dv_sy  += float(dv_true.sum().item())
        dv_sxx += float((dv_pred * dv_pred).sum().item())
        dv_syy += float((dv_true * dv_true).sum().item())
        dv_sxy += float((dv_pred * dv_true).sum().item())

        # top10 subset based on TRUE divergence magnitude
        sel = (torch.abs(div_t) >= thr)
        if sel.any():
            dd = ddiv[sel]
            top_sum_sq_div += float((dd ** 2).sum().item())
            c = int(sel.sum().item())
            top_n += c

            xt = div_t[sel].reshape(-1).double()
            xp = div_p[sel].reshape(-1).double()
            top_sx  += float(xp.sum().item())
            top_sy  += float(xt.sum().item())
            top_sxx += float((xp * xp).sum().item())
            top_syy += float((xt * xt).sum().item())
            top_sxy += float((xp * xt).sum().item())

        # keep n_pix for vector/speed metrics
        # (use python int to avoid overflow in long runs)
        # n_pix stored after loop by summing n each iteration
        # We'll just accumulate here:
        # (Note: store as python int outside; simplest is to compute from sum_sq_u etc,
        # but n_pix is needed for RMSE.)
        # We'll keep a separate accumulator:
        if "n_pix_accum" not in locals():
            n_pix_accum = 0
        n_pix_accum += n

    # n_pix for vector/speed
    n_pix = int(n_pix_accum)

    # aggregate metrics
    rmse_u = float(np.sqrt(sum_sq_u / max(n_pix, 1)))
    rmse_v = float(np.sqrt(sum_sq_v / max(n_pix, 1)))
    rmse_speed = float(np.sqrt(sum_sq_speed / max(n_pix, 1)))
    mae_speed = float(sum_abs_speed / max(n_pix, 1))
    mean_ang = float(sum_ang / max(ang_count, 1.0))

    rmse_div = float(np.sqrt(sum_sq_div / max(n_div_pix, 1)))
    pearson_speed = pearsonr_from_sums(sp_n, sp_sx, sp_sy, sp_sxx, sp_syy, sp_sxy)
    pearson_div = pearsonr_from_sums(dv_n, dv_sx, dv_sy, dv_sxx, dv_syy, dv_sxy)

    # skill vs persistence (vector MSE averaged over u,v)
    mse_model = float((sum_sq_u + sum_sq_v) / max(n_pix * 2.0, 1.0))
    mse_pers = float((sum_sq_u_pers + sum_sq_v_pers) / max(n_pix * 2.0, 1.0))
    skill_pers = float(1.0 - (mse_model / max(mse_pers, 1e-12)))

    # top10 divergence metrics
    rmse_div_top10 = float(np.sqrt(top_sum_sq_div / max(top_n, 1)))
    pearson_div_top10 = pearsonr_from_sums(top_n, top_sx, top_sy, top_sxx, top_syy, top_sxy)

    out = {
        "run_dir": run_dir,
        "checkpoint": ckpt_path,
        "checkpoint_epoch": ckpt.get("epoch", None),
        "checkpoint_val_loss": ckpt.get("val_loss", None),
        "denorm": bool(args.denorm),
        "dx": float(args.dx),
        "dy": float(args.dy),
        "angle_eps": float(args.angle_eps),

        "div_true_abs_quantile": float(args.topq),
        "div_true_abs_threshold_est": float(thr),
        "n_pixels_vector": int(n_pix),
        "n_pixels_divergence": int(n_div_pix),
        "top_subset_pixels": int(top_n),
        "top_subset_fraction": float(top_n / max(n_div_pix, 1)),

        "metrics": {
            "rmse_u": rmse_u,
            "rmse_v": rmse_v,
            "rmse_speed": rmse_speed,
            "mae_speed": mae_speed,
            "mean_abs_angle_deg": mean_ang,

            "rmse_divergence": rmse_div,
            "pearson_speed": float(pearson_speed),
            "pearson_divergence": float(pearson_div),

            "mse_vector_model": mse_model,
            "mse_vector_persistence": mse_pers,
            "skill_vs_persistence": skill_pers,

            "rmse_divergence_top10_true": rmse_div_top10,
            "pearson_divergence_top10_true": float(pearson_div_top10),
        },
    }

    out_path = os.path.join(run_dir, "val_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"Wrote {out_path}")
    logger.info(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()
