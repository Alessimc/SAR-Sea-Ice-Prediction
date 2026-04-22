#!/usr/bin/env python3
"""
Plot divergence panels (past drift, future wind, true/pred future drift) for a trained drift model,
using DriftWindSARDataset with SAR postprocessing that matches training.

This version mirrors your quiver script but replaces quivers with divergence maps.

Divergence is computed with central differences on the INTERIOR:
  div(u,v) = du/dx + dv/dy
and we plot (H-2, W-2) maps (cropped).

Example:
python plot_divergence_sarclip.py \
  --ckpt /.../checkpoints/best_val.pt \
  --val-index /.../index_val.jsonl \
  --norm-yaml /.../norm_stats_train.yaml \
  --outdir /.../plots \
  --model-module model_dev_main.src.models.Unet \
  --model-class UNet_4layers \
  --sar-channels "HH,HV,IA" \
  --sar-to-db \
  --device auto \
  --dx 100 --dy 100
"""

import os
import argparse
import random
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import yaml
import importlib

from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset


# -----------------------------
# utilities
# -----------------------------
def import_model(module: str, class_name: str):
    m = importlib.import_module(module)
    if not hasattr(m, class_name):
        raise AttributeError(f"Module '{module}' has no class '{class_name}'")
    return getattr(m, class_name)


def load_norm_yaml(norm_yaml_path: str) -> Tuple[Dict, Dict]:
    with open(norm_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("inputs", {}), cfg.get("targets", {})


def denorm_x_channel(x: torch.Tensor, ch_name: str, ch_idx: int, inputs_stats: Dict) -> torch.Tensor:
    if ch_name not in inputs_stats:
        raise KeyError(
            f"Channel '{ch_name}' not found in norm YAML inputs stats. "
            f"Available keys: {list(inputs_stats.keys())[:20]}..."
        )
    mean = float(inputs_stats[ch_name]["mean"])
    std = float(inputs_stats[ch_name]["std"])
    return x[ch_idx] * std + mean


def denorm_y(y: torch.Tensor, targets_stats: Dict, normalize_y: bool, device: torch.device) -> torch.Tensor:
    if not normalize_y:
        return y

    for k in ("future_drift_u", "future_drift_v"):
        if k not in targets_stats:
            raise KeyError(f"Target '{k}' not found in norm YAML targets stats.")

    y_mean = torch.tensor(
        [targets_stats["future_drift_u"]["mean"], targets_stats["future_drift_v"]["mean"]],
        dtype=torch.float32,
        device=device,
    ).view(-1, 1, 1)

    y_std = torch.tensor(
        [targets_stats["future_drift_u"]["std"], targets_stats["future_drift_v"]["std"]],
        dtype=torch.float32,
        device=device,
    ).view(-1, 1, 1)

    return y * y_std + y_mean


def divergence_centered(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Central differences on the interior.
    u,v: (H,W)
    returns: (H-2, W-2)
    """
    du_dx = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)   # (H, W-2)
    dv_dy = (v[2:, :] - v[:-2, :]) / (2.0 * dy)   # (H-2, W)
    du_dx = du_dx[1:-1, :]                        # (H-2, W-2)
    dv_dy = dv_dy[:, 1:-1]                        # (H-2, W-2)
    return du_dx + dv_dy


def pick_from_ckpt_config(ckpt_cfg: Dict[str, Any], key: str, default):
    return ckpt_cfg[key] if (ckpt_cfg is not None and key in ckpt_cfg and ckpt_cfg[key] is not None) else default


# -----------------------------
# SAR clip DB bounds (HARDCODED)
# -----------------------------
SAR_CLIP_DB = True
SAR_CLIP_DB_BOUNDS: Dict[str, Tuple[float, float]] = {
    "HH": (-18.88, -7.12),
    "HV": (-34.88, -17.88),
}


# -----------------------------
# plotting
# -----------------------------
def robust_sym_limits(a: np.ndarray, q: float = 0.99, min_v: float = 1e-12) -> float:
    """
    Pick symmetric color limits from robust quantile of absolute values.
    """
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1.0
    vmax = float(np.quantile(np.abs(a), q))
    if not np.isfinite(vmax) or vmax < min_v:
        vmax = 1.0
    return vmax


def plot_div(ax, div: np.ndarray, title: str, vlim: float):
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    im = ax.imshow(div, cmap="coolwarm", norm=norm, origin="upper", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-index", required=True)
    ap.add_argument("--norm-yaml", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--model-module", required=True)
    ap.add_argument("--model-class", required=True)

    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--force-cpu", action="store_true")

    ap.add_argument("--base-channels", type=int, default=None)
    ap.add_argument("--in-ch", type=int, default=None)
    ap.add_argument("--out-ch", type=int, default=None)

    ap.add_argument("--include-wspd", action="store_true", default=None)
    ap.add_argument("--normalize-y", action="store_true", default=None)

    ap.add_argument("--sar-channels", default="HH,HV,IA", help="Comma-separated, e.g. 'HH,HV,IA'")
    ap.add_argument("--sar-to-db", action="store_true", default=None)
    ap.add_argument("--sar-postprocess", action="store_true", default=None)
    ap.add_argument("--sar-zero-is-nodata", action="store_true", default=None)

    # NEW: divergence grid spacing (physical or pixel units)
    ap.add_argument("--dx", type=float, default=1.0, help="Grid spacing in x for divergence (e.g. 100 for 100m pixels)")
    ap.add_argument("--dy", type=float, default=1.0, help="Grid spacing in y for divergence (e.g. 100 for 100m pixels)")

    # NEW: color scaling
    ap.add_argument("--robust-q", type=float, default=0.99, help="Quantile for robust symmetric color limits")
    ap.add_argument("--shared-scale", action="store_true", help="Use one shared vlim across all 4 panels")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # device
    if args.force_cpu or args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {}) or {}

    include_wspd = pick_from_ckpt_config(ckpt_cfg, "include_wspd", False)
    normalize_y = pick_from_ckpt_config(ckpt_cfg, "normalize_y", True)
    base_channels = pick_from_ckpt_config(ckpt_cfg, "base_channels", 32)
    out_ch = pick_from_ckpt_config(ckpt_cfg, "out_ch", 2)

    sar_to_db = pick_from_ckpt_config(ckpt_cfg, "sar_to_db", True)
    sar_postprocess = pick_from_ckpt_config(ckpt_cfg, "sar_postprocess", True)
    sar_zero_is_nodata = pick_from_ckpt_config(ckpt_cfg, "sar_zero_is_nodata", False)

    if args.include_wspd is not None:
        include_wspd = bool(args.include_wspd)
    if args.normalize_y is not None:
        normalize_y = bool(args.normalize_y)
    if args.base_channels is not None:
        base_channels = int(args.base_channels)
    if args.out_ch is not None:
        out_ch = int(args.out_ch)

    if args.sar_to_db is not None:
        sar_to_db = bool(args.sar_to_db)
    if args.sar_postprocess is not None:
        sar_postprocess = bool(args.sar_postprocess)
    if args.sar_zero_is_nodata is not None:
        sar_zero_is_nodata = bool(args.sar_zero_is_nodata)

    sar_channels = tuple([c.strip() for c in args.sar_channels.split(",") if c.strip() != ""])

    print(f"[run ] device={device}")
    print(f"[data] include_wspd={include_wspd} normalize_y={normalize_y}")
    print(f"[sar ] sar_channels={sar_channels} sar_to_db={sar_to_db} sar_postprocess={sar_postprocess} "
          f"zero_is_nodata={sar_zero_is_nodata} sar_clip_db={SAR_CLIP_DB} bounds={SAR_CLIP_DB_BOUNDS}")
    print(f"[div ] dx={args.dx} dy={args.dy} robust_q={args.robust_q} shared_scale={args.shared_scale}")
    print(f"[model] module={args.model_module} class={args.model_class} base_channels={base_channels} out_ch={out_ch}")

    inputs_stats, targets_stats = load_norm_yaml(args.norm_yaml)

    val_ds = DriftWindSARDataset(
        args.val_index,
        norm_yaml_path=args.norm_yaml,
        normalize_y=normalize_y,
        include_wspd=include_wspd,
        return_meta=False,
        cache_size=0,
        sar_channels=sar_channels,
        sar_to_db=sar_to_db,
        sar_postprocess=sar_postprocess,
        sar_clip_percentiles=None,
        sar_zero_is_nodata=sar_zero_is_nodata,
        sar_clip_db=SAR_CLIP_DB,
        sar_clip_db_bounds=SAR_CLIP_DB_BOUNDS,
    )

    inferred_in_ch = len(getattr(val_ds, "x_channels", [])) or val_ds[0]["x"].shape[0]
    in_ch = int(args.in_ch) if args.in_ch is not None else inferred_in_ch

    if in_ch != inferred_in_ch:
        raise ValueError(
            f"in_ch={in_ch} but dataset provides {inferred_in_ch} channels. "
            f"Check include_wspd / sar_channels / sar_to_db / dataset config."
        )

    ModelClass = import_model(args.model_module, args.model_class)
    model = ModelClass(in_channels=in_ch, out_channels=out_ch, base_channels=base_channels)

    try:
        model = model.to(device)
    except torch.cuda.OutOfMemoryError:
        if args.device == "auto" and device.type == "cuda":
            print("[warn] CUDA OOM moving model to GPU; falling back to CPU.")
            device = torch.device("cpu")
            model = model.to(device)
        else:
            raise

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    use_amp = (device.type == "cuda")

    if not hasattr(val_ds, "x_channels"):
        raise AttributeError("DriftWindSARDataset is expected to expose x_channels.")
    x_ch: List[str] = list(val_ds.x_channels)

    def idx_of(name: str) -> int:
        if name not in x_ch:
            raise KeyError(f"Expected x channel '{name}' not found. Available: {x_ch}")
        return x_ch.index(name)

    i_past_u = idx_of("past_drift_u")
    i_past_v = idx_of("past_drift_v")
    i_wind_u = idx_of("future_wind_u10_mean")
    i_wind_v = idx_of("future_wind_v10_mean")

    n = len(val_ds)
    picks = random.sample(range(n), k=min(args.num_samples, n))

    for idx in picks:
        s = val_ds[idx]
        x = s["x"].to(device).float()
        y = s["y"].to(device).float()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        with torch.inference_mode():
            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    pred = model(x.unsqueeze(0)).squeeze(0)
            else:
                pred = model(x.unsqueeze(0)).squeeze(0)

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
            reserved_mem = torch.cuda.max_memory_reserved(device) / 1024**3
            print(f"[GPU] Peak allocated: {peak_mem:.3f} GB | Peak reserved: {reserved_mem:.3f} GB")
            
        # denorm vector fields
        past_u = denorm_x_channel(x, "past_drift_u", i_past_u, inputs_stats).cpu().numpy()
        past_v = denorm_x_channel(x, "past_drift_v", i_past_v, inputs_stats).cpu().numpy()
        wind_u = denorm_x_channel(x, "future_wind_u10_mean", i_wind_u, inputs_stats).cpu().numpy()
        wind_v = denorm_x_channel(x, "future_wind_v10_mean", i_wind_v, inputs_stats).cpu().numpy()

        y_raw = denorm_y(y, targets_stats, normalize_y, device)
        pred_raw = denorm_y(pred, targets_stats, normalize_y, device)

        fut_u, fut_v = y_raw[0].cpu().numpy(), y_raw[1].cpu().numpy()
        prd_u, prd_v = pred_raw[0].cpu().numpy(), pred_raw[1].cpu().numpy()

        # divergence maps (cropped to H-2, W-2)
        div_past = divergence_centered(past_u, past_v, dx=args.dx, dy=args.dy)
        div_wind = divergence_centered(wind_u, wind_v, dx=args.dx, dy=args.dy)
        div_true = divergence_centered(fut_u, fut_v, dx=args.dx, dy=args.dy)
        div_pred = divergence_centered(prd_u, prd_v, dx=args.dx, dy=args.dy)

        # color limits
        if args.shared_scale:
            allv = np.concatenate([
                div_past.ravel(), div_true.ravel(), div_pred.ravel()
            ])
            vlim = robust_sym_limits(allv, q=args.robust_q)
            vlims = (vlim, robust_sym_limits(div_wind, q=args.robust_q), vlim, vlim)
        else:
            vlims = (
                robust_sym_limits(div_past, q=args.robust_q),
                robust_sym_limits(div_wind, q=args.robust_q),
                robust_sym_limits(div_true, q=args.robust_q),
                robust_sym_limits(div_pred, q=args.robust_q),
            )

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

        im0 = plot_div(axes[0, 0], div_past, "Divergence: Past drift", vlims[0])
        im1 = plot_div(axes[0, 1], div_wind, "Divergence: Future wind", vlims[1])
        im2 = plot_div(axes[1, 0], div_true, "Divergence: True future drift", vlims[2])
        im3 = plot_div(axes[1, 1], div_pred, "Divergence: Predicted future drift", vlims[3])

        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.02).set_label("div [units]/m (or per pixel)")
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.02).set_label("div [units]/m (or per pixel)")
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.02).set_label("div [units]/m (or per pixel)")
        fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.02).set_label("div [units]/m (or per pixel)")

        sid = s.get("id", idx)
        t = s.get("t", "")
        fig.suptitle(f"Val sample id={sid} {t}".strip(), fontsize=14)

        out_path = os.path.join(
            args.outdir,
            f"val_divergence_{sid}_{t}.png".replace(":", "").replace("/", "_")
        )
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()