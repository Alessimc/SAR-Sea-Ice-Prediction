#!/usr/bin/env python3
"""
Plot quiver panels (past drift, future wind, true/pred future drift) for a trained drift model,
using DriftWindSARDataset with SAR postprocessing that matches training.

This version:
- Uses SAR clip-in-dB bounds (HH/HV) and sar_zero_is_nodata, and DOES NOT use percentile clipping.
- Lets you choose SAR channels via --sar-channels (e.g. "HH,HV,IA") to match your experiment.
- Pulls most defaults from the checkpoint's saved config if available, but you can override on CLI.

Example (your SAR-all run):
python plot_quiver_sarclip.py \
  --ckpt /.../checkpoints/best_val.pt \
  --val-index /.../index_val.jsonl \
  --norm-yaml /.../norm_stats_train.yaml \
  --outdir /.../plots \
  --model-module model_dev_main.src.models.Unet \
  --model-class UNet_4layers \
  --sar-channels "HH,HV,IA" \
  --sar-to-db \
  --device auto
"""

import os
import argparse
import random
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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


def vector_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u * u + v * v)


def denorm_x_channel(
    x: torch.Tensor,
    ch_name: str,
    ch_idx: int,
    inputs_stats: Dict,
) -> torch.Tensor:
    """
    Denormalize a single x channel: x[ch_idx] * std + mean
    x is (C,H,W) normalized
    returns (H,W) in original units
    """
    if ch_name not in inputs_stats:
        raise KeyError(
            f"Channel '{ch_name}' not found in norm YAML inputs stats. "
            f"Available keys: {list(inputs_stats.keys())[:20]}..."
        )
    mean = float(inputs_stats[ch_name]["mean"])
    std = float(inputs_stats[ch_name]["std"])
    return x[ch_idx] * std + mean


def denorm_y(
    y: torch.Tensor,
    targets_stats: Dict,
    normalize_y: bool,
    device: torch.device,
) -> torch.Tensor:
    """
    y: (2,H,W) normalized OR raw depending on normalize_y
    returns y_raw (2,H,W)
    """
    if not normalize_y:
        return y

    for k in ("future_drift_u", "future_drift_v"):
        if k not in targets_stats:
            raise KeyError(
                f"Target '{k}' not found in norm YAML targets stats. "
                f"Available keys: {list(targets_stats.keys())[:20]}..."
            )

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


def denorm_pred(
    pred: torch.Tensor,
    targets_stats: Dict,
    normalize_y: bool,
    device: torch.device,
) -> torch.Tensor:
    return denorm_y(pred, targets_stats, normalize_y, device)


def plot_mag_quiver(
    ax,
    u: np.ndarray,
    v: np.ndarray,
    mag: np.ndarray,
    title: str,
    norm: Normalize,
    step: int,
    quiver_scale=None,
    quiver_width: float = 0.004,
):
    """
    Data is already in image coordinates => NO flipping.
    Returns AxesImage for correct colorbar scaling.
    """
    H, W = u.shape

    im = ax.imshow(
        mag,
        cmap="viridis",
        norm=norm,
        origin="upper",
        interpolation="nearest",
    )

    yy, xx = np.mgrid[0:H:step, 0:W:step]
    uu = u[0:H:step, 0:W:step]
    vv = v[0:H:step, 0:W:step]

    ax.quiver(
        xx, yy, uu, vv,
        angles="xy",
        scale_units="xy",
        scale=quiver_scale,
        color="k",
        width=quiver_width,
        headwidth=4.5,
        headlength=6.0,
        headaxislength=5.0,
        pivot="mid",
    )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)

    return im


def pick_from_ckpt_config(ckpt_cfg: Dict[str, Any], key: str, default):
    return ckpt_cfg[key] if (ckpt_cfg is not None and key in ckpt_cfg and ckpt_cfg[key] is not None) else default


# -----------------------------
# SAR clip DB bounds (HARDCODED)
# -----------------------------
# These match your training config:
#   HH: [-18.88, -7.12]
#   HV: [-34.88, -17.88]
# IA is typically left unclipped here (dataset may handle it separately).
SAR_CLIP_DB = True
SAR_CLIP_DB_BOUNDS: Dict[str, Tuple[float, float]] = {
    "HH": (-18.88, -7.12),
    "HV": (-34.88, -17.88),
}


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

    # Plot controls
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=32)

    # Device controls
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--force-cpu", action="store_true")

    # Model args (optional overrides)
    ap.add_argument("--base-channels", type=int, default=None)
    ap.add_argument("--in-ch", type=int, default=None)
    ap.add_argument("--out-ch", type=int, default=None)

    # Data args (optional overrides)
    ap.add_argument("--include-wspd", action="store_true", default=None)
    ap.add_argument("--normalize-y", action="store_true", default=None)

    # SAR dataset args
    ap.add_argument("--sar-channels", default="HH,HV,IA", help="Comma-separated, e.g. 'HH,HV,IA'")
    ap.add_argument("--sar-to-db", action="store_true", default=None)

    ap.add_argument("--sar-postprocess", action="store_true", default=None)
    ap.add_argument("--sar-zero-is-nodata", action="store_true", default=None)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # device selection
    if args.force_cpu or args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load ckpt + config (if present)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {}) or {}

    # infer flags from ckpt when possible; allow CLI overrides
    include_wspd = pick_from_ckpt_config(ckpt_cfg, "include_wspd", False)
    normalize_y = pick_from_ckpt_config(ckpt_cfg, "normalize_y", True)
    base_channels = pick_from_ckpt_config(ckpt_cfg, "base_channels", 32)
    out_ch = pick_from_ckpt_config(ckpt_cfg, "out_ch", 2)

    # SAR defaults (match typical training defaults; clip DB is hardcoded above)
    sar_to_db = pick_from_ckpt_config(ckpt_cfg, "sar_to_db", True)
    sar_postprocess = pick_from_ckpt_config(ckpt_cfg, "sar_postprocess", True)
    sar_zero_is_nodata = pick_from_ckpt_config(ckpt_cfg, "sar_zero_is_nodata", False)

    # CLI overrides (only if user provided)
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

    # parse SAR channels
    sar_channels = tuple([c.strip() for c in args.sar_channels.split(",") if c.strip() != ""])

    print(f"[run ] device={device}")
    print(f"[data] include_wspd={include_wspd} normalize_y={normalize_y}")
    print(f"[sar ] sar_channels={sar_channels} sar_to_db={sar_to_db} sar_postprocess={sar_postprocess} "
          f"zero_is_nodata={sar_zero_is_nodata} sar_clip_db={SAR_CLIP_DB} bounds={SAR_CLIP_DB_BOUNDS}")
    print(f"[model] module={args.model_module} class={args.model_class} base_channels={base_channels} out_ch={out_ch}")

    # Load normalization stats
    inputs_stats, targets_stats = load_norm_yaml(args.norm_yaml)

    # Dataset (NOTE: percentile clipping is DISABLED / not passed)
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
        sar_clip_percentiles=None,  # <- explicitly disabled
        sar_zero_is_nodata=sar_zero_is_nodata,
        sar_clip_db=SAR_CLIP_DB,
        sar_clip_db_bounds=SAR_CLIP_DB_BOUNDS,
    )

    # Infer in_ch from dataset unless user forces it
    inferred_in_ch = len(getattr(val_ds, "x_channels", [])) or val_ds[0]["x"].shape[0]
    in_ch = int(args.in_ch) if args.in_ch is not None else inferred_in_ch

    if in_ch != inferred_in_ch:
        raise ValueError(
            f"in_ch={in_ch} but dataset provides {inferred_in_ch} channels. "
            f"Check include_wspd / sar_channels / sar_to_db / dataset config."
        )

    # model
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

    # channel index lookup (robust to SAR channel ordering)
    if not hasattr(val_ds, "x_channels"):
        raise AttributeError("DriftWindSARDataset is expected to expose x_channels. Add it or index by known order.")
    x_ch: List[str] = list(val_ds.x_channels)

    def idx_of(name: str) -> int:
        if name not in x_ch:
            raise KeyError(f"Expected x channel '{name}' not found. Available: {x_ch}")
        return x_ch.index(name)

    # These must exist irrespective of SAR
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

        with torch.inference_mode():
            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    pred = model(x.unsqueeze(0)).squeeze(0)
            else:
                pred = model(x.unsqueeze(0)).squeeze(0)

        # denorm only the channels we need for vector fields
        past_u = denorm_x_channel(x, "past_drift_u", i_past_u, inputs_stats).cpu().numpy()
        past_v = denorm_x_channel(x, "past_drift_v", i_past_v, inputs_stats).cpu().numpy()
        wind_u = denorm_x_channel(x, "future_wind_u10_mean", i_wind_u, inputs_stats).cpu().numpy()
        wind_v = denorm_x_channel(x, "future_wind_v10_mean", i_wind_v, inputs_stats).cpu().numpy()

        y_raw = denorm_y(y, targets_stats, normalize_y, device)
        pred_raw = denorm_pred(pred, targets_stats, normalize_y, device)

        fut_u, fut_v = y_raw[0].cpu().numpy(), y_raw[1].cpu().numpy()
        prd_u, prd_v = pred_raw[0].cpu().numpy(), pred_raw[1].cpu().numpy()

        past_mag = vector_magnitude(past_u, past_v)
        wind_mag = vector_magnitude(wind_u, wind_v)
        fut_mag = vector_magnitude(fut_u, fut_v)
        prd_mag = vector_magnitude(prd_u, prd_v)

        # Shared drift range WITHIN this figure (true/pred)
        future_drift_vmax = max(
            float(np.nanmax(fut_mag)),
            float(np.nanmax(prd_mag)),
        )
        if not np.isfinite(future_drift_vmax) or future_drift_vmax <= 0:
            future_drift_vmax = 1.0
        future_drift_norm = Normalize(vmin=0.0, vmax=future_drift_vmax)

        past_vmax = float(np.nanmax(past_mag))
        if not np.isfinite(past_vmax) or past_vmax <= 0:
            past_vmax = 1.0
        past_drift_norm = Normalize(vmin=0.0, vmax=past_vmax)

        wind_vmax = float(np.nanmax(wind_mag))
        if not np.isfinite(wind_vmax) or wind_vmax <= 0:
            wind_vmax = 1.0
        wind_norm = Normalize(vmin=0.0, vmax=wind_vmax)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

        im_past = plot_mag_quiver(axes[0, 0], past_u, past_v, past_mag, "Past drift (m/s)", past_drift_norm, args.step)
        im_wind = plot_mag_quiver(axes[0, 1], wind_u, wind_v, wind_mag, "Future wind (m/s)", wind_norm, args.step)
        im_fut = plot_mag_quiver(axes[1, 0], fut_u, fut_v, fut_mag, "True future drift (m/s)", future_drift_norm, args.step)
        im_prd = plot_mag_quiver(axes[1, 1], prd_u, prd_v, prd_mag, "Predicted future drift (m/s)", future_drift_norm, args.step)

        fig.colorbar(im_past, ax=axes[0, 0], fraction=0.046, pad=0.02).set_label("Drift speed (m/s)")
        fig.colorbar(im_wind, ax=axes[0, 1], fraction=0.046, pad=0.02).set_label("Wind speed (m/s)")
        fig.colorbar(im_fut, ax=axes[1, 0], fraction=0.046, pad=0.02).set_label("Drift speed (m/s)")
        fig.colorbar(im_prd, ax=axes[1, 1], fraction=0.046, pad=0.02).set_label("Drift speed (m/s)")

        sid = s.get("id", idx)
        t = s.get("t", "")
        fig.suptitle(f"Val sample id={sid} {t}".strip(), fontsize=14)

        out_path = os.path.join(
            args.outdir,
            f"val_quiver_sarclip_{sid}_{t}.png".replace(":", "").replace("/", "_")
        )
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
