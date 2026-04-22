#!/usr/bin/env python3
import os
import argparse
import random
from typing import Dict, Tuple, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import yaml
import importlib

from model_dev_main.src.dataloader.DriftWindDataset import DriftWindDataset


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


def denorm_xy(
    x: torch.Tensor,
    y: torch.Tensor,
    inputs_stats: Dict,
    targets_stats: Dict,
    include_wspd: bool,
    normalize_y: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (C,H,W) normalized
    y: (2,H,W) normalized OR raw depending on normalize_y
    returns x_raw, y_raw in original units (m/s)
    """
    x_names = [
        "past_drift_u",
        "past_drift_v",
        "future_wind_u10_mean",
        "future_wind_v10_mean",
    ]
    if include_wspd:
        x_names.append("future_wind_wspd_mean")

    y_names = ["future_drift_u", "future_drift_v"]

    x_mean = torch.tensor([inputs_stats[n]["mean"] for n in x_names], dtype=torch.float32, device=device).view(-1, 1, 1)
    x_std  = torch.tensor([inputs_stats[n]["std"]  for n in x_names], dtype=torch.float32, device=device).view(-1, 1, 1)
    x_raw = x * x_std + x_mean

    if normalize_y:
        y_mean = torch.tensor([targets_stats[n]["mean"] for n in y_names], dtype=torch.float32, device=device).view(-1, 1, 1)
        y_std  = torch.tensor([targets_stats[n]["std"]  for n in y_names], dtype=torch.float32, device=device).view(-1, 1, 1)
        y_raw = y * y_std + y_mean
    else:
        y_raw = y

    return x_raw, y_raw


def denorm_pred(pred: torch.Tensor, targets_stats: Dict, normalize_y: bool, device: torch.device) -> torch.Tensor:
    if not normalize_y:
        return pred
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
    return pred * y_std + y_mean


def vector_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u * u + v * v)


def pick_from_ckpt_config(ckpt_cfg: Dict[str, Any], key: str, default):
    return ckpt_cfg[key] if (ckpt_cfg is not None and key in ckpt_cfg and ckpt_cfg[key] is not None) else default


def plot_mag_quiver(ax, u, v, mag, title, norm: Normalize, step: int,
                    quiver_scale=None, quiver_width=0.004):
    """
    Data is already in image coordinates => NO flipping of v.
    Returns the AxesImage handle for making correct colorbars.
    """
    H, W = u.shape

    im = ax.imshow(
        mag,
        cmap="viridis",
        norm=norm,              # shared norm passed in
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
        scale=quiver_scale,     # None => auto
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
    ax.set_ylim(H - 0.5, -0.5)  # keep image-like y-direction

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

    ap.add_argument("--model-module", required=True, help="e.g. model_dev_main.src.models.Unet")
    ap.add_argument("--model-class", required=True, help="e.g. UNet")

    # Plot controls
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=16)

    # Range scan controls
    ap.add_argument("--max-samples-for-range", type=int, default=200)

    # Device controls
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--force-cpu", action="store_true")

    # Optional overrides
    ap.add_argument("--base-channels", type=int, default=None)
    ap.add_argument("--include-wspd", action="store_true", default=None)
    ap.add_argument("--normalize-y", action="store_true", default=None)
    ap.add_argument("--in-ch", type=int, default=None)
    ap.add_argument("--out-ch", type=int, default=None)

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

    # load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {}) or {}

    include_wspd = pick_from_ckpt_config(ckpt_cfg, "include_wspd", False)
    normalize_y = pick_from_ckpt_config(ckpt_cfg, "normalize_y", True)
    base_channels = pick_from_ckpt_config(ckpt_cfg, "base_channels", 32)
    in_ch = pick_from_ckpt_config(ckpt_cfg, "in_ch", 5 if include_wspd else 4)
    out_ch = pick_from_ckpt_config(ckpt_cfg, "out_ch", 2)

    # CLI overrides
    if args.include_wspd is not None:
        include_wspd = bool(args.include_wspd)
    if args.normalize_y is not None:
        normalize_y = bool(args.normalize_y)
    if args.base_channels is not None:
        base_channels = int(args.base_channels)
    if args.in_ch is not None:
        in_ch = int(args.in_ch)
    if args.out_ch is not None:
        out_ch = int(args.out_ch)

    if args.in_ch is None:
        in_ch = 5 if include_wspd else 4

    print(f"[run ] device={device}")
    print(f"[data] include_wspd={include_wspd} normalize_y={normalize_y}")
    print(f"[model] module={args.model_module} class={args.model_class} in_ch={in_ch} out_ch={out_ch} base_channels={base_channels}")

    # Build model
    ModelClass = import_model(args.model_module, args.model_class)
    model = ModelClass(in_channels=in_ch, out_channels=out_ch, base_channels=base_channels)

    # Move model with OOM fallback if auto
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

    # Load normalization stats
    inputs_stats, targets_stats = load_norm_yaml(args.norm_yaml)

    # Dataset
    val_ds = DriftWindDataset(
        args.val_index,
        norm_yaml_path=args.norm_yaml,
        normalize_y=normalize_y,
        include_wspd=include_wspd,
        return_meta=False,
        cache_size=0,
    )

    n = len(val_ds)
    picks = random.sample(range(n), k=min(args.num_samples, n))

    # ---------------------------------------------------------
    # 1) Compute a GLOBAL drift vmax over a scan of the dataset
    #    so that all drift panels share the same color range.
    # ---------------------------------------------------------
    scan_n = min(n, args.max_samples_for_range)
    # drift_vmax = 0.0

    for i in range(scan_n):
        s = val_ds[i]
        x = s["x"].to(device).float()
        y = s["y"].to(device).float()

        x_raw, y_raw = denorm_xy(x, y, inputs_stats, targets_stats, include_wspd, normalize_y, device)
        past_u, past_v = x_raw[0].cpu().numpy(), x_raw[1].cpu().numpy()
        fut_u, fut_v   = y_raw[0].cpu().numpy(), y_raw[1].cpu().numpy()


    # print(f"[plot] drift_norm vmin=0.0 vmax={case_drift_vmax:.4g} (from scan_n={scan_n})")

    # ---------------------------------------------------------
    # 2) Inference + plotting (wind uses per-figure norm)
    # ---------------------------------------------------------
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

        x_raw, y_raw = denorm_xy(x, y, inputs_stats, targets_stats, include_wspd, normalize_y, device)
        pred_raw = denorm_pred(pred, targets_stats, normalize_y, device)

        past_u, past_v = x_raw[0].cpu().numpy(), x_raw[1].cpu().numpy()
        wind_u, wind_v = x_raw[2].cpu().numpy(), x_raw[3].cpu().numpy()
        fut_u, fut_v   = y_raw[0].cpu().numpy(), y_raw[1].cpu().numpy()
        prd_u, prd_v   = pred_raw[0].cpu().numpy(), pred_raw[1].cpu().numpy()

        past_mag = vector_magnitude(past_u, past_v)
        wind_mag = vector_magnitude(wind_u, wind_v)
        fut_mag  = vector_magnitude(fut_u, fut_v)
        prd_mag  = vector_magnitude(prd_u, prd_v)

        future_drift_vmax = max(
            float(np.nanmax(fut_mag)),
            float(np.nanmax(prd_mag)),
        )

        if not np.isfinite(future_drift_vmax) or future_drift_vmax <= 0:
            future_drift_vmax = 1.0

        future_drift_norm = Normalize(vmin=0.0, vmax=future_drift_vmax)
        past_drift_norm = Normalize(vmin=0.0, vmax=float(np.nanmax(past_mag)))

        # wind gets its own range for THIS sample (continuous)
        wind_vmax = float(np.nanmax(wind_mag))
        if not np.isfinite(wind_vmax) or wind_vmax <= 0:
            wind_vmax = 1.0
        wind_norm = Normalize(vmin=0.0, vmax=wind_vmax)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

        im_past = plot_mag_quiver(axes[0, 0], past_u, past_v, past_mag, "Past drift (m/s)", past_drift_norm, args.step)
        im_wind = plot_mag_quiver(axes[0, 1], wind_u, wind_v, wind_mag, "Future wind (m/s)", wind_norm,  args.step)
        im_fut  = plot_mag_quiver(axes[1, 0], fut_u,  fut_v,  fut_mag,  "True future drift (m/s)", future_drift_norm, args.step)
        im_prd  = plot_mag_quiver(axes[1, 1], prd_u,  prd_v,  prd_mag,  "Predicted future drift (m/s)", future_drift_norm, args.step)

        fig.colorbar(im_past, ax=axes[0, 0], fraction=0.046, pad=0.02)\
        .set_label("Drift speed (m/s)")

        fig.colorbar(im_wind, ax=axes[0, 1], fraction=0.046, pad=0.02)\
        .set_label("Wind speed (m/s)")

        fig.colorbar(im_fut, ax=axes[1, 0], fraction=0.046, pad=0.02)\
        .set_label("Drift speed (m/s)")

        fig.colorbar(im_prd, ax=axes[1, 1], fraction=0.046, pad=0.02)\
        .set_label("Drift speed (m/s)")


        sid = s.get("id", idx)
        t = s.get("t", "")
        fig.suptitle(f"Val sample id={sid} {t}".strip(), fontsize=14)

        out_path = os.path.join(args.outdir, f"val_quiver_{sid}_{t}.png".replace(":", "").replace("/", "_"))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
