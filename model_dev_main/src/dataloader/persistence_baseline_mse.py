#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from model_dev_main.src.dataloader.DriftWindDataset import DriftWindDataset
from src.utils import init_logging

logger = init_logging()


def load_norm_stats(norm_yaml_path: str, eps: float = 1e-6):
    """
    Return x_mean/x_std for input channels and y_mean/y_std for target channels.
    Shapes:
      x_mean/x_std: (4,1,1) or (5,1,1) depending on include_wspd, but we only need wind u/v.
      y_mean/y_std: (2,1,1)
    """
    with open(norm_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    inputs = cfg["inputs"]
    targets = cfg["targets"]

    # These names must match your dataset channel naming in the YAML
    x_names = ["past_drift_u", "past_drift_v", "future_wind_u10_mean", "future_wind_v10_mean"]
    y_names = ["future_drift_u", "future_drift_v"]

    x_mean = torch.tensor([inputs[n]["mean"] for n in x_names], dtype=torch.float32).view(-1, 1, 1)
    x_std  = torch.tensor([inputs[n]["std"]  for n in x_names], dtype=torch.float32).view(-1, 1, 1).clamp_min(eps)

    y_mean = torch.tensor([targets[n]["mean"] for n in y_names], dtype=torch.float32).view(-1, 1, 1)
    y_std  = torch.tensor([targets[n]["std"]  for n in y_names], dtype=torch.float32).view(-1, 1, 1).clamp_min(eps)

    return x_mean, x_std, y_mean, y_std


def rotate_clockwise(u: torch.Tensor, v: torch.Tensor, degrees: float):
    """
    Rotate vectors (u,v) by 'degrees' clockwise.
    u,v: (...,H,W) tensors
    Returns (u_rot, v_rot).
    """
    theta = torch.deg2rad(torch.tensor(degrees, dtype=u.dtype, device=u.device))
    c = torch.cos(theta)
    s = torch.sin(theta)

    # Clockwise rotation by theta:
    # [u']   [ c  s][u]
    # [v'] = [-s  c][v]
    u_rot = c * u + s * v
    v_rot = -s * u + c * v
    return u_rot, v_rot


@torch.no_grad()
def baseline_mse(loader, device, x_mean, x_std, y_mean, y_std, wind_scale=0.02, wind_rot_deg=-45.0):
    """
    Computes:
      1) Persistence baseline: y_hat_persist = past drift = x[:,0:2]
      2) Wind baseline: y_hat_wind computed from denormed wind:
           - take 2% of wind magnitude
           - rotate wind direction 45 deg clockwise
         then normalize with target (y) stats so comparable to y (normalized)

    Returns:
      dict with total/u/v mse for both baselines
    """
    mse = nn.MSELoss(reduction="mean")

    # accumulators
    out = {
        "persist_total": 0.0, "persist_u": 0.0, "persist_v": 0.0,
        "wind_total": 0.0,    "wind_u": 0.0,    "wind_v": 0.0,
        "n_batches": 0
    }

    # move stats to device once
    x_mean = x_mean.to(device)
    x_std  = x_std.to(device)
    y_mean = y_mean.to(device)
    y_std  = y_std.to(device)

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True).float()  # (B,C,H,W) normalized
        y = batch["y"].to(device, non_blocking=True).float()  # (B,2,H,W) normalized

        # ------------------------
        # 1) Persistence baseline
        # ------------------------
        y_hat_p = x[:, 0:2]  # past drift u,v (already normalized with x stats, not y stats!)
        # NOTE: This is the same persistence baseline you've been using, evaluated in "dataset space".
        # It matches how you compare in training only if your model sees/outputs in that same normalized space.
        # (This is what you asked for and what you've been doing.)

        Lp  = mse(y_hat_p, y)
        Lpu = mse(y_hat_p[:, 0:1], y[:, 0:1])
        Lpv = mse(y_hat_p[:, 1:2], y[:, 1:2])

        # ------------------------
        # 2) Wind-based baseline
        # ------------------------
        # x channels: [past_u, past_v, wind_u, wind_v, ...]
        wind_u_n = x[:, 2:3]  # normalized wind u
        wind_v_n = x[:, 3:4]  # normalized wind v

        # denormalize wind to m/s using INPUT stats (wind channels are indices 2 and 3 in x_names)
        wind_u = wind_u_n * x_std[2:3] + x_mean[2:3]
        wind_v = wind_v_n * x_std[3:4] + x_mean[3:4]

        # rotate wind direction 45° to the right (clockwise) and take 2% magnitude
        wind_u_rot, wind_v_rot = rotate_clockwise(wind_u, wind_v, wind_rot_deg)
        drift_u_phys = wind_scale * wind_u_rot
        drift_v_phys = wind_scale * wind_v_rot

        # normalize drift baseline into Y-normalized space so it can be compared to y
        y_hat_w = torch.cat(
            [(drift_u_phys - y_mean[0:1]) / y_std[0:1],
             (drift_v_phys - y_mean[1:2]) / y_std[1:2]],
            dim=1
        )  # (B,2,H,W)

        Lw  = mse(y_hat_w, y)
        Lwu = mse(y_hat_w[:, 0:1], y[:, 0:1])
        Lwv = mse(y_hat_w[:, 1:2], y[:, 1:2])

        # accumulate
        out["persist_total"] += Lp.item()
        out["persist_u"]     += Lpu.item()
        out["persist_v"]     += Lpv.item()

        out["wind_total"] += Lw.item()
        out["wind_u"]     += Lwu.item()
        out["wind_v"]     += Lwv.item()

        out["n_batches"] += 1

    # average
    n = out["n_batches"]
    for k in list(out.keys()):
        if k != "n_batches":
            out[k] /= max(n, 1)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-index", required=True)
    ap.add_argument("--norm-yaml", required=True, help="Training normalization YAML (same as model)")
    ap.add_argument("--include-wspd", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cache-size", type=int, default=0)

    ap.add_argument("--wind-scale", type=float, default=0.02, help="Fraction of wind magnitude used as drift (default 0.02)")
    ap.add_argument("--wind-rot-deg", type=float, default=-30.0, help="Clockwise rotation degrees (default 45)")
                                                # using -45 here due to the image spave y axis
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset returns normalized x and y
    ds = DriftWindDataset(
        args.val_index,
        norm_yaml_path=args.norm_yaml,
        normalize_y=True,
        include_wspd=args.include_wspd,
        cache_size=args.cache_size,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    x_mean, x_std, y_mean, y_std = load_norm_stats(args.norm_yaml)

    res = baseline_mse(
        loader,
        device=device,
        x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std,
        wind_scale=args.wind_scale,
        wind_rot_deg=args.wind_rot_deg,
    )

    logger.info("Baselines MSE (normalized space):")
    logger.info(f"Persistence:")
    logger.info(f"  total: {res['persist_total']:.6e} | u: {res['persist_u']:.6e} | v: {res['persist_v']:.6e}")
    logger.info(f"Wind baseline (scale={args.wind_scale}, rot={args.wind_rot_deg}° clockwise):")
    logger.info(f"  total: {res['wind_total']:.6e} | u: {res['wind_u']:.6e} | v: {res['wind_v']:.6e}")
    logger.info(f"N samples: {len(ds)}")


if __name__ == "__main__":
    main()
