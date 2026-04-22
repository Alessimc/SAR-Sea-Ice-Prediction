#!/usr/bin/env python3
"""
Fit alpha and theta for the simple wind baseline:

    u_ice = alpha * R(theta) * u_wind

using the TRAINING dataset only.

The fitted model is:
    u_ice = A * u_wind - B * v_wind
    v_ice = B * u_wind + A * v_wind

where
    A = alpha * cos(theta)
    B = alpha * sin(theta)

After fitting:
    alpha = sqrt(A^2 + B^2)
    theta = atan2(B, A)

Example:
python fit_wind_baseline.py \
  --config /path/to/config_used.yaml \
  --out /path/to/wind_baseline_fit.json
"""

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def denorm_y(y_norm: torch.Tensor, ds: DriftWindSARDataset) -> torch.Tensor:
    if not getattr(ds, "do_norm", False) or not getattr(ds, "normalize_y", True):
        return y_norm
    return y_norm * ds.y_std.to(y_norm.device) + ds.y_mean.to(y_norm.device)


def denorm_x(x_norm: torch.Tensor, ds: DriftWindSARDataset) -> torch.Tensor:
    if not getattr(ds, "do_norm", False):
        return x_norm
    return x_norm * ds.x_std.to(x_norm.device) + ds.x_mean.to(x_norm.device)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config_used.yaml or training config yaml")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", required=True, help="Output JSON file")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    data_cfg = cfg.get("data", {})
    train_cfg = cfg["train"]

    # only need wind as input, and future drift as target
    train_ds = DriftWindSARDataset(
        paths["train_index"],
        norm_yaml_path=paths["norm_yaml"],
        normalize_y=bool(train_cfg.get("normalize_y", True)),
        include_wspd=bool(data_cfg.get("include_wspd", False)),
        return_meta=False,
        cache_size=int(data_cfg.get("cache_size_train", 0)),
        x_groups=["wind"],   # only wind needed for fitting this baseline
        sar_channels=tuple(data_cfg.get("sar_channels", ["HV"])),
        sar_to_db=bool(data_cfg.get("sar_to_db", True)),
        sar_postprocess=bool(data_cfg.get("sar_postprocess", True)),
        sar_clip_percentiles=tuple(data_cfg.get("sar_clip_percentiles", [0.01, 0.99]))
            if data_cfg.get("sar_clip_percentiles", None) is not None else None,
        sar_zero_is_nodata=bool(data_cfg.get("sar_zero_is_nodata", False)),
        sar_clip_db=bool(data_cfg.get("sar_clip_db", False)),
        sar_clip_db_bounds={str(k).upper(): tuple(v) for k, v in data_cfg.get("sar_clip_db_bounds", {}).items()},
    )

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 4)) if args.num_workers > 0 else None,
    )

    # Find wind channel indices explicitly
    x_ch = list(train_ds.x_channels)
    i_wu = x_ch.index("future_wind_u10_mean")
    i_wv = x_ch.index("future_wind_v10_mean")

    # Accumulate normal equations for least squares in A,B
    #
    # Minimize:
    #   u_i - (A u_w - B v_w)
    #   v_i - (B u_w + A v_w)
    #
    # Stack as y = X beta, beta=[A,B]
    #
    # row 1: [u_w, -v_w] -> u_i
    # row 2: [v_w,  u_w] -> v_i
    #
    XtX = np.zeros((2, 2), dtype=np.float64)
    Xty = np.zeros(2, dtype=np.float64)

    n_pix = 0

    for batch in loader:
        x = batch["x"].float()
        y = batch["y"].float()

        # Denormalize to physical units
        x = denorm_x(x, train_ds)
        y = denorm_y(y, train_ds)

        uw = x[:, i_wu].reshape(-1).double().cpu().numpy()
        vw = x[:, i_wv].reshape(-1).double().cpu().numpy()
        ui = y[:, 0].reshape(-1).double().cpu().numpy()
        vi = y[:, 1].reshape(-1).double().cpu().numpy()

        # Filter finite values
        m = np.isfinite(uw) & np.isfinite(vw) & np.isfinite(ui) & np.isfinite(vi)
        uw = uw[m]
        vw = vw[m]
        ui = ui[m]
        vi = vi[m]

        if uw.size == 0:
            continue

        # Build normal equations without explicitly stacking giant X
        # For row set:
        #   [uw, -vw] -> ui
        #   [vw,  uw] -> vi
        #
        # XtX contributions:
        #   sum(uw^2 + vw^2) on diagonal, off-diagonals cancel
        s = np.sum(uw * uw + vw * vw)

        XtX[0, 0] += s
        XtX[1, 1] += s

        # X^T y:
        # A coefficient:
        #   sum(uw*ui + vw*vi)
        # B coefficient:
        #   sum(-vw*ui + uw*vi)
        Xty[0] += np.sum(uw * ui + vw * vi)
        Xty[1] += np.sum(-vw * ui + uw * vi)

        n_pix += uw.size

    if n_pix == 0:
        raise RuntimeError("No valid pixels found for fitting alpha/theta.")

    beta = np.linalg.solve(XtX, Xty)
    A, B = float(beta[0]), float(beta[1])

    alpha = float(np.sqrt(A * A + B * B))
    theta_rad = float(np.arctan2(B, A))
    theta_deg = float(np.degrees(theta_rad))

    out = {
        "fit_dataset": paths["train_index"],
        "n_pixels_used": int(n_pix),
        "A": A,
        "B": B,
        "alpha": alpha,
        "theta_rad": theta_rad,
        "theta_deg": theta_deg,
        "model_equations": {
            "u_ice": "A * u_wind - B * v_wind",
            "v_ice": "B * u_wind + A * v_wind",
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()