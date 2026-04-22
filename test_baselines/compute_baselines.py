#!/usr/bin/env python3
"""
Compute test metrics for simple baselines:

1) Persistence:
       u_pred = past_drift_u
       v_pred = past_drift_v

2) Wind baseline:
       [u_pred, v_pred]^T = alpha * R(theta) [u_wind, v_wind]^T

The script uses a user-provided test index file and the normalization/settings
from a training config_used.yaml.

Examples
--------
# Persistence baseline
python eval_test_baseline_metrics.py \
  --config /path/to/config_used.yaml \
  --index_file /path/to/index_test.jsonl \
  --baseline persistence \
  --dx 100 --dy 100 \
  --denorm \
  --out persistence_test_metrics.json

# Wind baseline with fitted alpha/theta
python eval_test_baseline_metrics.py \
  --config /path/to/config_used.yaml \
  --index_file /path/to/index_test_carra.jsonl \
  --baseline wind \
  --wind_fit /path/to/carra_wind_fit.json \
  --dx 100 --dy 100 \
  --denorm \
  --out wind_carra_test_metrics.json
"""

import argparse
import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _ensure_groups(groups: Sequence[str]) -> List[str]:
    return [g.lower() for g in groups]


def denorm_y(y_norm: torch.Tensor, ds: DriftWindSARDataset) -> torch.Tensor:
    if not getattr(ds, "do_norm", False) or not getattr(ds, "normalize_y", True):
        return y_norm
    return y_norm * ds.y_std.to(y_norm.device) + ds.y_mean.to(y_norm.device)


def denorm_x(x_norm: torch.Tensor, ds: DriftWindSARDataset) -> torch.Tensor:
    if not getattr(ds, "do_norm", False):
        return x_norm
    return x_norm * ds.x_std.to(x_norm.device) + ds.x_mean.to(x_norm.device)


def divergence_centered_torch(u: torch.Tensor, v: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """
    Central differences on the interior only, matching your other scripts.
    Input:  (B,H,W)
    Output: (B,H-2,W-2)
    """
    du_dx = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dx)
    dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / (2.0 * dy)
    du_dx = du_dx[:, 1:-1, :]
    dv_dy = dv_dy[:, :, 1:-1]
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
    ap.add_argument("--config", required=True, help="Path to config_used.yaml (or compatible training config)")
    ap.add_argument("--index_file", required=True, help="Path to test index file")
    ap.add_argument("--baseline", required=True, choices=["persistence", "wind"])
    ap.add_argument("--wind_fit", default=None, help="JSON file with fitted alpha/theta for wind baseline")
    ap.add_argument("--dx", type=float, required=True)
    ap.add_argument("--dy", type=float, required=True)
    ap.add_argument("--denorm", action="store_true")
    ap.add_argument("--angle_eps", type=float, default=0.02)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--topq", type=float, default=0.90)
    ap.add_argument("--div_sample_per_batch", type=int, default=50000)
    ap.add_argument("--reservoir_size", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    if args.baseline == "wind" and args.wind_fit is None:
        raise ValueError("--wind_fit is required when --baseline wind")
    if not os.path.exists(args.config):
        raise FileNotFoundError(args.config)
    if not os.path.exists(args.index_file):
        raise FileNotFoundError(args.index_file)
    if args.wind_fit is not None and not os.path.exists(args.wind_fit):
        raise FileNotFoundError(args.wind_fit)

    cfg = load_yaml(args.config)
    paths = cfg["paths"]
    data_cfg = cfg.get("data", {})
    train_cfg = cfg["train"]

    include_wspd = bool(data_cfg.get("include_wspd", False))
    normalize_y = bool(train_cfg.get("normalize_y", True))
    sar_to_db = bool(data_cfg.get("sar_to_db", True))
    sar_postprocess = bool(data_cfg.get("sar_postprocess", True))
    sar_zero_is_nodata = bool(data_cfg.get("sar_zero_is_nodata", False))
    sar_channels = tuple(data_cfg.get("sar_channels", ["HV"]))
    sar_clip_percentiles = data_cfg.get("sar_clip_percentiles", [0.01, 0.99])
    sar_clip_percentiles = tuple(sar_clip_percentiles) if sar_clip_percentiles is not None else None
    sar_clip_db = bool(data_cfg.get("sar_clip_db", False))
    sar_clip_db_bounds = {str(k).upper(): tuple(v) for k, v in data_cfg.get("sar_clip_db_bounds", {}).items()}
    prefetch_factor = int(data_cfg.get("prefetch_factor", 4))

    # Need both drift and wind channels available
    x_groups = ["drift", "wind"]

    ds = DriftWindSARDataset(
        args.index_file,
        norm_yaml_path=paths["norm_yaml"],
        normalize_y=normalize_y,
        include_wspd=include_wspd,
        return_meta=False,
        cache_size=0,
        x_groups=x_groups,
        sar_channels=sar_channels,
        sar_to_db=sar_to_db,
        sar_postprocess=sar_postprocess,
        sar_clip_percentiles=sar_clip_percentiles,
        sar_zero_is_nodata=sar_zero_is_nodata,
        sar_clip_db=sar_clip_db,
        sar_clip_db_bounds=sar_clip_db_bounds,
    )

    bs = args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 4))
    nw = args.num_workers if args.num_workers is not None else int(train_cfg.get("num_workers", 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"

    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
        prefetch_factor=prefetch_factor if nw > 0 else None,
    )

    x_ch = list(ds.x_channels)
    i_pu = x_ch.index("past_drift_u")
    i_pv = x_ch.index("past_drift_v")
    i_wu = x_ch.index("future_wind_u10_mean")
    i_wv = x_ch.index("future_wind_v10_mean")

    alpha = None
    theta_rad = None
    A = None
    B = None
    if args.baseline == "wind":
        with open(args.wind_fit, "r") as f:
            fit = json.load(f)
        if "A" in fit and "B" in fit:
            A = float(fit["A"])
            B = float(fit["B"])
            alpha = float(np.sqrt(A * A + B * B))
            theta_rad = float(np.arctan2(B, A))
        else:
            alpha = float(fit["alpha"])
            theta_rad = float(fit["theta_rad"])
            A = float(alpha * np.cos(theta_rad))
            B = float(alpha * np.sin(theta_rad))

    # Pass 1: estimate threshold for |div_true|
    sampler = ReservoirSampler(k=args.reservoir_size, seed=args.seed)
    for batch in loader:
        y = batch["y"].to(device, non_blocking=True)
        if args.denorm:
            y = denorm_y(y, ds)

        u_t, v_t = y[:, 0], y[:, 1]
        div_t = divergence_centered_torch(u_t, v_t, args.dx, args.dy)
        abs_div = torch.abs(div_t).reshape(-1)

        n = abs_div.numel()
        k = min(args.div_sample_per_batch, n)
        if k > 0:
            idx = torch.randint(0, n, (k,), device=abs_div.device)
            samp = abs_div[idx].detach().float().cpu().numpy()
            samp = samp[np.isfinite(samp)]
            sampler.add(samp)

    thr = sampler.quantile(args.topq)
    if not np.isfinite(thr):
        raise RuntimeError("Could not estimate divergence threshold.")

    # Pass 2: metrics
    sum_sq_u = sum_sq_v = 0.0
    sum_sq_speed = 0.0
    sum_abs_speed = 0.0
    sum_ang = 0.0
    ang_count = 0.0

    sum_sq_div = 0.0
    n_div_pix = 0

    sp_n = 0
    sp_sx = sp_sy = sp_sxx = sp_syy = sp_sxy = 0.0
    dv_n = 0
    dv_sx = dv_sy = dv_sxx = dv_syy = dv_sxy = 0.0

    # persistence reference for skill
    sum_sq_u_pers = sum_sq_v_pers = 0.0

    top_n = 0
    top_sum_sq_div = 0.0
    top_sx = top_sy = top_sxx = top_syy = top_sxy = 0.0

    n_pix_accum = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        if args.denorm:
            x = denorm_x(x, ds)
            y = denorm_y(y, ds)

        u_t, v_t = y[:, 0], y[:, 1]

        # persistence reference
        u_pers = x[:, i_pu]
        v_pers = x[:, i_pv]

        # chosen baseline prediction
        if args.baseline == "persistence":
            u_p = u_pers
            v_p = v_pers
        else:
            uw = x[:, i_wu]
            vw = x[:, i_wv]
            u_p = A * uw - B * vw
            v_p = B * uw + A * vw

        Bsz, H, W = u_t.shape
        n = Bsz * H * W
        n_pix_accum += n

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

        m_ang = (sp_t >= args.angle_eps)
        ang_t = torch.atan2(v_t, u_t)
        ang_p = torch.atan2(v_p, u_p)
        dtheta = (ang_p - ang_t + np.pi) % (2 * np.pi) - np.pi
        ang_abs = torch.abs(dtheta) * (180.0 / np.pi)
        sum_ang += float(ang_abs[m_ang].sum().item())
        ang_count += float(m_ang.sum().item())

        sp_pred = sp_p.reshape(-1).double()
        sp_true = sp_t.reshape(-1).double()
        sp_n += sp_pred.numel()
        sp_sx += float(sp_pred.sum().item())
        sp_sy += float(sp_true.sum().item())
        sp_sxx += float((sp_pred * sp_pred).sum().item())
        sp_syy += float((sp_true * sp_true).sum().item())
        sp_sxy += float((sp_pred * sp_true).sum().item())

        div_t = divergence_centered_torch(u_t, v_t, args.dx, args.dy)
        div_p = divergence_centered_torch(u_p, v_p, args.dx, args.dy)
        ddiv = div_p - div_t

        sum_sq_div += float((ddiv ** 2).sum().item())
        n_div_pix += ddiv.numel()

        dv_pred = div_p.reshape(-1).double()
        dv_true = div_t.reshape(-1).double()
        dv_n += dv_pred.numel()
        dv_sx += float(dv_pred.sum().item())
        dv_sy += float(dv_true.sum().item())
        dv_sxx += float((dv_pred * dv_pred).sum().item())
        dv_syy += float((dv_true * dv_true).sum().item())
        dv_sxy += float((dv_pred * dv_true).sum().item())

        sel = (torch.abs(div_t) >= thr)
        if sel.any():
            dd = ddiv[sel]
            top_sum_sq_div += float((dd ** 2).sum().item())
            c = int(sel.sum().item())
            top_n += c

            xt = div_t[sel].reshape(-1).double()
            xp = div_p[sel].reshape(-1).double()
            top_sx += float(xp.sum().item())
            top_sy += float(xt.sum().item())
            top_sxx += float((xp * xp).sum().item())
            top_syy += float((xt * xt).sum().item())
            top_sxy += float((xp * xt).sum().item())

    n_pix = int(n_pix_accum)

    rmse_u = float(np.sqrt(sum_sq_u / max(n_pix, 1)))
    rmse_v = float(np.sqrt(sum_sq_v / max(n_pix, 1)))
    rmse_speed = float(np.sqrt(sum_sq_speed / max(n_pix, 1)))
    mae_speed = float(sum_abs_speed / max(n_pix, 1))
    mean_ang = float(sum_ang / max(ang_count, 1.0))

    rmse_div = float(np.sqrt(sum_sq_div / max(n_div_pix, 1)))
    pearson_speed = pearsonr_from_sums(sp_n, sp_sx, sp_sy, sp_sxx, sp_syy, sp_sxy)
    pearson_div = pearsonr_from_sums(dv_n, dv_sx, dv_sy, dv_sxx, dv_syy, dv_sxy)

    mse_model = float((sum_sq_u + sum_sq_v) / max(n_pix * 2.0, 1.0))
    mse_pers = float((sum_sq_u_pers + sum_sq_v_pers) / max(n_pix * 2.0, 1.0))

    if args.baseline == "persistence":
        skill_pers = 0.0
    else:
        skill_pers = float(1.0 - (mse_model / max(mse_pers, 1e-12)))

    rmse_div_top10 = float(np.sqrt(top_sum_sq_div / max(top_n, 1)))
    pearson_div_top10 = pearsonr_from_sums(top_n, top_sx, top_sy, top_sxx, top_syy, top_sxy)

    out = {
        "config": args.config,
        "index_file": args.index_file,
        "baseline": args.baseline,
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
            "skill_vs_persistence": float(skill_pers),
            "rmse_divergence_top10_true": rmse_div_top10,
            "pearson_divergence_top10_true": float(pearson_div_top10),
        },
    }

    if args.baseline == "wind":
        out["wind_fit"] = args.wind_fit
        out["alpha"] = float(alpha)
        out["theta_rad"] = float(theta_rad)
        out["theta_deg"] = float(np.degrees(theta_rad))

    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()