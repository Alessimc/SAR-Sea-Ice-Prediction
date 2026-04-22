#!/usr/bin/env python3
"""
Compute test metrics for a trained drift model using the BEST saved checkpoint.

- Loads run_dir/config_used.yaml (for normalization + data options)
- Loads run_dir/checkpoints/best_val.pt (or chosen ckpt)
- Uses a user-provided test index file
- Uses ckpt["config"] to mirror training settings when possible
- Computes metrics in physical units (denorm) if requested

Example:
python eval_best_test_metrics.py \
  --run_dir model_dev_main/runs/EXPERIMENT_NAME \
  --index_file model_dev_main/index_files/.../index_test.jsonl \
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
    if not getattr(ds, "do_norm", False) or not getattr(ds, "normalize_y", True):
        return y_norm
    return y_norm * ds.y_std.to(y_norm.device) + ds.y_mean.to(y_norm.device)


def denorm_x(x_norm: torch.Tensor, ds: DriftWindSARDataset) -> torch.Tensor:
    if not getattr(ds, "do_norm", False):
        return x_norm
    return x_norm * ds.x_std.to(x_norm.device) + ds.x_mean.to(x_norm.device)


def divergence_centered_torch(u: torch.Tensor, v: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
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
    ap.add_argument("--run_dir", required=True, help="Run dir containing config_used.yaml and checkpoints/")
    ap.add_argument("--index_file", required=True, help="Path to test index file")
    ap.add_argument("--ckpt", default="best_val.pt", help="Checkpoint name inside run_dir/checkpoints/")
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

    args = ap.parse_args()

    run_dir = args.run_dir
    cfg_path = os.path.join(run_dir, "config_used.yaml")
    ckpt_path = os.path.join(run_dir, "checkpoints", args.ckpt)

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    if not os.path.exists(args.index_file):
        raise FileNotFoundError(args.index_file)

    cfg = load_yaml(cfg_path)
    paths = cfg["paths"]
    data_cfg = cfg.get("data", {})
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {}) or {}

    include_wspd = bool(ckpt_cfg.get("include_wspd", data_cfg.get("include_wspd", False)))
    normalize_y = bool(ckpt_cfg.get("normalize_y", train_cfg.get("normalize_y", True)))
    base_channels = int(ckpt_cfg.get("base_channels", model_cfg.get("base_channels", 16)))
    out_ch = int(ckpt_cfg.get("out_ch", model_cfg.get("out_ch", 2)))

    sar_to_db = bool(ckpt_cfg.get("sar_to_db", data_cfg.get("sar_to_db", True)))
    sar_postprocess = bool(ckpt_cfg.get("sar_postprocess", data_cfg.get("sar_postprocess", True)))
    sar_zero_is_nodata = bool(ckpt_cfg.get("sar_zero_is_nodata", data_cfg.get("sar_zero_is_nodata", False)))

    sar_channels = tuple(data_cfg.get("sar_channels", ["HV"]))
    x_groups = data_cfg.get("x_groups", None)

    sar_clip_percentiles = data_cfg.get("sar_clip_percentiles", [0.01, 0.99])
    sar_clip_percentiles = tuple(sar_clip_percentiles) if sar_clip_percentiles is not None else None

    sar_clip_db = bool(data_cfg.get("sar_clip_db", False))
    sar_clip_db_bounds = {str(k).upper(): tuple(v) for k, v in data_cfg.get("sar_clip_db_bounds", {}).items()}
    prefetch_factor = int(data_cfg.get("prefetch_factor", 4))

    # test dataset for model
    test_ds_model = DriftWindSARDataset(
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

    # test dataset for persistence baseline
    groups = x_groups if x_groups is not None else ["drift", "wind", "sar"]
    groups = _ensure_groups(groups)
    if "drift" not in groups:
        groups = ["drift"] + groups

    test_ds_pers = DriftWindSARDataset(
        args.index_file,
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

    bs = args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 4))
    nw = args.num_workers if args.num_workers is not None else int(train_cfg.get("num_workers", 4))

    loader_model = DataLoader(
        test_ds_model,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
        prefetch_factor=prefetch_factor if nw > 0 else None,
    )
    loader_pers = DataLoader(
        test_ds_pers,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
        prefetch_factor=prefetch_factor if nw > 0 else None,
    )

    inferred_in_ch = len(getattr(test_ds_model, "x_channels", [])) or test_ds_model[0]["x"].shape[0]
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

    logger.info("Pass 1/2: estimating global |div_true| quantile threshold on test set...")
    sampler = ReservoirSampler(k=args.reservoir_size, seed=args.seed)

    for batch in loader_model:
        y = batch["y"].to(device, non_blocking=True)
        if args.denorm:
            y = denorm_y(y, test_ds_model)

        u_t, v_t = y[:, 0], y[:, 1]
        div_t = divergence_centered_torch(u_t, v_t, dx=args.dx, dy=args.dy)
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

    logger.info(f"Estimated global |div_true| quantile threshold: {thr:.6e} [1/s]")
    logger.info("Pass 2/2: computing test metrics...")

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

    sum_sq_u_pers = sum_sq_v_pers = 0.0

    top_n = 0
    top_sum_sq_div = 0.0
    top_sx = top_sy = top_sxx = top_syy = top_sxy = 0.0

    n_pix_accum = 0

    for batch_m, batch_p in zip(loader_model, loader_pers):
        x = batch_m["x"].to(device, non_blocking=True)
        y = batch_m["y"].to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                yhat = model(x)
        else:
            yhat = model(x)

        x_p = batch_p["x"].to(device, non_blocking=True)

        if args.denorm:
            y = denorm_y(y, test_ds_model)
            yhat = denorm_y(yhat, test_ds_model)
            x_p = denorm_x(x_p, test_ds_pers)

        u_t, v_t = y[:, 0], y[:, 1]
        u_p, v_p = yhat[:, 0], yhat[:, 1]

        u_pers = x_p[:, 0]
        v_pers = x_p[:, 1]

        B, H, W = u_t.shape
        n = B * H * W
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

        div_t = divergence_centered_torch(u_t, v_t, dx=args.dx, dy=args.dy)
        div_p = divergence_centered_torch(u_p, v_p, dx=args.dx, dy=args.dy)
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
    skill_pers = float(1.0 - (mse_model / max(mse_pers, 1e-12)))

    rmse_div_top10 = float(np.sqrt(top_sum_sq_div / max(top_n, 1)))
    pearson_div_top10 = pearsonr_from_sums(top_n, top_sx, top_sy, top_sxx, top_syy, top_sxy)

    out = {
        "run_dir": run_dir,
        "index_file": args.index_file,
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

    out_path = os.path.join(run_dir, "test_AROME_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"Wrote {out_path}")
    logger.info(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()