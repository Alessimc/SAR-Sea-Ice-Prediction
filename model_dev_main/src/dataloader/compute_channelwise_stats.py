"""
Compute channel-wise (pixel-weighted) mean/std for DriftWindDataset and save as YAML.

Usage (example):
  python compute_norm_stats_yaml.py \
    --index /path/to/index_train.jsonl \
    --out  norm_stats_train.yaml \
    --batch-size 4 --num-workers 4

Notes:
- Writes only YAML (human-readable). No NPZ.
- Channel order MUST match the dataset channel stacking order.
"""

import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import yaml

from model_dev_main.src.dataloader.DriftWindDataset import DriftWindDataset
from src.utils import init_logging

logger = init_logging()


@torch.no_grad()
def compute_channel_mean_std(dataloader, key="x", max_batches=None, device="cpu"):
    """
    Compute exact pixel-weighted mean/std per channel for a tensor batch shaped (B,C,H,W).
    Returns:
      mean: (C,)
      std : (C,)
    """
    sum_ = None
    sumsq = None
    n = 0  # total pixels per channel accumulated = sum over batches of (B*H*W)

    for b, batch in enumerate(dataloader):
        if max_batches is not None and b >= max_batches:
            break

        x = batch[key].to(device, non_blocking=True).float()  # (B,C,H,W)
        B, C, H, W = x.shape
        x = x.view(B, C, -1)  # (B,C,P)

        batch_sum = x.sum(dim=(0, 2))            # (C,)
        batch_sumsq = (x * x).sum(dim=(0, 2))    # (C,)

        if sum_ is None:
            sum_ = batch_sum
            sumsq = batch_sumsq
        else:
            sum_ += batch_sum
            sumsq += batch_sumsq

        n += B * H * W

    mean = sum_ / n
    var = (sumsq / n) - mean**2
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return mean.cpu(), std.cpu(), int(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Path to index_train.jsonl (or any split index).")
    parser.add_argument("--out", required=True, help="Output YAML path, e.g. norm_stats_train.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional: limit number of batches for faster estimate.")
    parser.add_argument("--include-wspd", action="store_true", help="If set, dataset includes wspd_mean as an extra X channel.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    # Channel naming MUST match DriftWindDataset stacking order
    x_channel_names = [
        "past_drift_u",
        "past_drift_v",
        "future_wind_u10_mean",
        "future_wind_v10_mean",
    ]
    if args.include_wspd:
        x_channel_names.append("future_wind_wspd_mean")

    y_channel_names = [
        "future_drift_u",
        "future_drift_v",
    ]

    logger.info(f"Loading dataset index: {args.index}")
    ds = DriftWindDataset(args.index, include_wspd=args.include_wspd, return_meta=False)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        persistent_workers=True if args.num_workers > 0 else False,
    )

    logger.info("Computing channel-wise stats for X...")
    x_mean, x_std, x_npix = compute_channel_mean_std(
        loader, key="x", max_batches=args.max_batches, device=args.device
    )

    logger.info("Computing channel-wise stats for Y...")
    y_mean, y_std, y_npix = compute_channel_mean_std(
        loader, key="y", max_batches=args.max_batches, device=args.device
    )

    # Clear per-channel logging
    logger.info("=== Channel-wise normalization statistics ===")
    for name, mu, sd in zip(x_channel_names, x_mean.tolist(), x_std.tolist()):
        logger.info(f"X | {name:>24s} | mean = {mu: .6e} | std = {sd: .6e}")
    for name, mu, sd in zip(y_channel_names, y_mean.tolist(), y_std.tolist()):
        logger.info(f"Y | {name:>24s} | mean = {mu: .6e} | std = {sd: .6e}")

    # Build human-readable YAML
    stats_yaml = {
        "inputs": {
            name: {"mean": float(mu), "std": float(sd)}
            for name, mu, sd in zip(x_channel_names, x_mean.tolist(), x_std.tolist())
        },
        "targets": {
            name: {"mean": float(mu), "std": float(sd)}
            for name, mu, sd in zip(y_channel_names, y_mean.tolist(), y_std.tolist())
        },
        "meta": {
            "index_path": os.path.abspath(args.index),
            "dataset_len": len(ds),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_batches": args.max_batches,
            "device_used_for_stats": args.device,
            "pixel_weighted": True,
            "x_num_channels": len(x_channel_names),
            "y_num_channels": len(y_channel_names),
            "x_total_pixels_per_channel_accumulated": x_npix,
            "y_total_pixels_per_channel_accumulated": y_npix,
            "created_utc": datetime.utcnow().isoformat() + "Z",
        },
    }

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(stats_yaml, f, sort_keys=False)

    logger.info(f"Saved human-readable stats to: {out_path}")


if __name__ == "__main__":
    main()
