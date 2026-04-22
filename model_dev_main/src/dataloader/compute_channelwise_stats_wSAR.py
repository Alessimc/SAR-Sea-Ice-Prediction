# """
# Compute channel-wise (pixel-weighted) mean/std for DriftWindSARDataset and save as YAML.

# Usage (example):
#   python compute_norm_stats_yaml_sar.py \
#     --index /path/to/index_train.jsonl \
#     --out  norm_stats_train.yaml \
#     --batch-size 4 --num-workers 4 \
#     --sar-channels HV \
#     --sar-to-db \
#     --device cuda

# Notes:
# - Writes YAML (human-readable).
# - Channel order/names MUST match the dataset stacking order.
#   This script uses ds.x_channels and ds.y_channels to guarantee correctness.
# """

# import argparse
# import os
# from datetime import datetime

# import torch
# from torch.utils.data import DataLoader
# import yaml

# # TODO: adjust import path as needed
# from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset
# from src.utils import init_logging

# logger = init_logging()


# @torch.no_grad()
# def compute_channel_mean_std(dataloader, key="x", max_batches=None, device="cpu"):
#     """
#     Compute pixel-weighted mean/std per channel for batch[key] shaped (B,C,H,W),
#     using ONLY finite values (ignores NaNs and +/-Infs).

#     Returns:
#       mean: (C,) torch.Tensor (cpu)
#       std : (C,) torch.Tensor (cpu)
#       n_pix_per_channel: (C,) torch.Tensor (cpu, int64)  # valid pixel counts per channel
#     """
#     sum_ = None          # (C,)
#     sumsq = None         # (C,)
#     count = None         # (C,) valid pixel counts

#     for b, batch in enumerate(dataloader):
#         if max_batches is not None and b >= max_batches:
#             break

#         x = batch[key].to(device, non_blocking=True).float()  # (B,C,H,W)
#         if x.ndim != 4:
#             raise ValueError(f"Expected batch['{key}'] to have shape (B,C,H,W), got {tuple(x.shape)}")

#         B, C, H, W = x.shape
#         x = x.view(B, C, -1)  # (B,C,P)

#         finite = torch.isfinite(x)               # (B,C,P) bool
#         x_safe = torch.where(finite, x, 0.0)     # replace invalid values with 0 for summation

#         batch_sum = x_safe.sum(dim=(0, 2))               # (C,)
#         batch_sumsq = (x_safe * x_safe).sum(dim=(0, 2))  # (C,)
#         batch_count = finite.sum(dim=(0, 2)).to(x.dtype) # (C,) as float for division

#         if sum_ is None:
#             sum_ = batch_sum
#             sumsq = batch_sumsq
#             count = batch_count
#         else:
#             sum_ += batch_sum
#             sumsq += batch_sumsq
#             count += batch_count

#     if sum_ is None:
#         raise RuntimeError("No batches processed. Check dataloader or max_batches=0.")

#     # Avoid divide-by-zero for channels with no valid pixels
#     eps = 1e-12
#     count_clamped = torch.clamp(count, min=1.0)

#     mean = sum_ / count_clamped
#     var = (sumsq / count_clamped) - mean**2
#     std = torch.sqrt(torch.clamp(var, min=eps))

#     # If a channel had zero valid pixels, set mean=0, std=1 to avoid nonsense
#     zero_valid = count < 0.5
#     mean = torch.where(zero_valid, torch.zeros_like(mean), mean)
#     std = torch.where(zero_valid, torch.ones_like(std), std)

#     return mean.cpu(), std.cpu(), count.to(torch.int64).cpu()



# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--index", required=True, help="Path to index_train.jsonl (or any split index).")
#     parser.add_argument("--out", required=True, help="Output YAML path, e.g. norm_stats_train.yaml")
#     parser.add_argument("--batch-size", type=int, default=4)
#     parser.add_argument("--num-workers", type=int, default=4)
#     parser.add_argument("--max-batches", type=int, default=None, help="Optional: limit number of batches for faster estimate.")
#     parser.add_argument("--include-wspd", action="store_true", help="Include wspd_mean as an extra X channel.")

#     # SAR options
#     parser.add_argument(
#         "--sar-channels",
#         nargs="+",
#         default=["HV"],
#         help="SAR channels to include. Choose any of: HH HV IA. Example: --sar-channels HH HV IA",
#     )
#     parser.add_argument(
#         "--sar-to-db",
#         action="store_true",
#         help="Convert HH/HV channels from linear to dB (10*log10). Recommended if your GeoTIFF stores linear sigma0.",
#     )
#     parser.add_argument(
#         "--sar-no-db",
#         dest="sar_to_db",
#         action="store_false",
#         help="Disable dB conversion (use SAR as-is). Use if GeoTIFF is already in dB.",
#     )
#     parser.set_defaults(sar_to_db=True)

#     parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
#     args = parser.parse_args()

#     if args.device == "cuda" and not torch.cuda.is_available():
#         logger.warning("CUDA requested but not available; falling back to CPU.")
#         args.device = "cpu"

#     sar_channels = [c.upper() for c in args.sar_channels]

#     logger.info(f"Loading dataset index: {args.index}")
#     ds = DriftWindSARDataset(
#         args.index,
#         include_wspd=args.include_wspd,
#         return_meta=False,
#         norm_yaml_path=None,      # IMPORTANT: do NOT normalize when computing stats
#         normalize_y=False,        # irrelevant when norm_yaml_path=None, but explicit
#         sar_channels=tuple(sar_channels),
#         sar_to_db=bool(args.sar_to_db),
#         cache_size=0,
#     )

#     # Use dataset-provided channel name lists to guarantee correct YAML naming
#     if not hasattr(ds, "x_channels") or not hasattr(ds, "y_channels"):
#         raise AttributeError("Dataset must expose x_channels and y_channels for correct YAML naming.")

#     x_channel_names = list(ds.x_channels)
#     y_channel_names = list(ds.y_channels)

#     logger.info(f"Dataset length: {len(ds)}")
#     logger.info(f"X channels ({len(x_channel_names)}): {x_channel_names}")
#     logger.info(f"Y channels ({len(y_channel_names)}): {y_channel_names}")

#     use_workers = args.num_workers > 0

#     loader = DataLoader(
#         ds,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=(args.device == "cuda"),
#         persistent_workers=True if use_workers else False,
#     )

#     logger.info("Computing channel-wise stats for X...")
#     x_mean, x_std, x_npix = compute_channel_mean_std(
#         loader, key="x", max_batches=args.max_batches, device=args.device
#     )

#     logger.info("Computing channel-wise stats for Y...")
#     y_mean, y_std, y_npix = compute_channel_mean_std(
#         loader, key="y", max_batches=args.max_batches, device=args.device
#     )

#     # Clear per-channel logging
#     logger.info("=== Channel-wise normalization statistics ===")
#     for name, mu, sd in zip(x_channel_names, x_mean.tolist(), x_std.tolist()):
#         logger.info(f"X | {name:>24s} | mean = {mu: .6e} | std = {sd: .6e}")
#     for name, mu, sd in zip(y_channel_names, y_mean.tolist(), y_std.tolist()):
#         logger.info(f"Y | {name:>24s} | mean = {mu: .6e} | std = {sd: .6e}")

#     # Build human-readable YAML
#     stats_yaml = {
#         "inputs": {
#             name: {"mean": float(mu), "std": float(sd)}
#             for name, mu, sd in zip(x_channel_names, x_mean.tolist(), x_std.tolist())
#         },
#         "targets": {
#             name: {"mean": float(mu), "std": float(sd)}
#             for name, mu, sd in zip(y_channel_names, y_mean.tolist(), y_std.tolist())
#         },
#         "meta": {
#             "index_path": os.path.abspath(args.index),
#             "dataset_len": len(ds),
#             "batch_size": args.batch_size,
#             "num_workers": args.num_workers,
#             "max_batches": args.max_batches,
#             "device_used_for_stats": args.device,
#             "pixel_weighted": True,
#             "include_wspd": bool(args.include_wspd),
#             "sar_channels": sar_channels,
#             "sar_to_db": bool(args.sar_to_db),
#             "x_num_channels": len(x_channel_names),
#             "y_num_channels": len(y_channel_names),
#             "x_total_pixels_per_channel_accumulated": x_npix,
#             "y_total_pixels_per_channel_accumulated": y_npix,
#             "created_utc": datetime.utcnow().isoformat() + "Z",
#         },
#     }

#     out_path = args.out
#     os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
#     with open(out_path, "w") as f:
#         yaml.safe_dump(stats_yaml, f, sort_keys=False)

#     logger.info(f"Saved human-readable stats to: {out_path}")


# if __name__ == "__main__":
#     main()





#!/usr/bin/env python3
"""
compute_norm_stats_yaml_sar.py

Compute channel-wise (pixel-weighted) mean/std for DriftWindSARDataset and save as YAML.

NEW:
- Optionally apply fixed SAR dB clipping (e.g. 1–99% bounds computed from train hist)
  BEFORE computing mean/std (recommended if you will also clip during training).

Key points:
- Clipping is applied ONLY to SAR HH/HV dB channels (not wind/drift).
- NaN/Inf are always ignored in stats.
- This script does NOT normalize the dataset (norm_yaml_path=None).
- This script does NOT rely on sar_postprocess in the dataset; we clip here explicitly
  to ensure stats are computed on clipped SAR.

Example:
  python compute_norm_stats_yaml_sar.py \
    --index /path/to/index_train.jsonl \
    --out norm_stats_train.yaml \
    --sar-channels HH HV IA \
    --sar-to-db \
    --sar-clip-db --sar-clip-qlo 0.01 --sar-clip-qhi 0.99 \
    --sar-clip-hh -18.88 -7.12 \
    --sar-clip-hv -34.88 -17.88 \
    --device cpu
"""

import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
import yaml

from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset
from src.utils import init_logging

logger = init_logging()


def _get_sar_clip_bounds(
    x_channel_names: List[str],
    enabled: bool,
    sar_to_db: bool,
    hh_bounds: Optional[Tuple[float, float]],
    hv_bounds: Optional[Tuple[float, float]],
) -> Dict[int, Tuple[float, float]]:
    """
    Return a mapping {channel_index: (lo, hi)} for SAR HH/HV channels to clip.
    We only clip if enabled=True and sar_to_db=True and bounds provided.

    We match channel names exactly as used in DriftWindSARDataset:
      - "sar_hh_db" / "sar_hv_db" when sar_to_db=True
      - "sar_hh" / "sar_hv" when sar_to_db=False (we do NOT clip in linear by default)
    """
    if not enabled:
        return {}
    if not sar_to_db:
        logger.warning("SAR clipping requested but sar_to_db=False. Skipping SAR clipping (bounds are in dB).")
        return {}

    bounds: Dict[int, Tuple[float, float]] = {}

    name_to_idx = {str(n): i for i, n in enumerate(x_channel_names)}
    hh_name = "sar_hh_db"
    hv_name = "sar_hv_db"

    if hh_bounds is not None and hh_name in name_to_idx:
        bounds[name_to_idx[hh_name]] = (float(hh_bounds[0]), float(hh_bounds[1]))
    if hv_bounds is not None and hv_name in name_to_idx:
        bounds[name_to_idx[hv_name]] = (float(hv_bounds[0]), float(hv_bounds[1]))

    if enabled and (hh_bounds is None and hv_bounds is None):
        logger.warning("SAR clipping enabled but no bounds provided; no SAR channels will be clipped.")

    return bounds


@torch.no_grad()
def compute_channel_mean_std(
    dataloader: DataLoader,
    key: str = "x",
    max_batches: Optional[int] = None,
    device: str = "cpu",
    clip_bounds_by_channel: Optional[Dict[int, Tuple[float, float]]] = None,
):
    """
    Compute pixel-weighted mean/std per channel for batch[key] shaped (B,C,H,W),
    using ONLY finite values (ignores NaNs and +/-Infs).

    Optionally clamps selected channels (e.g. SAR HH/HV) to fixed bounds
    BEFORE accumulation, but only for finite values.

    Returns:
      mean: (C,) torch.Tensor (cpu)
      std : (C,) torch.Tensor (cpu)
      n_pix_per_channel: (C,) torch.Tensor (cpu, int64)
    """
    sum_ = None          # (C,)
    sumsq = None         # (C,)
    count = None         # (C,) valid pixel counts

    clip_bounds_by_channel = clip_bounds_by_channel or {}

    for b, batch in enumerate(dataloader):
        if max_batches is not None and b >= max_batches:
            break

        x = batch[key].to(device, non_blocking=True).float()  # (B,C,H,W)
        if x.ndim != 4:
            raise ValueError(f"Expected batch['{key}'] to have shape (B,C,H,W), got {tuple(x.shape)}")

        B, C, H, W = x.shape
        x = x.view(B, C, -1)  # (B,C,P)

        finite = torch.isfinite(x)  # (B,C,P)

        # Apply per-channel clipping ONLY on finite values
        # (keeps NaN/Inf untouched so they remain excluded)
        if clip_bounds_by_channel:
            for ci, (lo, hi) in clip_bounds_by_channel.items():
                if ci < 0 or ci >= C:
                    continue
                m = finite[:, ci, :]
                if m.any():
                    x_ci = x[:, ci, :]
                    x_ci = torch.where(m, torch.clamp(x_ci, lo, hi), x_ci)
                    x[:, ci, :] = x_ci

        x_safe = torch.where(finite, x, 0.0)  # replace invalid with 0 for summation only

        batch_sum = x_safe.sum(dim=(0, 2))               # (C,)
        batch_sumsq = (x_safe * x_safe).sum(dim=(0, 2))  # (C,)
        batch_count = finite.sum(dim=(0, 2)).to(x.dtype) # (C,) float for division

        if sum_ is None:
            sum_ = batch_sum
            sumsq = batch_sumsq
            count = batch_count
        else:
            sum_ += batch_sum
            sumsq += batch_sumsq
            count += batch_count

        if (b + 1) % 50 == 0:
            logger.info(f"Processed {b+1} batches for '{key}' stats...")

    if sum_ is None:
        raise RuntimeError("No batches processed. Check dataloader or max_batches=0.")

    eps = 1e-12
    count_clamped = torch.clamp(count, min=1.0)

    mean = sum_ / count_clamped
    var = (sumsq / count_clamped) - mean**2
    std = torch.sqrt(torch.clamp(var, min=eps))

    zero_valid = count < 0.5
    mean = torch.where(zero_valid, torch.zeros_like(mean), mean)
    std = torch.where(zero_valid, torch.ones_like(std), std)

    return mean.cpu(), std.cpu(), count.to(torch.int64).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Path to index_train.jsonl (or any split index).")
    parser.add_argument("--out", required=True, help="Output YAML path, e.g. norm_stats_train.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional: limit batches for faster estimate.")
    parser.add_argument("--include-wspd", action="store_true", help="Include wspd_mean as an extra X channel.")

    # SAR options
    parser.add_argument(
        "--sar-channels",
        nargs="+",
        default=["HV"],
        help="SAR channels to include. Choose any of: HH HV IA. Example: --sar-channels HH HV IA",
    )
    parser.add_argument(
        "--sar-to-db",
        action="store_true",
        help="Convert HH/HV channels from linear to dB (10*log10). Recommended if GeoTIFF stores linear sigma0.",
    )
    parser.add_argument(
        "--sar-no-db",
        dest="sar_to_db",
        action="store_false",
        help="Disable dB conversion (use SAR as-is). Use if GeoTIFF is already in dB.",
    )
    parser.set_defaults(sar_to_db=True)

    # NEW: SAR clipping controls (dB bounds)
    parser.add_argument(
        "--sar-clip-db",
        action="store_true",
        help="Enable fixed dB clipping for SAR HH/HV channels before computing mean/std.",
    )
    parser.add_argument(
        "--sar-clip-hh",
        nargs=2,
        type=float,
        default=None,
        metavar=("HH_LO_DB", "HH_HI_DB"),
        help="Clip bounds for sar_hh_db. Example: --sar-clip-hh -18.88 -7.12",
    )
    parser.add_argument(
        "--sar-clip-hv",
        nargs=2,
        type=float,
        default=None,
        metavar=("HV_LO_DB", "HV_HI_DB"),
        help="Clip bounds for sar_hv_db. Example: --sar-clip-hv -34.88 -17.88",
    )

    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    sar_channels = [c.upper() for c in args.sar_channels]

    logger.info(f"Loading dataset index: {args.index}")
    ds = DriftWindSARDataset(
        args.index,
        include_wspd=args.include_wspd,
        return_meta=False,
        norm_yaml_path=None,      # IMPORTANT: do NOT normalize when computing stats
        normalize_y=False,
        sar_channels=tuple(sar_channels),
        sar_to_db=bool(args.sar_to_db),
        sar_postprocess=False,    # IMPORTANT: preserve NaN/Inf; we clip explicitly here
        cache_size=0,
    )

    if not hasattr(ds, "x_channels") or not hasattr(ds, "y_channels"):
        raise AttributeError("Dataset must expose x_channels and y_channels for correct YAML naming.")

    x_channel_names = list(ds.x_channels)
    y_channel_names = list(ds.y_channels)

    logger.info(f"Dataset length: {len(ds)}")
    logger.info(f"X channels ({len(x_channel_names)}): {x_channel_names}")
    logger.info(f"Y channels ({len(y_channel_names)}): {y_channel_names}")

    # Build per-channel clipping map for SAR HH/HV dB channels
    clip_map_x = _get_sar_clip_bounds(
        x_channel_names=x_channel_names,
        enabled=bool(args.sar_clip_db),
        sar_to_db=bool(args.sar_to_db),
        hh_bounds=tuple(args.sar_clip_hh) if args.sar_clip_hh is not None else None,
        hv_bounds=tuple(args.sar_clip_hv) if args.sar_clip_hv is not None else None,
    )
    if clip_map_x:
        pretty = {x_channel_names[k]: v for k, v in clip_map_x.items()}
        logger.info(f"Applying SAR clipping (X only) to: {pretty}")
    else:
        logger.info("SAR clipping disabled for stats.")

    use_workers = args.num_workers > 0
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        persistent_workers=True if use_workers else False,
    )

    logger.info("Computing channel-wise stats for X...")
    x_mean, x_std, x_npix = compute_channel_mean_std(
        loader,
        key="x",
        max_batches=args.max_batches,
        device=args.device,
        clip_bounds_by_channel=clip_map_x,
    )

    logger.info("Computing channel-wise stats for Y...")
    # No clipping on Y (drift)
    y_mean, y_std, y_npix = compute_channel_mean_std(
        loader, key="y", max_batches=args.max_batches, device=args.device, clip_bounds_by_channel=None
    )

    logger.info("=== Channel-wise normalization statistics ===")
    for name, mu, sd in zip(x_channel_names, x_mean.tolist(), x_std.tolist()):
        logger.info(f"X | {name:>24s} | mean = {mu: .6e} | std = {sd: .6e}")
    for name, mu, sd in zip(y_channel_names, y_mean.tolist(), y_std.tolist()):
        logger.info(f"Y | {name:>24s} | mean = {mu: .6e} | std = {sd: .6e}")

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
            "include_wspd": bool(args.include_wspd),
            "sar_channels": sar_channels,
            "sar_to_db": bool(args.sar_to_db),
            "sar_clip_db_enabled": bool(args.sar_clip_db),
            "sar_clip_hh_db": list(args.sar_clip_hh) if args.sar_clip_hh is not None else None,
            "sar_clip_hv_db": list(args.sar_clip_hv) if args.sar_clip_hv is not None else None,
            "x_num_channels": len(x_channel_names),
            "y_num_channels": len(y_channel_names),
            "x_total_pixels_per_channel_accumulated": x_npix.tolist(),
            "y_total_pixels_per_channel_accumulated": y_npix.tolist(),
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