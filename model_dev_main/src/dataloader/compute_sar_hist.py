#!/usr/bin/env python3
"""
compute_sar_hist_npz.py

Compute per-channel SAR (dB) distribution summaries for DriftWindSARDataset.

Outputs an NPZ containing:
- channel_names: list[str]
- bin_edges_db: (nbins+1,)
- bin_centers_db: (nbins,)
- counts: (C, nbins) int64              (global histogram over ALL finite pixels)
- pdf: (C, nbins) float64              (global density; integrates to ~1)
- cdf: (C, nbins) float64              (global cumulative mass; last ~1.0)

PLUS (for shaded "error band" plots):
- pdf_mean_across_batches: (C, nbins) float64  (mean batch-wise PDF)
- pdf_std_across_batches : (C, nbins) float64  (std  batch-wise PDF)
- n_batches_used_for_pdf_stats: int64

Notes:
- The "pdf_mean/std_across_batches" quantify variability of the estimated PDF across
  batches (or, more precisely, across batch-aggregated histograms). This is NOT the
  physical std of backscatter values.
- IMPORTANT: We disable SAR postprocessing (no nan_to_num / no percentile clip) and
  disable normalization to preserve the true distribution and ignore NaN/Inf via
  torch.isfinite.

Example:
  python compute_sar_hist_npz.py \
    --index /path/to/index_train.jsonl \
    --out /path/to/sar_hist_train.npz \
    --batch-size 4 --num-workers 4 \
    --sar-channels HH HV \
    --sar-to-db \
    --db-min -60 --db-max 10 --db-step 0.25 \
    --device cpu
"""

import argparse
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset
from src.utils import init_logging

logger = init_logging()


def _pick_sar_indices(x_channel_names: List[str], want: List[str]) -> List[int]:
    """
    Select SAR channel indices from ds.x_channels.
    'want' is a list like ["HH","HV","IA"].
    We match by substring in uppercased channel name.
    """
    want_u = [w.upper() for w in want]

    idx = []
    for i, name in enumerate(x_channel_names):
        n = str(name).upper()
        for w in want_u:
            if w in n:
                idx.append(i)
                break

    # Keep order consistent with want list if possible
    ordered: List[int] = []
    for w in want_u:
        for i in idx:
            if w in str(x_channel_names[i]).upper() and i not in ordered:
                ordered.append(i)
    for i in idx:
        if i not in ordered:
            ordered.append(i)

    return ordered


def _welford_init(C: int, nbins: int) -> Tuple[np.ndarray, np.ndarray, int]:
    mean = np.zeros((C, nbins), dtype=np.float64)
    M2 = np.zeros((C, nbins), dtype=np.float64)
    n = 0
    return mean, M2, n


def _welford_update(mean: np.ndarray, M2: np.ndarray, n: int, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Welford update for arrays. x has shape (C, nbins).
    """
    n_new = n + 1
    delta = x - mean
    mean = mean + delta / n_new
    delta2 = x - mean
    M2 = M2 + delta * delta2
    return mean, M2, n_new


def _welford_finalize(mean: np.ndarray, M2: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, int]:
    if n < 2:
        std = np.zeros_like(mean)
    else:
        std = np.sqrt(M2 / (n - 1))
    return mean, std, n


@torch.no_grad()
def compute_histograms_and_batch_pdf_stats(
    loader: DataLoader,
    sar_idx: List[int],
    bin_edges: np.ndarray,
    key: str = "x",
    max_batches: Optional[int] = None,
    device: str = "cpu",
):
    """
    Accumulate global histogram counts per SAR channel over finite pixels,
    and compute mean/std of batch-wise PDFs (for shaded bands).

    Returns:
      counts_global: (C_sar, nbins) int64
      n_valid_global: (C_sar,) int64
      pdf_mean_batches: (C_sar, nbins) float64
      pdf_std_batches:  (C_sar, nbins) float64
      n_batches_used: int
    """
    nbins = len(bin_edges) - 1
    bin_widths = np.diff(bin_edges).astype(np.float64)  # (nbins,)
    C_sar = len(sar_idx)

    counts_global = np.zeros((C_sar, nbins), dtype=np.int64)
    n_valid_global = np.zeros((C_sar,), dtype=np.int64)

    # Welford over per-batch PDFs
    pdf_mean, pdf_M2, pdf_n = _welford_init(C_sar, nbins)

    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break

        x = batch[key].to(device, non_blocking=True).float()  # (B,C,H,W)
        if x.ndim != 4:
            raise ValueError(f"Expected batch['{key}'] shape (B,C,H,W), got {tuple(x.shape)}")

        # Per-batch PDF for each SAR channel
        pdf_batch = np.zeros((C_sar, nbins), dtype=np.float64)

        for j, c in enumerate(sar_idx):
            vals = x[:, c, :, :].reshape(-1)
            finite = torch.isfinite(vals)
            vals = vals[finite]
            if vals.numel() == 0:
                continue

            vals_np = vals.detach().cpu().numpy()

            # batch histogram counts
            c_batch = np.histogram(vals_np, bins=bin_edges)[0].astype(np.int64)

            # update global
            counts_global[j] += c_batch
            n_valid_global[j] += vals_np.size

            # batch PDF (density): counts / (N * bin_width)
            N = c_batch.sum()
            if N > 0:
                pdf_batch[j] = c_batch.astype(np.float64) / (float(N) * bin_widths)

        # Update streaming mean/std of batch-wise PDFs.
        # Note: batches with no valid values for a channel contribute all-zeros for that channel.
        # If you want to ignore such batches per-channel, we can implement that, but this is usually fine.
        pdf_mean, pdf_M2, pdf_n = _welford_update(pdf_mean, pdf_M2, pdf_n, pdf_batch)

        if (b + 1) % 50 == 0:
            logger.info(f"Processed {b+1} batches...")

    pdf_mean, pdf_std, pdf_n = _welford_finalize(pdf_mean, pdf_M2, pdf_n)
    return counts_global, n_valid_global, pdf_mean, pdf_std, pdf_n


def hist_to_pdf_cdf(counts: np.ndarray, bin_edges: np.ndarray):
    """
    Convert global histogram counts into global PDF and CDF per channel.
    PDF integrates to 1 over the bin widths. CDF is cumulative PMF.
    """
    bin_widths = np.diff(bin_edges)  # (nbins,)
    N = counts.sum(axis=1, keepdims=True).astype(np.float64)  # (C,1)
    N_safe = np.where(N == 0, 1.0, N)
    pdf = counts.astype(np.float64) / (N_safe * bin_widths[None, :])

    pmf = counts.astype(np.float64) / N_safe
    cdf = np.cumsum(pmf, axis=1)
    return pdf, cdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, help="Path to index_train.jsonl (or split).")
    p.add_argument("--out", required=True, help="Output NPZ path, e.g. sar_hist_train.npz")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=None)

    p.add_argument("--sar-channels", nargs="+", default=["HV"], help="Which SAR channels to include: HH HV IA")
    p.add_argument("--sar-to-db", action="store_true", help="Convert HH/HV to dB in dataset.")
    p.add_argument("--sar-no-db", dest="sar_to_db", action="store_false", help="SAR already in dB.")
    p.set_defaults(sar_to_db=True)

    p.add_argument("--include-wspd", action="store_true", help="Doesn't matter for SAR hist; kept for parity.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    # histogram settings (dB)
    p.add_argument("--db-min", type=float, default=-40.0)
    p.add_argument("--db-max", type=float, default=0.0)
    p.add_argument("--db-step", type=float, default=0.25)

    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"

    sar_channels = [c.upper() for c in args.sar_channels]

    # IMPORTANT: no normalization; no SAR postprocess (no clip, no nan_to_num)
    ds = DriftWindSARDataset(
        args.index,
        include_wspd=args.include_wspd,
        return_meta=False,
        norm_yaml_path=None,
        normalize_y=False,
        sar_postprocess=False,     # critical for true distributions
        sar_channels=tuple(sar_channels),
        sar_to_db=bool(args.sar_to_db),
        cache_size=0,
    )

    if not hasattr(ds, "x_channels"):
        raise AttributeError("Dataset must expose x_channels so we can select SAR indices reliably.")
    x_channel_names = list(ds.x_channels)

    sar_idx = _pick_sar_indices(x_channel_names, sar_channels)
    if not sar_idx:
        raise RuntimeError(f"Could not find SAR channels {sar_channels} in ds.x_channels: {x_channel_names}")

    picked_names = [x_channel_names[i] for i in sar_idx]
    logger.info(f"Dataset length: {len(ds)}")
    logger.info(f"X channels: {x_channel_names}")
    logger.info(f"Selected SAR indices: {sar_idx}")
    logger.info(f"Selected SAR channels: {picked_names}")

    # bins
    # Use np.arange with a tiny epsilon to ensure inclusion when step doesn't divide range perfectly
    eps = 1e-12
    bin_edges = np.arange(args.db_min, args.db_max + args.db_step + eps, args.db_step, dtype=np.float64)
    if bin_edges.size < 2:
        raise ValueError("Invalid bin edges; check db-min/db-max/db-step.")
    # Ensure last edge is exactly db_max (optional but nice)
    bin_edges[-1] = float(args.db_max)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        persistent_workers=True if args.num_workers > 0 else False,
    )

    logger.info("Accumulating SAR histograms + batch-wise PDF stats...")
    counts, n_valid, pdf_mean_batches, pdf_std_batches, n_batches_used = compute_histograms_and_batch_pdf_stats(
        loader=loader,
        sar_idx=sar_idx,
        bin_edges=bin_edges,
        key="x",
        max_batches=args.max_batches,
        device=args.device,
    )

    pdf, cdf = hist_to_pdf_cdf(counts, bin_edges)

    meta = {
        "index_path": os.path.abspath(args.index),
        "dataset_len": len(ds),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_batches": args.max_batches,
        "device_used": args.device,
        "sar_channels_requested": sar_channels,
        "sar_channels_picked": [str(n) for n in picked_names],
        "sar_to_db": bool(args.sar_to_db),
        "db_min": float(args.db_min),
        "db_max": float(args.db_max),
        "db_step": float(args.db_step),
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "n_valid_pixels_per_channel": n_valid.tolist(),
        "n_batches_used_for_pdf_stats": int(n_batches_used),
        "pdf_band_definition": "mean/std of batch-wise PDFs (density) over bins; not physical backscatter std",
    }

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        channel_names=np.array([str(n) for n in picked_names], dtype=object),
        bin_edges_db=bin_edges,
        bin_centers_db=bin_centers,
        counts=counts,
        pdf=pdf,
        cdf=cdf,
        pdf_mean_across_batches=pdf_mean_batches,
        pdf_std_across_batches=pdf_std_batches,
        n_batches_used_for_pdf_stats=np.int64(n_batches_used),
        meta=np.array([meta], dtype=object),
    )

    logger.info(f"Saved SAR histogram summary to: {out_path}")


if __name__ == "__main__":
    main()