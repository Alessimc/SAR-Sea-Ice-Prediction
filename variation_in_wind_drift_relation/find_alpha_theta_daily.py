#!/usr/bin/env python3
"""
Fit daily alpha and theta for the simple wind baseline

    u_ice = alpha * R(theta) * u_wind

using train / val / test index files directly.

This version computes one alpha/theta value per calendar day, separately for:
  - train
  - val
  - test

Model:
    u_ice = A * u_wind - B * v_wind
    v_ice = B * u_wind + A * v_wind

with
    A = alpha * cos(theta)
    B = alpha * sin(theta)

so that
    alpha = sqrt(A^2 + B^2)
    theta = atan2(B, A)

Important:
- Tailored to DriftWindSARDataset, where time comes from batch["t"].
- Uses x_groups=["wind"], so only wind inputs are loaded.
- Denormalizes x and y using --norm_yaml before fitting.
- Outputs one CSV per split, sorted by date.

Example:
python find_alpha_theta_daily.py \
  --train_index /path/to/index_train.jsonl \
  --val_index /path/to/index_val.jsonl \
  --test_index /path/to/index_test.jsonl \
  --norm_yaml /path/to/norm_stats.yaml \
  --out_dir /path/to/daily_alpha_theta_csvs
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def denorm_y(y_norm: torch.Tensor, ds: DriftWindSARDataset) -> torch.Tensor:
    if not getattr(ds, "do_norm", False) or not getattr(ds, "normalize_y", True):
        return y_norm
    return y_norm * ds.y_std.to(y_norm.device) + ds.y_mean.to(y_norm.device)


def denorm_x(x_norm: torch.Tensor, ds: DriftWindSARDataset) -> torch.Tensor:
    if not getattr(ds, "do_norm", False):
        return x_norm
    return x_norm * ds.x_std.to(x_norm.device) + ds.x_mean.to(x_norm.device)


def parse_datetime_value(v: Any) -> datetime:
    """
    Parse one timestamp value from batch["t"].

    Supports:
      - python datetime
      - numpy datetime64
      - unix timestamps in s / ms / us / ns
      - strings like:
          20141107T0807
          20141107T080700
          2014-11-07T08:07
          2014-11-07 08:07:00
          2014-11-07

    Returns naive UTC datetime.
    """
    if v is None:
        raise ValueError("Timestamp is None")

    if isinstance(v, datetime):
        dt = v

    elif isinstance(v, np.datetime64):
        s = np.datetime_as_string(v, unit="s")
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))

    elif isinstance(v, (np.integer, int, np.floating, float)):
        vv = float(v)
        av = abs(vv)

        # Heuristic: infer timestamp unit.
        if av > 1e17:        # ns
            vv /= 1e9
        elif av > 1e14:      # us
            vv /= 1e6
        elif av > 1e11:      # ms
            vv /= 1e3
        # else assume seconds

        dt = datetime.fromtimestamp(vv, tz=timezone.utc)

    elif isinstance(v, bytes):
        return parse_datetime_value(v.decode("utf-8"))

    elif isinstance(v, str):
        s = v.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            fmts = [
                "%Y%m%dT%H%M",
                "%Y%m%dT%H%M%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M",
                "%Y-%m-%d",
                "%Y%m%d%H%M%S",
                "%Y%m%d%H%M",
                "%Y%m%d",
            ]

            dt = None
            for fmt in fmts:
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except ValueError:
                    continue

            if dt is None:
                raise ValueError(f"Could not parse timestamp: {v!r}")

    else:
        raise TypeError(f"Unsupported timestamp type: {type(v)}")

    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

    return dt


def batch_t_to_datetimes(batch_t: Any, batch_size: int) -> List[datetime]:
    """
    Convert batch["t"] into a list of datetimes, one per sample.
    Works with the default DataLoader collate behavior.
    """
    if isinstance(batch_t, torch.Tensor):
        if batch_t.ndim == 0:
            vals = [batch_t.item()] * batch_size
        else:
            vals = batch_t.cpu().tolist()

    elif isinstance(batch_t, np.ndarray):
        if batch_t.ndim == 0:
            vals = [batch_t.item()] * batch_size
        else:
            vals = batch_t.tolist()

    elif isinstance(batch_t, (list, tuple)):
        vals = list(batch_t)

    else:
        vals = [batch_t] * batch_size

    if len(vals) != batch_size:
        raise ValueError(
            f"batch['t'] length mismatch: got {len(vals)} values for batch_size={batch_size}"
        )

    return [parse_datetime_value(v) for v in vals]


@dataclass
class FitAccumulator:
    s: float = 0.0
    xty_A: float = 0.0
    xty_B: float = 0.0
    n_pixels: int = 0
    n_samples: int = 0

    def update(self, uw: np.ndarray, vw: np.ndarray, ui: np.ndarray, vi: np.ndarray) -> None:
        """
        Accumulate sufficient statistics for the constrained least-squares fit.
        """
        m = np.isfinite(uw) & np.isfinite(vw) & np.isfinite(ui) & np.isfinite(vi)
        if not np.any(m):
            return

        uw = uw[m].astype(np.float64, copy=False)
        vw = vw[m].astype(np.float64, copy=False)
        ui = ui[m].astype(np.float64, copy=False)
        vi = vi[m].astype(np.float64, copy=False)

        s_here = np.sum(uw * uw + vw * vw)
        if not np.isfinite(s_here) or s_here <= 0.0:
            return

        self.s += float(s_here)
        self.xty_A += float(np.sum(uw * ui + vw * vi))
        self.xty_B += float(np.sum(-vw * ui + uw * vi))
        self.n_pixels += int(uw.size)
        self.n_samples += 1

    def finalize(self) -> Optional[Dict[str, Any]]:
        if self.n_pixels == 0 or self.s <= 0.0:
            return None

        A = self.xty_A / self.s
        B = self.xty_B / self.s

        alpha = math.hypot(A, B)
        theta_rad = math.atan2(B, A)
        theta_deg = math.degrees(theta_rad)

        return {
            "n_samples_used": int(self.n_samples),
            "n_pixels_used": int(self.n_pixels),
            "A": float(A),
            "B": float(B),
            "alpha": float(alpha),
            "theta_rad": float(theta_rad),
            "theta_deg": float(theta_deg),
        }


def build_dataset(index_path: str, norm_yaml: str) -> DriftWindSARDataset:
    """
    Tailored to this baseline:
    - only wind inputs are needed
    - no SAR input is loaded
    - no metadata needed, because time is returned as top-level batch["t"]
    """
    return DriftWindSARDataset(
        index_jsonl=index_path,
        include_wspd=False,
        return_meta=False,
        norm_yaml_path=norm_yaml,
        normalize_y=True,
        cache_size=0,
        x_groups=["wind"],
    )


def build_loader(ds: DriftWindSARDataset, batch_size: int, num_workers: int) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    if num_workers > 0:
        kwargs["prefetch_factor"] = 4

    return DataLoader(ds, **kwargs)


def write_daily_csv(
    daily_store: Dict[str, FitAccumulator],
    split_name: str,
    csv_path: str,
) -> None:
    """
    Write one row per day.

    The resulting CSV is directly convenient for plotting:

        date, alpha
        date, theta_deg
    """
    ensure_parent_dir(csv_path)

    fieldnames = [
        "split",
        "date",
        "year",
        "month",
        "day",
        "n_samples_used",
        "n_pixels_used",
        "A",
        "B",
        "alpha",
        "theta_rad",
        "theta_deg",
    ]

    rows: List[Dict[str, Any]] = []

    for date_key in sorted(daily_store.keys()):
        fit = daily_store[date_key].finalize()
        if fit is None:
            continue

        dt = datetime.strptime(date_key, "%Y-%m-%d")

        rows.append(
            {
                "split": split_name,
                "date": date_key,
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "n_samples_used": fit["n_samples_used"],
                "n_pixels_used": fit["n_pixels_used"],
                "A": fit["A"],
                "B": fit["B"],
                "alpha": fit["alpha"],
                "theta_rad": fit["theta_rad"],
                "theta_deg": fit["theta_deg"],
            }
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} daily fits to:")
    print(f"  {csv_path}")


@torch.no_grad()
def process_split(
    split_name: str,
    index_path: str,
    norm_yaml: str,
    batch_size: int,
    num_workers: int,
    csv_path: str,
) -> None:
    print(f"\nProcessing split: {split_name}")
    print(f"  index: {index_path}")

    ds = build_dataset(index_path=index_path, norm_yaml=norm_yaml)
    loader = build_loader(ds=ds, batch_size=batch_size, num_workers=num_workers)

    x_ch = list(ds.x_channels)

    required_channels = ["future_wind_u10_mean", "future_wind_v10_mean"]
    missing = [ch for ch in required_channels if ch not in x_ch]
    if missing:
        raise ValueError(
            f"Missing required wind channels in dataset x_channels: {missing}. "
            f"Available x_channels: {x_ch}"
        )

    i_wu = x_ch.index("future_wind_u10_mean")
    i_wv = x_ch.index("future_wind_v10_mean")

    daily_store: Dict[str, FitAccumulator] = defaultdict(FitAccumulator)

    n_batches = len(loader)

    for ib, batch in enumerate(loader):
        x = batch["x"].float()
        y = batch["y"].float()

        x = denorm_x(x, ds)
        y = denorm_y(y, ds)

        this_bs = x.shape[0]

        if "t" not in batch:
            raise KeyError(
                f"Batch does not contain key 't'. Available keys: {list(batch.keys())}"
            )

        dts = batch_t_to_datetimes(batch["t"], this_bs)

        x_np = x.double().cpu().numpy()
        y_np = y.double().cpu().numpy()

        for i, dt in enumerate(dts):
            date_key = dt.date().isoformat()

            uw = x_np[i, i_wu].reshape(-1)
            vw = x_np[i, i_wv].reshape(-1)
            ui = y_np[i, 0].reshape(-1)
            vi = y_np[i, 1].reshape(-1)

            daily_store[date_key].update(uw, vw, ui, vi)

        if ib == 0:
            if isinstance(batch["t"], (list, tuple)):
                example_t = batch["t"][0]
            elif isinstance(batch["t"], torch.Tensor) and batch["t"].ndim > 0:
                example_t = batch["t"][0]
            else:
                example_t = batch["t"]

            print(f"  example batch['t']: {example_t}")

        if (ib + 1) % 25 == 0 or (ib + 1) == n_batches:
            print(f"  batch {ib + 1}/{n_batches}")

    write_daily_csv(
        daily_store=daily_store,
        split_name=split_name,
        csv_path=csv_path,
    )


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_index", required=True, help="Path to train index JSONL")
    ap.add_argument("--val_index", required=True, help="Path to val index JSONL")
    ap.add_argument("--test_index", required=True, help="Path to test index JSONL")
    ap.add_argument("--norm_yaml", required=True, help="Normalization YAML used by the dataset")
    ap.add_argument("--out_dir", required=True, help="Directory where daily CSV files are written")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    split_paths = {
        "train": args.train_index,
        "val": args.val_index,
        "test": args.test_index,
    }

    for split_name, index_path in split_paths.items():
        csv_path = os.path.join(args.out_dir, f"{split_name}_daily_alpha_theta.csv")

        process_split(
            split_name=split_name,
            index_path=index_path,
            norm_yaml=args.norm_yaml,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            csv_path=csv_path,
        )

    print("\nDone. Daily alpha/theta CSV files are in:")
    print(args.out_dir)


if __name__ == "__main__":
    main()