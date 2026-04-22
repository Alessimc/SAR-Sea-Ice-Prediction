#!/usr/bin/env python3
"""
Fit alpha and theta for the simple wind baseline

    u_ice = alpha * R(theta) * u_wind

using train / val / test index files directly, and group the fit by:
  - overall
  - calendar year
  - season (DJF, MAM, JJA, SON)
  - season-year (e.g. 2014_DJF, 2014_MAM, ...)

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
- This version is tailored to DriftWindSARDataset, where time comes from batch["t"].
- It uses x_groups=["wind"], so only wind inputs are loaded.
- It denormalizes x and y using --norm_yaml before fitting.

Example:
python find_alpha_theta_yearly_seasonally.py \
  --train_index /path/to/index_train.jsonl \
  --val_index /path/to/index_val.jsonl \
  --test_index /path/to/index_test.jsonl \
  --norm_yaml /path/to/norm_stats.yaml \
  --out /path/to/alpha_theta_grouped.json \
  --csv_out /path/to/alpha_theta_grouped.csv
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset


SEASON_ORDER = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}


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

        # Heuristic: infer unit
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

        # First try ISO parsing
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            fmts = [
                "%Y%m%dT%H%M",      # 20141107T0807
                "%Y%m%dT%H%M%S",    # 20141107T080700
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
        # Scalar fallback
        vals = [batch_t] * batch_size

    if len(vals) != batch_size:
        raise ValueError(
            f"batch['t'] length mismatch: got {len(vals)} values for batch_size={batch_size}"
        )

    return [parse_datetime_value(v) for v in vals]


def season_and_season_year(dt: datetime) -> Tuple[str, int]:
    """
    Meteorological seasons.
    December is assigned to the following DJF year:
      2020-12 -> DJF 2021
    """
    m = dt.month

    if m in (12, 1, 2):
        season = "DJF"
        season_year = dt.year + 1 if m == 12 else dt.year
    elif m in (3, 4, 5):
        season = "MAM"
        season_year = dt.year
    elif m in (6, 7, 8):
        season = "JJA"
        season_year = dt.year
    else:
        season = "SON"
        season_year = dt.year

    return season, season_year


@dataclass
class FitAccumulator:
    s: float = 0.0
    xty_A: float = 0.0
    xty_B: float = 0.0
    n_pixels: int = 0
    n_samples: int = 0

    def update(self, uw: np.ndarray, vw: np.ndarray, ui: np.ndarray, vi: np.ndarray) -> None:
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

        # XtX = s * I for this specific model
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


def make_store() -> Dict[str, Any]:
    return {
        "overall": FitAccumulator(),
        "per_year": defaultdict(FitAccumulator),
        "per_season": defaultdict(FitAccumulator),
        "per_year_season": defaultdict(FitAccumulator),
    }


def update_store(
    store: Dict[str, Any],
    dt: datetime,
    uw: np.ndarray,
    vw: np.ndarray,
    ui: np.ndarray,
    vi: np.ndarray,
) -> None:
    year_key = str(dt.year)
    season, season_year = season_and_season_year(dt)
    year_season_key = f"{season_year:04d}_{season}"

    store["overall"].update(uw, vw, ui, vi)
    store["per_year"][year_key].update(uw, vw, ui, vi)
    store["per_season"][season].update(uw, vw, ui, vi)
    store["per_year_season"][year_season_key].update(uw, vw, ui, vi)


def materialize_store(store: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    out["overall"] = store["overall"].finalize()

    per_year = {}
    for year in sorted(store["per_year"].keys(), key=int):
        fit = store["per_year"][year].finalize()
        if fit is not None:
            per_year[year] = fit
    out["per_year"] = per_year

    per_season = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        if season in store["per_season"]:
            fit = store["per_season"][season].finalize()
            if fit is not None:
                per_season[season] = fit
    out["per_season"] = per_season

    def ys_sort_key(k: str) -> Tuple[int, int]:
        year_str, season = k.split("_")
        return int(year_str), SEASON_ORDER[season]

    per_year_season = {}
    for key in sorted(store["per_year_season"].keys(), key=ys_sort_key):
        fit = store["per_year_season"][key].finalize()
        if fit is not None:
            per_year_season[key] = fit
    out["per_year_season"] = per_year_season

    return out


def flatten_results_for_csv(results: Dict[str, Any], split_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add_row(group_type: str, group_key: str, fit: Optional[Dict[str, Any]]) -> None:
        if fit is None:
            return

        row = {
            "split": split_name,
            "group_type": group_type,
            "group_key": group_key,
            "calendar_year": "",
            "season": "",
            "season_year": "",
            "n_samples_used": fit["n_samples_used"],
            "n_pixels_used": fit["n_pixels_used"],
            "A": fit["A"],
            "B": fit["B"],
            "alpha": fit["alpha"],
            "theta_rad": fit["theta_rad"],
            "theta_deg": fit["theta_deg"],
        }

        if group_type == "year":
            row["calendar_year"] = group_key
        elif group_type == "season":
            row["season"] = group_key
        elif group_type == "year_season":
            season_year, season = group_key.split("_")
            row["season_year"] = season_year
            row["season"] = season

        rows.append(row)

    add_row("overall", "all", results.get("overall"))

    for year, fit in results.get("per_year", {}).items():
        add_row("year", year, fit)

    for season, fit in results.get("per_season", {}).items():
        add_row("season", season, fit)

    for ys, fit in results.get("per_year_season", {}).items():
        add_row("year_season", ys, fit)

    return rows


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    ensure_parent_dir(path)
    if not rows:
        return

    fieldnames = [
        "split",
        "group_type",
        "group_key",
        "calendar_year",
        "season",
        "season_year",
        "n_samples_used",
        "n_pixels_used",
        "A",
        "B",
        "alpha",
        "theta_rad",
        "theta_deg",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(index_path: str, norm_yaml: str) -> DriftWindSARDataset:
    """
    Tailored to this baseline:
    - only wind inputs are needed
    - no SAR is loaded
    - no metadata needed, because time is already returned as top-level 't'
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
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )


@torch.no_grad()
def process_split(
    split_name: str,
    index_path: str,
    norm_yaml: str,
    batch_size: int,
    num_workers: int,
    global_store: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    print(f"\nProcessing split: {split_name}")
    print(f"  index: {index_path}")

    ds = build_dataset(index_path=index_path, norm_yaml=norm_yaml)
    loader = build_loader(ds=ds, batch_size=batch_size, num_workers=num_workers)

    x_ch = list(ds.x_channels)
    i_wu = x_ch.index("future_wind_u10_mean")
    i_wv = x_ch.index("future_wind_v10_mean")

    split_store = make_store()
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
            uw = x_np[i, i_wu].reshape(-1)
            vw = x_np[i, i_wv].reshape(-1)
            ui = y_np[i, 0].reshape(-1)
            vi = y_np[i, 1].reshape(-1)

            update_store(split_store, dt, uw, vw, ui, vi)
            update_store(global_store, dt, uw, vw, ui, vi)

        if ib == 0:
            example_t = batch["t"][0] if isinstance(batch["t"], (list, tuple)) else batch["t"]
            print(f"  example batch['t']: {example_t}")

        if (ib + 1) % 25 == 0 or (ib + 1) == n_batches:
            print(f"  batch {ib + 1}/{n_batches}")

    split_result = materialize_store(split_store)
    split_result["fit_dataset"] = index_path
    rows = flatten_results_for_csv(split_result, split_name)
    return split_result, rows


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_index", required=True, help="Path to train index JSONL")
    ap.add_argument("--val_index", required=True, help="Path to val index JSONL")
    ap.add_argument("--test_index", required=True, help="Path to test index JSONL")
    ap.add_argument("--norm_yaml", required=True, help="Normalization YAML used by the dataset")
    ap.add_argument("--out", required=True, help="Output JSON file")
    ap.add_argument("--csv_out", default=None, help="Optional flat CSV output")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    split_paths = {
        "train": args.train_index,
        "val": args.val_index,
        "test": args.test_index,
    }

    global_store = make_store()
    per_split_results: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    for split_name, index_path in split_paths.items():
        split_result, rows = process_split(
            split_name=split_name,
            index_path=index_path,
            norm_yaml=args.norm_yaml,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            global_store=global_store,
        )
        per_split_results[split_name] = split_result
        csv_rows.extend(rows)

    combined_results = materialize_store(global_store)
    csv_rows.extend(flatten_results_for_csv(combined_results, "all"))

    out = {
        "time_source": "batch['t']",
        "season_definition": {
            "type": "meteorological",
            "seasons": ["DJF", "MAM", "JJA", "SON"],
            "djf_rule": "December is assigned to the following season-year",
            "example": "2020-12-15 -> DJF 2021",
        },
        "splits_loaded": split_paths,
        "combined": combined_results,
        "per_split": per_split_results,
        "model_equations": {
            "u_ice": "A * u_wind - B * v_wind",
            "v_ice": "B * u_wind + A * v_wind",
        },
    }

    ensure_parent_dir(args.out)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    if args.csv_out is not None:
        write_csv(csv_rows, args.csv_out)

    print("\nSaved JSON:")
    print(args.out)

    if args.csv_out is not None:
        print("Saved CSV:")
        print(args.csv_out)

    print("\nCombined yearly fits:")
    for year, fit in out["combined"]["per_year"].items():
        print(
            f"  {year}: alpha={fit['alpha']:.6f}, "
            f"theta_deg={fit['theta_deg']:.3f}, "
            f"n_pixels={fit['n_pixels_used']}"
        )

    print("\nCombined season-year fits:")
    for ys, fit in out["combined"]["per_year_season"].items():
        print(
            f"  {ys}: alpha={fit['alpha']:.6f}, "
            f"theta_deg={fit['theta_deg']:.3f}, "
            f"n_pixels={fit['n_pixels_used']}"
        )


if __name__ == "__main__":
    main()