#!/usr/bin/env python3
"""
Compare a wind baseline against one ML drift model, saving two figures per sample.

Figure A: drift speed + SAR
---------------------------
Row 1:
  [0,0] Target drift speed + vectors
  [0,1] Wind baseline drift speed + vectors
  [0,2] ML drift speed + vectors
  [0,3] Shared colorbar for drift speed

Row 2:
  [1,0] Observed SAR at end time
  [1,1] Start SAR warped by wind baseline
  [1,2] Start SAR warped by ML drift
  [1,3] Shared colorbar for SAR

Figure B: divergence + shear
----------------------------
Row 1:
  [0,0] Target divergence
  [0,1] Wind baseline divergence
  [0,2] ML divergence
  [0,3] Shared colorbar for divergence

Row 2:
  [1,0] Target shear
  [1,1] Wind baseline shear
  [1,2] ML shear
  [1,3] Shared colorbar for shear

Notes
-----
- The start SAR comes from the dataset input channels.
- The end SAR comes from:
      index_jsonl -> future_drift_path -> npz["meta"].item()["end_path"]
- The drift lead time comes from:
      npz["meta"].item()["dt_seconds"]
- TIFF band index: 0 = HH, 1 = HV
- Divergence and shear are computed on a triangulation from a subsampled grid.
- Divergence and shear are converted from 1/s to 1/day.
"""

import json
import os
import argparse
import random
import warnings
import importlib
from functools import lru_cache
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, Normalize
from matplotlib.tri import Triangulation
from scipy.ndimage import map_coordinates
import yaml

from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset


SECONDS_PER_DAY = 86400.0
TEXTWIDTH_PT = 418.25368
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27

KM_DAY_CONVERSION_FACTOR = SECONDS_PER_DAY / 1000.0

def setup_pub_style(fontsize=9):
    mpl.rcParams.update({
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def fig_textwidth(height_ratio=0.5):
    return (TEXTWIDTH_IN, TEXTWIDTH_IN * height_ratio)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


@lru_cache(maxsize=8)
def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_norm_yaml(norm_yaml_path: str) -> Tuple[Dict, Dict]:
    cfg = load_yaml(norm_yaml_path)
    return cfg.get("inputs", {}), cfg.get("targets", {})


def sanitize_filename(s: str) -> str:
    s = str(s)
    for bad in ["/", "\\", ":", " ", "(", ")", "[", "]", "{", "}", ",", ";"]:
        s = s.replace(bad, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def normalize_sar_clip_bounds(bounds: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    if bounds is None:
        return {}
    out = {}
    for k, v in bounds.items():
        out[k] = (float(v[0]), float(v[1]))
    return out


def pick_cfg(model_cfg: Dict[str, Any], defaults: Dict[str, Any], ckpt_cfg: Dict[str, Any], key: str, default=None):
    if key in model_cfg and model_cfg[key] is not None:
        return model_cfg[key]
    if key in defaults and defaults[key] is not None:
        return defaults[key]
    if ckpt_cfg is not None and key in ckpt_cfg and ckpt_cfg[key] is not None:
        return ckpt_cfg[key]
    return default


def import_model(module: str, class_name: str):
    m = importlib.import_module(module)
    if not hasattr(m, class_name):
        raise AttributeError(f"Module '{module}' has no class '{class_name}'")
    return getattr(m, class_name)


def infer_in_ch_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
    for k, v in state_dict.items():
        if torch.is_tensor(v) and v.ndim == 4 and k.endswith("weight"):
            return int(v.shape[1])
    raise KeyError("Could not infer input channels from checkpoint state_dict.")


def resolve_channel_name(x_channels: List[str], requested: str) -> Optional[str]:
    if requested in x_channels:
        return requested

    lower_map = {c.lower(): c for c in x_channels}
    if requested.lower() in lower_map:
        return lower_map[requested.lower()]

    aliases = {
        "future_wind_u10_mean": ["future_wind_u10_mean", "wind_u10", "u10", "u_wind", "wind_u"],
        "future_wind_v10_mean": ["future_wind_v10_mean", "wind_v10", "v10", "v_wind", "wind_v"],
        "sar_hh_db": ["sar_hh_db", "sar_hh", "HH", "hh"],
        "sar_hv_db": ["sar_hv_db", "sar_hv", "HV", "hv"],
    }

    candidates = aliases.get(requested, [requested])
    for cand in candidates:
        if cand in x_channels:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    return None


def resolve_required_channel(x_channels: List[str], requested: str) -> str:
    out = resolve_channel_name(x_channels, requested)
    if out is None:
        raise KeyError(f"Channel '{requested}' not found in x_channels={x_channels}")
    return out


def resolve_input_channel_selection(ds, model_cfg: Dict[str, Any], defaults: Dict[str, Any],
                                    ckpt_cfg: Dict[str, Any], ckpt_state: Dict[str, torch.Tensor]):
    x_channels = list(ds.x_channels)

    requested_names = model_cfg.get("input_channel_names", defaults.get("input_channel_names", None))
    if isinstance(requested_names, str):
        requested_names = [c.strip() for c in requested_names.split(",") if c.strip()]

    expected_in_ch = pick_cfg(model_cfg, defaults, ckpt_cfg, "in_ch", None)
    if expected_in_ch is None:
        expected_in_ch = infer_in_ch_from_state_dict(ckpt_state)
    expected_in_ch = int(expected_in_ch)

    if requested_names is not None:
        indices = [x_channels.index(c) for c in requested_names]
        names = list(requested_names)
    else:
        if expected_in_ch == len(x_channels):
            indices = list(range(len(x_channels)))
            names = list(x_channels)
        elif expected_in_ch == 2:
            names = ["future_wind_u10_mean", "future_wind_v10_mean"]
            indices = [x_channels.index(c) for c in names]
        else:
            raise ValueError(
                f"Checkpoint expects in_ch={expected_in_ch}, dataset has {len(x_channels)} channels, "
                "and no input_channel_names were specified."
            )

    if len(indices) != expected_in_ch:
        raise ValueError(f"Selected {len(indices)} channels, checkpoint expects {expected_in_ch}")

    return indices, names, expected_in_ch, x_channels


def denorm_y(y: torch.Tensor, targets_stats: Dict, normalize_y: bool) -> torch.Tensor:
    if not normalize_y:
        return y

    y_mean = torch.tensor(
        [targets_stats["future_drift_u"]["mean"], targets_stats["future_drift_v"]["mean"]],
        dtype=torch.float32,
        device=y.device,
    ).view(-1, 1, 1)

    y_std = torch.tensor(
        [targets_stats["future_drift_u"]["std"], targets_stats["future_drift_v"]["std"]],
        dtype=torch.float32,
        device=y.device,
    ).view(-1, 1, 1)

    return y * y_std + y_mean


def denorm_x_channel(x: torch.Tensor, x_channels: List[str], inputs_stats: Dict, ch_name: str) -> torch.Tensor:
    ch_idx = x_channels.index(ch_name)
    x_ch = x[ch_idx]

    if ch_name not in inputs_stats:
        return x_ch

    mean = float(inputs_stats[ch_name]["mean"])
    std = float(inputs_stats[ch_name]["std"])
    return x_ch * std + mean


def robust_sym_limits(arrays: List[np.ndarray], q: float = 0.99, min_v: float = 1e-12) -> float:
    vals = []
    for a in arrays:
        c = a.compressed() if isinstance(a, np.ma.MaskedArray) else np.asarray(a)
        c = c[np.isfinite(c)]
        if c.size:
            vals.append(np.abs(c))
    if not vals:
        return 1.0
    vals = np.concatenate(vals)
    vmax = float(np.quantile(vals, q))
    return vmax if np.isfinite(vmax) and vmax >= min_v else 1.0


def robust_abs_quantile(arrays: List[np.ndarray], q: float, min_v: float = 1e-12) -> float:
    vals = []
    for a in arrays:
        c = a.compressed() if isinstance(a, np.ma.MaskedArray) else np.asarray(a)
        c = c[np.isfinite(c)]
        if c.size:
            vals.append(np.abs(c))
    if not vals:
        return 1.0
    vals = np.concatenate(vals)
    out = float(np.quantile(vals, q))
    return out if np.isfinite(out) and out >= min_v else min_v


def robust_upper_quantile(arrays: List[np.ndarray], q: float = 0.99, min_v: float = 1e-12) -> float:
    vals = []
    for a in arrays:
        c = np.asarray(a)
        c = c[np.isfinite(c)]
        if c.size:
            vals.append(c)
    if not vals:
        return 1.0
    vals = np.concatenate(vals)
    out = float(np.quantile(vals, q))
    return out if np.isfinite(out) and out >= min_v else max(min_v, 1.0)


def robust_image_limits_multi(images: List[np.ndarray], q_low: float = 0.02, q_high: float = 0.98) -> Tuple[float, float]:
    vals = []
    for img in images:
        c = np.asarray(img)
        c = c[np.isfinite(c)]
        if c.size:
            vals.append(c)
    if not vals:
        return 0.0, 1.0

    vals = np.concatenate(vals)
    vmin = float(np.quantile(vals, q_low))
    vmax = float(np.quantile(vals, q_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if vmin == vmax:
            vmax = vmin + 1.0
    return vmin, vmax


# -----------------------------
# deformation / divergence / shear
# -----------------------------
def get_deformation_elems(x, y, u, v, a):
    ux = uy = vx = vy = 0.0
    for i0, i1 in zip([1, 2, 0], [0, 1, 2]):
        ux += (u[i0] + u[i1]) * (y[i0] - y[i1])
        uy -= (u[i0] + u[i1]) * (x[i0] - x[i1])
        vx += (v[i0] + v[i1]) * (y[i0] - y[i1])
        vy -= (v[i0] + v[i1]) * (x[i0] - x[i1])

    with np.errstate(divide="ignore", invalid="ignore"):
        ux, uy, vx, vy = [i / (2.0 * a) for i in (ux, uy, vx, vy)]

    e1 = ux + vy
    e2 = np.sqrt((ux - vy) ** 2 + (uy + vx) ** 2)
    e3 = vx - uy
    return e1, e2, e3


def get_deformation_on_triangulation(x, y, u, v, t):
    xt, yt, ut, vt = [i[t].T for i in (x, y, u, v)]

    tri_x = np.diff(np.vstack([xt, xt[0]]), axis=0)
    tri_y = np.diff(np.vstack([yt, yt[0]]), axis=0)
    tri_s = np.hypot(tri_x, tri_y)

    tri_p = np.sum(tri_s, axis=0)
    s = tri_p / 2.0

    with np.errstate(invalid="ignore"):
        tri_a = np.sqrt(np.maximum(s * (s - tri_s[0]) * (s - tri_s[1]) * (s - tri_s[2]), 0.0))

    e1, e2, e3 = get_deformation_elems(xt, yt, ut, vt, tri_a)
    return e1, e2, e3, tri_a, tri_p


def make_subsampled_triangulation(h: int, w: int, dx: float, dy: float, stride: int):
    rows = np.arange(0, h, stride, dtype=np.int64)
    cols = np.arange(0, w, stride, dtype=np.int64)

    if rows[-1] != h - 1:
        rows = np.append(rows, h - 1)
    if cols[-1] != w - 1:
        cols = np.append(cols, w - 1)

    xx, yy = np.meshgrid(cols.astype(np.float64) * dx, rows.astype(np.float64) * dy)
    x = xx.ravel()
    y = yy.ravel()
    tri = Triangulation(x, y)

    extent = (-0.5 * dx, (w - 0.5) * dx, (h - 0.5) * dy, -0.5 * dy)

    return {
        "rows": rows,
        "cols": cols,
        "x": x,
        "y": y,
        "triangles": tri.triangles,
        "extent": extent,
    }


def deformation_from_triangulation(u: np.ndarray, v: np.ndarray, tri_info: Dict[str, Any]):
    u_sub = u[np.ix_(tri_info["rows"], tri_info["cols"])].astype(np.float64).ravel()
    v_sub = v[np.ix_(tri_info["rows"], tri_info["cols"])].astype(np.float64).ravel()

    e1, e2, e3, tri_a, tri_p = get_deformation_on_triangulation(
        tri_info["x"], tri_info["y"], u_sub, v_sub, tri_info["triangles"]
    )

    e1 = e1 * SECONDS_PER_DAY
    e2 = e2 * SECONDS_PER_DAY
    e3 = e3 * SECONDS_PER_DAY

    mask = (~np.isfinite(e1)) | (~np.isfinite(e2)) | (~np.isfinite(tri_a)) | (tri_a <= 0)
    div = np.ma.masked_array(e1, mask=mask)
    shear = np.ma.masked_array(e2, mask=mask)
    vort = np.ma.masked_array(e3, mask=mask)
    return div, shear, vort


# -----------------------------
# SAR loading / preprocessing
# -----------------------------
def resolve_sar_tiff_channel_index(requested: str) -> int:
    s = requested.lower()
    if "hh" in s:
        return 0
    if "hv" in s:
        return 1
    raise ValueError(f"Only HH/HV TIFF loading is supported, got sar_channel={requested}")


def load_tiff_band(path: str, band_index_0based: int) -> np.ndarray:
    try:
        import tifffile
        arr = tifffile.imread(path)
        arr = np.asarray(arr)

        if arr.ndim == 2:
            if band_index_0based != 0:
                raise ValueError(f"Single-band TIFF at {path}, requested band {band_index_0based}")
            return arr.astype(np.float32)

        if arr.ndim == 3:
            if arr.shape[0] <= 4 and arr.shape[1] > 16 and arr.shape[2] > 16:
                return arr[band_index_0based].astype(np.float32)
            if arr.shape[-1] <= 4 and arr.shape[0] > 16 and arr.shape[1] > 16:
                return arr[..., band_index_0based].astype(np.float32)

        raise ValueError(f"Unsupported TIFF shape {arr.shape} for {path}")

    except Exception:
        import rasterio
        with rasterio.open(path) as src:
            return src.read(band_index_0based + 1).astype(np.float32)


def preprocess_sar_band(
    img: np.ndarray,
    channel_name: str,
    sar_to_db: bool,
    sar_zero_is_nodata: bool,
    sar_clip_db: bool,
    sar_clip_db_bounds: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    out = np.asarray(img, dtype=np.float32).copy()

    if sar_zero_is_nodata:
        out[out == 0] = np.nan

    if sar_to_db:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = 10.0 * np.log10(out)

    if sar_clip_db:
        key = "HH" if "hh" in channel_name.lower() else "HV"
        if key in sar_clip_db_bounds:
            lo, hi = sar_clip_db_bounds[key]
            out = np.clip(out, lo, hi)

    return out


def get_future_npz_meta(bundle: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    rec = bundle["records"][sample_idx]
    future_drift_path = rec["future_drift_path"]

    with np.load(future_drift_path, allow_pickle=True) as npz:
        meta = npz["meta"].item()

    if "end_path" not in meta or "dt_seconds" not in meta:
        raise KeyError(f"future_drift meta missing end_path or dt_seconds in {future_drift_path}")

    return meta


def extract_start_sar_image(bundle: Dict[str, Any], sample_idx: int, sar_channel: str) -> np.ndarray:
    s = bundle["dataset"][sample_idx]
    x_full = s["x"].float()

    resolved_name = resolve_required_channel(bundle["x_channels"], sar_channel)
    img = denorm_x_channel(
        x_full,
        bundle["x_channels"],
        bundle["inputs_stats"],
        resolved_name,
    )
    return img.cpu().numpy()


def extract_end_sar_image(bundle: Dict[str, Any], sample_idx: int, sar_channel: str) -> np.ndarray:
    meta = get_future_npz_meta(bundle, sample_idx)
    end_path = meta["end_path"]

    band_idx = resolve_sar_tiff_channel_index(sar_channel)
    raw = load_tiff_band(end_path, band_idx)

    return preprocess_sar_band(
        raw,
        channel_name=sar_channel,
        sar_to_db=bundle["sar_to_db"],
        sar_zero_is_nodata=bundle["sar_zero_is_nodata"],
        sar_clip_db=bundle["sar_clip_db"],
        sar_clip_db_bounds=bundle["sar_clip_db_bounds"],
    )


# -----------------------------
# warp
# -----------------------------
def warp_with_forward_flow(img, u, v, n_iter=8, order=1, mode="constant", cval=np.nan):
    rows, cols = img.shape
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")

    r = rr.astype(np.float64)
    c = cc.astype(np.float64)

    for _ in range(n_iter):
        v_rc = map_coordinates(v, [r, c], order=1, mode="nearest")
        u_rc = map_coordinates(u, [r, c], order=1, mode="nearest")
        r = rr - v_rc
        c = cc - u_rc

    valid = np.isfinite(img).astype(np.float64)
    img_filled = np.where(np.isfinite(img), img, 0.0).astype(np.float64)

    warped_num = map_coordinates(img_filled, [r, c], order=order, mode=mode, cval=0.0)
    warped_den = map_coordinates(valid, [r, c], order=0, mode=mode, cval=0.0)

    return np.where(warped_den > 0.5, warped_num, cval)


# -----------------------------
# plotting helpers
# -----------------------------
def _subsample_indices(n: int, stride: int) -> np.ndarray:
    idx = np.arange(0, n, stride, dtype=np.int64)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def plot_vector_field_with_bg(
    ax,
    u: np.ndarray,
    v: np.ndarray,
    tri_info: Dict[str, Any],
    title: str,
    dx: float,
    dy: float,
    stride: int,
    quiver_scale: float,
    bg_vmin: float,
    bg_vmax: float,
    cmap: str = "viridis",
    vector_color: str = "white",
    key_value: Optional[float] = None,
):
    
    u_plot = u * KM_DAY_CONVERSION_FACTOR
    v_plot = v * KM_DAY_CONVERSION_FACTOR
    speed = np.hypot(u_plot, v_plot)

    xmin, xmax, ymax, ymin = tri_info["extent"]
    im = ax.imshow(
        speed,
        cmap=cmap,
        origin="upper",
        interpolation="nearest",
        vmin=bg_vmin,
        vmax=bg_vmax,
        extent=(xmin, xmax, ymax, ymin),
        aspect="auto",
        zorder=1,
    )

    h, w = u.shape
    rows = _subsample_indices(h, stride)
    cols = _subsample_indices(w, stride)

    xx, yy = np.meshgrid(cols.astype(np.float64) * dx, rows.astype(np.float64) * dy)
    uu = u_plot[np.ix_(rows, cols)]
    vv = v_plot[np.ix_(rows, cols)]

    q = ax.quiver(
        xx, yy, uu, vv,
        angles="xy",
        scale_units="width",
        scale=quiver_scale,
        color=vector_color,
        pivot="mid",
        width=0.0045,
        headwidth=3.5,
        headlength=4.6,
        headaxislength=4.1,
        alpha=0.95,
        zorder=2,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("auto")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if key_value is not None and np.isfinite(key_value) and key_value > 0:
        ax.quiverkey(
            q,
            X=0.80,
            Y=1.04,
            U=key_value,
            label=f"{key_value:.2f} m/s",
            labelpos="E",
            coordinates="axes",
            color=vector_color,
        )

    return im


def plot_tripcolor_faces(ax, tri_info: Dict[str, Any], field_faces: np.ma.MaskedArray,
                         title: str, cmap: str, norm):
    tri_plot = Triangulation(
        tri_info["x"],
        tri_info["y"],
        triangles=tri_info["triangles"],
        mask=np.ma.getmaskarray(field_faces),
    )

    im = ax.tripcolor(
        tri_plot,
        facecolors=np.asarray(field_faces.filled(np.nan), dtype=float),
        cmap=cmap,
        norm=norm,
        shading="flat",
    )

    xmin, xmax, ymax, ymin = tri_info["extent"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("auto")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_sar(ax, sar_img: np.ndarray, tri_info: Dict[str, Any], title: str,
             vmin: float, vmax: float, cmap: str = "gray"):
    xmin, xmax, ymax, ymin = tri_info["extent"]
    im = ax.imshow(
        sar_img,
        cmap=cmap,
        origin="upper",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        extent=(xmin, xmax, ymax, ymin),
        aspect="auto",
    )
    ax.set_title(title)
    grid_step = 25000.0  # 25 km in meters
    ax.set_xticks(np.arange(0, xmax, grid_step), minor=True)
    ax.set_yticks(np.arange(0, ymax, grid_step), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, alpha=1.0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


# -----------------------------
# model / prediction
# -----------------------------
def build_model_bundle(compare_yaml: str):
    cfg = load_yaml(compare_yaml)
    defaults = cfg.get("defaults", {}) or {}
    models = cfg.get("models", [])
    if not models:
        raise ValueError("compare YAML must contain a non-empty models list")

    model_cfg = models[0]
    label = model_cfg.get("label", "ML model")

    ckpt_path = model_cfg.get("ckpt")
    test_index = model_cfg.get("test_index")
    norm_yaml = model_cfg.get("norm_yaml")
    model_module = model_cfg.get("model_module", defaults.get("model_module"))
    model_class = model_cfg.get("model_class", defaults.get("model_class"))

    missing = [k for k, v in {
        "ckpt": ckpt_path,
        "test_index": test_index,
        "norm_yaml": norm_yaml,
        "model_module": model_module,
        "model_class": model_class,
    }.items() if v is None]
    if missing:
        raise ValueError(f"Model '{label}' is missing: {missing}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {}) or {}

    include_wspd = pick_cfg(model_cfg, defaults, ckpt_cfg, "include_wspd", False)
    normalize_y = pick_cfg(model_cfg, defaults, ckpt_cfg, "normalize_y", True)
    base_channels = pick_cfg(model_cfg, defaults, ckpt_cfg, "base_channels", 32)
    out_ch = pick_cfg(model_cfg, defaults, ckpt_cfg, "out_ch", 2)

    sar_channels = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_channels", ["HH", "HV", "IA"])
    sar_to_db = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_to_db", True)
    sar_postprocess = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_postprocess", True)
    sar_zero_is_nodata = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_zero_is_nodata", False)
    sar_clip_db = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_clip_db", True)
    sar_clip_db_bounds = normalize_sar_clip_bounds(
        pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_clip_db_bounds", {})
    )

    if isinstance(sar_channels, str):
        sar_channels = [c.strip() for c in sar_channels.split(",") if c.strip()]

    inputs_stats, targets_stats = load_norm_yaml(norm_yaml)

    ds = DriftWindSARDataset(
        test_index,
        norm_yaml_path=norm_yaml,
        normalize_y=normalize_y,
        include_wspd=include_wspd,
        return_meta=False,
        cache_size=0,
        sar_channels=tuple(sar_channels),
        sar_to_db=sar_to_db,
        sar_postprocess=sar_postprocess,
        sar_clip_percentiles=None,
        sar_zero_is_nodata=sar_zero_is_nodata,
        sar_clip_db=sar_clip_db,
        sar_clip_db_bounds=sar_clip_db_bounds,
    )

    input_channel_indices, input_channel_names, in_ch, x_channels = resolve_input_channel_selection(
        ds=ds,
        model_cfg=model_cfg,
        defaults=defaults,
        ckpt_cfg=ckpt_cfg,
        ckpt_state=ckpt["model_state"],
    )

    ModelClass = import_model(model_module, model_class)
    model = ModelClass(
        in_channels=int(in_ch),
        out_channels=int(out_ch),
        base_channels=int(base_channels),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    model.cpu()

    print(
        f"[load] {label}\n"
        f"       dataset_in_ch={len(x_channels)} model_in_ch={in_ch} out_ch={out_ch} base_channels={base_channels}\n"
        f"       selected_inputs={input_channel_names}\n"
        f"       dataset_x_channels={x_channels}"
    )

    return {
        "label": label,
        "dataset": ds,
        "records": load_jsonl_records(test_index),
        "model": model,
        "inputs_stats": inputs_stats,
        "targets_stats": targets_stats,
        "normalize_y": bool(normalize_y),
        "input_channel_indices": input_channel_indices,
        "x_channels": x_channels,
        "sar_to_db": bool(sar_to_db),
        "sar_zero_is_nodata": bool(sar_zero_is_nodata),
        "sar_clip_db": bool(sar_clip_db),
        "sar_clip_db_bounds": sar_clip_db_bounds,
    }


def infer_ml_prediction(bundle: Dict[str, Any], sample_idx: int, device: torch.device,
                        use_amp: bool, keep_models_on_device: bool):
    s = bundle["dataset"][sample_idx]
    x_full = s["x"].float()
    x_model = x_full[bundle["input_channel_indices"]]
    y = s["y"].float()

    sid = s.get("id", sample_idx)
    t = s.get("t", "")

    model = bundle["model"].to(device)

    with torch.inference_mode():
        x_dev = x_model.unsqueeze(0).to(device)
        if use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                pred = model(x_dev).squeeze(0)
        else:
            pred = model(x_dev).squeeze(0)

    pred = pred.detach().cpu()

    if device.type == "cuda" and not keep_models_on_device:
        model.cpu()
        torch.cuda.empty_cache()

    y_raw = denorm_y(y, bundle["targets_stats"], bundle["normalize_y"])
    pred_raw = denorm_y(pred, bundle["targets_stats"], bundle["normalize_y"])

    return {
        "sid": sid,
        "t": t,
        "target_u": y_raw[0].cpu().numpy(),
        "target_v": y_raw[1].cpu().numpy(),
        "pred_u": pred_raw[0].cpu().numpy(),
        "pred_v": pred_raw[1].cpu().numpy(),
    }


def extract_future_wind(bundle: Dict[str, Any], sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    s = bundle["dataset"][sample_idx]
    x_full = s["x"].float()

    u_name = resolve_required_channel(bundle["x_channels"], "future_wind_u10_mean")
    v_name = resolve_required_channel(bundle["x_channels"], "future_wind_v10_mean")

    u_wind = denorm_x_channel(x_full, bundle["x_channels"], bundle["inputs_stats"], u_name).cpu().numpy()
    v_wind = denorm_x_channel(x_full, bundle["x_channels"], bundle["inputs_stats"], v_name).cpu().numpy()
    return u_wind, v_wind


def baseline_from_wind(u_wind: np.ndarray, v_wind: np.ndarray, baseline_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    A = float(baseline_cfg["A"])
    B = float(baseline_cfg["B"])
    u_ice = A * u_wind - B * v_wind
    v_ice = B * u_wind + A * v_wind
    return u_ice, v_ice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-yaml", required=True)
    ap.add_argument("--baseline-json", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--sample-idx", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--keep-models-on-device", action="store_true")

    ap.add_argument("--dx", type=float, default=100.0)
    ap.add_argument("--dy", type=float, default=100.0)

    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--quiver-stride", type=int, default=16)

    ap.add_argument("--quiver-ref-q", type=float, default=0.95)
    ap.add_argument("--quiver-ref-frac", type=float, default=0.12)
    ap.add_argument("--show-quiver-key", action="store_true")

    ap.add_argument("--vel-q", type=float, default=0.99)

    ap.add_argument("--div-q", type=float, default=0.99)
    ap.add_argument("--div-scale", choices=["symlog", "linear"], default="symlog")
    ap.add_argument("--symlog-linthresh-q", type=float, default=0.80)

    ap.add_argument("--shear-q", type=float, default=0.99)

    ap.add_argument("--sar-q-low", type=float, default=0.02)
    ap.add_argument("--sar-q-high", type=float, default=0.98)
    ap.add_argument("--sar-channel", default="sar_hh_db")

    ap.add_argument("--warp-n-iter", type=int, default=8)
    ap.add_argument("--warp-order", type=int, default=1)
    ap.add_argument("--warp-mode", default="constant", choices=["constant", "nearest", "reflect", "mirror", "wrap"])
    ap.add_argument("--flip-v-for-warp", action="store_true")

    ap.add_argument("--fontsize", type=int, default=9)
    ap.add_argument("--height-ratio-drift-sar", type=float, default=0.72)
    ap.add_argument("--height-ratio-div-shear", type=float, default=0.72)

    args = ap.parse_args()

    setup_pub_style(fontsize=args.fontsize)
    os.makedirs(args.outdir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.force_cpu or args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = (device.type == "cuda")

    bundle = build_model_bundle(args.compare_yaml)
    baseline_cfg = load_json(args.baseline_json)

    if args.keep_models_on_device:
        bundle["model"] = bundle["model"].to(device)

    ds_len = len(bundle["dataset"])
    if args.sample_idx is not None:
        if args.sample_idx < 0 or args.sample_idx >= ds_len:
            raise ValueError(f"--sample-idx={args.sample_idx} out of range for dataset length {ds_len}")
        picks = [args.sample_idx]
    else:
        picks = random.sample(range(ds_len), k=min(args.num_samples, ds_len))

    print(f"[run ] device={device}")
    print(f"[grid] dx={args.dx} dy={args.dy}")
    print(f"[defo] stride={args.stride}")
    print(f"[quiv] stride={args.quiver_stride}")

    for idx in picks:
        ml_res = infer_ml_prediction(
            bundle=bundle,
            sample_idx=idx,
            device=device,
            use_amp=use_amp,
            keep_models_on_device=args.keep_models_on_device,
        )

        sid = ml_res["sid"]
        t = ml_res["t"]

        target_u = ml_res["target_u"]
        target_v = ml_res["target_v"]
        ml_u = ml_res["pred_u"]
        ml_v = ml_res["pred_v"]

        wind_u, wind_v = extract_future_wind(bundle, idx)
        base_u, base_v = baseline_from_wind(wind_u, wind_v, baseline_cfg)

        meta = get_future_npz_meta(bundle, idx)
        dt_seconds = float(meta["dt_seconds"])

        h, w = target_u.shape
        tri_info = make_subsampled_triangulation(h, w, args.dx, args.dy, args.stride)

        div_target, shear_target, _ = deformation_from_triangulation(target_u, target_v, tri_info)
        div_base, shear_base, _ = deformation_from_triangulation(base_u, base_v, tri_info)
        div_ml, shear_ml, _ = deformation_from_triangulation(ml_u, ml_v, tri_info)

        div_vlim = robust_sym_limits([div_target, div_base, div_ml], q=args.div_q)
        div_linthresh = robust_abs_quantile([div_target, div_base, div_ml], q=args.symlog_linthresh_q, min_v=1e-10)

        shear_vmax = robust_upper_quantile([shear_target, shear_base, shear_ml], q=args.shear_q)

        speed_target = np.hypot(target_u, target_v) * KM_DAY_CONVERSION_FACTOR
        speed_base = np.hypot(base_u, base_v) * KM_DAY_CONVERSION_FACTOR
        speed_ml = np.hypot(ml_u, ml_v) * KM_DAY_CONVERSION_FACTOR

        vel_vmax = robust_upper_quantile([speed_target, speed_base, speed_ml], q=args.vel_q)
        speed_ref = robust_upper_quantile([speed_target, speed_base, speed_ml], q=args.quiver_ref_q, min_v=1e-6)
        quiver_scale = speed_ref / max(args.quiver_ref_frac, 1e-6)

        sar_start = extract_start_sar_image(bundle, idx, args.sar_channel)
        sar_end = extract_end_sar_image(bundle, idx, args.sar_channel)

        sign_v = -1.0 if args.flip_v_for_warp else 1.0
        base_u_pix = base_u * dt_seconds / args.dx
        base_v_pix = sign_v * base_v * dt_seconds / args.dy
        ml_u_pix = ml_u * dt_seconds / args.dx
        ml_v_pix = sign_v * ml_v * dt_seconds / args.dy

        sar_warp_base = warp_with_forward_flow(
            sar_start, base_u_pix, base_v_pix,
            n_iter=args.warp_n_iter, order=args.warp_order, mode=args.warp_mode, cval=np.nan
        )
        sar_warp_ml = warp_with_forward_flow(
            sar_start, ml_u_pix, ml_v_pix,
            n_iter=args.warp_n_iter, order=args.warp_order, mode=args.warp_mode, cval=np.nan
        )

        sar_vmin, sar_vmax = robust_image_limits_multi(
            [sar_end, sar_warp_base, sar_warp_ml],
            q_low=args.sar_q_low,
            q_high=args.sar_q_high,
        )

        print(f"[sample] idx={idx} id={sid} dt_seconds={dt_seconds:.1f}")

        # -------- Figure 1: drift speed + SAR --------
        fig1 = plt.figure(figsize=fig_textwidth(args.height_ratio_drift_sar), constrained_layout=True)
        gs1 = fig1.add_gridspec(
            2, 4,
            width_ratios=[1.0, 1.0, 1.0, 0.06],
            height_ratios=[1.0, 1.0],
        )

        ax_tvec = fig1.add_subplot(gs1[0, 0])
        ax_bvec = fig1.add_subplot(gs1[0, 1])
        ax_mvec = fig1.add_subplot(gs1[0, 2])
        cax_vel = fig1.add_subplot(gs1[0, 3])

        ax_sar0 = fig1.add_subplot(gs1[1, 0])
        ax_sar_b = fig1.add_subplot(gs1[1, 1])
        ax_sar_m = fig1.add_subplot(gs1[1, 2])
        cax_sar = fig1.add_subplot(gs1[1, 3])

        for ax in [ax_tvec, ax_bvec, ax_mvec, ax_sar0, ax_sar_b, ax_sar_m]:
            ax.set_box_aspect(1)

        im_vel = plot_vector_field_with_bg(
            ax_tvec, target_u, target_v, tri_info, "Target",
            dx=args.dx, dy=args.dy, stride=args.quiver_stride,
            quiver_scale=quiver_scale, bg_vmin=0.0, bg_vmax=vel_vmax,
            key_value=(speed_ref if args.show_quiver_key else None),
        )
        plot_vector_field_with_bg(
            ax_bvec, base_u, base_v, tri_info, "Wind baseline",
            dx=args.dx, dy=args.dy, stride=args.quiver_stride,
            quiver_scale=quiver_scale, bg_vmin=0.0, bg_vmax=vel_vmax,
        )
        plot_vector_field_with_bg(
            ax_mvec, ml_u, ml_v, tri_info, f"U-Net",

            dx=args.dx, dy=args.dy, stride=args.quiver_stride,
            quiver_scale=quiver_scale, bg_vmin=0.0, bg_vmax=vel_vmax,
        )
        cbar_vel = fig1.colorbar(im_vel, cax=cax_vel)
        cbar_vel.set_label("Drift speed [km/day]")

        im_sar = plot_sar(
            ax_sar0, sar_end, tri_info,
            title=r"SAR at $t_0 + \Delta t$",
            vmin=sar_vmin, vmax=sar_vmax
        )
        plot_sar(
            ax_sar_b, sar_warp_base, tri_info,
            title=r"Warped SAR from $t_0$",
            vmin=sar_vmin, vmax=sar_vmax
        )
        plot_sar(
            ax_sar_m, sar_warp_ml, tri_info,
            title=r"Warped SAR from $t_0$",
            vmin=sar_vmin, vmax=sar_vmax
        )
        cbar_sar = fig1.colorbar(im_sar, cax=cax_sar)
        if "db" in args.sar_channel.lower():
            cbar_sar.set_label("SAR [dB]")
        else:
            cbar_sar.set_label("SAR")

        out1 = os.path.join(
            args.outdir,
            f"drift_sar_idx{idx}_id{sanitize_filename(sid)}"
            + (f"_{sanitize_filename(t)}" if str(t).strip() else "")
            + ".png"
        )
        fig1.savefig(out1, bbox_inches="tight")
        plt.close(fig1)
        print(f"Saved: {out1}")

        # -------- Figure 2: divergence + shear --------
        fig2 = plt.figure(figsize=fig_textwidth(args.height_ratio_div_shear), constrained_layout=True)
        gs2 = fig2.add_gridspec(
            2, 4,
            width_ratios=[1.0, 1.0, 1.0, 0.06],
            height_ratios=[1.0, 1.0],
        )

        ax_tdiv = fig2.add_subplot(gs2[0, 0])
        ax_bdiv = fig2.add_subplot(gs2[0, 1])
        ax_mdiv = fig2.add_subplot(gs2[0, 2])
        cax_div = fig2.add_subplot(gs2[0, 3])

        ax_tshr = fig2.add_subplot(gs2[1, 0])
        ax_bshr = fig2.add_subplot(gs2[1, 1])
        ax_mshr = fig2.add_subplot(gs2[1, 2])
        cax_shr = fig2.add_subplot(gs2[1, 3])

        for ax in [ax_tdiv, ax_bdiv, ax_mdiv, ax_tshr, ax_bshr, ax_mshr]:
            ax.set_box_aspect(1)

        if args.div_scale == "symlog":
            div_norm = SymLogNorm(
                linthresh=div_linthresh,
                linscale=1.0,
                vmin=-div_vlim,
                vmax=div_vlim,
                base=10,
            )
        else:
            div_norm = Normalize(vmin=-div_vlim, vmax=div_vlim)

        shear_norm = Normalize(vmin=0.0, vmax=shear_vmax)

        im_div = plot_tripcolor_faces(
            ax_tdiv, tri_info, div_target, "Target",
            cmap="RdBu_r", norm=div_norm
        )
        plot_tripcolor_faces(
            ax_bdiv, tri_info, div_base, "Wind baseline",
            cmap="RdBu_r", norm=div_norm
        )
        plot_tripcolor_faces(
            ax_mdiv, tri_info, div_ml, f"U-Net",
            cmap="RdBu_r", norm=div_norm
        )
        cbar_div = fig2.colorbar(im_div, cax=cax_div)
        cbar_div.set_label("Divergence [1/day]")

        im_shr = plot_tripcolor_faces(
            ax_tshr, tri_info, shear_target, "Target",
            cmap="magma", norm=shear_norm
        )
        plot_tripcolor_faces(
            ax_bshr, tri_info, shear_base, "Wind baseline",
            cmap="magma", norm=shear_norm
        )
        plot_tripcolor_faces(
            ax_mshr, tri_info, shear_ml, f"U-Net",
            cmap="magma", norm=shear_norm
        )
        cbar_shr = fig2.colorbar(im_shr, cax=cax_shr)
        cbar_shr.set_label("Shear [1/day]")

        out2 = os.path.join(
            args.outdir,
            f"divergence_shear_idx{idx}_id{sanitize_filename(sid)}"
            + (f"_{sanitize_filename(t)}" if str(t).strip() else "")
            + ".png"
        )
        fig2.savefig(out2, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {out2}")


if __name__ == "__main__":
    main()