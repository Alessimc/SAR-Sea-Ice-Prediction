#!/usr/bin/env python3
"""
Compare target divergence and predicted divergence from multiple trained drift models,
using triangulation-based deformation on a subsampled grid, and show SAR HH in the
bottom-left panel.

Fixed 2x3 layout:
  top row:
    [0,0] Target divergence
    [0,1] Only wind (MSE+Div)
    [0,2] Only wind (MSE)

  bottom row:
    [1,0] SAR HH
    [1,1] All predictors (MSE+Div)
    [1,2] All predictors (MSE)

Features
--------
- Shared symmetric log color scale across divergence panels
- Triangulation-based divergence on a subsampled grid
- Divergence converted from 1/s to 1/day
- SAR HH shown without colorbar
- Per-model input channel selection so 2-channel wind-only checkpoints and
  7-channel all-predictor checkpoints can be compared in one figure
- Panels fill the subplot boxes using a common full-image extent
- Assumes validation datasets are aligned by sample index across models
"""

import os
import argparse
import random
import warnings
import importlib
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.tri import Triangulation
import yaml

from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset


SECONDS_PER_DAY = 86400.0

TEXTWIDTH_PT = 418.25368
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27


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


def fig_textwidth(height_ratio=0.62):
    return (TEXTWIDTH_IN, TEXTWIDTH_IN * height_ratio)


# -----------------------------
# triangulation-based deformation
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

    extent = (
        -0.5 * dx,
        (w - 0.5) * dx,
        (h - 0.5) * dy,
        -0.5 * dy,
    )

    return {
        "rows": rows,
        "cols": cols,
        "x": x,
        "y": y,
        "triangles": tri.triangles,
        "extent": extent,
    }


def divergence_from_triangulation(u: np.ndarray, v: np.ndarray, tri_info: Dict[str, Any]) -> np.ma.MaskedArray:
    u_sub = u[np.ix_(tri_info["rows"], tri_info["cols"])].astype(np.float64).ravel()
    v_sub = v[np.ix_(tri_info["rows"], tri_info["cols"])].astype(np.float64).ravel()

    e1, _, _, tri_a, _ = get_deformation_on_triangulation(
        tri_info["x"], tri_info["y"], u_sub, v_sub, tri_info["triangles"]
    )

    e1 = e1 * SECONDS_PER_DAY  # 1/s -> 1/day

    mask = (~np.isfinite(e1)) | (~np.isfinite(tri_a)) | (tri_a <= 0)
    return np.ma.masked_array(e1, mask=mask)


# -----------------------------
# utilities
# -----------------------------
def import_model(module: str, class_name: str):
    m = importlib.import_module(module)
    if not hasattr(m, class_name):
        raise AttributeError(f"Module '{module}' has no class '{class_name}'")
    return getattr(m, class_name)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_norm_yaml(norm_yaml_path: str) -> Tuple[Dict, Dict]:
    cfg = load_yaml(norm_yaml_path)
    return cfg.get("inputs", {}), cfg.get("targets", {})


def denorm_y(y: torch.Tensor, targets_stats: Dict, normalize_y: bool) -> torch.Tensor:
    if not normalize_y:
        return y

    for k in ("future_drift_u", "future_drift_v"):
        if k not in targets_stats:
            raise KeyError(f"Target '{k}' not found in norm YAML targets stats.")

    device = y.device
    y_mean = torch.tensor(
        [targets_stats["future_drift_u"]["mean"], targets_stats["future_drift_v"]["mean"]],
        dtype=torch.float32,
        device=device,
    ).view(-1, 1, 1)

    y_std = torch.tensor(
        [targets_stats["future_drift_u"]["std"], targets_stats["future_drift_v"]["std"]],
        dtype=torch.float32,
        device=device,
    ).view(-1, 1, 1)

    return y * y_std + y_mean


def denorm_x_channel(x: torch.Tensor, x_channels: List[str], inputs_stats: Dict, ch_name: str) -> torch.Tensor:
    if ch_name not in x_channels:
        raise KeyError(f"Channel '{ch_name}' not found in x_channels={x_channels}")

    ch_idx = x_channels.index(ch_name)
    x_ch = x[ch_idx]

    if ch_name not in inputs_stats:
        warnings.warn(f"Channel '{ch_name}' not found in input norm stats. Returning raw tensor values.")
        return x_ch

    mean = float(inputs_stats[ch_name]["mean"])
    std = float(inputs_stats[ch_name]["std"])
    return x_ch * std + mean


def robust_sym_limits(arrays: List[np.ma.MaskedArray], q: float = 0.99, min_v: float = 1e-12) -> float:
    vals = []
    for a in arrays:
        c = a.compressed() if isinstance(a, np.ma.MaskedArray) else np.asarray(a)
        c = c[np.isfinite(c)]
        if c.size > 0:
            vals.append(np.abs(c))

    if len(vals) == 0:
        return 1.0

    vals = np.concatenate(vals)
    vmax = float(np.quantile(vals, q))
    if not np.isfinite(vmax) or vmax < min_v:
        vmax = 1.0
    return vmax


def robust_abs_quantile(arrays: List[np.ma.MaskedArray], q: float, min_v: float = 1e-12) -> float:
    vals = []
    for a in arrays:
        c = a.compressed() if isinstance(a, np.ma.MaskedArray) else np.asarray(a)
        c = c[np.isfinite(c)]
        if c.size > 0:
            vals.append(np.abs(c))

    if len(vals) == 0:
        return 1.0

    vals = np.concatenate(vals)
    out = float(np.quantile(vals, q))
    if not np.isfinite(out) or out < min_v:
        out = min_v
    return out


def robust_image_limits(img: np.ndarray, q_low: float = 0.02, q_high: float = 0.98) -> Tuple[float, float]:
    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.quantile(vals, q_low))
    vmax = float(np.quantile(vals, q_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if vmin == vmax:
            vmax = vmin + 1.0
    return vmin, vmax


def pick_cfg(model_cfg: Dict[str, Any], defaults: Dict[str, Any], ckpt_cfg: Dict[str, Any], key: str, default=None):
    if key in model_cfg and model_cfg[key] is not None:
        return model_cfg[key]
    if key in defaults and defaults[key] is not None:
        return defaults[key]
    if ckpt_cfg is not None and key in ckpt_cfg and ckpt_cfg[key] is not None:
        return ckpt_cfg[key]
    return default


def normalize_sar_clip_bounds(bounds: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    if bounds is None:
        return {}
    out = {}
    for k, v in bounds.items():
        if len(v) != 2:
            raise ValueError(f"sar_clip_db_bounds[{k}] must have length 2, got {v}")
        out[k] = (float(v[0]), float(v[1]))
    return out


def sanitize_filename(s: str) -> str:
    s = str(s)
    for bad in ["/", "\\", ":", " ", "(", ")", "[", "]", "{", "}", ",", ";"]:
        s = s.replace(bad, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


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
        "HH": ["HH", "hh", "sar_hh_db", "sar_hh", "sar_HH", "HH_db"],
        "HV": ["HV", "hv", "sar_hv_db", "sar_hv", "sar_HV", "HV_db"],
        "IA": ["IA", "ia", "sar_incidence_angle", "sar_ia", "sar_IA"],
    }

    candidates = aliases.get(requested, [requested])
    for cand in candidates:
        if cand in x_channels:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    return None


def resolve_input_channel_selection(ds, model_cfg: Dict[str, Any], defaults: Dict[str, Any],
                                    ckpt_cfg: Dict[str, Any], ckpt_state: Dict[str, torch.Tensor]):
    if not hasattr(ds, "x_channels"):
        raise AttributeError("Dataset must expose x_channels for channel selection.")

    x_channels = list(ds.x_channels)

    requested_names = model_cfg.get("input_channel_names", defaults.get("input_channel_names", None))
    if isinstance(requested_names, str):
        requested_names = [c.strip() for c in requested_names.split(",") if c.strip()]

    expected_in_ch = pick_cfg(model_cfg, defaults, ckpt_cfg, "in_ch", None)
    if expected_in_ch is None:
        expected_in_ch = infer_in_ch_from_state_dict(ckpt_state)
    expected_in_ch = int(expected_in_ch)

    if requested_names is not None:
        missing = [c for c in requested_names if c not in x_channels]
        if missing:
            raise KeyError(f"Requested input_channel_names {missing} not found in dataset x_channels={x_channels}")
        indices = [x_channels.index(c) for c in requested_names]
        names = list(requested_names)
    else:
        if expected_in_ch == len(x_channels):
            indices = list(range(len(x_channels)))
            names = list(x_channels)
        elif expected_in_ch == 2:
            wind_names = ["future_wind_u10_mean", "future_wind_v10_mean"]
            missing = [c for c in wind_names if c not in x_channels]
            if missing:
                raise KeyError(
                    f"Checkpoint expects 2 input channels, but wind channels {missing} were not found "
                    f"in dataset x_channels={x_channels}"
                )
            indices = [x_channels.index(c) for c in wind_names]
            names = wind_names
        else:
            raise ValueError(
                f"Checkpoint expects in_ch={expected_in_ch}, but dataset provides {len(x_channels)} channels "
                f"and no input_channel_names were specified.\nDataset channels: {x_channels}"
            )

    if len(indices) != expected_in_ch:
        raise ValueError(f"Selected {len(indices)} channels {names}, but checkpoint expects in_ch={expected_in_ch}")

    return indices, names, expected_in_ch, x_channels


# -----------------------------
# plotting
# -----------------------------
def plot_div_faces_symlog(ax, tri_info: Dict[str, Any], div_faces: np.ma.MaskedArray,
                          title: str, vlim: float, linthresh: float):
    tri_plot = Triangulation(
        tri_info["x"],
        tri_info["y"],
        triangles=tri_info["triangles"],
        mask=np.ma.getmaskarray(div_faces),
    )

    norm = SymLogNorm(
        linthresh=linthresh,
        linscale=1.0,
        vmin=-vlim,
        vmax=vlim,
        base=10,
    )

    im = ax.tripcolor(
        tri_plot,
        facecolors=np.asarray(div_faces.filled(np.nan), dtype=float),
        cmap="RdBu_r",
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


def plot_sar(ax, sar_img: np.ndarray, tri_info: Dict[str, Any], title: str = "SAR HH"):
    vmin, vmax = robust_image_limits(sar_img, q_low=0.02, q_high=0.98)
    xmin, xmax, ymax, ymin = tri_info["extent"]
    ax.imshow(
        sar_img,
        cmap="gray",
        origin="upper",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        extent=(xmin, xmax, ymax, ymin),
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


# -----------------------------
# model loading
# -----------------------------
def build_model_bundle(model_cfg: Dict[str, Any], defaults: Dict[str, Any]):
    label = model_cfg.get("label")
    if label is None:
        raise ValueError("Each model entry in compare YAML must have a 'label'.")

    ckpt_path = model_cfg.get("ckpt")
    val_index = model_cfg.get("val_index")
    norm_yaml = model_cfg.get("norm_yaml")
    model_module = model_cfg.get("model_module", defaults.get("model_module"))
    model_class = model_cfg.get("model_class", defaults.get("model_class"))

    if ckpt_path is None or val_index is None or norm_yaml is None or model_module is None or model_class is None:
        raise ValueError(f"Model '{label}' is missing one of ckpt/val_index/norm_yaml/model_module/model_class.")

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
        val_index,
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

    sample0 = ds[0]
    dataset_in_ch = sample0["x"].shape[0]

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
        f"       dataset_in_ch={dataset_in_ch} model_in_ch={in_ch} out_ch={out_ch} base_channels={base_channels}\n"
        f"       selected_inputs={input_channel_names}\n"
        f"       dataset_x_channels={x_channels}"
    )

    return {
        "label": label,
        "dataset": ds,
        "model": model,
        "inputs_stats": inputs_stats,
        "targets_stats": targets_stats,
        "normalize_y": bool(normalize_y),
        "input_channel_indices": input_channel_indices,
        "x_channels": x_channels,
    }


# -----------------------------
# inference + SAR
# -----------------------------
def infer_prediction(bundle: Dict[str, Any], sample_idx: int, device: torch.device,
                     use_amp: bool, keep_models_on_device: bool):
    ds = bundle["dataset"]
    model = bundle["model"]

    s = ds[sample_idx]
    x_full = s["x"].float()
    x_model = x_full[bundle["input_channel_indices"]]
    y = s["y"].float()

    sid = s.get("id", sample_idx)
    t = s.get("t", "")

    model = model.to(device)

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


def extract_sar_image(bundle: Dict[str, Any], sample_idx: int, sar_channel: str) -> np.ndarray:
    s = bundle["dataset"][sample_idx]
    x_full = s["x"].float()

    resolved_name = resolve_channel_name(bundle["x_channels"], sar_channel)
    if resolved_name is None:
        raise KeyError(f"SAR channel '{sar_channel}' not found in x_channels={bundle['x_channels']}")

    img = denorm_x_channel(
        x=x_full,
        x_channels=bundle["x_channels"],
        inputs_stats=bundle["inputs_stats"],
        ch_name=resolved_name,
    )
    return img.cpu().numpy()


def find_bundle_with_channel(bundles: List[Dict[str, Any]], ch_name: str) -> Optional[Tuple[Dict[str, Any], str]]:
    for b in bundles:
        resolved = resolve_channel_name(b["x_channels"], ch_name)
        if resolved is not None:
            return b, resolved
    return None


# -----------------------------
# panel ordering
# -----------------------------
def _norm_label(s: str) -> str:
    return s.lower().replace("-", " ").replace("_", " ")


def select_pred_div(pred_divs: List[Tuple[str, np.ma.MaskedArray]], group: str, with_div: bool):
    matches = []
    for label, div in pred_divs:
        l = _norm_label(label)
        is_wind = "wind" in l
        is_all = ("all" in l) or ("predictor" in l)
        has_div = "div" in l

        if group == "wind" and is_wind and (has_div == with_div):
            matches.append((label, div))
        if group == "all" and is_all and (has_div == with_div):
            matches.append((label, div))

    if len(matches) != 1:
        available = [lbl for lbl, _ in pred_divs]
        raise ValueError(
            f"Could not uniquely resolve panel for group='{group}', with_div={with_div}. "
            f"Available labels: {available}"
        )
    return matches[0]


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-yaml", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--sample-idx", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--force-cpu", action="store_true")

    ap.add_argument("--dx", type=float, default=100.0)
    ap.add_argument("--dy", type=float, default=100.0)
    ap.add_argument("--stride", type=int, default=8)

    ap.add_argument("--robust-q", type=float, default=0.99)
    ap.add_argument("--symlog-linthresh-q", type=float, default=0.60)

    ap.add_argument("--sar-channel", default="sar_hh_db")
    ap.add_argument("--fontsize", type=int, default=9)
    ap.add_argument("--height-ratio", type=float, default=0.62)

    ap.add_argument("--keep-models-on-device", action="store_true")

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

    cfg = load_yaml(args.compare_yaml)
    defaults = cfg.get("defaults", {}) or {}
    models_cfg = cfg.get("models", None)
    if not isinstance(models_cfg, list) or len(models_cfg) == 0:
        raise ValueError("compare YAML must contain a non-empty 'models:' list.")

    print(f"[run ] device={device}")
    print(f"[div ] dx={args.dx} dy={args.dy} stride={args.stride}")
    print(f"[sar ] requested_channel={args.sar_channel}")

    bundles = [build_model_bundle(mcfg, defaults) for mcfg in models_cfg]

    lengths = [len(b["dataset"]) for b in bundles]
    min_len = min(lengths)
    if len(set(lengths)) != 1:
        warnings.warn(
            f"Validation dataset lengths differ across models: {lengths}. "
            f"Make sure sample indices are aligned across val_index files."
        )

    if args.sample_idx is not None:
        if args.sample_idx < 0 or args.sample_idx >= min_len:
            raise ValueError(f"--sample-idx={args.sample_idx} out of range for smallest dataset length {min_len}")
        picks = [args.sample_idx]
    else:
        picks = random.sample(range(min_len), k=min(args.num_samples, min_len))

    if args.keep_models_on_device:
        for b in bundles:
            b["model"] = b["model"].to(device)

    sar_bundle_info = find_bundle_with_channel(bundles, args.sar_channel)
    if sar_bundle_info is None:
        warnings.warn(f"No bundle contains SAR channel '{args.sar_channel}'. Final panel will be blank.")
        sar_bundle = None
        sar_channel_resolved = None
    else:
        sar_bundle, sar_channel_resolved = sar_bundle_info
        print(f"[sar ] resolved_channel={sar_channel_resolved}")

    for idx in picks:
        results = []
        for b in bundles:
            res = infer_prediction(
                b,
                sample_idx=idx,
                device=device,
                use_amp=use_amp,
                keep_models_on_device=args.keep_models_on_device,
            )
            results.append((b["label"], res))

        sid = results[0][1]["sid"]
        t = results[0][1]["t"]

        target_u = results[0][1]["target_u"]
        target_v = results[0][1]["target_v"]

        h, w = target_u.shape
        tri_info = make_subsampled_triangulation(h=h, w=w, dx=args.dx, dy=args.dy, stride=args.stride)

        div_target = divergence_from_triangulation(target_u, target_v, tri_info)

        pred_divs = []
        for label, res in results:
            div_pred = divergence_from_triangulation(res["pred_u"], res["pred_v"], tri_info)
            pred_divs.append((label, div_pred))

        # fixed panel order
        wind_div_label, wind_div = select_pred_div(pred_divs, group="wind", with_div=True)
        wind_mse_label, wind_mse = select_pred_div(pred_divs, group="wind", with_div=False)
        all_div_label, all_div = select_pred_div(pred_divs, group="all", with_div=True)
        all_mse_label, all_mse = select_pred_div(pred_divs, group="all", with_div=False)

        div_arrays = [div_target, wind_div, wind_mse, all_div, all_mse]
        vlim = robust_sym_limits(div_arrays, q=args.robust_q)
        linthresh = robust_abs_quantile(div_arrays, q=args.symlog_linthresh_q, min_v=1e-10)

        fig, axes = plt.subplots(
            2,
            3,
            figsize=fig_textwidth(height_ratio=args.height_ratio),
            constrained_layout=True,
        )
        axes = axes.ravel()

        for ax in axes:
            ax.set_box_aspect(1)

        div_axes = []

        im = plot_div_faces_symlog(
            axes[0], tri_info, div_target, "Target divergence", vlim=vlim, linthresh=linthresh
        )
        div_axes.append(axes[0])

        plot_div_faces_symlog(
            axes[1], tri_info, wind_div, wind_div_label, vlim=vlim, linthresh=linthresh
        )
        div_axes.append(axes[1])

        plot_div_faces_symlog(
            axes[2], tri_info, wind_mse, wind_mse_label, vlim=vlim, linthresh=linthresh
        )
        div_axes.append(axes[2])

        if sar_bundle is not None and sar_channel_resolved is not None:
            sar_img = extract_sar_image(sar_bundle, idx, sar_channel=sar_channel_resolved)
            plot_sar(axes[3], sar_img, tri_info, title="SAR HH")
        else:
            axes[3].text(0.5, 0.5, f"SAR {args.sar_channel}\nnot available", ha="center", va="center")
            axes[3].set_xticks([])
            axes[3].set_yticks([])

        plot_div_faces_symlog(
            axes[4], tri_info, all_div, all_div_label, vlim=vlim, linthresh=linthresh
        )
        div_axes.append(axes[4])

        plot_div_faces_symlog(
            axes[5], tri_info, all_mse, all_mse_label, vlim=vlim, linthresh=linthresh
        )
        div_axes.append(axes[5])

        # draw once so subplot positions are finalized
        fig.canvas.draw()

        # union of all divergence axes positions
        x0 = min(ax.get_position().x0 for ax in div_axes)
        y0 = min(ax.get_position().y0 for ax in div_axes)
        x1 = max(ax.get_position().x1 for ax in div_axes)
        y1 = max(ax.get_position().y1 for ax in div_axes)

        # manual colorbar axis: make it a bit taller than the divergence panel block
        pad = 0.012
        cbar_width = 0.018
        extra = 0.0   # increase this to extend both upward and downward

        cax = fig.add_axes([
            x1 + pad,          # left
            y0 - extra,        # bottom
            cbar_width,        # width
            (y1 - y0) + 2*extra  # height
        ])

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Divergence [1/day]")

        # fig.suptitle(f"Val sample id={sid} idx={idx} {t}".strip())

        out_name = f"val_div_compare_idx{idx}_id{sanitize_filename(sid)}"
        if str(t).strip():
            out_name += f"_{sanitize_filename(t)}"
        out_path = os.path.join(args.outdir, out_name + ".png")

        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()