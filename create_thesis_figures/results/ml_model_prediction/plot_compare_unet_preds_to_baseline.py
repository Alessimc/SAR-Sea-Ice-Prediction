# #!/usr/bin/env python3
# """
# Compare a fitted wind-rotation/scaling baseline against one best ML drift model.

# Figure layout (3 rows)
# ----------------------
# Row 1:
#   [0,0] Target drift speed + vectors
#   [0,1] Wind baseline drift speed + vectors
#   [0,2] ML drift speed + vectors
#   [0,3] Shared colorbar for drift speed

# Row 2:
#   [1,0] Target divergence
#   [1,1] Wind baseline divergence
#   [1,2] ML divergence
#   [1,3] Shared colorbar for divergence

# Row 3:
#   [2,0] SAR HH
#   [2,1] SAR HH warped by wind baseline
#   [2,2] SAR HH warped by ML model
#   [2,3] Shared SAR colorbar

# Notes
# -----
# - Divergence is computed on a subsampled triangulation and reported in 1/day
# - Divergence scaling can be:
#       --div-scale symlog   (default)
#       --div-scale linear
# - Vector panels show drift-speed magnitude as background, with plain single-color arrows on top
# - SAR warping uses drift [m/s] converted to pixel displacement:
#       u_pix = u * dt_seconds / dx
#       v_pix = v * dt_seconds / dy
# - The baseline is:
#       u_ice = A * u_wind - B * v_wind
#       v_ice = B * u_wind + A * v_wind
# - The ML model is loaded from the first entry in --compare-yaml
# """

# import json
# import os
# import re
# import argparse
# import random
# import warnings
# import importlib
# from datetime import datetime, timezone
# from functools import lru_cache
# from typing import Dict, Tuple, Any, List, Optional

# import numpy as np
# import torch
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib.colors import SymLogNorm, Normalize
# from matplotlib.tri import Triangulation
# from scipy.ndimage import map_coordinates
# import yaml

# from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset


# SECONDS_PER_DAY = 86400.0
# TEXTWIDTH_PT = 418.25368
# TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27


# def setup_pub_style(fontsize=9):
#     mpl.rcParams.update({
#         "font.size": fontsize,
#         "axes.titlesize": fontsize,
#         "axes.labelsize": fontsize,
#         "xtick.labelsize": fontsize - 1,
#         "ytick.labelsize": fontsize - 1,
#         "legend.fontsize": fontsize - 1,
#         "figure.dpi": 300,
#         "savefig.dpi": 300,
#     })


# def fig_textwidth(height_ratio=1.05):
#     return (TEXTWIDTH_IN, TEXTWIDTH_IN * height_ratio)


# # -----------------------------
# # triangulation-based deformation
# # -----------------------------
# def get_deformation_elems(x, y, u, v, a):
#     ux = uy = vx = vy = 0.0
#     for i0, i1 in zip([1, 2, 0], [0, 1, 2]):
#         ux += (u[i0] + u[i1]) * (y[i0] - y[i1])
#         uy -= (u[i0] + u[i1]) * (x[i0] - x[i1])
#         vx += (v[i0] + v[i1]) * (y[i0] - y[i1])
#         vy -= (v[i0] + v[i1]) * (x[i0] - x[i1])

#     with np.errstate(divide="ignore", invalid="ignore"):
#         ux, uy, vx, vy = [i / (2.0 * a) for i in (ux, uy, vx, vy)]

#     e1 = ux + vy
#     e2 = np.sqrt((ux - vy) ** 2 + (uy + vx) ** 2)
#     e3 = vx - uy
#     return e1, e2, e3


# def get_deformation_on_triangulation(x, y, u, v, t):
#     xt, yt, ut, vt = [i[t].T for i in (x, y, u, v)]

#     tri_x = np.diff(np.vstack([xt, xt[0]]), axis=0)
#     tri_y = np.diff(np.vstack([yt, yt[0]]), axis=0)
#     tri_s = np.hypot(tri_x, tri_y)

#     tri_p = np.sum(tri_s, axis=0)
#     s = tri_p / 2.0

#     with np.errstate(invalid="ignore"):
#         tri_a = np.sqrt(np.maximum(s * (s - tri_s[0]) * (s - tri_s[1]) * (s - tri_s[2]), 0.0))

#     e1, e2, e3 = get_deformation_elems(xt, yt, ut, vt, tri_a)
#     return e1, e2, e3, tri_a, tri_p


# def make_subsampled_triangulation(h: int, w: int, dx: float, dy: float, stride: int):
#     rows = np.arange(0, h, stride, dtype=np.int64)
#     cols = np.arange(0, w, stride, dtype=np.int64)

#     if rows[-1] != h - 1:
#         rows = np.append(rows, h - 1)
#     if cols[-1] != w - 1:
#         cols = np.append(cols, w - 1)

#     xx, yy = np.meshgrid(cols.astype(np.float64) * dx, rows.astype(np.float64) * dy)
#     x = xx.ravel()
#     y = yy.ravel()

#     tri = Triangulation(x, y)

#     extent = (
#         -0.5 * dx,
#         (w - 0.5) * dx,
#         (h - 0.5) * dy,
#         -0.5 * dy,
#     )

#     return {
#         "rows": rows,
#         "cols": cols,
#         "x": x,
#         "y": y,
#         "triangles": tri.triangles,
#         "extent": extent,
#     }


# def divergence_from_triangulation(u: np.ndarray, v: np.ndarray, tri_info: Dict[str, Any]) -> np.ma.MaskedArray:
#     u_sub = u[np.ix_(tri_info["rows"], tri_info["cols"])].astype(np.float64).ravel()
#     v_sub = v[np.ix_(tri_info["rows"], tri_info["cols"])].astype(np.float64).ravel()

#     e1, _, _, tri_a, _ = get_deformation_on_triangulation(
#         tri_info["x"], tri_info["y"], u_sub, v_sub, tri_info["triangles"]
#     )

#     e1 = e1 * SECONDS_PER_DAY

#     mask = (~np.isfinite(e1)) | (~np.isfinite(tri_a)) | (tri_a <= 0)
#     return np.ma.masked_array(e1, mask=mask)


# # -----------------------------
# # utilities
# # -----------------------------
# def import_model(module: str, class_name: str):
#     m = importlib.import_module(module)
#     if not hasattr(m, class_name):
#         raise AttributeError(f"Module '{module}' has no class '{class_name}'")
#     return getattr(m, class_name)


# def load_yaml(path: str) -> Dict[str, Any]:
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def load_json(path: str) -> Dict[str, Any]:
#     with open(path, "r") as f:
#         return json.load(f)


# @lru_cache(maxsize=8)
# def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
#     records = []
#     with open(path, "r") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 records.append(json.loads(line))
#     return records


# def load_norm_yaml(norm_yaml_path: str) -> Tuple[Dict, Dict]:
#     cfg = load_yaml(norm_yaml_path)
#     return cfg.get("inputs", {}), cfg.get("targets", {})


# def denorm_y(y: torch.Tensor, targets_stats: Dict, normalize_y: bool) -> torch.Tensor:
#     if not normalize_y:
#         return y

#     for k in ("future_drift_u", "future_drift_v"):
#         if k not in targets_stats:
#             raise KeyError(f"Target '{k}' not found in norm YAML targets stats.")

#     device = y.device
#     y_mean = torch.tensor(
#         [targets_stats["future_drift_u"]["mean"], targets_stats["future_drift_v"]["mean"]],
#         dtype=torch.float32,
#         device=device,
#     ).view(-1, 1, 1)

#     y_std = torch.tensor(
#         [targets_stats["future_drift_u"]["std"], targets_stats["future_drift_v"]["std"]],
#         dtype=torch.float32,
#         device=device,
#     ).view(-1, 1, 1)

#     return y * y_std + y_mean


# def denorm_x_channel(x: torch.Tensor, x_channels: List[str], inputs_stats: Dict, ch_name: str) -> torch.Tensor:
#     if ch_name not in x_channels:
#         raise KeyError(f"Channel '{ch_name}' not found in x_channels={x_channels}")

#     ch_idx = x_channels.index(ch_name)
#     x_ch = x[ch_idx]

#     if ch_name not in inputs_stats:
#         warnings.warn(f"Channel '{ch_name}' not found in input norm stats. Returning raw tensor values.")
#         return x_ch

#     mean = float(inputs_stats[ch_name]["mean"])
#     std = float(inputs_stats[ch_name]["std"])
#     return x_ch * std + mean


# def robust_sym_limits(arrays: List[np.ma.MaskedArray], q: float = 0.99, min_v: float = 1e-12) -> float:
#     vals = []
#     for a in arrays:
#         c = a.compressed() if isinstance(a, np.ma.MaskedArray) else np.asarray(a)
#         c = c[np.isfinite(c)]
#         if c.size > 0:
#             vals.append(np.abs(c))

#     if len(vals) == 0:
#         return 1.0

#     vals = np.concatenate(vals)
#     vmax = float(np.quantile(vals, q))
#     if not np.isfinite(vmax) or vmax < min_v:
#         vmax = 1.0
#     return vmax


# def robust_abs_quantile(arrays: List[np.ndarray], q: float, min_v: float = 1e-12) -> float:
#     vals = []
#     for a in arrays:
#         c = a.compressed() if isinstance(a, np.ma.MaskedArray) else np.asarray(a)
#         c = c[np.isfinite(c)]
#         if c.size > 0:
#             vals.append(np.abs(c))

#     if len(vals) == 0:
#         return 1.0

#     vals = np.concatenate(vals)
#     out = float(np.quantile(vals, q))
#     if not np.isfinite(out) or out < min_v:
#         out = min_v
#     return out


# def robust_upper_quantile(arrays: List[np.ndarray], q: float = 0.99, min_v: float = 1e-12) -> float:
#     vals = []
#     for a in arrays:
#         c = np.asarray(a)
#         c = c[np.isfinite(c)]
#         if c.size > 0:
#             vals.append(c)

#     if len(vals) == 0:
#         return 1.0

#     vals = np.concatenate(vals)
#     out = float(np.quantile(vals, q))
#     if not np.isfinite(out) or out < min_v:
#         out = max(min_v, 1.0)
#     return out


# def robust_image_limits(img: np.ndarray, q_low: float = 0.02, q_high: float = 0.98) -> Tuple[float, float]:
#     vals = img[np.isfinite(img)]
#     if vals.size == 0:
#         return 0.0, 1.0
#     vmin = float(np.quantile(vals, q_low))
#     vmax = float(np.quantile(vals, q_high))
#     if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
#         vmin = float(np.nanmin(vals))
#         vmax = float(np.nanmax(vals))
#         if vmin == vmax:
#             vmax = vmin + 1.0
#     return vmin, vmax


# def robust_image_limits_multi(images: List[np.ndarray], q_low: float = 0.02, q_high: float = 0.98) -> Tuple[float, float]:
#     vals = []
#     for img in images:
#         c = np.asarray(img)
#         c = c[np.isfinite(c)]
#         if c.size > 0:
#             vals.append(c)

#     if len(vals) == 0:
#         return 0.0, 1.0

#     vals = np.concatenate(vals)
#     vmin = float(np.quantile(vals, q_low))
#     vmax = float(np.quantile(vals, q_high))

#     if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
#         vmin = float(np.nanmin(vals))
#         vmax = float(np.nanmax(vals))
#         if vmin == vmax:
#             vmax = vmin + 1.0

#     return vmin, vmax


# def pick_cfg(model_cfg: Dict[str, Any], defaults: Dict[str, Any], ckpt_cfg: Dict[str, Any], key: str, default=None):
#     if key in model_cfg and model_cfg[key] is not None:
#         return model_cfg[key]
#     if key in defaults and defaults[key] is not None:
#         return defaults[key]
#     if ckpt_cfg is not None and key in ckpt_cfg and ckpt_cfg[key] is not None:
#         return ckpt_cfg[key]
#     return default


# def normalize_sar_clip_bounds(bounds: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
#     if bounds is None:
#         return {}
#     out = {}
#     for k, v in bounds.items():
#         if len(v) != 2:
#             raise ValueError(f"sar_clip_db_bounds[{k}] must have length 2, got {v}")
#         out[k] = (float(v[0]), float(v[1]))
#     return out


# def sanitize_filename(s: str) -> str:
#     s = str(s)
#     for bad in ["/", "\\", ":", " ", "(", ")", "[", "]", "{", "}", ",", ";"]:
#         s = s.replace(bad, "_")
#     while "__" in s:
#         s = s.replace("__", "_")
#     return s.strip("_")


# def infer_in_ch_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
#     for k, v in state_dict.items():
#         if torch.is_tensor(v) and v.ndim == 4 and k.endswith("weight"):
#             return int(v.shape[1])
#     raise KeyError("Could not infer input channels from checkpoint state_dict.")


# def resolve_channel_name(x_channels: List[str], requested: str) -> Optional[str]:
#     if requested in x_channels:
#         return requested

#     lower_map = {c.lower(): c for c in x_channels}
#     if requested.lower() in lower_map:
#         return lower_map[requested.lower()]

#     aliases = {
#         "HH": ["HH", "hh", "sar_hh_db", "sar_hh", "sar_HH", "HH_db"],
#         "HV": ["HV", "hv", "sar_hv_db", "sar_hv", "sar_HV", "HV_db"],
#         "IA": ["IA", "ia", "sar_incidence_angle", "sar_ia", "sar_IA"],
#         "future_wind_u10_mean": ["future_wind_u10_mean", "wind_u10", "u10", "u_wind", "wind_u"],
#         "future_wind_v10_mean": ["future_wind_v10_mean", "wind_v10", "v10", "v_wind", "wind_v"],
#     }

#     candidates = aliases.get(requested, [requested])
#     for cand in candidates:
#         if cand in x_channels:
#             return cand
#         if cand.lower() in lower_map:
#             return lower_map[cand.lower()]

#     return None


# def resolve_required_channel(x_channels: List[str], requested: str) -> str:
#     resolved = resolve_channel_name(x_channels, requested)
#     if resolved is None:
#         raise KeyError(f"Channel '{requested}' not found in x_channels={x_channels}")
#     return resolved


# def resolve_input_channel_selection(ds, model_cfg: Dict[str, Any], defaults: Dict[str, Any],
#                                     ckpt_cfg: Dict[str, Any], ckpt_state: Dict[str, torch.Tensor]):
#     if not hasattr(ds, "x_channels"):
#         raise AttributeError("Dataset must expose x_channels for channel selection.")

#     x_channels = list(ds.x_channels)

#     requested_names = model_cfg.get("input_channel_names", defaults.get("input_channel_names", None))
#     if isinstance(requested_names, str):
#         requested_names = [c.strip() for c in requested_names.split(",") if c.strip()]

#     expected_in_ch = pick_cfg(model_cfg, defaults, ckpt_cfg, "in_ch", None)
#     if expected_in_ch is None:
#         expected_in_ch = infer_in_ch_from_state_dict(ckpt_state)
#     expected_in_ch = int(expected_in_ch)

#     if requested_names is not None:
#         missing = [c for c in requested_names if c not in x_channels]
#         if missing:
#             raise KeyError(f"Requested input_channel_names {missing} not found in dataset x_channels={x_channels}")
#         indices = [x_channels.index(c) for c in requested_names]
#         names = list(requested_names)
#     else:
#         if expected_in_ch == len(x_channels):
#             indices = list(range(len(x_channels)))
#             names = list(x_channels)
#         elif expected_in_ch == 2:
#             wind_names = ["future_wind_u10_mean", "future_wind_v10_mean"]
#             missing = [c for c in wind_names if c not in x_channels]
#             if missing:
#                 raise KeyError(
#                     f"Checkpoint expects 2 input channels, but wind channels {missing} were not found "
#                     f"in dataset x_channels={x_channels}"
#                 )
#             indices = [x_channels.index(c) for c in wind_names]
#             names = wind_names
#         else:
#             raise ValueError(
#                 f"Checkpoint expects in_ch={expected_in_ch}, but dataset provides {len(x_channels)} channels "
#                 f"and no input_channel_names were specified.\nDataset channels: {x_channels}"
#             )

#     if len(indices) != expected_in_ch:
#         raise ValueError(f"Selected {len(indices)} channels {names}, but checkpoint expects in_ch={expected_in_ch}")

#     return indices, names, expected_in_ch, x_channels


# # -----------------------------
# # time / warp helpers
# # -----------------------------
# def _find_key_recursive(obj: Any, wanted_keys: List[str]):
#     wanted = {k.lower() for k in wanted_keys}

#     if isinstance(obj, dict):
#         for k, v in obj.items():
#             if str(k).lower() in wanted:
#                 return v
#         for _, v in obj.items():
#             out = _find_key_recursive(v, wanted_keys)
#             if out is not None:
#                 return out

#     elif isinstance(obj, list):
#         for v in obj:
#             out = _find_key_recursive(v, wanted_keys)
#             if out is not None:
#                 return out

#     return None


# def _extract_all_strings(obj: Any) -> List[str]:
#     out = []
#     if isinstance(obj, dict):
#         for _, v in obj.items():
#             out.extend(_extract_all_strings(v))
#     elif isinstance(obj, list):
#         for v in obj:
#             out.extend(_extract_all_strings(v))
#     elif isinstance(obj, str):
#         out.append(obj)
#     return out


# def _normalize_datetime(dt: datetime) -> datetime:
#     if dt.tzinfo is not None:
#         return dt.astimezone(timezone.utc).replace(tzinfo=None)
#     return dt


# def parse_datetime_flexible(text: str) -> Optional[datetime]:
#     if not isinstance(text, str):
#         return None

#     s = text.strip()
#     candidates = [s, os.path.basename(s)]

#     for cand in candidates:
#         try:
#             return _normalize_datetime(datetime.fromisoformat(cand.replace("Z", "+00:00")))
#         except Exception:
#             pass

#     regex_fmts = [
#         (r"\d{8}T\d{6}", "%Y%m%dT%H%M%S"),
#         (r"\d{8}T\d{4}", "%Y%m%dT%H%M"),
#         (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "%Y-%m-%dT%H:%M:%S"),
#         (r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}", "%Y-%m-%d_%H:%M:%S"),
#         (r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "%Y-%m-%d %H:%M:%S"),
#         (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", "%Y-%m-%dT%H:%M"),
#         (r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", "%Y-%m-%d %H:%M"),
#     ]

#     for cand in candidates:
#         for pat, fmt in regex_fmts:
#             m = re.search(pat, cand)
#             if m:
#                 try:
#                     return datetime.strptime(m.group(0), fmt)
#                 except Exception:
#                     pass

#     return None


# def infer_dt_seconds_from_record(rec: Dict[str, Any]) -> Optional[float]:
#     sec_keys = [
#         "dt_seconds", "delta_t_seconds", "future_dt_seconds", "lead_time_seconds",
#         "time_diff_seconds", "drift_dt_seconds", "forecast_seconds"
#     ]
#     val = _find_key_recursive(rec, sec_keys)
#     if val is not None:
#         try:
#             return float(val)
#         except Exception:
#             pass

#     hour_keys = [
#         "dt_hours", "delta_t_hours", "future_dt_hours", "lead_time_hours",
#         "time_diff_hours", "drift_dt_hours", "forecast_hours"
#     ]
#     val = _find_key_recursive(rec, hour_keys)
#     if val is not None:
#         try:
#             return 3600.0 * float(val)
#         except Exception:
#             pass

#     pair_keys = [
#         ("t1", "t2"),
#         ("time1", "time2"),
#         ("start_time", "end_time"),
#         ("source_time", "target_time"),
#         ("current_time", "future_time"),
#         ("obs_time", "forecast_time"),
#         ("sar_time_1", "sar_time_2"),
#         ("sar_time1", "sar_time2"),
#         ("file1", "file2"),
#         ("path1", "path2"),
#         ("source_path", "target_path"),
#         ("sar1_path", "sar2_path"),
#         ("input_path", "target_path"),
#     ]

#     for k1, k2 in pair_keys:
#         v1 = _find_key_recursive(rec, [k1])
#         v2 = _find_key_recursive(rec, [k2])
#         if v1 is not None and v2 is not None:
#             dt1 = parse_datetime_flexible(str(v1))
#             dt2 = parse_datetime_flexible(str(v2))
#             if dt1 is not None and dt2 is not None:
#                 return abs((dt2 - dt1).total_seconds())

#     parsed = []
#     for s in _extract_all_strings(rec):
#         dt = parse_datetime_flexible(s)
#         if dt is not None:
#             parsed.append(dt)

#     parsed = sorted(set(parsed))
#     if len(parsed) == 2:
#         return abs((parsed[1] - parsed[0]).total_seconds())

#     return None


# def get_dt_seconds_for_sample(bundle: Dict[str, Any], sample_idx: int,
#                               cli_drift_seconds: Optional[float],
#                               cli_drift_hours: Optional[float]) -> float:
#     if cli_drift_seconds is not None:
#         return float(cli_drift_seconds)

#     if cli_drift_hours is not None:
#         return 3600.0 * float(cli_drift_hours)

#     records = load_jsonl_records(bundle["test_index"])
#     if sample_idx < 0 or sample_idx >= len(records):
#         raise IndexError(f"sample_idx={sample_idx} out of range for test_index records")

#     dt_seconds = infer_dt_seconds_from_record(records[sample_idx])
#     if dt_seconds is None:
#         raise ValueError(
#             "Could not infer the drift lead time from the index record. "
#             "Please pass --drift-seconds or --drift-hours explicitly."
#         )

#     return float(dt_seconds)


# def warp_with_forward_flow(img, u, v, n_iter=8, order=1, mode="constant", cval=np.nan):
#     """
#     Warp img using a forward flow (u,v) defined on the source grid:
#         source p -> target p + (u(p), v(p))
#     Produces output on the same grid as img via approximate inversion.

#     u = column displacement in pixels
#     v = row displacement in pixels
#     """
#     rows, cols = img.shape
#     rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")

#     r = rr.astype(np.float64)
#     c = cc.astype(np.float64)

#     for _ in range(n_iter):
#         v_rc = map_coordinates(v, [r, c], order=1, mode="nearest")
#         u_rc = map_coordinates(u, [r, c], order=1, mode="nearest")
#         r = rr - v_rc
#         c = cc - u_rc

#     valid = np.isfinite(img).astype(np.float64)
#     img_filled = np.where(np.isfinite(img), img, 0.0).astype(np.float64)

#     warped_num = map_coordinates(img_filled, [r, c], order=order, mode=mode, cval=0.0)
#     warped_den = map_coordinates(valid, [r, c], order=0, mode=mode, cval=0.0)

#     warped = np.where(warped_den > 0.5, warped_num, cval)
#     return warped


# # -----------------------------
# # plotting helpers
# # -----------------------------
# def _subsample_indices(n: int, stride: int) -> np.ndarray:
#     idx = np.arange(0, n, stride, dtype=np.int64)
#     if idx[-1] != n - 1:
#         idx = np.append(idx, n - 1)
#     return idx


# def plot_vector_field_with_bg(
#     ax,
#     u: np.ndarray,
#     v: np.ndarray,
#     tri_info: Dict[str, Any],
#     title: str,
#     dx: float,
#     dy: float,
#     stride: int,
#     quiver_scale: float,
#     bg_vmin: float,
#     bg_vmax: float,
#     cmap: str = "viridis",
#     vector_color: str = "white",
#     key_value: Optional[float] = None,
# ):
#     speed = np.hypot(u, v)

#     xmin, xmax, ymax, ymin = tri_info["extent"]
#     im = ax.imshow(
#         speed,
#         cmap=cmap,
#         origin="upper",
#         interpolation="nearest",
#         vmin=bg_vmin,
#         vmax=bg_vmax,
#         extent=(xmin, xmax, ymax, ymin),
#         aspect="auto",
#         zorder=1,
#     )

#     h, w = u.shape
#     rows = _subsample_indices(h, stride)
#     cols = _subsample_indices(w, stride)

#     xx, yy = np.meshgrid(cols.astype(np.float64) * dx, rows.astype(np.float64) * dy)
#     uu = u[np.ix_(rows, cols)]
#     vv = v[np.ix_(rows, cols)]

#     q = ax.quiver(
#         xx,
#         yy,
#         uu,
#         vv,
#         angles="xy",
#         scale_units="width",
#         scale=quiver_scale,
#         color=vector_color,
#         pivot="mid",
#         width=0.0045,
#         headwidth=3.5,
#         headlength=4.6,
#         headaxislength=4.1,
#         alpha=0.95,
#         zorder=2,
#     )

#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymax, ymin)
#     ax.set_aspect("auto")
#     ax.set_title(title)
#     ax.set_xticks([])
#     ax.set_yticks([])

#     if key_value is not None and np.isfinite(key_value) and key_value > 0:
#         ax.quiverkey(
#             q,
#             X=0.80,
#             Y=1.04,
#             U=key_value,
#             label=f"{key_value:.2f} m/s",
#             labelpos="E",
#             coordinates="axes",
#             color=vector_color,
#         )

#     return im


# def plot_div_faces(ax, tri_info: Dict[str, Any], div_faces: np.ma.MaskedArray, title: str,
#                    vlim: float, div_scale: str = "symlog", linthresh: float = 1e-3,
#                    cmap: str = "RdBu_r"):
#     tri_plot = Triangulation(
#         tri_info["x"],
#         tri_info["y"],
#         triangles=tri_info["triangles"],
#         mask=np.ma.getmaskarray(div_faces),
#     )

#     if div_scale == "symlog":
#         norm = SymLogNorm(
#             linthresh=linthresh,
#             linscale=1.0,
#             vmin=-vlim,
#             vmax=vlim,
#             base=10,
#         )
#     elif div_scale == "linear":
#         norm = Normalize(vmin=-vlim, vmax=vlim)
#     else:
#         raise ValueError(f"Unsupported div_scale='{div_scale}'")

#     im = ax.tripcolor(
#         tri_plot,
#         facecolors=np.asarray(div_faces.filled(np.nan), dtype=float),
#         cmap=cmap,
#         norm=norm,
#         shading="flat",
#     )

#     xmin, xmax, ymax, ymin = tri_info["extent"]
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymax, ymin)
#     ax.set_aspect("auto")
#     ax.set_title(title)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     return im


# def plot_sar(ax, sar_img: np.ndarray, tri_info: Dict[str, Any], title: str,
#              vmin: float, vmax: float, cmap: str = "gray"):
#     xmin, xmax, ymax, ymin = tri_info["extent"]
#     im = ax.imshow(
#         sar_img,
#         cmap=cmap,
#         origin="upper",
#         interpolation="nearest",
#         vmin=vmin,
#         vmax=vmax,
#         extent=(xmin, xmax, ymax, ymin),
#         aspect="auto",
#     )
#     ax.set_title(title)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     return im


# # -----------------------------
# # model loading
# # -----------------------------
# def build_model_bundle(model_cfg: Dict[str, Any], defaults: Dict[str, Any]):
#     label = model_cfg.get("label", "ML model")

#     ckpt_path = model_cfg.get("ckpt")
#     test_index = model_cfg.get("test_index")
#     norm_yaml = model_cfg.get("norm_yaml")
#     model_module = model_cfg.get("model_module", defaults.get("model_module"))
#     model_class = model_cfg.get("model_class", defaults.get("model_class"))

#     missing = []
#     if ckpt_path is None:
#         missing.append("ckpt")
#     if test_index is None:
#         missing.append("test_index")
#     if norm_yaml is None:
#         missing.append("norm_yaml")
#     if model_module is None:
#         missing.append("model_module")
#     if model_class is None:
#         missing.append("model_class")
#     if missing:
#         raise ValueError(f"Model '{label}' is missing: {missing}")

#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     ckpt_cfg = ckpt.get("config", {}) or {}

#     include_wspd = pick_cfg(model_cfg, defaults, ckpt_cfg, "include_wspd", False)
#     normalize_y = pick_cfg(model_cfg, defaults, ckpt_cfg, "normalize_y", True)
#     base_channels = pick_cfg(model_cfg, defaults, ckpt_cfg, "base_channels", 32)
#     out_ch = pick_cfg(model_cfg, defaults, ckpt_cfg, "out_ch", 2)

#     sar_channels = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_channels", ["HH", "HV", "IA"])
#     sar_to_db = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_to_db", True)
#     sar_postprocess = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_postprocess", True)
#     sar_zero_is_nodata = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_zero_is_nodata", False)
#     sar_clip_db = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_clip_db", True)
#     sar_clip_db_bounds = normalize_sar_clip_bounds(
#         pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_clip_db_bounds", {})
#     )

#     if isinstance(sar_channels, str):
#         sar_channels = [c.strip() for c in sar_channels.split(",") if c.strip()]

#     inputs_stats, targets_stats = load_norm_yaml(norm_yaml)

#     ds = DriftWindSARDataset(
#         test_index,
#         norm_yaml_path=norm_yaml,
#         normalize_y=normalize_y,
#         include_wspd=include_wspd,
#         return_meta=False,
#         cache_size=0,
#         sar_channels=tuple(sar_channels),
#         sar_to_db=sar_to_db,
#         sar_postprocess=sar_postprocess,
#         sar_clip_percentiles=None,
#         sar_zero_is_nodata=sar_zero_is_nodata,
#         sar_clip_db=sar_clip_db,
#         sar_clip_db_bounds=sar_clip_db_bounds,
#     )

#     sample0 = ds[0]
#     dataset_in_ch = sample0["x"].shape[0]

#     input_channel_indices, input_channel_names, in_ch, x_channels = resolve_input_channel_selection(
#         ds=ds,
#         model_cfg=model_cfg,
#         defaults=defaults,
#         ckpt_cfg=ckpt_cfg,
#         ckpt_state=ckpt["model_state"],
#     )

#     ModelClass = import_model(model_module, model_class)
#     model = ModelClass(
#         in_channels=int(in_ch),
#         out_channels=int(out_ch),
#         base_channels=int(base_channels),
#     )
#     model.load_state_dict(ckpt["model_state"], strict=True)
#     model.eval()
#     model.cpu()

#     print(
#         f"[load] {label}\n"
#         f"       dataset_in_ch={dataset_in_ch} model_in_ch={in_ch} out_ch={out_ch} base_channels={base_channels}\n"
#         f"       selected_inputs={input_channel_names}\n"
#         f"       dataset_x_channels={x_channels}"
#     )

#     return {
#         "label": label,
#         "dataset": ds,
#         "model": model,
#         "inputs_stats": inputs_stats,
#         "targets_stats": targets_stats,
#         "normalize_y": bool(normalize_y),
#         "input_channel_indices": input_channel_indices,
#         "x_channels": x_channels,
#         "sar_to_db": bool(sar_to_db),
#         "test_index": test_index,
#     }


# # -----------------------------
# # inference + baseline + SAR
# # -----------------------------
# def infer_ml_prediction(bundle: Dict[str, Any], sample_idx: int, device: torch.device,
#                         use_amp: bool, keep_models_on_device: bool):
#     ds = bundle["dataset"]
#     model = bundle["model"]

#     s = ds[sample_idx]
#     x_full = s["x"].float()
#     x_model = x_full[bundle["input_channel_indices"]]
#     y = s["y"].float()

#     sid = s.get("id", sample_idx)
#     t = s.get("t", "")

#     model = model.to(device)

#     with torch.inference_mode():
#         x_dev = x_model.unsqueeze(0).to(device)
#         if use_amp:
#             with torch.amp.autocast("cuda", enabled=True):
#                 pred = model(x_dev).squeeze(0)
#         else:
#             pred = model(x_dev).squeeze(0)

#     pred = pred.detach().cpu()

#     if device.type == "cuda" and not keep_models_on_device:
#         model.cpu()
#         torch.cuda.empty_cache()

#     y_raw = denorm_y(y, bundle["targets_stats"], bundle["normalize_y"])
#     pred_raw = denorm_y(pred, bundle["targets_stats"], bundle["normalize_y"])

#     return {
#         "sid": sid,
#         "t": t,
#         "target_u": y_raw[0].cpu().numpy(),
#         "target_v": y_raw[1].cpu().numpy(),
#         "pred_u": pred_raw[0].cpu().numpy(),
#         "pred_v": pred_raw[1].cpu().numpy(),
#     }


# def extract_sar_image(bundle: Dict[str, Any], sample_idx: int, sar_channel: str) -> np.ndarray:
#     s = bundle["dataset"][sample_idx]
#     x_full = s["x"].float()

#     resolved_name = resolve_required_channel(bundle["x_channels"], sar_channel)
#     img = denorm_x_channel(
#         x=x_full,
#         x_channels=bundle["x_channels"],
#         inputs_stats=bundle["inputs_stats"],
#         ch_name=resolved_name,
#     )
#     return img.cpu().numpy()


# def extract_future_wind(bundle: Dict[str, Any], sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
#     s = bundle["dataset"][sample_idx]
#     x_full = s["x"].float()

#     u_name = resolve_required_channel(bundle["x_channels"], "future_wind_u10_mean")
#     v_name = resolve_required_channel(bundle["x_channels"], "future_wind_v10_mean")

#     u_wind = denorm_x_channel(
#         x=x_full,
#         x_channels=bundle["x_channels"],
#         inputs_stats=bundle["inputs_stats"],
#         ch_name=u_name,
#     ).cpu().numpy()

#     v_wind = denorm_x_channel(
#         x=x_full,
#         x_channels=bundle["x_channels"],
#         inputs_stats=bundle["inputs_stats"],
#         ch_name=v_name,
#     ).cpu().numpy()

#     return u_wind, v_wind


# def baseline_from_wind(u_wind: np.ndarray, v_wind: np.ndarray, baseline_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
#     A = float(baseline_cfg["A"])
#     B = float(baseline_cfg["B"])

#     u_ice = A * u_wind - B * v_wind
#     v_ice = B * u_wind + A * v_wind
#     return u_ice, v_ice


# # -----------------------------
# # main
# # -----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--compare-yaml", required=True, help="YAML with the best ML model as the first entry in models:")
#     ap.add_argument("--baseline-json", required=True, help="JSON containing at least A and B")
#     ap.add_argument("--outdir", required=True)

#     ap.add_argument("--num-samples", type=int, default=10)
#     ap.add_argument("--sample-idx", type=int, default=None)
#     ap.add_argument("--seed", type=int, default=0)

#     ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
#     ap.add_argument("--force-cpu", action="store_true")

#     ap.add_argument("--dx", type=float, default=100.0)
#     ap.add_argument("--dy", type=float, default=100.0)

#     ap.add_argument("--stride", type=int, default=8, help="Triangulation stride for divergence")
#     ap.add_argument("--quiver-stride", type=int, default=16, help="Subsampling stride for vector arrows")

#     ap.add_argument("--quiver-ref-q", type=float, default=0.95,
#                     help="Quantile used to define the quiver key magnitude")
#     ap.add_argument("--quiver-ref-frac", type=float, default=0.12,
#                     help="Approximate fraction of axes width used for the quiver key arrow")
#     ap.add_argument("--show-quiver-key", action="store_true",
#                     help="Show quiver key on the target vector panel")

#     ap.add_argument("--vel-q", type=float, default=0.99,
#                     help="Robust upper quantile for shared velocity color scale")
#     ap.add_argument("--div-q", type=float, default=0.99,
#                     help="Robust upper quantile for shared divergence color scale")
#     ap.add_argument("--div-scale", choices=["symlog", "linear"], default="symlog",
#                     help="Use symmetric-log or linear scaling for divergence")
#     ap.add_argument("--symlog-linthresh-q", type=float, default=0.80,
#                     help="Quantile for symlog linear threshold")

#     ap.add_argument("--sar-q-low", type=float, default=0.02)
#     ap.add_argument("--sar-q-high", type=float, default=0.98)

#     ap.add_argument("--drift-seconds", type=float, default=None,
#                     help="Lead time in seconds used to convert drift [m/s] to pixel displacement")
#     ap.add_argument("--drift-hours", type=float, default=None,
#                     help="Lead time in hours used to convert drift [m/s] to pixel displacement")
#     ap.add_argument("--warp-n-iter", type=int, default=8)
#     ap.add_argument("--warp-order", type=int, default=1)
#     ap.add_argument("--warp-mode", default="constant", choices=["constant", "nearest", "reflect", "mirror", "wrap"])
#     ap.add_argument("--flip-v-for-warp", action="store_true",
#                     help="Flip sign of v when converting to row displacement, if vertical warp direction looks wrong")

#     ap.add_argument("--sar-channel", default="sar_hh_db")
#     ap.add_argument("--fontsize", type=int, default=9)
#     ap.add_argument("--height-ratio", type=float, default=1.05)

#     ap.add_argument("--keep-models-on-device", action="store_true")

#     args = ap.parse_args()

#     if args.drift_seconds is not None and args.drift_hours is not None:
#         raise ValueError("Pass only one of --drift-seconds or --drift-hours, not both.")

#     setup_pub_style(fontsize=args.fontsize)
#     os.makedirs(args.outdir, exist_ok=True)

#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     if args.force_cpu or args.device == "cpu":
#         device = torch.device("cpu")
#     elif args.device == "cuda":
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     use_amp = (device.type == "cuda")

#     cfg = load_yaml(args.compare_yaml)
#     defaults = cfg.get("defaults", {}) or {}
#     models_cfg = cfg.get("models", None)
#     if not isinstance(models_cfg, list) or len(models_cfg) == 0:
#         raise ValueError("compare YAML must contain a non-empty 'models:' list.")

#     if len(models_cfg) > 1:
#         warnings.warn("More than one model found in compare YAML. Only the first model will be used.")

#     ml_bundle = build_model_bundle(models_cfg[0], defaults)
#     baseline_cfg = load_json(args.baseline_json)

#     if "A" not in baseline_cfg or "B" not in baseline_cfg:
#         raise KeyError(f"--baseline-json must contain keys 'A' and 'B'. Got keys={list(baseline_cfg.keys())}")

#     print(f"[run ] device={device}")
#     print(f"[grid] dx={args.dx} dy={args.dy}")
#     print(f"[div ] stride={args.stride} scale={args.div_scale}")
#     print(f"[quiv] stride={args.quiver_stride} ref_q={args.quiver_ref_q} ref_frac={args.quiver_ref_frac}")
#     print(f"[vel ] q={args.vel_q}")
#     print(f"[sar ] requested_channel={args.sar_channel}")
#     print(f"[base] A={baseline_cfg['A']} B={baseline_cfg['B']}")

#     if args.keep_models_on_device:
#         ml_bundle["model"] = ml_bundle["model"].to(device)

#     ds_len = len(ml_bundle["dataset"])
#     if args.sample_idx is not None:
#         if args.sample_idx < 0 or args.sample_idx >= ds_len:
#             raise ValueError(f"--sample-idx={args.sample_idx} out of range for dataset length {ds_len}")
#         picks = [args.sample_idx]
#     else:
#         picks = random.sample(range(ds_len), k=min(args.num_samples, ds_len))

#     for idx in picks:
#         ml_res = infer_ml_prediction(
#             ml_bundle,
#             sample_idx=idx,
#             device=device,
#             use_amp=use_amp,
#             keep_models_on_device=args.keep_models_on_device,
#         )

#         sid = ml_res["sid"]
#         t = ml_res["t"]

#         target_u = ml_res["target_u"]
#         target_v = ml_res["target_v"]
#         ml_u = ml_res["pred_u"]
#         ml_v = ml_res["pred_v"]

#         wind_u, wind_v = extract_future_wind(ml_bundle, idx)
#         base_u, base_v = baseline_from_wind(wind_u, wind_v, baseline_cfg)

#         dt_seconds = get_dt_seconds_for_sample(
#             ml_bundle,
#             sample_idx=idx,
#             cli_drift_seconds=args.drift_seconds,
#             cli_drift_hours=args.drift_hours,
#         )

#         h, w = target_u.shape
#         tri_info = make_subsampled_triangulation(h=h, w=w, dx=args.dx, dy=args.dy, stride=args.stride)

#         # divergence
#         div_target = divergence_from_triangulation(target_u, target_v, tri_info)
#         div_base = divergence_from_triangulation(base_u, base_v, tri_info)
#         div_ml = divergence_from_triangulation(ml_u, ml_v, tri_info)

#         div_arrays = [div_target, div_base, div_ml]
#         div_vlim = robust_sym_limits(div_arrays, q=args.div_q)
#         div_linthresh = robust_abs_quantile(div_arrays, q=args.symlog_linthresh_q, min_v=1e-10)

#         # speed for vector backgrounds
#         speed_target = np.hypot(target_u, target_v)
#         speed_base = np.hypot(base_u, base_v)
#         speed_ml = np.hypot(ml_u, ml_v)

#         vel_vmax = robust_upper_quantile([speed_target, speed_base, speed_ml], q=args.vel_q)
#         vel_vmin = 0.0

#         speed_ref = robust_upper_quantile([speed_target, speed_base, speed_ml], q=args.quiver_ref_q, min_v=1e-6)
#         quiver_scale = speed_ref / max(args.quiver_ref_frac, 1e-6)

#         # SAR
#         sar_img = extract_sar_image(ml_bundle, idx, sar_channel=args.sar_channel)

#         sign_v = -1.0 if args.flip_v_for_warp else 1.0
#         base_u_pix = base_u * dt_seconds / args.dx
#         base_v_pix = sign_v * base_v * dt_seconds / args.dy
#         ml_u_pix = ml_u * dt_seconds / args.dx
#         ml_v_pix = sign_v * ml_v * dt_seconds / args.dy

#         sar_warp_base = warp_with_forward_flow(
#             sar_img,
#             base_u_pix,
#             base_v_pix,
#             n_iter=args.warp_n_iter,
#             order=args.warp_order,
#             mode=args.warp_mode,
#             cval=np.nan,
#         )

#         sar_warp_ml = warp_with_forward_flow(
#             sar_img,
#             ml_u_pix,
#             ml_v_pix,
#             n_iter=args.warp_n_iter,
#             order=args.warp_order,
#             mode=args.warp_mode,
#             cval=np.nan,
#         )

#         sar_vmin, sar_vmax = robust_image_limits_multi(
#             [sar_img, sar_warp_base, sar_warp_ml],
#             q_low=args.sar_q_low,
#             q_high=args.sar_q_high,
#         )

#         print(
#             f"[sample] idx={idx} id={sid} t={t} "
#             f"dt_seconds={dt_seconds:.1f} "
#             f"vel_vmax={vel_vmax:.4g} div_vlim={div_vlim:.4g}"
#         )

#         fig = plt.figure(
#             figsize=fig_textwidth(height_ratio=args.height_ratio),
#             constrained_layout=True,
#         )
#         gs = fig.add_gridspec(
#             3, 4,
#             width_ratios=[1.0, 1.0, 1.0, 0.06],
#             height_ratios=[1.0, 1.0, 1.0],
#         )

#         # row 1
#         ax_tvec = fig.add_subplot(gs[0, 0])
#         ax_bvec = fig.add_subplot(gs[0, 1])
#         ax_mvec = fig.add_subplot(gs[0, 2])
#         cax_vel = fig.add_subplot(gs[0, 3])

#         # row 2
#         ax_tdiv = fig.add_subplot(gs[1, 0])
#         ax_bdiv = fig.add_subplot(gs[1, 1])
#         ax_mdiv = fig.add_subplot(gs[1, 2])
#         cax_div = fig.add_subplot(gs[1, 3])

#         # row 3
#         ax_sar0 = fig.add_subplot(gs[2, 0])
#         ax_sar_b = fig.add_subplot(gs[2, 1])
#         ax_sar_m = fig.add_subplot(gs[2, 2])
#         cax_sar = fig.add_subplot(gs[2, 3])

#         for ax in [ax_tvec, ax_bvec, ax_mvec, ax_tdiv, ax_bdiv, ax_mdiv, ax_sar0, ax_sar_b, ax_sar_m]:
#             ax.set_box_aspect(1)

#         # row 1: vector fields with speed background
#         im_vel = plot_vector_field_with_bg(
#             ax_tvec,
#             target_u,
#             target_v,
#             tri_info,
#             "Target drift vector field",
#             dx=args.dx,
#             dy=args.dy,
#             stride=args.quiver_stride,
#             quiver_scale=quiver_scale,
#             bg_vmin=vel_vmin,
#             bg_vmax=vel_vmax,
#             cmap="viridis",
#             vector_color="white",
#             key_value=(speed_ref if args.show_quiver_key else None),
#         )

#         plot_vector_field_with_bg(
#             ax_bvec,
#             base_u,
#             base_v,
#             tri_info,
#             "Wind baseline vector field",
#             dx=args.dx,
#             dy=args.dy,
#             stride=args.quiver_stride,
#             quiver_scale=quiver_scale,
#             bg_vmin=vel_vmin,
#             bg_vmax=vel_vmax,
#             cmap="viridis",
#             vector_color="white",
#             key_value=None,
#         )

#         plot_vector_field_with_bg(
#             ax_mvec,
#             ml_u,
#             ml_v,
#             tri_info,
#             f"{ml_bundle['label']} vector field",
#             dx=args.dx,
#             dy=args.dy,
#             stride=args.quiver_stride,
#             quiver_scale=quiver_scale,
#             bg_vmin=vel_vmin,
#             bg_vmax=vel_vmax,
#             cmap="viridis",
#             vector_color="white",
#             key_value=None,
#         )

#         cbar_vel = fig.colorbar(im_vel, cax=cax_vel)
#         cbar_vel.set_label("Drift speed [m/s]")

#         # row 2: divergence
#         im_div = plot_div_faces(
#             ax_tdiv,
#             tri_info,
#             div_target,
#             "Target divergence",
#             vlim=div_vlim,
#             div_scale=args.div_scale,
#             linthresh=div_linthresh,
#             cmap="RdBu_r",
#         )

#         plot_div_faces(
#             ax_bdiv,
#             tri_info,
#             div_base,
#             "Wind baseline divergence",
#             vlim=div_vlim,
#             div_scale=args.div_scale,
#             linthresh=div_linthresh,
#             cmap="RdBu_r",
#         )

#         plot_div_faces(
#             ax_mdiv,
#             tri_info,
#             div_ml,
#             f"{ml_bundle['label']} divergence",
#             vlim=div_vlim,
#             div_scale=args.div_scale,
#             linthresh=div_linthresh,
#             cmap="RdBu_r",
#         )

#         cbar_div = fig.colorbar(im_div, cax=cax_div)
#         cbar_div.set_label("Divergence [1/day]")

#         # row 3: SAR
#         im_sar = plot_sar(
#             ax_sar0,
#             sar_img,
#             tri_info,
#             title="SAR HH",
#             vmin=sar_vmin,
#             vmax=sar_vmax,
#             cmap="gray",
#         )

#         plot_sar(
#             ax_sar_b,
#             sar_warp_base,
#             tri_info,
#             title="SAR HH warped by wind baseline",
#             vmin=sar_vmin,
#             vmax=sar_vmax,
#             cmap="gray",
#         )

#         plot_sar(
#             ax_sar_m,
#             sar_warp_ml,
#             tri_info,
#             title="SAR HH warped by ML model",
#             vmin=sar_vmin,
#             vmax=sar_vmax,
#             cmap="gray",
#         )

#         cbar_sar = fig.colorbar(im_sar, cax=cax_sar)
#         if "db" in args.sar_channel.lower():
#             cbar_sar.set_label("SAR HH [dB]")
#         else:
#             cbar_sar.set_label("SAR HH")

#         out_name = f"baseline_vs_ml_warped_sar_idx{idx}_id{sanitize_filename(sid)}"
#         if str(t).strip():
#             out_name += f"_{sanitize_filename(t)}"
#         out_path = os.path.join(args.outdir, out_name + ".png")

#         fig.savefig(out_path, bbox_inches="tight")
#         plt.close(fig)
#         print(f"Saved: {out_path}")


# if __name__ == "__main__":
#     main()

















# #!/usr/bin/env python3
# """
# Compare a fitted wind-rotation/scaling baseline against one best ML drift model.

# Figure layout (3 rows)
# ----------------------
# Row 1:
#   [0,0] Target drift speed + vectors
#   [0,1] Wind baseline drift speed + vectors
#   [0,2] ML drift speed + vectors
#   [0,3] Shared colorbar for drift speed

# Row 2:
#   [1,0] Target divergence
#   [1,1] Wind baseline divergence
#   [1,2] ML divergence
#   [1,3] Shared colorbar for divergence

# Row 3:
#   [2,0] SAR HH (currently start image)
#   [2,1] SAR HH warped by wind baseline
#   [2,2] SAR HH warped by ML model
#   [2,3] Shared SAR colorbar

# Debugging
# ---------
# Use:
#   --print-future-drift-metadata
# to inspect the NPZ referenced by future_drift_path in the index JSONL.

# Use:
#   --stop-after-future-drift-metadata
# to print metadata and exit without plotting.

# Notes
# -----
# - Divergence is computed on a subsampled triangulation and reported in 1/day
# - Divergence scaling can be:
#       --div-scale symlog   (default)
#       --div-scale linear
# - Vector panels show drift-speed magnitude as background, with plain single-color arrows on top
# - SAR warping uses drift [m/s] converted to pixel displacement:
#       u_pix = u * dt_seconds / dx
#       v_pix = v * dt_seconds / dy
# - The baseline is:
#       u_ice = A * u_wind - B * v_wind
#       v_ice = B * u_wind + A * v_wind
# - The ML model is loaded from the first entry in --compare-yaml
# """

# import json
# import os
# import re
# import argparse
# import random
# import warnings
# import importlib
# from datetime import datetime, timezone
# from functools import lru_cache
# from typing import Dict, Tuple, Any, List, Optional

# import numpy as np
# import torch
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib.colors import SymLogNorm, Normalize
# from matplotlib.tri import Triangulation
# from scipy.ndimage import map_coordinates
# import yaml

# from model_dev_main.src.dataloader.DriftWindSARDataset import DriftWindSARDataset


# SECONDS_PER_DAY = 86400.0
# TEXTWIDTH_PT = 418.25368
# TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27


# def setup_pub_style(fontsize=9):
#     mpl.rcParams.update({
#         "font.size": fontsize,
#         "axes.titlesize": fontsize,
#         "axes.labelsize": fontsize,
#         "xtick.labelsize": fontsize - 1,
#         "ytick.labelsize": fontsize - 1,
#         "legend.fontsize": fontsize - 1,
#         "figure.dpi": 300,
#         "savefig.dpi": 300,
#     })


# def fig_textwidth(height_ratio=1.05):
#     return (TEXTWIDTH_IN, TEXTWIDTH_IN * height_ratio)


# # -----------------------------
# # triangulation-based deformation
# # -----------------------------
# def get_deformation_elems(x, y, u, v, a):
#     ux = uy = vx = vy = 0.0
#     for i0, i1 in zip([1, 2, 0], [0, 1, 2]):
#         ux += (u[i0] + u[i1]) * (y[i0] - y[i1])
#         uy -= (u[i0] + u[i1]) * (x[i0] - x[i1])
#         vx += (v[i0] + v[i1]) * (y[i0] - y[i1])
#         vy -= (v[i0] + v[i1]) * (x[i0] - x[i1])

#     with np.errstate(divide="ignore", invalid="ignore"):
#         ux, uy, vx, vy = [i / (2.0 * a) for i in (ux, uy, vx, vy)]

#     e1 = ux + vy
#     e2 = np.sqrt((ux - vy) ** 2 + (uy + vx) ** 2)
#     e3 = vx - uy
#     return e1, e2, e3


# def get_deformation_on_triangulation(x, y, u, v, t):
#     xt, yt, ut, vt = [i[t].T for i in (x, y, u, v)]

#     tri_x = np.diff(np.vstack([xt, xt[0]]), axis=0)
#     tri_y = np.diff(np.vstack([yt, yt[0]]), axis=0)
#     tri_s = np.hypot(tri_x, tri_y)

#     tri_p = np.sum(tri_s, axis=0)
#     s = tri_p / 2.0

#     with np.errstate(invalid="ignore"):
#         tri_a = np.sqrt(np.maximum(s * (s - tri_s[0]) * (s - tri_s[1]) * (s - tri_s[2]), 0.0))

#     e1, e2, e3 = get_deformation_elems(xt, yt, ut, vt, tri_a)
#     return e1, e2, e3, tri_a, tri_p


# def make_subsampled_triangulation(h: int, w: int, dx: float, dy: float, stride: int):
#     rows = np.arange(0, h, stride, dtype=np.int64)
#     cols = np.arange(0, w, stride, dtype=np.int64)

#     if rows[-1] != h - 1:
#         rows = np.append(rows, h - 1)
#     if cols[-1] != w - 1:
#         cols = np.append(cols, w - 1)

#     xx, yy = np.meshgrid(cols.astype(np.float64) * dx, rows.astype(np.float64) * dy)
#     x = xx.ravel()
#     y = yy.ravel()

#     tri = Triangulation(x, y)

#     extent = (
#         -0.5 * dx,
#         (w - 0.5) * dx,
#         (h - 0.5) * dy,
#         -0.5 * dy,
#     )

#     return {
#         "rows": rows,
#         "cols": cols,
#         "x": x,
#         "y": y,
#         "triangles": tri.triangles,
#         "extent": extent,
#     }


# def divergence_from_triangulation(u: np.ndarray, v: np.ndarray, tri_info: Dict[str, Any]) -> np.ma.MaskedArray:
#     u_sub = u[np.ix_(tri_info["rows"], tri_info["cols"])].astype(np.float64).ravel()
#     v_sub = v[np.ix_(tri_info["rows"], tri_info["cols"])].astype(np.float64).ravel()

#     e1, _, _, tri_a, _ = get_deformation_on_triangulation(
#         tri_info["x"], tri_info["y"], u_sub, v_sub, tri_info["triangles"]
#     )

#     e1 = e1 * SECONDS_PER_DAY

#     mask = (~np.isfinite(e1)) | (~np.isfinite(tri_a)) | (tri_a <= 0)
#     return np.ma.masked_array(e1, mask=mask)


# # -----------------------------
# # utilities
# # -----------------------------
# def import_model(module: str, class_name: str):
#     m = importlib.import_module(module)
#     if not hasattr(m, class_name):
#         raise AttributeError(f"Module '{module}' has no class '{class_name}'")
#     return getattr(m, class_name)


# def load_yaml(path: str) -> Dict[str, Any]:
#     with open(path, "r") as f:
#         return yaml.safe_load(f)


# def load_json(path: str) -> Dict[str, Any]:
#     with open(path, "r") as f:
#         return json.load(f)


# @lru_cache(maxsize=8)
# def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
#     records = []
#     with open(path, "r") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 records.append(json.loads(line))
#     return records


# def load_norm_yaml(norm_yaml_path: str) -> Tuple[Dict, Dict]:
#     cfg = load_yaml(norm_yaml_path)
#     return cfg.get("inputs", {}), cfg.get("targets", {})


# def denorm_y(y: torch.Tensor, targets_stats: Dict, normalize_y: bool) -> torch.Tensor:
#     if not normalize_y:
#         return y

#     for k in ("future_drift_u", "future_drift_v"):
#         if k not in targets_stats:
#             raise KeyError(f"Target '{k}' not found in norm YAML targets stats.")

#     device = y.device
#     y_mean = torch.tensor(
#         [targets_stats["future_drift_u"]["mean"], targets_stats["future_drift_v"]["mean"]],
#         dtype=torch.float32,
#         device=device,
#     ).view(-1, 1, 1)

#     y_std = torch.tensor(
#         [targets_stats["future_drift_u"]["std"], targets_stats["future_drift_v"]["std"]],
#         dtype=torch.float32,
#         device=device,
#     ).view(-1, 1, 1)

#     return y * y_std + y_mean


# def denorm_x_channel(x: torch.Tensor, x_channels: List[str], inputs_stats: Dict, ch_name: str) -> torch.Tensor:
#     if ch_name not in x_channels:
#         raise KeyError(f"Channel '{ch_name}' not found in x_channels={x_channels}")

#     ch_idx = x_channels.index(ch_name)
#     x_ch = x[ch_idx]

#     if ch_name not in inputs_stats:
#         warnings.warn(f"Channel '{ch_name}' not found in input norm stats. Returning raw tensor values.")
#         return x_ch

#     mean = float(inputs_stats[ch_name]["mean"])
#     std = float(inputs_stats[ch_name]["std"])
#     return x_ch * std + mean


# def robust_sym_limits(arrays: List[np.ma.MaskedArray], q: float = 0.99, min_v: float = 1e-12) -> float:
#     vals = []
#     for a in arrays:
#         c = a.compressed() if isinstance(a, np.ma.MaskedArray) else np.asarray(a)
#         c = c[np.isfinite(c)]
#         if c.size > 0:
#             vals.append(np.abs(c))

#     if len(vals) == 0:
#         return 1.0

#     vals = np.concatenate(vals)
#     vmax = float(np.quantile(vals, q))
#     if not np.isfinite(vmax) or vmax < min_v:
#         vmax = 1.0
#     return vmax


# def robust_abs_quantile(arrays: List[np.ndarray], q: float, min_v: float = 1e-12) -> float:
#     vals = []
#     for a in arrays:
#         c = a.compressed() if isinstance(a, np.ma.MaskedArray) else np.asarray(a)
#         c = c[np.isfinite(c)]
#         if c.size > 0:
#             vals.append(np.abs(c))

#     if len(vals) == 0:
#         return 1.0

#     vals = np.concatenate(vals)
#     out = float(np.quantile(vals, q))
#     if not np.isfinite(out) or out < min_v:
#         out = min_v
#     return out


# def robust_upper_quantile(arrays: List[np.ndarray], q: float = 0.99, min_v: float = 1e-12) -> float:
#     vals = []
#     for a in arrays:
#         c = np.asarray(a)
#         c = c[np.isfinite(c)]
#         if c.size > 0:
#             vals.append(c)

#     if len(vals) == 0:
#         return 1.0

#     vals = np.concatenate(vals)
#     out = float(np.quantile(vals, q))
#     if not np.isfinite(out) or out < min_v:
#         out = max(min_v, 1.0)
#     return out


# def robust_image_limits(img: np.ndarray, q_low: float = 0.02, q_high: float = 0.98) -> Tuple[float, float]:
#     vals = img[np.isfinite(img)]
#     if vals.size == 0:
#         return 0.0, 1.0
#     vmin = float(np.quantile(vals, q_low))
#     vmax = float(np.quantile(vals, q_high))
#     if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
#         vmin = float(np.nanmin(vals))
#         vmax = float(np.nanmax(vals))
#         if vmin == vmax:
#             vmax = vmin + 1.0
#     return vmin, vmax


# def robust_image_limits_multi(images: List[np.ndarray], q_low: float = 0.02, q_high: float = 0.98) -> Tuple[float, float]:
#     vals = []
#     for img in images:
#         c = np.asarray(img)
#         c = c[np.isfinite(c)]
#         if c.size > 0:
#             vals.append(c)

#     if len(vals) == 0:
#         return 0.0, 1.0

#     vals = np.concatenate(vals)
#     vmin = float(np.quantile(vals, q_low))
#     vmax = float(np.quantile(vals, q_high))

#     if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
#         vmin = float(np.nanmin(vals))
#         vmax = float(np.nanmax(vals))
#         if vmin == vmax:
#             vmax = vmin + 1.0

#     return vmin, vmax


# def pick_cfg(model_cfg: Dict[str, Any], defaults: Dict[str, Any], ckpt_cfg: Dict[str, Any], key: str, default=None):
#     if key in model_cfg and model_cfg[key] is not None:
#         return model_cfg[key]
#     if key in defaults and defaults[key] is not None:
#         return defaults[key]
#     if ckpt_cfg is not None and key in ckpt_cfg and ckpt_cfg[key] is not None:
#         return ckpt_cfg[key]
#     return default


# def normalize_sar_clip_bounds(bounds: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
#     if bounds is None:
#         return {}
#     out = {}
#     for k, v in bounds.items():
#         if len(v) != 2:
#             raise ValueError(f"sar_clip_db_bounds[{k}] must have length 2, got {v}")
#         out[k] = (float(v[0]), float(v[1]))
#     return out


# def sanitize_filename(s: str) -> str:
#     s = str(s)
#     for bad in ["/", "\\", ":", " ", "(", ")", "[", "]", "{", "}", ",", ";"]:
#         s = s.replace(bad, "_")
#     while "__" in s:
#         s = s.replace("__", "_")
#     return s.strip("_")


# def infer_in_ch_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
#     for k, v in state_dict.items():
#         if torch.is_tensor(v) and v.ndim == 4 and k.endswith("weight"):
#             return int(v.shape[1])
#     raise KeyError("Could not infer input channels from checkpoint state_dict.")


# def resolve_channel_name(x_channels: List[str], requested: str) -> Optional[str]:
#     if requested in x_channels:
#         return requested

#     lower_map = {c.lower(): c for c in x_channels}
#     if requested.lower() in lower_map:
#         return lower_map[requested.lower()]

#     aliases = {
#         "HH": ["HH", "hh", "sar_hh_db", "sar_hh", "sar_HH", "HH_db"],
#         "HV": ["HV", "hv", "sar_hv_db", "sar_hv", "sar_HV", "HV_db"],
#         "IA": ["IA", "ia", "sar_incidence_angle", "sar_ia", "sar_IA"],
#         "future_wind_u10_mean": ["future_wind_u10_mean", "wind_u10", "u10", "u_wind", "wind_u"],
#         "future_wind_v10_mean": ["future_wind_v10_mean", "wind_v10", "v10", "v_wind", "wind_v"],
#     }

#     candidates = aliases.get(requested, [requested])
#     for cand in candidates:
#         if cand in x_channels:
#             return cand
#         if cand.lower() in lower_map:
#             return lower_map[cand.lower()]

#     return None


# def resolve_required_channel(x_channels: List[str], requested: str) -> str:
#     resolved = resolve_channel_name(x_channels, requested)
#     if resolved is None:
#         raise KeyError(f"Channel '{requested}' not found in x_channels={x_channels}")
#     return resolved


# def resolve_input_channel_selection(ds, model_cfg: Dict[str, Any], defaults: Dict[str, Any],
#                                     ckpt_cfg: Dict[str, Any], ckpt_state: Dict[str, torch.Tensor]):
#     if not hasattr(ds, "x_channels"):
#         raise AttributeError("Dataset must expose x_channels for channel selection.")

#     x_channels = list(ds.x_channels)

#     requested_names = model_cfg.get("input_channel_names", defaults.get("input_channel_names", None))
#     if isinstance(requested_names, str):
#         requested_names = [c.strip() for c in requested_names.split(",") if c.strip()]

#     expected_in_ch = pick_cfg(model_cfg, defaults, ckpt_cfg, "in_ch", None)
#     if expected_in_ch is None:
#         expected_in_ch = infer_in_ch_from_state_dict(ckpt_state)
#     expected_in_ch = int(expected_in_ch)

#     if requested_names is not None:
#         missing = [c for c in requested_names if c not in x_channels]
#         if missing:
#             raise KeyError(f"Requested input_channel_names {missing} not found in dataset x_channels={x_channels}")
#         indices = [x_channels.index(c) for c in requested_names]
#         names = list(requested_names)
#     else:
#         if expected_in_ch == len(x_channels):
#             indices = list(range(len(x_channels)))
#             names = list(x_channels)
#         elif expected_in_ch == 2:
#             wind_names = ["future_wind_u10_mean", "future_wind_v10_mean"]
#             missing = [c for c in wind_names if c not in x_channels]
#             if missing:
#                 raise KeyError(
#                     f"Checkpoint expects 2 input channels, but wind channels {missing} were not found "
#                     f"in dataset x_channels={x_channels}"
#                 )
#             indices = [x_channels.index(c) for c in wind_names]
#             names = wind_names
#         else:
#             raise ValueError(
#                 f"Checkpoint expects in_ch={expected_in_ch}, but dataset provides {len(x_channels)} channels "
#                 f"and no input_channel_names were specified.\nDataset channels: {x_channels}"
#             )

#     if len(indices) != expected_in_ch:
#         raise ValueError(f"Selected {len(indices)} channels {names}, but checkpoint expects in_ch={expected_in_ch}")

#     return indices, names, expected_in_ch, x_channels


# # -----------------------------
# # time / metadata / warp helpers
# # -----------------------------
# def _find_key_recursive(obj: Any, wanted_keys: List[str]):
#     wanted = {k.lower() for k in wanted_keys}

#     if isinstance(obj, dict):
#         for k, v in obj.items():
#             if str(k).lower() in wanted:
#                 return v
#         for _, v in obj.items():
#             out = _find_key_recursive(v, wanted_keys)
#             if out is not None:
#                 return out

#     elif isinstance(obj, list):
#         for v in obj:
#             out = _find_key_recursive(v, wanted_keys)
#             if out is not None:
#                 return out

#     return None


# def _extract_all_strings(obj: Any) -> List[str]:
#     out = []
#     if isinstance(obj, dict):
#         for _, v in obj.items():
#             out.extend(_extract_all_strings(v))
#     elif isinstance(obj, list):
#         for v in obj:
#             out.extend(_extract_all_strings(v))
#     elif isinstance(obj, str):
#         out.append(obj)
#     return out


# def _normalize_datetime(dt: datetime) -> datetime:
#     if dt.tzinfo is not None:
#         return dt.astimezone(timezone.utc).replace(tzinfo=None)
#     return dt


# def parse_datetime_flexible(text: str) -> Optional[datetime]:
#     if not isinstance(text, str):
#         return None

#     s = text.strip()
#     candidates = [s, os.path.basename(s)]

#     for cand in candidates:
#         try:
#             return _normalize_datetime(datetime.fromisoformat(cand.replace("Z", "+00:00")))
#         except Exception:
#             pass

#     regex_fmts = [
#         (r"\d{8}T\d{6}", "%Y%m%dT%H%M%S"),
#         (r"\d{8}T\d{4}", "%Y%m%dT%H%M"),
#         (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "%Y-%m-%dT%H:%M:%S"),
#         (r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}", "%Y-%m-%d_%H:%M:%S"),
#         (r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "%Y-%m-%d %H:%M:%S"),
#         (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", "%Y-%m-%dT%H:%M"),
#         (r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", "%Y-%m-%d %H:%M"),
#     ]

#     for cand in candidates:
#         for pat, fmt in regex_fmts:
#             m = re.search(pat, cand)
#             if m:
#                 try:
#                     return datetime.strptime(m.group(0), fmt)
#                 except Exception:
#                     pass

#     return None


# def infer_dt_seconds_from_record(rec: Dict[str, Any]) -> Optional[float]:
#     sec_keys = [
#         "dt_seconds", "delta_t_seconds", "future_dt_seconds", "lead_time_seconds",
#         "time_diff_seconds", "drift_dt_seconds", "forecast_seconds"
#     ]
#     val = _find_key_recursive(rec, sec_keys)
#     if val is not None:
#         try:
#             return float(val)
#         except Exception:
#             pass

#     hour_keys = [
#         "dt_hours", "delta_t_hours", "future_dt_hours", "lead_time_hours",
#         "time_diff_hours", "drift_dt_hours", "forecast_hours"
#     ]
#     val = _find_key_recursive(rec, hour_keys)
#     if val is not None:
#         try:
#             return 3600.0 * float(val)
#         except Exception:
#             pass

#     pair_keys = [
#         ("t1", "t2"),
#         ("time1", "time2"),
#         ("start_time", "end_time"),
#         ("source_time", "target_time"),
#         ("current_time", "future_time"),
#         ("obs_time", "forecast_time"),
#         ("sar_time_1", "sar_time_2"),
#         ("sar_time1", "sar_time2"),
#         ("file1", "file2"),
#         ("path1", "path2"),
#         ("source_path", "target_path"),
#         ("sar1_path", "sar2_path"),
#         ("input_path", "target_path"),
#     ]

#     for k1, k2 in pair_keys:
#         v1 = _find_key_recursive(rec, [k1])
#         v2 = _find_key_recursive(rec, [k2])
#         if v1 is not None and v2 is not None:
#             dt1 = parse_datetime_flexible(str(v1))
#             dt2 = parse_datetime_flexible(str(v2))
#             if dt1 is not None and dt2 is not None:
#                 return abs((dt2 - dt1).total_seconds())

#     parsed = []
#     for s in _extract_all_strings(rec):
#         dt = parse_datetime_flexible(s)
#         if dt is not None:
#             parsed.append(dt)

#     parsed = sorted(set(parsed))
#     if len(parsed) == 2:
#         return abs((parsed[1] - parsed[0]).total_seconds())

#     return None


# def get_dt_seconds_for_sample(bundle: Dict[str, Any], sample_idx: int,
#                               cli_drift_seconds: Optional[float],
#                               cli_drift_hours: Optional[float]) -> float:
#     if cli_drift_seconds is not None:
#         return float(cli_drift_seconds)

#     if cli_drift_hours is not None:
#         return 3600.0 * float(cli_drift_hours)

#     records = load_jsonl_records(bundle["test_index"])
#     if sample_idx < 0 or sample_idx >= len(records):
#         raise IndexError(f"sample_idx={sample_idx} out of range for test_index records")

#     dt_seconds = infer_dt_seconds_from_record(records[sample_idx])
#     if dt_seconds is None:
#         raise ValueError(
#             "Could not infer the drift lead time from the index record. "
#             "Please pass --drift-seconds or --drift-hours explicitly."
#         )

#     return float(dt_seconds)


# def _pretty_print_obj(obj, prefix=""):
#     if isinstance(obj, dict):
#         for k, v in obj.items():
#             print(f"{prefix}{k}: {type(v)}")
#             _pretty_print_obj(v, prefix + "  ")
#     elif isinstance(obj, (list, tuple)):
#         print(f"{prefix}len={len(obj)}")
#         for i, v in enumerate(obj[:10]):
#             print(f"{prefix}[{i}]: {type(v)}")
#             _pretty_print_obj(v, prefix + "  ")
#         if len(obj) > 10:
#             print(f"{prefix}... ({len(obj) - 10} more)")
#     else:
#         print(f"{prefix}{repr(obj)}")


# def print_future_drift_npz_metadata(bundle: Dict[str, Any], sample_idx: int):
#     records = load_jsonl_records(bundle["test_index"])
#     rec = records[sample_idx]

#     future_drift_path = _find_key_recursive(rec, ["future_drift_path"])
#     if future_drift_path is None:
#         raise KeyError(f"No 'future_drift_path' found in record for sample_idx={sample_idx}")

#     print("=" * 80)
#     print(f"[debug] sample_idx={sample_idx}")
#     print(f"[debug] future_drift_path={future_drift_path}")
#     print(f"[debug] record keys={list(rec.keys())}")
#     print("=" * 80)

#     with np.load(future_drift_path, allow_pickle=True) as npz:
#         print(f"[debug] npz keys: {list(npz.files)}")

#         for k in npz.files:
#             arr = npz[k]
#             print("-" * 80)
#             print(f"[debug] key='{k}'")
#             print(f"        shape={arr.shape}, dtype={arr.dtype}")

#             if arr.dtype == object:
#                 try:
#                     obj = arr.item()
#                     print(f"        object type={type(obj)}")
#                     _pretty_print_obj(obj, prefix="        ")
#                 except Exception:
#                     try:
#                         print(f"        first object repr={repr(arr.flat[0])}")
#                     except Exception:
#                         print("        could not decode object contents")
#             else:
#                 if arr.size <= 20:
#                     print(f"        values={arr}")
#                 else:
#                     if np.issubdtype(arr.dtype, np.number):
#                         finite = arr[np.isfinite(arr)]
#                         if finite.size > 0:
#                             print(
#                                 f"        min={finite.min():.6g}, max={finite.max():.6g}, "
#                                 f"mean={finite.mean():.6g}"
#                             )
#                     else:
#                         print("        non-numeric array")
#     print("=" * 80)


# def warp_with_forward_flow(img, u, v, n_iter=8, order=1, mode="constant", cval=np.nan):
#     """
#     Warp img using a forward flow (u,v) defined on the source grid:
#         source p -> target p + (u(p), v(p))
#     Produces output on the same grid as img via approximate inversion.

#     u = column displacement in pixels
#     v = row displacement in pixels
#     """
#     rows, cols = img.shape
#     rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")

#     r = rr.astype(np.float64)
#     c = cc.astype(np.float64)

#     for _ in range(n_iter):
#         v_rc = map_coordinates(v, [r, c], order=1, mode="nearest")
#         u_rc = map_coordinates(u, [r, c], order=1, mode="nearest")
#         r = rr - v_rc
#         c = cc - u_rc

#     valid = np.isfinite(img).astype(np.float64)
#     img_filled = np.where(np.isfinite(img), img, 0.0).astype(np.float64)

#     warped_num = map_coordinates(img_filled, [r, c], order=order, mode=mode, cval=0.0)
#     warped_den = map_coordinates(valid, [r, c], order=0, mode=mode, cval=0.0)

#     warped = np.where(warped_den > 0.5, warped_num, cval)
#     return warped


# # -----------------------------
# # plotting helpers
# # -----------------------------
# def _subsample_indices(n: int, stride: int) -> np.ndarray:
#     idx = np.arange(0, n, stride, dtype=np.int64)
#     if idx[-1] != n - 1:
#         idx = np.append(idx, n - 1)
#     return idx


# def plot_vector_field_with_bg(
#     ax,
#     u: np.ndarray,
#     v: np.ndarray,
#     tri_info: Dict[str, Any],
#     title: str,
#     dx: float,
#     dy: float,
#     stride: int,
#     quiver_scale: float,
#     bg_vmin: float,
#     bg_vmax: float,
#     cmap: str = "viridis",
#     vector_color: str = "white",
#     key_value: Optional[float] = None,
# ):
#     speed = np.hypot(u, v)

#     xmin, xmax, ymax, ymin = tri_info["extent"]
#     im = ax.imshow(
#         speed,
#         cmap=cmap,
#         origin="upper",
#         interpolation="nearest",
#         vmin=bg_vmin,
#         vmax=bg_vmax,
#         extent=(xmin, xmax, ymax, ymin),
#         aspect="auto",
#         zorder=1,
#     )

#     h, w = u.shape
#     rows = _subsample_indices(h, stride)
#     cols = _subsample_indices(w, stride)

#     xx, yy = np.meshgrid(cols.astype(np.float64) * dx, rows.astype(np.float64) * dy)
#     uu = u[np.ix_(rows, cols)]
#     vv = v[np.ix_(rows, cols)]

#     q = ax.quiver(
#         xx,
#         yy,
#         uu,
#         vv,
#         angles="xy",
#         scale_units="width",
#         scale=quiver_scale,
#         color=vector_color,
#         pivot="mid",
#         width=0.0045,
#         headwidth=3.5,
#         headlength=4.6,
#         headaxislength=4.1,
#         alpha=0.95,
#         zorder=2,
#     )

#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymax, ymin)
#     ax.set_aspect("auto")
#     ax.set_title(title)
#     ax.set_xticks([])
#     ax.set_yticks([])

#     if key_value is not None and np.isfinite(key_value) and key_value > 0:
#         ax.quiverkey(
#             q,
#             X=0.80,
#             Y=1.04,
#             U=key_value,
#             label=f"{key_value:.2f} m/s",
#             labelpos="E",
#             coordinates="axes",
#             color=vector_color,
#         )

#     return im


# def plot_div_faces(ax, tri_info: Dict[str, Any], div_faces: np.ma.MaskedArray, title: str,
#                    vlim: float, div_scale: str = "symlog", linthresh: float = 1e-3,
#                    cmap: str = "RdBu_r"):
#     tri_plot = Triangulation(
#         tri_info["x"],
#         tri_info["y"],
#         triangles=tri_info["triangles"],
#         mask=np.ma.getmaskarray(div_faces),
#     )

#     if div_scale == "symlog":
#         norm = SymLogNorm(
#             linthresh=linthresh,
#             linscale=1.0,
#             vmin=-vlim,
#             vmax=vlim,
#             base=10,
#         )
#     elif div_scale == "linear":
#         norm = Normalize(vmin=-vlim, vmax=vlim)
#     else:
#         raise ValueError(f"Unsupported div_scale='{div_scale}'")

#     im = ax.tripcolor(
#         tri_plot,
#         facecolors=np.asarray(div_faces.filled(np.nan), dtype=float),
#         cmap=cmap,
#         norm=norm,
#         shading="flat",
#     )

#     xmin, xmax, ymax, ymin = tri_info["extent"]
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymax, ymin)
#     ax.set_aspect("auto")
#     ax.set_title(title)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     return im


# def plot_sar(ax, sar_img: np.ndarray, tri_info: Dict[str, Any], title: str,
#              vmin: float, vmax: float, cmap: str = "gray"):
#     xmin, xmax, ymax, ymin = tri_info["extent"]
#     im = ax.imshow(
#         sar_img,
#         cmap=cmap,
#         origin="upper",
#         interpolation="nearest",
#         vmin=vmin,
#         vmax=vmax,
#         extent=(xmin, xmax, ymax, ymin),
#         aspect="auto",
#     )
#     ax.set_title(title)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     return im


# # -----------------------------
# # model loading
# # -----------------------------
# def build_model_bundle(model_cfg: Dict[str, Any], defaults: Dict[str, Any]):
#     label = model_cfg.get("label", "ML model")

#     ckpt_path = model_cfg.get("ckpt")
#     test_index = model_cfg.get("test_index")
#     norm_yaml = model_cfg.get("norm_yaml")
#     model_module = model_cfg.get("model_module", defaults.get("model_module"))
#     model_class = model_cfg.get("model_class", defaults.get("model_class"))

#     missing = []
#     if ckpt_path is None:
#         missing.append("ckpt")
#     if test_index is None:
#         missing.append("test_index")
#     if norm_yaml is None:
#         missing.append("norm_yaml")
#     if model_module is None:
#         missing.append("model_module")
#     if model_class is None:
#         missing.append("model_class")
#     if missing:
#         raise ValueError(f"Model '{label}' is missing: {missing}")

#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     ckpt_cfg = ckpt.get("config", {}) or {}

#     include_wspd = pick_cfg(model_cfg, defaults, ckpt_cfg, "include_wspd", False)
#     normalize_y = pick_cfg(model_cfg, defaults, ckpt_cfg, "normalize_y", True)
#     base_channels = pick_cfg(model_cfg, defaults, ckpt_cfg, "base_channels", 32)
#     out_ch = pick_cfg(model_cfg, defaults, ckpt_cfg, "out_ch", 2)

#     sar_channels = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_channels", ["HH", "HV", "IA"])
#     sar_to_db = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_to_db", True)
#     sar_postprocess = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_postprocess", True)
#     sar_zero_is_nodata = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_zero_is_nodata", False)
#     sar_clip_db = pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_clip_db", True)
#     sar_clip_db_bounds = normalize_sar_clip_bounds(
#         pick_cfg(model_cfg, defaults, ckpt_cfg, "sar_clip_db_bounds", {})
#     )

#     if isinstance(sar_channels, str):
#         sar_channels = [c.strip() for c in sar_channels.split(",") if c.strip()]

#     inputs_stats, targets_stats = load_norm_yaml(norm_yaml)

#     ds = DriftWindSARDataset(
#         test_index,
#         norm_yaml_path=norm_yaml,
#         normalize_y=normalize_y,
#         include_wspd=include_wspd,
#         return_meta=False,
#         cache_size=0,
#         sar_channels=tuple(sar_channels),
#         sar_to_db=sar_to_db,
#         sar_postprocess=sar_postprocess,
#         sar_clip_percentiles=None,
#         sar_zero_is_nodata=sar_zero_is_nodata,
#         sar_clip_db=sar_clip_db,
#         sar_clip_db_bounds=sar_clip_db_bounds,
#     )

#     sample0 = ds[0]
#     dataset_in_ch = sample0["x"].shape[0]

#     input_channel_indices, input_channel_names, in_ch, x_channels = resolve_input_channel_selection(
#         ds=ds,
#         model_cfg=model_cfg,
#         defaults=defaults,
#         ckpt_cfg=ckpt_cfg,
#         ckpt_state=ckpt["model_state"],
#     )

#     ModelClass = import_model(model_module, model_class)
#     model = ModelClass(
#         in_channels=int(in_ch),
#         out_channels=int(out_ch),
#         base_channels=int(base_channels),
#     )
#     model.load_state_dict(ckpt["model_state"], strict=True)
#     model.eval()
#     model.cpu()

#     print(
#         f"[load] {label}\n"
#         f"       dataset_in_ch={dataset_in_ch} model_in_ch={in_ch} out_ch={out_ch} base_channels={base_channels}\n"
#         f"       selected_inputs={input_channel_names}\n"
#         f"       dataset_x_channels={x_channels}"
#     )

#     return {
#         "label": label,
#         "dataset": ds,
#         "model": model,
#         "inputs_stats": inputs_stats,
#         "targets_stats": targets_stats,
#         "normalize_y": bool(normalize_y),
#         "input_channel_indices": input_channel_indices,
#         "x_channels": x_channels,
#         "sar_to_db": bool(sar_to_db),
#         "test_index": test_index,
#     }


# # -----------------------------
# # inference + baseline + SAR
# # -----------------------------
# def infer_ml_prediction(bundle: Dict[str, Any], sample_idx: int, device: torch.device,
#                         use_amp: bool, keep_models_on_device: bool):
#     ds = bundle["dataset"]
#     model = bundle["model"]

#     s = ds[sample_idx]
#     x_full = s["x"].float()
#     x_model = x_full[bundle["input_channel_indices"]]
#     y = s["y"].float()

#     sid = s.get("id", sample_idx)
#     t = s.get("t", "")

#     model = model.to(device)

#     with torch.inference_mode():
#         x_dev = x_model.unsqueeze(0).to(device)
#         if use_amp:
#             with torch.amp.autocast("cuda", enabled=True):
#                 pred = model(x_dev).squeeze(0)
#         else:
#             pred = model(x_dev).squeeze(0)

#     pred = pred.detach().cpu()

#     if device.type == "cuda" and not keep_models_on_device:
#         model.cpu()
#         torch.cuda.empty_cache()

#     y_raw = denorm_y(y, bundle["targets_stats"], bundle["normalize_y"])
#     pred_raw = denorm_y(pred, bundle["targets_stats"], bundle["normalize_y"])

#     return {
#         "sid": sid,
#         "t": t,
#         "target_u": y_raw[0].cpu().numpy(),
#         "target_v": y_raw[1].cpu().numpy(),
#         "pred_u": pred_raw[0].cpu().numpy(),
#         "pred_v": pred_raw[1].cpu().numpy(),
#     }


# def extract_sar_image(bundle: Dict[str, Any], sample_idx: int, sar_channel: str) -> np.ndarray:
#     s = bundle["dataset"][sample_idx]
#     x_full = s["x"].float()

#     resolved_name = resolve_required_channel(bundle["x_channels"], sar_channel)
#     img = denorm_x_channel(
#         x=x_full,
#         x_channels=bundle["x_channels"],
#         inputs_stats=bundle["inputs_stats"],
#         ch_name=resolved_name,
#     )
#     return img.cpu().numpy()


# def extract_future_wind(bundle: Dict[str, Any], sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
#     s = bundle["dataset"][sample_idx]
#     x_full = s["x"].float()

#     u_name = resolve_required_channel(bundle["x_channels"], "future_wind_u10_mean")
#     v_name = resolve_required_channel(bundle["x_channels"], "future_wind_v10_mean")

#     u_wind = denorm_x_channel(
#         x=x_full,
#         x_channels=bundle["x_channels"],
#         inputs_stats=bundle["inputs_stats"],
#         ch_name=u_name,
#     ).cpu().numpy()

#     v_wind = denorm_x_channel(
#         x=x_full,
#         x_channels=bundle["x_channels"],
#         inputs_stats=bundle["inputs_stats"],
#         ch_name=v_name,
#     ).cpu().numpy()

#     return u_wind, v_wind


# def baseline_from_wind(u_wind: np.ndarray, v_wind: np.ndarray, baseline_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
#     A = float(baseline_cfg["A"])
#     B = float(baseline_cfg["B"])

#     u_ice = A * u_wind - B * v_wind
#     v_ice = B * u_wind + A * v_wind
#     return u_ice, v_ice


# # -----------------------------
# # main
# # -----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--compare-yaml", required=True, help="YAML with the best ML model as the first entry in models:")
#     ap.add_argument("--baseline-json", required=True, help="JSON containing at least A and B")
#     ap.add_argument("--outdir", required=True)

#     ap.add_argument("--num-samples", type=int, default=10)
#     ap.add_argument("--sample-idx", type=int, default=None)
#     ap.add_argument("--seed", type=int, default=0)

#     ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
#     ap.add_argument("--force-cpu", action="store_true")

#     ap.add_argument("--dx", type=float, default=100.0)
#     ap.add_argument("--dy", type=float, default=100.0)

#     ap.add_argument("--stride", type=int, default=8, help="Triangulation stride for divergence")
#     ap.add_argument("--quiver-stride", type=int, default=16, help="Subsampling stride for vector arrows")

#     ap.add_argument("--quiver-ref-q", type=float, default=0.95,
#                     help="Quantile used to define the quiver key magnitude")
#     ap.add_argument("--quiver-ref-frac", type=float, default=0.12,
#                     help="Approximate fraction of axes width used for the quiver key arrow")
#     ap.add_argument("--show-quiver-key", action="store_true",
#                     help="Show quiver key on the target vector panel")

#     ap.add_argument("--vel-q", type=float, default=0.99,
#                     help="Robust upper quantile for shared velocity color scale")
#     ap.add_argument("--div-q", type=float, default=0.99,
#                     help="Robust upper quantile for shared divergence color scale")
#     ap.add_argument("--div-scale", choices=["symlog", "linear"], default="symlog",
#                     help="Use symmetric-log or linear scaling for divergence")
#     ap.add_argument("--symlog-linthresh-q", type=float, default=0.80,
#                     help="Quantile for symlog linear threshold")

#     ap.add_argument("--sar-q-low", type=float, default=0.02)
#     ap.add_argument("--sar-q-high", type=float, default=0.98)

#     ap.add_argument("--drift-seconds", type=float, default=None,
#                     help="Lead time in seconds used to convert drift [m/s] to pixel displacement")
#     ap.add_argument("--drift-hours", type=float, default=None,
#                     help="Lead time in hours used to convert drift [m/s] to pixel displacement")
#     ap.add_argument("--warp-n-iter", type=int, default=8)
#     ap.add_argument("--warp-order", type=int, default=1)
#     ap.add_argument("--warp-mode", default="constant", choices=["constant", "nearest", "reflect", "mirror", "wrap"])
#     ap.add_argument("--flip-v-for-warp", action="store_true",
#                     help="Flip sign of v when converting to row displacement, if vertical warp direction looks wrong")

#     ap.add_argument("--print-future-drift-metadata", action="store_true",
#                     help="Print metadata from future_drift_path NPZ for the chosen sample")
#     ap.add_argument("--stop-after-future-drift-metadata", action="store_true",
#                     help="Exit after printing future_drift_path metadata")

#     ap.add_argument("--sar-channel", default="sar_hh_db")
#     ap.add_argument("--fontsize", type=int, default=9)
#     ap.add_argument("--height-ratio", type=float, default=1.05)

#     ap.add_argument("--keep-models-on-device", action="store_true")

#     args = ap.parse_args()

#     if args.drift_seconds is not None and args.drift_hours is not None:
#         raise ValueError("Pass only one of --drift-seconds or --drift-hours, not both.")

#     setup_pub_style(fontsize=args.fontsize)
#     os.makedirs(args.outdir, exist_ok=True)

#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     if args.force_cpu or args.device == "cpu":
#         device = torch.device("cpu")
#     elif args.device == "cuda":
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     use_amp = (device.type == "cuda")

#     cfg = load_yaml(args.compare_yaml)
#     defaults = cfg.get("defaults", {}) or {}
#     models_cfg = cfg.get("models", None)
#     if not isinstance(models_cfg, list) or len(models_cfg) == 0:
#         raise ValueError("compare YAML must contain a non-empty 'models:' list.")

#     if len(models_cfg) > 1:
#         warnings.warn("More than one model found in compare YAML. Only the first model will be used.")

#     ml_bundle = build_model_bundle(models_cfg[0], defaults)
#     baseline_cfg = load_json(args.baseline_json)

#     if "A" not in baseline_cfg or "B" not in baseline_cfg:
#         raise KeyError(f"--baseline-json must contain keys 'A' and 'B'. Got keys={list(baseline_cfg.keys())}")

#     if args.print_future_drift_metadata:
#         inspect_idx = args.sample_idx if args.sample_idx is not None else 0
#         print_future_drift_npz_metadata(ml_bundle, inspect_idx)
#         if args.stop_after_future_drift_metadata:
#             return

#     print(f"[run ] device={device}")
#     print(f"[grid] dx={args.dx} dy={args.dy}")
#     print(f"[div ] stride={args.stride} scale={args.div_scale}")
#     print(f"[quiv] stride={args.quiver_stride} ref_q={args.quiver_ref_q} ref_frac={args.quiver_ref_frac}")
#     print(f"[vel ] q={args.vel_q}")
#     print(f"[sar ] requested_channel={args.sar_channel}")
#     print(f"[base] A={baseline_cfg['A']} B={baseline_cfg['B']}")

#     if args.keep_models_on_device:
#         ml_bundle["model"] = ml_bundle["model"].to(device)

#     ds_len = len(ml_bundle["dataset"])
#     if args.sample_idx is not None:
#         if args.sample_idx < 0 or args.sample_idx >= ds_len:
#             raise ValueError(f"--sample-idx={args.sample_idx} out of range for dataset length {ds_len}")
#         picks = [args.sample_idx]
#     else:
#         picks = random.sample(range(ds_len), k=min(args.num_samples, ds_len))

#     for idx in picks:
#         ml_res = infer_ml_prediction(
#             ml_bundle,
#             sample_idx=idx,
#             device=device,
#             use_amp=use_amp,
#             keep_models_on_device=args.keep_models_on_device,
#         )

#         sid = ml_res["sid"]
#         t = ml_res["t"]

#         target_u = ml_res["target_u"]
#         target_v = ml_res["target_v"]
#         ml_u = ml_res["pred_u"]
#         ml_v = ml_res["pred_v"]

#         wind_u, wind_v = extract_future_wind(ml_bundle, idx)
#         base_u, base_v = baseline_from_wind(wind_u, wind_v, baseline_cfg)

#         dt_seconds = get_dt_seconds_for_sample(
#             ml_bundle,
#             sample_idx=idx,
#             cli_drift_seconds=args.drift_seconds,
#             cli_drift_hours=args.drift_hours,
#         )

#         h, w = target_u.shape
#         tri_info = make_subsampled_triangulation(h=h, w=w, dx=args.dx, dy=args.dy, stride=args.stride)

#         # divergence
#         div_target = divergence_from_triangulation(target_u, target_v, tri_info)
#         div_base = divergence_from_triangulation(base_u, base_v, tri_info)
#         div_ml = divergence_from_triangulation(ml_u, ml_v, tri_info)

#         div_arrays = [div_target, div_base, div_ml]
#         div_vlim = robust_sym_limits(div_arrays, q=args.div_q)
#         div_linthresh = robust_abs_quantile(div_arrays, q=args.symlog_linthresh_q, min_v=1e-10)

#         # speed for vector backgrounds
#         speed_target = np.hypot(target_u, target_v)
#         speed_base = np.hypot(base_u, base_v)
#         speed_ml = np.hypot(ml_u, ml_v)

#         vel_vmax = robust_upper_quantile([speed_target, speed_base, speed_ml], q=args.vel_q)
#         vel_vmin = 0.0

#         speed_ref = robust_upper_quantile([speed_target, speed_base, speed_ml], q=args.quiver_ref_q, min_v=1e-6)
#         quiver_scale = speed_ref / max(args.quiver_ref_frac, 1e-6)

#         # SAR
#         sar_img = extract_sar_image(ml_bundle, idx, sar_channel=args.sar_channel)

#         sign_v = -1.0 if args.flip_v_for_warp else 1.0
#         base_u_pix = base_u * dt_seconds / args.dx
#         base_v_pix = sign_v * base_v * dt_seconds / args.dy
#         ml_u_pix = ml_u * dt_seconds / args.dx
#         ml_v_pix = sign_v * ml_v * dt_seconds / args.dy

#         sar_warp_base = warp_with_forward_flow(
#             sar_img,
#             base_u_pix,
#             base_v_pix,
#             n_iter=args.warp_n_iter,
#             order=args.warp_order,
#             mode=args.warp_mode,
#             cval=np.nan,
#         )

#         sar_warp_ml = warp_with_forward_flow(
#             sar_img,
#             ml_u_pix,
#             ml_v_pix,
#             n_iter=args.warp_n_iter,
#             order=args.warp_order,
#             mode=args.warp_mode,
#             cval=np.nan,
#         )

#         sar_vmin, sar_vmax = robust_image_limits_multi(
#             [sar_img, sar_warp_base, sar_warp_ml],
#             q_low=args.sar_q_low,
#             q_high=args.sar_q_high,
#         )

#         print(
#             f"[sample] idx={idx} id={sid} t={t} "
#             f"dt_seconds={dt_seconds:.1f} "
#             f"vel_vmax={vel_vmax:.4g} div_vlim={div_vlim:.4g}"
#         )

#         fig = plt.figure(
#             figsize=fig_textwidth(height_ratio=args.height_ratio),
#             constrained_layout=True,
#         )
#         gs = fig.add_gridspec(
#             3, 4,
#             width_ratios=[1.0, 1.0, 1.0, 0.06],
#             height_ratios=[1.0, 1.0, 1.0],
#         )

#         # row 1
#         ax_tvec = fig.add_subplot(gs[0, 0])
#         ax_bvec = fig.add_subplot(gs[0, 1])
#         ax_mvec = fig.add_subplot(gs[0, 2])
#         cax_vel = fig.add_subplot(gs[0, 3])

#         # row 2
#         ax_tdiv = fig.add_subplot(gs[1, 0])
#         ax_bdiv = fig.add_subplot(gs[1, 1])
#         ax_mdiv = fig.add_subplot(gs[1, 2])
#         cax_div = fig.add_subplot(gs[1, 3])

#         # row 3
#         ax_sar0 = fig.add_subplot(gs[2, 0])
#         ax_sar_b = fig.add_subplot(gs[2, 1])
#         ax_sar_m = fig.add_subplot(gs[2, 2])
#         cax_sar = fig.add_subplot(gs[2, 3])

#         for ax in [ax_tvec, ax_bvec, ax_mvec, ax_tdiv, ax_bdiv, ax_mdiv, ax_sar0, ax_sar_b, ax_sar_m]:
#             ax.set_box_aspect(1)

#         # row 1: vector fields with speed background
#         im_vel = plot_vector_field_with_bg(
#             ax_tvec,
#             target_u,
#             target_v,
#             tri_info,
#             "Target drift vector field",
#             dx=args.dx,
#             dy=args.dy,
#             stride=args.quiver_stride,
#             quiver_scale=quiver_scale,
#             bg_vmin=vel_vmin,
#             bg_vmax=vel_vmax,
#             cmap="viridis",
#             vector_color="white",
#             key_value=(speed_ref if args.show_quiver_key else None),
#         )

#         plot_vector_field_with_bg(
#             ax_bvec,
#             base_u,
#             base_v,
#             tri_info,
#             "Wind baseline vector field",
#             dx=args.dx,
#             dy=args.dy,
#             stride=args.quiver_stride,
#             quiver_scale=quiver_scale,
#             bg_vmin=vel_vmin,
#             bg_vmax=vel_vmax,
#             cmap="viridis",
#             vector_color="white",
#             key_value=None,
#         )

#         plot_vector_field_with_bg(
#             ax_mvec,
#             ml_u,
#             ml_v,
#             tri_info,
#             f"{ml_bundle['label']} vector field",
#             dx=args.dx,
#             dy=args.dy,
#             stride=args.quiver_stride,
#             quiver_scale=quiver_scale,
#             bg_vmin=vel_vmin,
#             bg_vmax=vel_vmax,
#             cmap="viridis",
#             vector_color="white",
#             key_value=None,
#         )

#         cbar_vel = fig.colorbar(im_vel, cax=cax_vel)
#         cbar_vel.set_label("Drift speed [m/s]")

#         # row 2: divergence
#         im_div = plot_div_faces(
#             ax_tdiv,
#             tri_info,
#             div_target,
#             "Target divergence",
#             vlim=div_vlim,
#             div_scale=args.div_scale,
#             linthresh=div_linthresh,
#             cmap="RdBu_r",
#         )

#         plot_div_faces(
#             ax_bdiv,
#             tri_info,
#             div_base,
#             "Wind baseline divergence",
#             vlim=div_vlim,
#             div_scale=args.div_scale,
#             linthresh=div_linthresh,
#             cmap="RdBu_r",
#         )

#         plot_div_faces(
#             ax_mdiv,
#             tri_info,
#             div_ml,
#             f"{ml_bundle['label']} divergence",
#             vlim=div_vlim,
#             div_scale=args.div_scale,
#             linthresh=div_linthresh,
#             cmap="RdBu_r",
#         )

#         cbar_div = fig.colorbar(im_div, cax=cax_div)
#         cbar_div.set_label("Divergence [1/day]")

#         # row 3: SAR
#         im_sar = plot_sar(
#             ax_sar0,
#             sar_img,
#             tri_info,
#             title="SAR HH",
#             vmin=sar_vmin,
#             vmax=sar_vmax,
#             cmap="gray",
#         )

#         plot_sar(
#             ax_sar_b,
#             sar_warp_base,
#             tri_info,
#             title="SAR HH warped by wind baseline",
#             vmin=sar_vmin,
#             vmax=sar_vmax,
#             cmap="gray",
#         )

#         plot_sar(
#             ax_sar_m,
#             sar_warp_ml,
#             tri_info,
#             title="SAR HH warped by ML model",
#             vmin=sar_vmin,
#             vmax=sar_vmax,
#             cmap="gray",
#         )

#         cbar_sar = fig.colorbar(im_sar, cax=cax_sar)
#         if "db" in args.sar_channel.lower():
#             cbar_sar.set_label("SAR HH [dB]")
#         else:
#             cbar_sar.set_label("SAR HH")

#         out_name = f"baseline_vs_ml_warped_sar_idx{idx}_id{sanitize_filename(sid)}"
#         if str(t).strip():
#             out_name += f"_{sanitize_filename(t)}"
#         out_path = os.path.join(args.outdir, out_name + ".png")

#         fig.savefig(out_path, bbox_inches="tight")
#         plt.close(fig)
#         print(f"Saved: {out_path}")


# if __name__ == "__main__":
#     main()





















#!/usr/bin/env python3
"""
Compare a wind baseline against one ML drift model.

Layout
------
Row 1:
  [0,0] Target drift speed + vectors
  [0,1] Wind baseline drift speed + vectors
  [0,2] ML drift speed + vectors
  [0,3] Shared colorbar for drift speed

Row 2:
  [1,0] Target divergence
  [1,1] Wind baseline divergence
  [1,2] ML divergence
  [1,3] Shared colorbar for divergence

Row 3:
  [2,0] Observed SAR at end time
  [2,1] Start SAR warped by wind baseline
  [2,2] Start SAR warped by ML drift
  [2,3] Shared SAR colorbar

Notes
-----
- The start SAR comes from the dataset input channels.
- The end SAR comes from:
      index_jsonl -> future_drift_path -> npz["meta"].item()["end_path"]
- The warp uses:
      u_pix = u * dt_seconds / dx
      v_pix = v * dt_seconds / dy
  where dt_seconds comes from npz["meta"].item()["dt_seconds"].
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


def fig_textwidth(height_ratio=1.05):
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
# deformation / divergence
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


def divergence_from_triangulation(u: np.ndarray, v: np.ndarray, tri_info: Dict[str, Any]) -> np.ma.MaskedArray:
    u_sub = u[np.ix_(tri_info["rows"], tri_info["cols"])].astype(np.float64).ravel()
    v_sub = v[np.ix_(tri_info["rows"], tri_info["cols"])].astype(np.float64).ravel()

    e1, _, _, tri_a, _ = get_deformation_on_triangulation(
        tri_info["x"], tri_info["y"], u_sub, v_sub, tri_info["triangles"]
    )

    e1 = e1 * SECONDS_PER_DAY
    mask = (~np.isfinite(e1)) | (~np.isfinite(tri_a)) | (tri_a <= 0)
    return np.ma.masked_array(e1, mask=mask)


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
        x=x_full,
        x_channels=bundle["x_channels"],
        inputs_stats=bundle["inputs_stats"],
        ch_name=resolved_name,
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
    """
    Warp img using a forward flow (u,v) defined on the source grid:
        source p -> target p + (u(p), v(p))

    u = column displacement in pixels
    v = row displacement in pixels
    """
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
    speed = np.hypot(u, v)

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
    uu = u[np.ix_(rows, cols)]
    vv = v[np.ix_(rows, cols)]

    q = ax.quiver(
        xx,
        yy,
        uu,
        vv,
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


def plot_div_faces(ax, tri_info: Dict[str, Any], div_faces: np.ma.MaskedArray, title: str,
                   vlim: float, div_scale: str = "symlog", linthresh: float = 1e-3,
                   cmap: str = "RdBu_r"):
    tri_plot = Triangulation(
        tri_info["x"],
        tri_info["y"],
        triangles=tri_info["triangles"],
        mask=np.ma.getmaskarray(div_faces),
    )

    if div_scale == "symlog":
        norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-vlim, vmax=vlim, base=10)
    else:
        norm = Normalize(vmin=-vlim, vmax=vlim)

    im = ax.tripcolor(
        tri_plot,
        facecolors=np.asarray(div_faces.filled(np.nan), dtype=float),
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
        "test_index": test_index,
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

    ap.add_argument("--sar-q-low", type=float, default=0.02)
    ap.add_argument("--sar-q-high", type=float, default=0.98)
    ap.add_argument("--sar-channel", default="sar_hh_db")

    ap.add_argument("--warp-n-iter", type=int, default=8)
    ap.add_argument("--warp-order", type=int, default=1)
    ap.add_argument("--warp-mode", default="constant", choices=["constant", "nearest", "reflect", "mirror", "wrap"])
    ap.add_argument("--flip-v-for-warp", action="store_true")

    ap.add_argument("--fontsize", type=int, default=9)
    ap.add_argument("--height-ratio", type=float, default=1.05)

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
        picks = [args.sample_idx]
    else:
        picks = random.sample(range(ds_len), k=min(args.num_samples, ds_len))

    print(f"[run ] device={device}")
    print(f"[grid] dx={args.dx} dy={args.dy}")
    print(f"[div ] stride={args.stride} scale={args.div_scale}")
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

        div_target = divergence_from_triangulation(target_u, target_v, tri_info)
        div_base = divergence_from_triangulation(base_u, base_v, tri_info)
        div_ml = divergence_from_triangulation(ml_u, ml_v, tri_info)

        div_vlim = robust_sym_limits([div_target, div_base, div_ml], q=args.div_q)
        div_linthresh = robust_abs_quantile([div_target, div_base, div_ml], q=args.symlog_linthresh_q, min_v=1e-10)

        speed_target = np.hypot(target_u, target_v)
        speed_base = np.hypot(base_u, base_v)
        speed_ml = np.hypot(ml_u, ml_v)

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

        fig = plt.figure(figsize=fig_textwidth(args.height_ratio), constrained_layout=True)
        gs = fig.add_gridspec(
            3, 4,
            width_ratios=[1.0, 1.0, 1.0, 0.06],
            height_ratios=[1.0, 1.0, 1.0],
        )

        ax_tvec = fig.add_subplot(gs[0, 0])
        ax_bvec = fig.add_subplot(gs[0, 1])
        ax_mvec = fig.add_subplot(gs[0, 2])
        cax_vel = fig.add_subplot(gs[0, 3])

        ax_tdiv = fig.add_subplot(gs[1, 0])
        ax_bdiv = fig.add_subplot(gs[1, 1])
        ax_mdiv = fig.add_subplot(gs[1, 2])
        cax_div = fig.add_subplot(gs[1, 3])

        ax_sar0 = fig.add_subplot(gs[2, 0])
        ax_sar_b = fig.add_subplot(gs[2, 1])
        ax_sar_m = fig.add_subplot(gs[2, 2])
        cax_sar = fig.add_subplot(gs[2, 3])

        for ax in [ax_tvec, ax_bvec, ax_mvec, ax_tdiv, ax_bdiv, ax_mdiv, ax_sar0, ax_sar_b, ax_sar_m]:
            ax.set_box_aspect(1)

        im_vel = plot_vector_field_with_bg(
            ax_tvec, target_u, target_v, tri_info, "Target drift vector field",
            dx=args.dx, dy=args.dy, stride=args.quiver_stride,
            quiver_scale=quiver_scale, bg_vmin=0.0, bg_vmax=vel_vmax,
            key_value=(speed_ref if args.show_quiver_key else None),
        )
        plot_vector_field_with_bg(
            ax_bvec, base_u, base_v, tri_info, "Wind baseline vector field",
            dx=args.dx, dy=args.dy, stride=args.quiver_stride,
            quiver_scale=quiver_scale, bg_vmin=0.0, bg_vmax=vel_vmax,
        )
        plot_vector_field_with_bg(
            ax_mvec, ml_u, ml_v, tri_info, f"{bundle['label']} vector field",
            dx=args.dx, dy=args.dy, stride=args.quiver_stride,
            quiver_scale=quiver_scale, bg_vmin=0.0, bg_vmax=vel_vmax,
        )
        cbar_vel = fig.colorbar(im_vel, cax=cax_vel)
        cbar_vel.set_label("Drift speed [m/s]")

        im_div = plot_div_faces(
            ax_tdiv, tri_info, div_target, "Target divergence",
            vlim=div_vlim, div_scale=args.div_scale, linthresh=div_linthresh
        )
        plot_div_faces(
            ax_bdiv, tri_info, div_base, "Wind baseline divergence",
            vlim=div_vlim, div_scale=args.div_scale, linthresh=div_linthresh
        )
        plot_div_faces(
            ax_mdiv, tri_info, div_ml, f"{bundle['label']} divergence",
            vlim=div_vlim, div_scale=args.div_scale, linthresh=div_linthresh
        )
        cbar_div = fig.colorbar(im_div, cax=cax_div)
        cbar_div.set_label("Divergence [1/day]")

        im_sar = plot_sar(
            ax_sar0, sar_end, tri_info,
            title="Observed SAR HH at end time",
            vmin=sar_vmin, vmax=sar_vmax
        )
        plot_sar(
            ax_sar_b, sar_warp_base, tri_info,
            title="Start SAR HH warped by wind baseline",
            vmin=sar_vmin, vmax=sar_vmax
        )
        plot_sar(
            ax_sar_m, sar_warp_ml, tri_info,
            title="Start SAR HH warped by ML model",
            vmin=sar_vmin, vmax=sar_vmax
        )
        cbar_sar = fig.colorbar(im_sar, cax=cax_sar)
        if "db" in args.sar_channel.lower():
            cbar_sar.set_label("SAR HH [dB]")
        else:
            cbar_sar.set_label("SAR HH")

        out_name = f"baseline_vs_ml_endsar_idx{idx}_id{sanitize_filename(sid)}"
        if str(t).strip():
            out_name += f"_{sanitize_filename(t)}"
        out_path = os.path.join(args.outdir, out_name + ".png")

        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()