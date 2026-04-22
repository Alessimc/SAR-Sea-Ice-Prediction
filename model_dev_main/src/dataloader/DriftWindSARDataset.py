import json
from collections import OrderedDict
from typing import Optional, Dict, Any, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

import rioxarray as rxr


class DriftWindSARDataset(Dataset):
    """
    Dataset returning:
      X: (C,H,W) = [past_drift_u, past_drift_v, future_wind_u10_mean, future_wind_v10_mean, ... SAR ...]
      Y: (2,H,W) = [future_drift_u, future_drift_v]

    SAR handling:
      - Reads SAR GeoTIFF (rioxarray)
      - Converts HH/HV to dB if sar_to_db=True: 10*log10(max(x, sar_db_eps))
      - Normalizes using YAML mean/std (requires matching channel names)
      - ONLY for SAR HH/HV channels (after normalization):
          * percentile clip (default 1st99th) computed over finite values only
          * then nan/inf -> 0

    Notes:
      - Percentile clip and nan_to_num are NOT applied to drift/wind channels.
      - IA (incidence angle) is NOT postprocessed unless you extend the indices.
    """

    DEFAULT_X_CHANNELS = [
        "past_drift_u",
        "past_drift_v",
        "future_wind_u10_mean",
        "future_wind_v10_mean",
    ]
    DEFAULT_Y_CHANNELS = [
        "future_drift_u",
        "future_drift_v",
    ]

    # You said: band0=HH, band1=HV, band2=incidence_angle
    SAR_BAND_INDEX = {"HH": 0, "HV": 1, "IA": 2}

    def __init__(
        self,
        index_jsonl: str,
        include_wspd: bool = False,
        return_meta: bool = False,
        norm_yaml_path: Optional[str] = None,
        normalize_y: bool = True,
        eps: float = 1e-6,
        cache_size: int = 0,
        x_groups: Optional[Sequence[str]] = None,

        # SAR controls
        sar_channels: Sequence[str] = ("HV",),   # choose from ("HH","HV","IA")
        sar_to_db: bool = True,
        sar_db_eps: float = 1e-6,

        # SAR postprocessing (ONLY applied to SAR HH/HV channels)
        sar_postprocess: bool = True,
        sar_clip_percentiles: Optional[Tuple[float, float]] = (0.01, 0.99),  #  None to disable
        sar_zero_is_nodata: bool = False,  # if True: raw==0 for HH/HV becomes NaN before dB

        sar_clip_db: bool = False,
        sar_clip_db_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.include_wspd = bool(include_wspd)
        self.return_meta = bool(return_meta)
        self.normalize_y = bool(normalize_y)

        self.x_groups = [g.lower() for g in (x_groups or ("drift","wind","sar"))]

        self.sar_channels = [c.upper() for c in sar_channels]
        for c in self.sar_channels:
            if c not in self.SAR_BAND_INDEX:
                raise ValueError(f"Unknown sar channel '{c}'. Choose from {list(self.SAR_BAND_INDEX.keys())}")
    
        self.sar_to_db = bool(sar_to_db)
        self.sar_db_eps = float(sar_db_eps)

        self.sar_postprocess = bool(sar_postprocess)
        self.sar_clip_percentiles = sar_clip_percentiles
        self.sar_zero_is_nodata = bool(sar_zero_is_nodata)

        self.sar_clip_db = bool(sar_clip_db)
        self.sar_clip_db_bounds = sar_clip_db_bounds or {}

        if self.sar_clip_db and self.sar_to_db:
            for c in ("HH", "HV"):
                if c in self.sar_channels and c not in self.sar_clip_db_bounds:
                    raise ValueError(f"sar_clip_db_bounds missing for {c}")

        # Build channel name list (must match YAML keys)
        drift_ch = ["past_drift_u", "past_drift_v"]

        wind_ch = ["future_wind_u10_mean", "future_wind_v10_mean"]
        if self.include_wspd:
            wind_ch.append("future_wind_wspd_mean")

        sar_ch = []
        for c in self.sar_channels:
            if c == "HH":
                sar_ch.append("sar_hh_db" if self.sar_to_db else "sar_hh")
            elif c == "HV":
                sar_ch.append("sar_hv_db" if self.sar_to_db else "sar_hv")
            elif c == "IA":
                sar_ch.append("sar_incidence_angle")

        self.x_channels = []
        if "drift" in self.x_groups:
            self.x_channels += drift_ch
        if "wind" in self.x_groups:
            self.x_channels += wind_ch
        if "sar" in self.x_groups:
            self.x_channels += sar_ch

        self.y_channels = self.DEFAULT_Y_CHANNELS.copy()

        # Pre-parse JSONL into lists for fast indexing
        self.wind_paths: List[str] = []
        self.past_paths: List[str] = []
        self.future_paths: List[str] = []
        self.sar_t_paths: List[str] = []
        self.ids: List[Any] = []
        self.ts: List[Any] = []

        with open(index_jsonl, "r") as f:
            for line in f:
                r = json.loads(line)
                self.wind_paths.append(r["future_wind_path"])
                self.past_paths.append(r["past_drift_path"])
                self.future_paths.append(r["future_drift_path"])
                self.sar_t_paths.append(r["sar_t_path"])
                self.ids.append(r.get("id"))
                self.ts.append(r.get("t"))

        # Normalization params (CPU tensors shaped (C,1,1))
        self.do_norm = norm_yaml_path is not None
        if self.do_norm:
            self.x_mean, self.x_std, self.y_mean, self.y_std = self._load_norm_yaml(norm_yaml_path, eps=eps)

        # Compute indices for SAR HH/HV channels inside x (so we can clip/sanitize only those)
        hh_name = "sar_hh_db" if self.sar_to_db else "sar_hh"
        hv_name = "sar_hv_db" if self.sar_to_db else "sar_hv"
        self._sar_hh_hv_x_idx: List[int] = [
            i for i, name in enumerate(self.x_channels) if name in (hh_name, hv_name)
        ]

        # Simple per-worker LRU cache
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()

    def _load_norm_yaml(self, path: str, eps: float):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        inputs = cfg.get("inputs", {})
        targets = cfg.get("targets", {})

        missing_x = [c for c in self.x_channels if c not in inputs]
        missing_y = [c for c in self.y_channels if c not in targets]
        if missing_x:
            raise KeyError(f"Normalization YAML missing input channels: {missing_x}")
        if missing_y:
            raise KeyError(f"Normalization YAML missing target channels: {missing_y}")

        x_mean = torch.tensor([inputs[c]["mean"] for c in self.x_channels], dtype=torch.float32).view(-1, 1, 1)
        x_std  = torch.tensor([inputs[c]["std"]  for c in self.x_channels], dtype=torch.float32).view(-1, 1, 1)#.clamp_min(eps)
        y_mean = torch.tensor([targets[c]["mean"] for c in self.y_channels], dtype=torch.float32).view(-1, 1, 1)
        y_std  = torch.tensor([targets[c]["std"]  for c in self.y_channels], dtype=torch.float32).view(-1, 1, 1)#.clamp_min(eps)
        return x_mean, x_std, y_mean, y_std

    def __len__(self):
        return len(self.wind_paths)

    def _cache_get(self, idx: int):
        if self.cache_size <= 0:
            return None
        v = self._cache.get(idx)
        if v is not None:
            self._cache.move_to_end(idx)
        return v

    def _cache_put(self, idx: int, value: Dict[str, Any]):
        if self.cache_size <= 0:
            return
        self._cache[idx] = value
        self._cache.move_to_end(idx)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _to_float32_torch(arr: np.ndarray) -> torch.Tensor:
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return torch.from_numpy(arr)

    @staticmethod
    def _clip_channel_percentiles_ignore_nonfinite(
        x_ch: torch.Tensor, q_lo: float, q_hi: float
    ) -> torch.Tensor:
        """
        x_ch: (H,W) tensor
        Clips finite values only, quantiles computed over finite pixels only.
        Leaves non-finite untouched (so you can nan_to_num after).
        """
        finite = torch.isfinite(x_ch)
        if finite.sum() < 10:
            return x_ch

        vals = x_ch[finite]
        lo = torch.quantile(vals, q_lo)
        hi = torch.quantile(vals, q_hi)

        if (not torch.isfinite(lo)) or (not torch.isfinite(hi)) or (hi <= lo):
            return x_ch

        out = x_ch.clone()
        out[finite] = torch.clamp(out[finite], lo, hi)
        return out

    def _read_sar_channels(self, sar_t_path: str) -> List[torch.Tensor]:
        """
        Returns a list of torch tensors (H,W) in the same order as self.sar_channels.
        - Converts HH/HV to dB if sar_to_db=True
        - Optionally treats raw zeros as nodata for HH/HV (sar_zero_is_nodata=True)
        - Leaves incidence angle as-is (float32)
        """
        da = rxr.open_rasterio(sar_t_path)  # DataArray: (band, y, x)
        out: List[torch.Tensor] = []

        for c in self.sar_channels:
            b = self.SAR_BAND_INDEX[c]
            arr = da[b].values  # numpy (H,W)

            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)

            # Optionally mark zeros as nodata for HH/HV *before* dB conversion
            if self.sar_zero_is_nodata and c in ("HH", "HV"):
                arr = arr.copy()  # need a writeable array
                arr[arr == 0.0] = np.nan

            if c in ("HH", "HV") and self.sar_to_db:
                # convert only finite entries; preserve NaNs
                finite = np.isfinite(arr)
                arr_db = np.full_like(arr, np.nan, dtype=np.float32)
                lin = np.maximum(arr[finite], self.sar_db_eps)
                arr_db[finite] = (10.0 * np.log10(lin)).astype(np.float32, copy=False)
                arr = arr_db

                if self.sar_clip_db:
                    bounds = self.sar_clip_db_bounds.get(c)
                    if bounds is not None:
                        lo, hi = float(bounds[0]), float(bounds[1])
                        m = np.isfinite(arr)
                        arr[m] = np.clip(arr[m], lo, hi)

            out.append(torch.from_numpy(arr))

        return out

    def __getitem__(self, idx: int):
        cached = self._cache_get(idx)
        if cached is not None:
            return cached

        wind_path = self.wind_paths[idx]
        past_path = self.past_paths[idx]
        fut_path  = self.future_paths[idx]
        sar_path  = self.sar_t_paths[idx]

        # Load NPZ arrays
        with np.load(wind_path, allow_pickle=True) as w, \
             np.load(past_path, allow_pickle=True) as p, \
             np.load(fut_path,  allow_pickle=True) as f:

            past_u = self._to_float32_torch(p["u"])
            past_v = self._to_float32_torch(p["v"])
            wind_u = self._to_float32_torch(w["u10_mean"])
            wind_v = self._to_float32_torch(w["v10_mean"])

            x_tensors = []
            if "drift" in self.x_groups:
                x_tensors += [past_u, past_v]
                
            if "wind" in self.x_groups:
                x_tensors += [wind_u, wind_v]
                if self.include_wspd:
                    x_tensors.append(self._to_float32_torch(w["wspd_mean"]))

            if "sar" in self.x_groups:
                sar_tensors = self._read_sar_channels(sar_path)
                x_tensors.extend(sar_tensors)

            # if self.include_wspd:
            #     x_tensors.append(self._to_float32_torch(w["wspd_mean"]))

            fut_u = self._to_float32_torch(f["u"])
            fut_v = self._to_float32_torch(f["v"])
            y = torch.stack([fut_u, fut_v], dim=0)  # (2,H,W)

            meta_out = None
            if self.return_meta:
                meta_out = {
                    "future_wind_attrs": w["attrs"].item() if "attrs" in w else None,
                    "drift_meta_past_drift": p["meta"].item() if "meta" in p else None,
                    "drift_meta_future_drift": f["meta"].item() if "meta" in f else None,
                    "sar_t_path": sar_path,
                }

        # Read SAR channels (GeoTIFF) and append to x
        # sar_tensors = self._read_sar_channels(sar_path)
        # x_tensors.extend(sar_tensors)

        x = torch.stack(x_tensors, dim=0)  # (C,H,W)

        # Normalize (uses YAML stats, including SAR HH/HV dB channels)
        if self.do_norm:
            x = (x - self.x_mean) / self.x_std
            if self.normalize_y:
                y = (y - self.y_mean) / self.y_std

        # --- SAR HH/HV ONLY: clip percentiles + nan/inf -> 0 ---
        if self.sar_postprocess and len(self._sar_hh_hv_x_idx) > 0:
            # 1) percentile clip (finite values only)
            if self.sar_clip_percentiles is not None:
                q_lo, q_hi = self.sar_clip_percentiles
                for ci in self._sar_hh_hv_x_idx:
                    x[ci] = self._clip_channel_percentiles_ignore_nonfinite(x[ci], q_lo, q_hi)

            # 2) sanitize remaining NaN/Inf -> 0
            for ci in self._sar_hh_hv_x_idx:
                x[ci] = torch.nan_to_num(x[ci], nan=0.0, posinf=0.0, neginf=0.0)
        
        if "sar_incidence_angle" in self.x_channels:
            ia_idx = self.x_channels.index("sar_incidence_angle")
            x[ia_idx] = torch.nan_to_num(x[ia_idx], nan=0.0, posinf=0.0, neginf=0.0)


        out = {
            "x": x,
            "y": y,
            "t": self.ts[idx],
            "id": self.ids[idx],
        }
        if meta_out is not None:
            out["meta"] = meta_out

        self._cache_put(idx, out)
        return out
