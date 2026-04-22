import json
from collections import OrderedDict
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml


class DriftWindDataset(Dataset):
    """
    Optimized Dataset:
    - Pre-parses JSONL into lists (no per-sample dict lookups)
    - Avoids np.stack copies by stacking in torch
    - Avoids redundant astype(float32) copies (only converts if needed)
    - Optional small per-worker LRU cache (useful across epochs)
    - Keeps normalization tensors on CPU (broadcasting is cheap)

    X = [past_drift_u, past_drift_v, future_wind_u10_mean, future_wind_v10_mean] -> (4,H,W)
    Y = [future_drift_u, future_drift_v]                                         -> (2,H,W)
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

    def __init__(
        self,
        index_jsonl: str,
        include_wspd: bool = False,
        return_meta: bool = False,
        norm_yaml_path: Optional[str] = None,
        normalize_y: bool = True,
        eps: float = 1e-6,
        cache_size: int = 0,  # 0 disables caching; try 16/32
    ):
        self.include_wspd = include_wspd
        self.return_meta = return_meta
        self.normalize_y = normalize_y

        # Channel ordering used to build x/y
        self.x_channels = self.DEFAULT_X_CHANNELS.copy()
        if self.include_wspd:
            self.x_channels.append("future_wind_wspd_mean")
        self.y_channels = self.DEFAULT_Y_CHANNELS.copy()

        # Pre-parse JSONL into lists for fast indexing
        self.wind_paths: List[str] = []
        self.past_paths: List[str] = []
        self.future_paths: List[str] = []
        self.ids: List[Any] = []
        self.ts: List[Any] = []

        with open(index_jsonl, "r") as f:
            for line in f:
                r = json.loads(line)
                # NOTE: keys must match the index jsonl
                self.wind_paths.append(r["future_wind_path"])
                self.past_paths.append(r["past_drift_path"])
                self.future_paths.append(r["future_drift_path"])
                self.ids.append(r.get("id"))
                self.ts.append(r.get("t"))

        # Normalization params (CPU tensors shaped (C,1,1))
        self.do_norm = norm_yaml_path is not None
        if self.do_norm:
            self.x_mean, self.x_std, self.y_mean, self.y_std = self._load_norm_yaml(norm_yaml_path, eps=eps)

        # Simple per-worker LRU cache (each DataLoader worker has its own dataset instance)
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
        x_std  = torch.tensor([inputs[c]["std"]  for c in self.x_channels], dtype=torch.float32).view(-1, 1, 1).clamp_min(eps)
        y_mean = torch.tensor([targets[c]["mean"] for c in self.y_channels], dtype=torch.float32).view(-1, 1, 1)
        y_std  = torch.tensor([targets[c]["std"]  for c in self.y_channels], dtype=torch.float32).view(-1, 1, 1).clamp_min(eps)

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
        # Avoid copies if already float32
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int):
        cached = self._cache_get(idx)
        if cached is not None:
            return cached

        wind_path = self.wind_paths[idx]
        past_path = self.past_paths[idx]
        fut_path  = self.future_paths[idx]

        # Load arrays; keep inside the 'with' so files close immediately
        with np.load(wind_path, allow_pickle=True) as w, \
             np.load(past_path, allow_pickle=True) as p, \
             np.load(fut_path,  allow_pickle=True) as f:

            # Convert to torch without unnecessary copies
            past_u = self._to_float32_torch(p["u"])
            past_v = self._to_float32_torch(p["v"])
            wind_u = self._to_float32_torch(w["u10_mean"])
            wind_v = self._to_float32_torch(w["v10_mean"])

            x_tensors = [past_u, past_v, wind_u, wind_v]
            if self.include_wspd:
                x_tensors.append(self._to_float32_torch(w["wspd_mean"]))

            x = torch.stack(x_tensors, dim=0)  # (C,H,W)

            fut_u = self._to_float32_torch(f["u"])
            fut_v = self._to_float32_torch(f["v"])
            y = torch.stack([fut_u, fut_v], dim=0)  # (2,H,W)

            meta_out = None
            if self.return_meta:
                # Grab meta now (no second reopen)
                meta_out = {
                    "future_wind_attrs": w["attrs"].item() if "attrs" in w else None,
                    "drift_meta_past_drift": p["meta"].item() if "meta" in p else None,
                    "drift_meta_future_drift": f["meta"].item() if "meta" in f else None,
                }

        # Normalize (broadcasted on CPU; later batch is moved to GPU by DataLoader/training loop)
        if self.do_norm:
            x = (x - self.x_mean) / self.x_std
            if self.normalize_y:
                y = (y - self.y_mean) / self.y_std

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
