import os, re, json, glob
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple

TS_RE = re.compile(r"(\d{8}T\d{4})")
YEAR_RE = re.compile(r"/(19|20)\d{2}/")

def t_from_wind(path: str) -> Optional[str]:
    m = TS_RE.search(Path(path).name)
    return m.group(1) if m else None

def t_from_past(path: str) -> Optional[str]:
    # ".../A__B_past.npz" -> t = B
    name = Path(path).name
    if "__" not in name:
        return None
    right = name.split("__", 1)[1]
    m = TS_RE.search(right)
    return m.group(1) if m else None

def t_from_future(path: str) -> Optional[str]:
    # ".../A__B_future.npz" -> t = A
    name = Path(path).name
    if "__" not in name:
        return None
    left = name.split("__", 1)[0]
    m = TS_RE.search(left)
    return m.group(1) if m else None

def year_from_path(path: str) -> Optional[int]:
    """
    Extract year from a path containing .../<YYYY>/<MM>/<DD>/...
    """
    # robust: look for '/YYYY/' pattern
    m = re.search(r"/((19|20)\d{2})/", path)
    if not m:
        return None
    return int(m.group(1))

def region_from_path(path: str) -> Optional[str]:
    """
    Extract region folder name: the first directory component that starts with 'region-'
    """
    parts = Path(path).parts
    for p in parts:
        if p.startswith("region-"):
            return p
    return None

def load_shapes_ok(wind_path: str, past_path: str, future_path: str) -> bool:
    import numpy as np
    w = np.load(wind_path, allow_pickle=True)
    p = np.load(past_path, allow_pickle=True)
    f = np.load(future_path, allow_pickle=True)
    shapes = [
        w["u10_mean"].shape, w["v10_mean"].shape,
        p["u"].shape, p["v"].shape,
        f["u"].shape, f["v"].shape
    ]
    return len(set(shapes)) == 1

def build_split_indices_jsonl_all_regions(
    wind_root: str,
    drift_root: str,
    out_train_jsonl: str,
    out_val_jsonl: str,
    out_test_jsonl: str,
    train_years: Set[int],
    val_years: Set[int],
    test_years: Set[int],
    require_same_shape: bool = True,
    id_unique_across_splits: bool = True,
    verbose: bool = True,
):
    """
    wind_root example:
      <WIND_ROOT>/MEAN_CARRA_WIND_8steps

    drift_root example:
      <DRIFT_ROOT>/VECTOR_FIELDS_24h_pairs_velocity/HV
    """

    # sanity: no overlap between splits
    overlap = (train_years & val_years) | (train_years & test_years) | (val_years & test_years)
    if overlap:
        raise ValueError(f"Year sets overlap across splits: {sorted(overlap)}")

    # gather files from ALL regions
    wind_files = glob.glob(os.path.join(wind_root,  "region-*", "**", "*_future.npz"), recursive=True)
    past_files = glob.glob(os.path.join(drift_root, "region-*", "**", "*_past.npz"),   recursive=True)
    fut_files  = glob.glob(os.path.join(drift_root, "region-*", "**", "*_future.npz"), recursive=True)

    # map (region, t) -> path
    wind_map: Dict[Tuple[str, str], str] = {}
    for p in wind_files:
        t = t_from_wind(p)
        r = region_from_path(p)
        if t and r:
            wind_map[(r, t)] = p

    past_map: Dict[Tuple[str, str], str] = {}
    for p in past_files:
        t = t_from_past(p)
        r = region_from_path(p)
        if t and r:
            past_map[(r, t)] = p

    fut_map: Dict[Tuple[str, str], str] = {}
    for p in fut_files:
        t = t_from_future(p)
        r = region_from_path(p)
        if t and r:
            fut_map[(r, t)] = p

    # candidate keys are those present in all three maps
    keys = sorted(set(wind_map.keys()) & set(past_map.keys()) & set(fut_map.keys()))

    # writers
    outs = {
        "train": open(out_train_jsonl, "w"),
        "val":   open(out_val_jsonl, "w"),
        "test":  open(out_test_jsonl, "w"),
    }

    # counters
    written = {"train": 0, "val": 0, "test": 0}
    skipped = {"year_unknown": 0, "year_not_in_split": 0, "shape": 0}

    global_id = 0
    local_id = {"train": 0, "val": 0, "test": 0}

    for (region, t) in keys:
        wind_path = wind_map[(region, t)]
        past_path = past_map[(region, t)]
        future_path = fut_map[(region, t)]

        # Determine year (prefer drift future path)
        y = year_from_path(future_path) or year_from_path(past_path) or year_from_path(wind_path)
        if y is None:
            skipped["year_unknown"] += 1
            continue

        if y in train_years:
            split = "train"
        elif y in val_years:
            split = "val"
        elif y in test_years:
            split = "test"
        else:
            skipped["year_not_in_split"] += 1
            continue

        if require_same_shape and not load_shapes_ok(wind_path, past_path, future_path):
            skipped["shape"] += 1
            continue

        sid = global_id if id_unique_across_splits else local_id[split]

        row = {
            "id": sid,
            "t": t,
            "year": y,
            "region": region,
            "future_wind_path": wind_path,
            "past_drift_path": past_path,
            "future_drift_path": future_path,
        }

        outs[split].write(json.dumps(row) + "\n")
        written[split] += 1

        if id_unique_across_splits:
            global_id += 1
        else:
            local_id[split] += 1

    for f in outs.values():
        f.close()

    if verbose:
        print("Scanned:")
        print(f"  wind files : {len(wind_files)}")
        print(f"  past files : {len(past_files)}")
        print(f"  future files: {len(fut_files)}")
        print(f"  matched triplets (region,t): {len(keys)}")
        print("Written:")
        print(f"  train: {written['train']} -> {out_train_jsonl}")
        print(f"  val  : {written['val']} -> {out_val_jsonl}")
        print(f"  test : {written['test']} -> {out_test_jsonl}")
        print("Skipped:")
        for k, v in skipped.items():
            print(f"  {k}: {v}")


wind_root  = os.environ.get("WIND_ROOT", "path/to/MEAN_CARRA_WIND_8steps")
drift_root = os.environ.get("DRIFT_ROOT", "path/to/VECTOR_FIELDS_24h_pairs_velocity/HV")

train_years = set(range(2014, 2020))   # 2014–2019
val_years   = {2020}
test_years  = {2021}

build_split_indices_jsonl_all_regions(
    wind_root=wind_root,
    drift_root=drift_root,
    out_train_jsonl="model_dev_main/index_files/full_wind_drift_dataset/index_train.jsonl",
    out_val_jsonl="model_dev_main/index_files/full_wind_drift_dataset/index_val.jsonl",
    out_test_jsonl="model_dev_main/index_files/full_wind_drift_dataset/index_test.jsonl",
    train_years=train_years,
    val_years=val_years,
    test_years=test_years,
    require_same_shape=False,
    id_unique_across_splits=False,
)
