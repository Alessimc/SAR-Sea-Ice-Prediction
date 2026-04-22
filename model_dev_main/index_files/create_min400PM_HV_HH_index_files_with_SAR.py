import os, re, json, glob
from pathlib import Path
from typing import Optional, Dict, Tuple, Set

import numpy as np
import pandas as pd

# --- helpers copied from your other scripts ---
PAIR_RE = re.compile(r"""\(\s*'(?P<region>[^']+)'\s*,\s*Timestamp\('(?P<ts>[^']+)'\)\s*\)""")
TS_RE = re.compile(r"(\d{8}T\d{4})")

def parse_pair_key(s: str) -> Tuple[str, pd.Timestamp]:
    m = PAIR_RE.search(s)
    if not m:
        raise ValueError(f"Could not parse pair_key: {s}")
    region = m.group("region")
    t0 = pd.to_datetime(m.group("ts"))
    return region, t0

def region_from_path(path: str) -> Optional[str]:
    for p in Path(path).parts:
        if p.startswith("region-"):
            return p
    return None

def t_from_wind(path: str) -> Optional[str]:
    m = TS_RE.search(Path(path).name)
    return m.group(1) if m else None

def sar_t_from_future_npz(future_path: str) -> Optional[str]:
    """meta['start_path'] from future drift NPZ (SAR at time t)."""
    try:
        with np.load(future_path, allow_pickle=True) as f:
            if "meta" not in f:
                return None
            meta = f["meta"].item()
            return meta.get("start_path")
    except Exception:
        return None

def build_wind_map(wind_root: str) -> Dict[Tuple[str, str], str]:
    wind_files = glob.glob(os.path.join(wind_root, "region-*", "**", "*_future.npz"), recursive=True)
    wind_map: Dict[Tuple[str, str], str] = {}
    for p in wind_files:
        r = region_from_path(p)
        t = t_from_wind(p)
        if r and t:
            wind_map[(r, t)] = p
    return wind_map

def write_jsonl(rows, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

# --- main ---
def build_indices_from_goodpairs_csv(
    csv_path: str,
    wind_root: str,
    out_train_jsonl: str,
    out_val_jsonl: str,
    out_test_jsonl: str,
    train_years: Set[int],
    val_years: Set[int],
    test_years: Set[int],
    min_pm_good: int = 400,
    add_sar_t_path: bool = True,
    id_unique_across_splits: bool = False,
    require_wind: bool = True,
):
    # sanity: no overlap between splits
    overlap = (train_years & val_years) | (train_years & test_years) | (val_years & test_years)
    if overlap:
        raise ValueError(f"Year sets overlap across splits: {sorted(overlap)}")

    df = pd.read_csv(csv_path)

    # If the CSV is already filtered, this is redundant but safe:
    if "future_n_pm_good" in df.columns:
        df = df[df["future_n_pm_good"] >= min_pm_good].copy()
    if "past_n_pm_good" in df.columns:
        df = df[df["past_n_pm_good"] >= min_pm_good].copy()

    # parse (region, t0)
    df[["region", "t0"]] = df["pair_key"].apply(lambda s: pd.Series(parse_pair_key(s)))
    df["t"] = df["t0"].dt.strftime("%Y%m%dT%H%M")
    df["year"] = df["t0"].dt.year

    wind_map = build_wind_map(wind_root)

    out_rows = {"train": [], "val": [], "test": []}
    skipped = {"year_not_in_split": 0, "wind_missing": 0, "sar_missing": 0}

    global_id = 0
    local_id = {"train": 0, "val": 0, "test": 0}

    for _, row in df.iterrows():
        y = int(row["year"])
        if y in train_years:
            split = "train"
        elif y in val_years:
            split = "val"
        elif y in test_years:
            split = "test"
        else:
            skipped["year_not_in_split"] += 1
            continue

        region = row["region"]
        t = row["t"]

        wind_path = wind_map.get((region, t))
        if wind_path is None:
            if require_wind:
                skipped["wind_missing"] += 1
                continue
            # else allow None
        past_path = row["past_npz_path"]
        future_path = row["future_npz_path"]

        sar_t_path = None
        if add_sar_t_path:
            sar_t_path = sar_t_from_future_npz(future_path)
            if sar_t_path is None:
                skipped["sar_missing"] += 1
                continue

        sid = global_id if id_unique_across_splits else local_id[split]

        rec = {
            "id": sid,
            "t": t,
            "year": y,
            "region": region,
            "future_wind_path": wind_path,
            "past_drift_path": past_path,
            "future_drift_path": future_path,
        }
        if add_sar_t_path:
            rec["sar_t_path"] = sar_t_path

        out_rows[split].append(rec)

        if id_unique_across_splits:
            global_id += 1
        else:
            local_id[split] += 1

    write_jsonl(out_rows["train"], out_train_jsonl)
    write_jsonl(out_rows["val"], out_val_jsonl)
    write_jsonl(out_rows["test"], out_test_jsonl)

    print("CSV rows after filtering:", len(df))
    print("Written:", {k: len(v) for k, v in out_rows.items()})
    print("Skipped:", skipped)

wind_root = os.environ.get("WIND_ROOT", "path/to/MEAN_CARRA_WIND_8steps")

train_years = set(range(2014, 2020))
val_years   = {2020}
test_years  = {2021}

build_indices_from_goodpairs_csv(
    csv_path=os.environ.get(
        "PAIRS_CSV",
        "path/to/VECTOR_FIELDS_24h_pairs_velocity/HV_HH/paired_HV_HH_min400PM.csv",
    ),
    wind_root=wind_root,
    out_train_jsonl="model_dev_main/index_files/min400PM_wind_drift_SAR_dataset/index_train.jsonl",
    out_val_jsonl="model_dev_main/index_files/min400PM_wind_drift_SAR_dataset/index_val.jsonl",
    out_test_jsonl="model_dev_main/index_files/min400PM_wind_drift_SAR_dataset/index_test.jsonl",
    train_years=train_years,
    val_years=val_years,
    test_years=test_years,
    min_pm_good=400,
    add_sar_t_path=True,
    id_unique_across_splits=False,
    require_wind=True,
)

