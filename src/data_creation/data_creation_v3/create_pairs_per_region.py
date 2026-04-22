import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import csv
import yaml
from src.utils import init_logging

logger = init_logging()

# CONFIG
parser = argparse.ArgumentParser(description="Group SAR scenes into 24-hour pairs.")
parser.add_argument("--data_paths_config", default="configs/data_paths.yaml",
                    help="Path to data_paths.yaml (default: configs/data_paths.yaml)")
args = parser.parse_args()

with open(args.data_paths_config, "r") as f:
    path_config = yaml.safe_load(f)

data_dir = path_config.get("SAR_sea_ice_dataset")
out_dir = data_dir + "/pairs"

# dt is the time difference for the first interval (t-dt,t)
dt = 24
tol = 1
dt_target1 = timedelta(hours=dt)
tolerance = timedelta(hours=tol)  # +-tol hours

logger.info(f"Loading regions from: {data_dir}")
logger.info(f"Writing csv pair files to: {out_dir}")
# Make out_dir if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

logger.info(f"Using dt = {dt} hours and tolerance = ±{tolerance}.")

# FIND ALL REGION DIRECTORIES
region_dirs = [
    d for d in os.listdir(data_dir)
    if d.startswith("region") and os.path.isdir(os.path.join(data_dir, d))
]

logger.info("Found regions:")
for r in sorted(region_dirs):
    logger.info(f"  {r}")


# PROCESS EACH REGION
for region in sorted(region_dirs):
    region_path = data_dir + "/" + region
    logger.info(f"\nProcessing {region} ...")


    # COLLECT ALL TIFF FILES RECURSIVELY

    tiff_files = list(Path(region_path).rglob("*.tif")) + list(Path(region_path).rglob("*.tiff"))

    if len(tiff_files) == 0:
        logger.info(f"  No TIFF files in {region}, skipping.")
        continue
    else:
        logger.info(f"  Found {len(tiff_files)} TIFF files.")


    # PARSE FILENAMES → DATETIME

    entries = []
    for f in tiff_files:
        name = f.stem  # e.g., "20150227T0557"
        try:
            t = datetime.strptime(name, "%Y%m%dT%H%M")
            entries.append((t, f))
        except Exception:
            # Ignore files that don't follow naming convention
            continue

    if len(entries) < 3:
        logger.info(f"  Not enough files in {region} for pairs.")
        continue

    # Sort by timestamp
    entries.sort(key=lambda x: x[0])


    # FIND t−24 → t PAIRS

    pairs = []

    times = [e[0] for e in entries]

    for i, (t, f) in enumerate(entries):

        # 1) Find candidate t−24
        t_minus_min = t - dt_target1 - tolerance
        t_minus_max = t - dt_target1 + tolerance

        # All files within tolerance windows
        t_minus_candidates = [
            f2 for (t2, f2) in entries
            if t_minus_min <= t2 <= t_minus_max
        ]

        # Pair them: take closest in time
        if len(t_minus_candidates) == 0:
            continue

        # Choose the ones closest to target time
        t_minus = min(t_minus_candidates, key=lambda x: abs((x.stat().st_mtime)))

        pairs.append((t_minus, f))

    logger.info(f"  Found {len(pairs)} pairs.")

    if len(pairs) == 0:
        continue


    # WRITE CSV FOR THIS REGION

    csv_path = out_dir + f"/{region}_pairs_dt{dt}h_tol{tol}h.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"t_minus_{dt}", "t"])

        for f1, f2 in pairs:
            writer.writerow([str(f1), str(f2)])

    logger.info(f"  Saved CSV: {csv_path}")
