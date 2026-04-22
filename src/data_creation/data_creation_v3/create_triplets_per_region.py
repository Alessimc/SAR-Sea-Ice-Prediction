import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import csv
import yaml
from src.utils import init_logging

logger = init_logging()

# CONFIG
parser_cfg = argparse.ArgumentParser(description="Group SAR scenes into triplets for backward drift.")
parser_cfg.add_argument("--data_paths_config", default="configs/data_paths.yaml",
                        help="Path to data_paths.yaml (default: configs/data_paths.yaml)")
args = parser_cfg.parse_args()

with open(args.data_paths_config, "r") as f:
    path_config = yaml.safe_load(f)

data_dir = path_config.get("SAR_sea_ice_dataset")
out_dir = data_dir + "/triplets"

# dt1 is the time difference for the first interval (t-dt,t)
# dt2 is the time difference for the second interval (t,t+dt)
dt1 = 24
dt2 = 6
tol = 1
dt_target1 = timedelta(hours=dt1)
dt_target2 = timedelta(hours=dt2)
tolerance = timedelta(hours=tol)  # +-2 hours

logger.info(f"Loading regions from: {data_dir}")
logger.info(f"Writing csv triplet files to: {out_dir}")
# Make out_dir if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

logger.info(f"Using dt1 = {dt1} hours, dt2 = {dt2} hours and tolerance = ±{tolerance}.")

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
        logger.info(f"  Not enough files in {region} for triplets.")
        continue

    # Sort by timestamp
    entries.sort(key=lambda x: x[0])


    # FIND t−24 → t → t+24 TRIPLETS

    triplets = []

    times = [e[0] for e in entries]

    for i, (t, f) in enumerate(entries):

        # 1) Find candidate t−24
        t_minus_min = t - dt_target1 - tolerance
        t_minus_max = t - dt_target1 + tolerance

        # 2) Find candidate t+24
        t_plus_min = t + dt_target2 - tolerance
        t_plus_max = t + dt_target2 + tolerance

        # All files within tolerance windows
        t_minus_candidates = [
            f2 for (t2, f2) in entries
            if t_minus_min <= t2 <= t_minus_max
        ]

        t_plus_candidates = [
            f2 for (t2, f2) in entries
            if t_plus_min <= t2 <= t_plus_max
        ]

        # Pair them: take closest in time
        if len(t_minus_candidates) == 0 or len(t_plus_candidates) == 0:
            continue

        # Choose the ones closest to target time
        t_minus = min(t_minus_candidates, key=lambda x: abs((x.stat().st_mtime)))
        t_plus = min(t_plus_candidates, key=lambda x: abs((x.stat().st_mtime)))

        triplets.append((t_minus, f, t_plus))

    logger.info(f"  Found {len(triplets)} triplets.")

    if len(triplets) == 0:
        continue


    # WRITE CSV FOR THIS REGION

    csv_path = out_dir + f"/{region}_triplets_dt{dt1}h_{dt2}h_tol{tol}h.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"t_minus_{dt1}", "t", f"t_plus_{dt2}"])

        for f1, f2, f3 in triplets:
            writer.writerow([str(f1), str(f2), str(f3)])

    logger.info(f"  Saved CSV: {csv_path}")
