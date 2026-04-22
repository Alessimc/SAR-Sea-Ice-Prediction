import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import rioxarray as rxr
from collections import defaultdict
from src.sea_ice_drift.adapted_dualPol import SeaIceDriftFromTiff
from src.utils import init_logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--polarization_mode",
    default="HV",
    choices=["HV", "HH", "HV+HH"],
    help="Polarization mode"
)
parser.add_argument(
    "--triplet_csv",
    default=None,
    help="Path to CSV file containing triplet list for a region"
)
parser.add_argument(
    "--output_root",
    default=None,
    help="Root directory for output NPZ files. Defaults to <SAR_sea_ice_dataset>/VECTOR_FIELDS_24h_pairs/<pol_folder>",
)
parser.add_argument(
    "--data_paths_config",
    default="configs/data_paths.yaml",
    help="Path to data_paths.yaml (default: configs/data_paths.yaml)",
)

args = parser.parse_args()

pol_mode = args.polarization_mode
grid_step_pix = 25
img_size = 25
min_border = 40 
max_border = 80
threads = 4

if pol_mode == "HV+HH":
    pol_folder = "HV_HH"
else:
    pol_folder = pol_mode

if args.output_root is not None:
    output_root = Path(args.output_root)
else:
    import yaml
    with open(args.data_paths_config) as f:
        _paths = yaml.safe_load(f)
    output_root = Path(_paths["SAR_sea_ice_dataset"]) / f"VECTOR_FIELDS_24h_pairs/{pol_folder}"
logger = init_logging()

# Load triplet list
triplet_csv = args.triplet_csv
df = pd.read_csv(triplet_csv)

logger.info(f"Loaded {len(df)} triplets")
logger.info(f"Triplet CSV: {triplet_csv}")
logger.info(f"Polarization mode: {pol_mode}")
logger.info(f"Output root: {output_root}")


def compute_vector_field(file_a, file_b, pol_mode):
    """
    Compute dense drift field between file_a to file_b.
    Returns (u_dense, v_dense) or None if failed.
    """
    try:
        tA = datetime.strptime(Path(file_a).stem, "%Y%m%dT%H%M")
        tB = datetime.strptime(Path(file_b).stem, "%Y%m%dT%H%M")

        sid = SeaIceDriftFromTiff(file_a, file_b, time1=tA, time2=tB,
                                  pixel_size_m=100.0, pol_mode=pol_mode)

        # Feature tracking
        c1, r1, du, dv = sid.get_drift_FT()
        n_ft = len(c1)
        logger.info(f"Number of FT vectors: {n_ft}")
        
        img1 = sid.n1[1]

        # Pattern matching
        u_pix, v_pix, a_deg, r_mcc, h_hess, pm_cols, pm_rows = sid.get_drift_PM(
            grid_step_pix=grid_step_pix,
            img_size=img_size,
            min_border=min_border,
            max_border=max_border,
            threads=threads
        )

        n_pm_total = pm_cols.size
        logger.info(f"Total PM grid points: {n_pm_total}")

        good = (r_mcc * h_hess) > 4
        n_pm_good = int(np.nansum(good))
        logger.info(f"Valid PM points: {n_pm_good}")

        if np.sum(good) == 0:
            logger.info("No valid PM vectors; skipping.")
            return None

        # Dense interpolation to image grid
        u_dense, v_dense = sid.interpolate_to_dense_image_grid(
            pm_cols, pm_rows,
            u_pix, v_pix,
            good_mask=good,
            img_shape=img1.shape,
            method='linear',
            fill_method='nearest'
        )

        info = dict(
            n_ft=n_ft,
            n_pm_total=n_pm_total,
            n_pm_good=n_pm_good
        )

        return u_dense.astype(np.float32), v_dense.astype(np.float32), info

    except Exception as e:
        logger.info(f"Vector field computation failed: {e}")
        return None


def save_vector_field(u, v, out_path, meta):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        u=u,
        v=v,
        meta=meta
    )
    logger.info(f"Saved {out_path}")


# PROCESS TRIPLETS
# stats = defaultdict(list)

# stats["n_ft"]        # list of FT counts
# stats["n_pm_total"] # list of total PM grid points
# stats["n_pm_good"]  # list of valid PM points
# stats["npz_path"]

for idx in df.index:
    f_minus = df.loc[idx, "t_minus_24"]
    f_center = df.loc[idx, "t"]
    f_plus = df.loc[idx, "t_plus_24"]

    logger.info(f"\n=== Triplet {idx} ===")

    # Build output directory structure
    center_path = Path(f_center)
    region_root = "/".join(center_path.parts[-5:-4])
    date_str = center_path.parts[-4] + "/" + center_path.parts[-3]
    day_folder = center_path.stem[6:8]     

    out_dir = output_root / region_root / date_str / day_folder

    # Compute PAST drift: t_minus_24 to t
    vf_past = compute_vector_field(f_minus, f_center, pol_mode)

    # Compute FUTURE drift: t to t_plus_24
    vf_future = compute_vector_field(f_center, f_plus, pol_mode)

    # only save if both past and future are available
    if vf_past is not None and vf_future is not None:
        # past
        u_p, v_p, info_past = vf_past

        outpath_past = out_dir / f"{Path(f_minus).stem}__{Path(f_center).stem}_past.npz"

        # stats["n_ft"].append(info_past["n_ft"])
        # stats["n_pm_total"].append(info_past["n_pm_total"])
        # stats["n_pm_good"].append(info_past["n_pm_good"])
        # stats["npz_path"].append(str(outpath_past))

        save_vector_field(
            u_p, v_p,
            outpath_past,
            meta=dict(
                start_path=str(Path(f_minus)),
                end_path=str(Path(f_center)),
                start_stem=Path(f_minus).stem,
                end_stem=Path(f_center).stem,
                pol_mode=pol_mode,
                grid_step_pix=grid_step_pix,
                img_size=img_size
            )
        )
        # future
        u_f, v_f, info_future = vf_future

        outpath_future = out_dir / f"{Path(f_center).stem}__{Path(f_plus).stem}_future.npz"

        # stats["n_ft"].append(info_future["n_ft"])
        # stats["n_pm_total"].append(info_future["n_pm_total"])
        # stats["n_pm_good"].append(info_future["n_pm_good"])
        # stats["npz_path"].append(str(outpath_future))

        save_vector_field(
            u_f, v_f,
            outpath_future,
            meta=dict(
                start_path=str(Path(f_center)),
                end_path=str(Path(f_plus)),
                start_stem=Path(f_center).stem,
                end_stem=Path(f_plus).stem,
                pol_mode=pol_mode,
                grid_step_pix=grid_step_pix,
                img_size=img_size
            )
        )

        new_rows = [{
            "npz_path": str(outpath_past),
            "n_ft": info_past["n_ft"],
            "n_pm_total": info_past["n_pm_total"],
            "n_pm_good": info_past["n_pm_good"]
        }, {
            "npz_path": str(outpath_future),
            "n_ft": info_future["n_ft"],
            "n_pm_total": info_future["n_pm_total"],
            "n_pm_good": info_future["n_pm_good"]
        }]

        stats_df = pd.DataFrame(new_rows)
        stats_out = output_root / region_root / "summary_stats.csv"

        stats_out.parent.mkdir(parents=True, exist_ok=True)
        write_header = not stats_out.exists()

        stats_df.to_csv(stats_out, mode="a", header=write_header, index=False)
        
        logger.info(f"Saved FT/PM statistics to {stats_out}")

logger.info("Dataset generation complete.")



