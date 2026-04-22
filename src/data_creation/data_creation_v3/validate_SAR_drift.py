"""
This script is used to validate SAR drift data against International Arctic Buoy Programme (IABP) buoy derived drift.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import rioxarray as rxr
from src.sea_ice_drift.adapted_dualPol import SeaIceDriftFromTiff, warp_image_with_flow
from src.utils import init_logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--polarization_mode",
    default="HV+HH",
    choices=["HV", "HH", "HV+HH"],
    help="Polarization mode"
)
parser.add_argument(
    "--buoy_pairs_csv",
    required=True,
    help="CSV of IABP buoy-SAR drift pairs (e.g. buoy_dataset/valid_buoy_drift_pairs_dt6h_to_dt26h.csv)",
)
parser.add_argument(
    "--output-dir",
    default="buoy_dataset",
    help="Directory to write the output validation CSV (default: buoy_dataset/)",
)

args = parser.parse_args()

polarization_mode = args.polarization_mode
dt_h = 24  # target drift time in hours
tol = 2  # hours tolerance around 24h drift
if polarization_mode == "HV+HH":
    pol_str = "HV_HH"
elif polarization_mode == "HH":
    pol_str = "HH"
elif polarization_mode == "HV":
    pol_str = "HV"
else:
    raise ValueError(f"Unknown polarization mode: {polarization_mode}")
output_csv = f"{args.output_dir}/valid_{pol_str}_{dt_h}h_tol{tol}_drift_pairs_with_errors.csv"

# Initialize logger and read buoy-SAR pairs
logger = init_logging()
pairs_df = pd.read_csv(args.buoy_pairs_csv)

# logger info
logger.info(f"Total buoy-SAR drift pairs loaded: {len(pairs_df)}")
logger.info(f"Validating SAR drift against IABP buoy drift for {dt_h}h ± {tol}h pairs.")
logger.info(f"Polarization mode: {polarization_mode}")
logger.info(f"Output CSV will be saved to: {output_csv}")

# Helper functions
def lonlat_to_pixel(da, lon, lat):
    """
    Convert lon/lat to piixel coordinates for a geocoded SAR TIFF.
    da is the rioxarray DataArray (4 bands, y,x).
    """
    # find nearest x-index (lon)
    px = int(np.argmin(np.abs(da.x.values - lon)))
    
    # find nearest y-index (lat)
    py = int(np.argmin(np.abs(da.y.values - lat)))
    
    return px, py


def compute_buoy_sar_error_meters(
    u_dense, v_dense, 
    bx0, by0, bx1, by1,
    dt_hours,
    pixel_size_m=100.0
):
    """
    Compare buoy drift vector to SAR dense motion vector,
    converting everything into METERS, including endpoint drift error.
    """

    # --- BUOY DRIFT (pixels) ---
    buoy_dx_pix = bx1 - bx0
    buoy_dy_pix = by1 - by0

    # --- BUOY DRIFT (meters) ---
    buoy_dx_m = buoy_dx_pix * pixel_size_m
    buoy_dy_m = buoy_dy_pix * pixel_size_m
    mag_buoy_m = np.hypot(buoy_dx_m, buoy_dy_m)

    # --- SAR DRIFT at buoy start pixel ---
    sar_dx_pix = u_dense[int(by0), int(bx0)]
    sar_dy_pix = v_dense[int(by0), int(bx0)]

    sar_dx_m = sar_dx_pix * pixel_size_m
    sar_dy_m = sar_dy_pix * pixel_size_m
    mag_sar_m = np.hypot(sar_dx_m, sar_dy_m)

    # --- Magnitude error (meters) ---
    mag_error_m = mag_sar_m - mag_buoy_m
    abs_mag_error_m = abs(mag_error_m)

    # --- Direction (angle) error ---
    def angle(dx, dy):
        return np.degrees(np.arctan2(dy, dx))

    angle_buoy = angle(buoy_dx_m, buoy_dy_m)
    angle_sar  = angle(sar_dx_m, sar_dy_m)

    angle_error = (angle_sar - angle_buoy + 180) % 360 - 180

    # --- RMSE (meters) ---
    rmse_pix = np.sqrt((sar_dx_pix - buoy_dx_pix)**2 +
                       (sar_dy_pix - buoy_dy_pix)**2)
    rmse_m = rmse_pix * pixel_size_m

    # --- ENDPOINT ERROR ---
    xf = bx0 + sar_dx_pix   # SAR-predicted final x pixel
    yf = by0 + sar_dy_pix   # SAR-predicted final y pixel

    xb = bx1                # buoy final
    yb = by1

    endpoint_err_pix = np.hypot(xf - xb, yf - yb)
    endpoint_err_m   = endpoint_err_pix * pixel_size_m

    # --- Relative endpoint error ---
    if mag_buoy_m > 0:
        rel_endpoint_err = endpoint_err_m / mag_buoy_m
    else:
        rel_endpoint_err = np.nan

    # --- Speeds (m/s) ---
    dt_seconds = dt_hours * 3600.0
    buoy_speed_ms = mag_buoy_m / dt_seconds
    sar_speed_ms  = mag_sar_m / dt_seconds
    speed_error_ms = sar_speed_ms - buoy_speed_ms

    return {
        # Pixel drift components
        "buoy_dx_pix": buoy_dx_pix,
        "buoy_dy_pix": buoy_dy_pix,
        "sar_dx_pix": sar_dx_pix,
        "sar_dy_pix": sar_dy_pix,

        # Drift vectors in meters
        "buoy_dx_m": buoy_dx_m,
        "buoy_dy_m": buoy_dy_m,
        "sar_dx_m": sar_dx_m,
        "sar_dy_m": sar_dy_m,

        # Magnitudes
        "mag_buoy_m": mag_buoy_m,
        "mag_sar_m": mag_sar_m,
        "mag_error_m": mag_error_m,
        "abs_mag_error_m": abs_mag_error_m,

        # Angular difference
        "angle_buoy_deg": angle_buoy,
        "angle_sar_deg": angle_sar,
        "angle_error_deg": angle_error,

        # RMSE of vector difference
        "rmse_pix": rmse_pix,
        "rmse_m": rmse_m,

        # ENDPOINT ERROR
        "endpoint_err_pix": endpoint_err_pix,
        "endpoint_err_m": endpoint_err_m,
        "rel_endpoint_err": rel_endpoint_err,

        # Speeds
        "buoy_speed_ms": buoy_speed_ms,
        "sar_speed_ms": sar_speed_ms,
        "speed_error_ms": speed_error_ms
    }

# ---------------------------------------------------------------------------

# here we only keep the 'valid' pairs with drift time around 24 hours +-2 hours
min_dt_hours = dt_h - tol
max_dt_hours = dt_h + tol

valid_pairs = pairs_df[
    (pairs_df["dt_hours"] >= min_dt_hours) &
    (pairs_df["dt_hours"] <= max_dt_hours)
].copy()

logger.info(f"Valid {dt_h}±{tol}h drift pairs: {len(valid_pairs)}")

reset_idx_valid_pairs = valid_pairs.reset_index(drop=True)

df = reset_idx_valid_pairs.copy()

# Add output columns
df["abs_mag_error_m"] = np.nan
df["angle_error_deg"] = np.nan
df["endpoint_err_m"] = np.nan
df["rel_endpoint_err"] = np.nan
df["n_ft"] = np.nan
df["n_pm"] = np.nan
df["buoy_dx_m"] = np.nan
df["buoy_dy_m"] = np.nan
df["sar_dx_m"] = np.nan
df["sar_dy_m"] = np.nan
df["rmse_m"] = np.nan


for idx in df.index:

    logger.info(f"\n=== Processing index {idx} ===")

    try:
        # Paths and timestamps
        file1 = Path(df.loc[idx, "tiff0_path"])
        file2 = Path(df.loc[idx, "tiff1_path"])

        t1 = datetime.strptime(file1.stem, "%Y%m%dT%H%M")
        t2 = datetime.strptime(file2.stem, "%Y%m%dT%H%M")

        # Sea ice drift object
        sid = SeaIceDriftFromTiff(
            file1, file2,
            time1=t1, time2=t2,
            pixel_size_m=100.0,
            pol_mode=polarization_mode
        )

        # Feature Tracking
        c1, r1, du_ft, dv_ft = sid.get_drift_FT()
        img1 = sid.n1[1]  # Needed for dense interpolation target shape
        n_ft = len(c1)

        # Pattern Matching
        u_pix, v_pix, a_deg, r_mcc, h_hess, pm_cols, pm_rows = sid.get_drift_PM(
            grid_step_pix=25,
            img_size=25,
            min_border=40,
            max_border=80,
            threads=4
        )

        # Quality filter !NOTE: from github example. filters better than paper for swath seams and gaps.
        good = (r_mcc * h_hess) > 4
        n_pm = int(np.sum(good))

        if np.sum(good) == 0:
            logger.info(f"[WARNING] No valid PM points for idx={idx}, skipping.")
            continue  # go to next index

        # Dense interpolation to full grid
        u_dense, v_dense = sid.interpolate_to_dense_image_grid(
            pm_cols, pm_rows,
            u_pix, v_pix,
            good_mask=good,
            img_shape=img1.shape,
            method='linear',
            fill_method='nearest'
        )

        # Buoy position in pixel coordinates
        da1 = rxr.open_rasterio(file1)
        da2 = rxr.open_rasterio(file2)

        bx0, by0 = lonlat_to_pixel(da1, df.loc[idx, "t0_lon"], df.loc[idx, "t0_lat"])
        bx1, by1 = lonlat_to_pixel(da2, df.loc[idx, "t1_lon"], df.loc[idx, "t1_lat"])

        # Compute error between SAR drift and buoy drift
        err = compute_buoy_sar_error_meters(
            u_dense, v_dense,
            bx0, by0,
            bx1, by1,
            dt_hours=df.loc[idx, "dt_hours"]
        )

        # Store results
        df.loc[idx, "abs_mag_error_m"] = err["abs_mag_error_m"]
        df.loc[idx, "angle_error_deg"] = err["angle_error_deg"]
        df.loc[idx, "endpoint_err_m"] = err["endpoint_err_m"]
        df.loc[idx, "rel_endpoint_err"] = err["rel_endpoint_err"]
        df.loc[idx, "n_ft"] = n_ft
        df.loc[idx, "n_pm"] = n_pm
        df.loc[idx, "buoy_dx_m"] = err["buoy_dx_m"]
        df.loc[idx, "buoy_dy_m"] = err["buoy_dy_m"]
        df.loc[idx, "sar_dx_m"] = err["sar_dx_m"]         
        df.loc[idx, "sar_dy_m"] = err["sar_dy_m"]
        df.loc[idx, "rmse_m"] = err["rmse_m"]


        logger.info(f"[OK] idx={idx}: endpoint error = {err['endpoint_err_m']:.1f} m, relative endpoint error = {err['rel_endpoint_err']:.2f}")

    except Exception as e:
        # If anything crashes, we skip gracefully
        logger.info(f"[ERROR] idx={idx} failed with: {e}")
        logger.info("Continuing to next index...")
        continue

# save to CSV
df.to_csv(output_csv, index=False)
df.head()