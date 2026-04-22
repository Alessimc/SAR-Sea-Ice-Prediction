"""
create_future_mean_wind.py

Create "future" mean wind datasets using CARRA1 10m wind fields.

For each row in paired_stats.csv (future drift fields):
  - parse (region, t0) from pair_key
  - find closest CARRA time to t0
  - compute mean of that timestep + next 7 timesteps (8 total; 24h for 3-hourly CARRA)
  - rotate grid-relative (Lambert grid) winds -> earth-relative (east/north) using meridian convergence
  - reproject earth-relative winds onto a fixed SAR EPSG:4326 grid for the region
  - save NPZ under:
      .../MEAN_CARRA_WIND_8steps/...

Designed for SGE array jobs:
  #$ -t 1-208
  python3 -m src.data.data_creation_v3.create_future_mean_wind --task-id $SGE_TASK_ID --n-tasks $SGE_TASK_LAST

Memory-safe: never concatenates more than 8 timesteps.
"""

import os
import re
import argparse
from functools import lru_cache
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import pyproj
from affine import Affine
from rasterio.warp import reproject, Resampling
from src.utils import init_logging

logger = init_logging()


PAIR_RE = re.compile(
    r"""\(\s*'(?P<region>[^']+)'\s*,\s*Timestamp\('(?P<ts>[^']+)'\)\s*\)"""
)

def parse_pair_key(s: str):
    m = PAIR_RE.search(s)
    if not m:
        raise ValueError(f"Could not parse pair_key: {s}")
    region = m.group("region")
    t0 = pd.to_datetime(m.group("ts"))
    return region, t0


def select_rows_for_task(df: pd.DataFrame, task_id: int, n_tasks: int) -> pd.DataFrame:
    """
    Deterministic split:
      keep row i if i % n_tasks == (task_id - 1)
    task_id is 1-based.
    """
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    if task_id < 1 or task_id > n_tasks:
        raise ValueError(f"task_id must be in [1, {n_tasks}], got {task_id}")

    idx = np.arange(len(df))
    mask = (idx % n_tasks) == (task_id - 1)
    return df.loc[mask].copy()


# Date/month helpers
def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

def add_month(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    y = ts.year + (ts.month - 1 + n) // 12
    m = (ts.month - 1 + n) % 12 + 1
    return ts.replace(year=y, month=m, day=1, hour=0, minute=0, second=0, microsecond=0)

def carra_month_path(base_dir: str, var: str, month_ts: pd.Timestamp) -> str:
    # like <carra_base_dir>/<YYYY>/an_10u_<YYYYMM>.nc
    y = month_ts.year
    yyyymm = f"{month_ts.year}{month_ts.month:02d}"
    return os.path.join(base_dir, f"{y}", f"an_{var}_{yyyymm}.nc")


# CARRA CRS from CF metadata
def carra_crs_from_ds(ds: xr.Dataset) -> pyproj.CRS:
    lcc = ds["Lambert_Conformal"]
    central_lon = float(lcc.longitude_of_central_meridian)
    central_lat = float(lcc.latitude_of_projection_origin)
    std_parallel = float(lcc.standard_parallel)
    false_easting = float(lcc.false_easting)
    false_northing = float(lcc.false_northing)
    earth_radius = float(lcc.earth_radius)

    return pyproj.CRS.from_proj4(
        f"+proj=lcc +lat_1={std_parallel} +lat_2={std_parallel} "
        f"+lat_0={central_lat} +lon_0={central_lon} "
        f"+x_0={false_easting} +y_0={false_northing} "
        f"+R={earth_radius} +units=m +no_defs"
    )


@lru_cache(maxsize=64)
def load_sar_grid(sar_path: str):
    sar = rioxarray.open_rasterio(sar_path)
    target_crs = sar.rio.crs
    target_transform = sar.rio.transform()
    H, W = sar.shape[-2], sar.shape[-1]
    sar.close()
    return target_crs, target_transform, H, W


@lru_cache(maxsize=512)
def open_carra_single_month(carra_base_dir: str, year: int, month: int, var: str) -> xr.Dataset:
    """
    Open exactly one monthly CARRA file. Cached for speed.
    NOTE: Do NOT close returned datasets; they are cached.
    """
    m0 = pd.Timestamp(year=year, month=month, day=1)
    path = carra_month_path(carra_base_dir, var, m0)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_dataset(path)


def nearest_time_index(time_values: np.ndarray, t0: pd.Timestamp) -> int:
    t0_np = np.datetime64(pd.Timestamp(t0).to_datetime64())
    return int(np.argmin(np.abs(time_values - t0_np)))


# CARRA georeferencing helpers
def _carra_affine_from_xy(x: np.ndarray, y: np.ndarray) -> Affine:
    """
    Build affine transform consistent with CARRA x/y being pixel centers.
    Matches your diagnostics (e.g. dx=dy=2500, origin -1250,-1250).
    """
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return Affine(
        dx, 0.0, float(x[0] - dx / 2),
        0.0, dy, float(y[0] - dy / 2),
    )

def attach_carra_geo(da: xr.DataArray, carra_crs: pyproj.CRS) -> xr.DataArray:
    """
    Ensure da has correct spatial dims, CRS, and affine transform consistent with its x/y coords.
    """
    x = da["x"].values
    y = da["y"].values
    transform = _carra_affine_from_xy(x, y)

    return (
        da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
          .rio.write_crs(carra_crs)
          .rio.write_transform(transform)
    )

def meridian_convergence_gamma(u_da: xr.DataArray, carra_crs: pyproj.CRS) -> np.ndarray:
    """
    Compute meridian convergence gamma (radians) on the CARRA grid.
    Uses the sign you validated: gamma = -meridian_convergence.
    """
    x = u_da["x"].values
    y = u_da["y"].values
    XX, YY = np.meshgrid(x, y)

    to_ll = pyproj.Transformer.from_crs(carra_crs, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(XX, YY)

    P = pyproj.Proj(carra_crs)
    factors = P.get_factors(lon, lat)

    gamma = -np.deg2rad(factors.meridian_convergence)
    return gamma


# Reproject scalar field to SAR grid
def reproject_scalar_to_sar_grid(da: xr.DataArray, target_crs, target_transform, H, W, source_crs) -> np.ndarray:
    """
    Reproject a single scalar field (already earth-relative component) onto SAR grid.
    NO vector rotation here.
    """
    da = da.rio.write_crs(source_crs)
    src_transform = da.rio.transform()

    out = np.zeros((H, W), dtype=np.float32)

    reproject(
        source=da.values,
        destination=out,
        src_transform=src_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
    )
    return out


# Core computation (memory-safe)
def compute_case_8step_mean_on_sar(
    t0: pd.Timestamp,
    sar_path: str,
    carra_base_dir: str,
    n_steps: int = 8,
    time_name: str = "time",
    u_var: str = "10u",
    v_var: str = "10v",
    height_dim: str = "height",
    height_index: int = 0,
):
    
    target_crs, target_transform, H, W = load_sar_grid(sar_path)

    ds_u0 = open_carra_single_month(carra_base_dir, t0.year, t0.month, "10u")
    ds_v0 = open_carra_single_month(carra_base_dir, t0.year, t0.month, "10v")

    times0 = ds_u0[time_name].values
    idx0 = nearest_time_index(times0, t0)

    n_time0 = times0.shape[0]
    end_idx0 = idx0 + (n_steps - 1)

    if end_idx0 < n_time0:
        sel_idx0 = np.arange(idx0, end_idx0 + 1)
        need_next = 0
    else:
        sel_idx0 = np.arange(idx0, n_time0)
        need_next = (end_idx0 + 1) - n_time0

    if height_dim in ds_u0[u_var].dims:
        u0 = ds_u0[u_var].isel({time_name: sel_idx0, height_dim: height_index})
        v0 = ds_v0[v_var].isel({time_name: sel_idx0, height_dim: height_index})
    else:
        u0 = ds_u0[u_var].isel({time_name: sel_idx0})
        v0 = ds_v0[v_var].isel({time_name: sel_idx0})

    sel_times0 = ds_u0[time_name].isel({time_name: sel_idx0}).values

    if need_next > 0:
        m1 = add_month(month_start(t0), 1)

        ds_u1 = open_carra_single_month(carra_base_dir, m1.year, m1.month, "10u")
        ds_v1 = open_carra_single_month(carra_base_dir, m1.year, m1.month, "10v")

        if height_dim in ds_u1[u_var].dims:
            u1 = ds_u1[u_var].isel({time_name: slice(0, need_next), height_dim: height_index})
            v1 = ds_v1[v_var].isel({time_name: slice(0, need_next), height_dim: height_index})
        else:
            u1 = ds_u1[u_var].isel({time_name: slice(0, need_next)})
            v1 = ds_v1[v_var].isel({time_name: slice(0, need_next)})

        sel_times1 = ds_u1[time_name].isel({time_name: slice(0, need_next)}).values

        u_stack = xr.concat([u0, u1], dim=time_name)
        v_stack = xr.concat([v0, v1], dim=time_name)
        sel_times = np.concatenate([sel_times0, sel_times1], axis=0)
    else:
        u_stack = u0
        v_stack = v0
        sel_times = sel_times0

    u_mean = u_stack.mean(time_name)
    v_mean = v_stack.mean(time_name)

    source_crs = carra_crs_from_ds(ds_u0)

    u_mean = attach_carra_geo(u_mean, source_crs)
    v_mean = attach_carra_geo(v_mean, source_crs)

    # Rotate grid-relative -> earth-relative (east/north) using meridian convergence
    gamma = meridian_convergence_gamma(u_mean, source_crs)
    cosg = np.cos(gamma)
    sing = np.sin(gamma)

    ug = u_mean.values
    vg = v_mean.values

    u_east  = ug * cosg - vg * sing
    v_north = ug * sing + vg * cosg

    u_east_da = xr.DataArray(u_east, coords=u_mean.coords, dims=u_mean.dims, name="u_east")
    v_north_da = xr.DataArray(v_north, coords=v_mean.coords, dims=v_mean.dims, name="v_north")

    u_east_da = attach_carra_geo(u_east_da, source_crs)
    v_north_da = attach_carra_geo(v_north_da, source_crs)

    # Reproject earth-relative components to SAR grid
    u_sar = reproject_scalar_to_sar_grid(u_east_da,  target_crs, target_transform, H, W, source_crs)
    v_sar = reproject_scalar_to_sar_grid(v_north_da, target_crs, target_transform, H, W, source_crs)

    #flip v sign to match SAR y-axis direction in image convension
    u_sar_img = u_sar
    v_sar_img = -v_sar

    carra_time_nearest = np.datetime_as_string(times0[idx0], unit="m")
    carra_time_window_start = np.datetime_as_string(sel_times[0], unit="m")
    carra_time_window_end = np.datetime_as_string(sel_times[-1], unit="m")

    ds_out = xr.Dataset(
        data_vars=dict(
            u10_mean=(("y", "x"), u_sar_img),
            v10_mean=(("y", "x"), v_sar_img),
            wspd_mean=(("y", "x"), np.hypot(u_sar_img, v_sar_img).astype(np.float32)),
        ),
        coords=dict(
            x=np.arange(W, dtype=np.int32),
            y=np.arange(H, dtype=np.int32),
        ),
        attrs=dict(
            t0_requested=str(pd.Timestamp(t0)),
            carra_time_nearest=str(carra_time_nearest),
            carra_time_window_start=str(carra_time_window_start),
            carra_time_window_end=str(carra_time_window_end),
            n_steps=int(n_steps),
            sar_path=sar_path,
        ),
    )

    return ds_out


# Matches either:
#   HV/region-.../YYYY/MM/DD/<file>.npz
#   HV_HH/region-.../YYYY/MM/DD/<file>.npz
# also stuff like
#   wBackwardPastDrift/HV_HH/region-.../YYYY/MM/DD/<file>.npz
REL_RE = re.compile(
    r"^(?:[^/]+/)?(?P<pol>HV(?:_HH)?)/(?P<rest>region-[^/]+/\d{4}/\d{2}/\d{2}/[^/]+\.npz)$"
)

def build_mean_wind_output_path(row, out_root: str):
    drift_path = row["future_npz_path"]

    split_token = "VECTOR_FIELDS_24h_pairs"
    if split_token not in drift_path:
        raise ValueError(f"Unexpected drift path format: {drift_path}")

    rel_path = drift_path.split(split_token, 1)[1].lstrip("/_")  # handles "/HV..." and "_wBackwardPastDrift/..."

    m = REL_RE.match(rel_path)
    if not m:
        raise ValueError(f"Unexpected drift relative path format: {rel_path}")

    # rest starts at region-..., so NO HV/HV_HH folder in output
    rest = m.group("rest")
    rel_dir = os.path.dirname(rest)

    _, t0 = parse_pair_key(row["pair_key"])
    tstamp = pd.Timestamp(t0).strftime("%Y%m%dT%H%M")
    fname = f"{tstamp}_mean_wind_8steps_future.npz"

    out_dir = os.path.join(out_root, rel_dir)
    out_path = os.path.join(out_dir, fname)
    return out_dir, out_path

def save_mean_wind_npz(ds_case: xr.Dataset, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        u10_mean=ds_case["u10_mean"].values.astype(np.float32),
        v10_mean=ds_case["v10_mean"].values.astype(np.float32),
        wspd_mean=ds_case["wspd_mean"].values.astype(np.float32),
        attrs=dict(ds_case.attrs),
    )


def run_from_csv(
    csv_path: str,
    region_to_sar: dict,
    out_root: str,
    carra_base_dir: str,
    max_rows=None,
    task_id: int = 1,
    n_tasks: int = 1,
):
    df = pd.read_csv(csv_path)
    df[["region", "t0"]] = df["pair_key"].apply(lambda s: pd.Series(parse_pair_key(s)))
    df = df.sort_values(["region", "t0"]).reset_index(drop=True)

    if max_rows is not None:
        df = df.iloc[:max_rows].copy()

    df_task = select_rows_for_task(df, task_id=task_id, n_tasks=n_tasks)
    logger.info(f"Task {task_id}/{n_tasks}: processing {len(df_task)} rows (of {len(df)})")

    for _, row in df_task.iterrows():
        region = row["region"]
        t0 = row["t0"]

        sar_path = region_to_sar.get(region)
        if sar_path is None:
            raise KeyError(f"No SAR tiff provided for region={region}")

        _, out_path = build_mean_wind_output_path(row, out_root)
        if os.path.exists(out_path):
            logger.info(f"Task {task_id}/{n_tasks}: exists, skipping {out_path}")
            continue

        ds_case = compute_case_8step_mean_on_sar(
            t0=t0,
            sar_path=sar_path,
            carra_base_dir=carra_base_dir,
            n_steps=8,
        )

        save_mean_wind_npz(ds_case, out_path)
        logger.info(f"Task {task_id}/{n_tasks}: wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create future mean wind (8-step) NPZ files from CARRA.")
    parser.add_argument("--task-id", type=int, default=int(os.environ.get("SGE_TASK_ID", "1")))
    parser.add_argument("--n-tasks", type=int, default=int(os.environ.get("SGE_TASK_LAST", "1")))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--csv-path",
        required=True,
        help="CSV of paired drift files (output of create_pairs_per_region.py), e.g. paired_HV_HH_min400PM.csv",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Root output directory for mean wind NPZ files, e.g. /data/MEAN_CARRA_WIND_8steps",
    )
    parser.add_argument(
        "--carra-dir",
        required=True,
        help=(
            "Root directory of CARRA monthly NetCDF files. "
            "Expected structure: <carra-dir>/<YYYY>/an_10u_<YYYYMM>.nc"
        ),
    )
    parser.add_argument(
        "--region-sar-json",
        default=None,
        help=(
            "JSON file mapping region names to a representative SAR TIFF path used for "
            "reprojection. If omitted, one TIFF is looked up automatically from the CSV."
        ),
    )
    args = parser.parse_args()

    import json as _json

    csv_path = args.csv_path
    out_root = args.out_root
    carra_base_dir = args.carra_dir

    if args.region_sar_json:
        with open(args.region_sar_json) as _f:
            region_to_sar = _json.load(_f)
    else:
        # Auto-discover one SAR tiff per region from the drift CSV
        import pandas as _pd
        _df = _pd.read_csv(csv_path)
        region_to_sar = {}
        for _, _row in _df.iterrows():
            try:
                _reg, _ = parse_pair_key(_row["pair_key"])
            except Exception:
                continue
            if _reg not in region_to_sar:
                # Use the start_path of the future field as a representative SAR scene
                _p = _row.get("start_path") or _row.get("future_start_path")
                if _p and os.path.exists(str(_p)):
                    region_to_sar[_reg] = str(_p)

    logger.info(f"Starting task {args.task_id}/{args.n_tasks} (max_rows={args.max_rows})")
    logger.info(f"CARRA dir: {carra_base_dir}")
    logger.info(f"Output root: {out_root}")

    run_from_csv(
        csv_path=csv_path,
        region_to_sar=region_to_sar,
        out_root=out_root,
        carra_base_dir=carra_base_dir,
        max_rows=args.max_rows,
        task_id=args.task_id,
        n_tasks=args.n_tasks,
    )

    logger.info(f"Finished task {args.task_id}/{args.n_tasks}")
