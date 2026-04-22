"""
create_arome_arctic_future_mean_wind_from_index.py

Create AROME-Arctic future mean wind NPZ files from an existing JSONL index file,
and write a parallel JSONL index where `future_wind_path` points to the AROME product.

For each JSONL row:
  - read SAR timestamp from `t`
  - use SAR TIFF from `sar_t_path` as target grid
  - choose latest AROME-Arctic forecast cycle <= SAR time (3-hourly cycle files)
  - open that forecast via OPeNDAP, with fallback archive products
  - find latest valid time <= SAR time
  - compute a mean wind window spanning 21 hours from that selected start time
    (this gives 8 samples for 3-hourly data, and the corresponding number of
     samples for hourly products)
  - rotate grid-relative Lambert winds -> earth-relative east/north
  - reproject east/north winds onto SAR grid
  - save NPZ under the directory set by AROME_WIND_OUT_ROOT environment variable
    (default: output/MEAN_AROME_ARCTIC_WIND_8steps)
  - write a new JSONL row identical to input except for `future_wind_path`

Safety / robustness improvements
--------------------------------
- No dataset caching for OPeNDAP xarray datasets
- Datasets are explicitly closed in finally blocks
- SAR grids are cached safely by path
- Added checks that the opened dataset matches the requested cycle date
- Added detailed metadata and logging per row
- Output JSONL is written incrementally
- Existing NPZ outputs are reused unless --overwrite-npz is passed
- OPeNDAP file fallback order:
    1) arome_arctic_extracted_2_5km_...
    2) arome_arctic_det_2_5km_...
    3) arome_arctic_pp_2_5km_...
- Handles different internal dimension names across archive products, e.g.
    extracted: x_wind_10m[time, height3, y, x]
    det/pp   : x_wind_10m[time, height7, y1, x1]

Example
-------
python3 -m src.data.data_creation_v3.create_arome_arctic_future_mean_wind_from_index \
    --input-jsonl /path/to/test_index.jsonl \
    --output-jsonl /path/to/test_index_arome.jsonl
"""

import os
import json
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

OUT_ROOT = os.environ.get(
    "AROME_WIND_OUT_ROOT",
    "output/MEAN_AROME_ARCTIC_WIND_8steps",
)


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def parse_index_time(t_str: str) -> pd.Timestamp:
    """Parse timestamps like '20210313T0702'."""
    return pd.to_datetime(t_str, format="%Y%m%dT%H%M")


def floor_to_3h_cycle(t0: pd.Timestamp) -> pd.Timestamp:
    """
    Latest 3-hourly AROME-Arctic cycle <= t0.
    Example:
      2021-03-13 07:02 -> 2021-03-13 06:00
    """
    t0 = pd.Timestamp(t0)
    floored_hour = (t0.hour // 3) * 3
    return t0.replace(hour=floored_hour, minute=0, second=0, microsecond=0)


def cycle_to_opendap_urls(cycle_time: pd.Timestamp):
    """
    Return candidate OPeNDAP URLs in fallback order:
      extracted -> det -> pp
    """
    base = (
        f"https://thredds.met.no/thredds/dodsC/"
        f"aromearcticarchive/{cycle_time:%Y/%m/%d}/"
    )

    urls = [
        f"{base}arome_arctic_extracted_2_5km_{cycle_time:%Y%m%dT%H}Z.nc",
        f"{base}arome_arctic_det_2_5km_{cycle_time:%Y%m%dT%H}Z.nc",
        f"{base}arome_arctic_pp_2_5km_{cycle_time:%Y%m%dT%H}Z.nc",
    ]
    return urls


def open_arome_dataset(url: str) -> xr.Dataset:
    """
    Open one AROME-Arctic OPeNDAP dataset.
    Intentionally NOT cached.
    """
    logger.info(f"Opening AROME-Arctic dataset: {url}")
    return xr.open_dataset(url, decode_times=True)


def open_arome_dataset_with_fallback(cycle_time: pd.Timestamp):
    """
    Try candidate archive products in fallback order and return:
      ds, url
    """
    urls = cycle_to_opendap_urls(cycle_time)

    last_error = None
    for i, url in enumerate(urls):
        label = ["extracted", "det", "pp"][i]
        try:
            ds = open_arome_dataset(url)
            logger.info(f"Using AROME-Arctic {label} archive: {url}")
            return ds, url
        except Exception as e:
            last_error = e
            logger.warning(f"Failed opening AROME-Arctic {label} archive: {url} ({e})")

    raise RuntimeError(
        f"Could not open any AROME-Arctic archive for cycle {cycle_time}. "
        f"Tried: {urls}. Last error: {last_error}"
    )


@lru_cache(maxsize=256)
def load_sar_grid(sar_path: str):
    """
    Cache only SAR grid metadata by file path.
    This is safe and avoids reopening the same TIFF repeatedly.
    """
    sar = rioxarray.open_rasterio(sar_path)
    try:
        target_crs = sar.rio.crs
        target_transform = sar.rio.transform()
        H, W = sar.shape[-2], sar.shape[-1]
    finally:
        sar.close()
    return target_crs, target_transform, H, W


def latest_time_index_leq(valid_times: pd.DatetimeIndex, t0: pd.Timestamp) -> int:
    idx = np.where(valid_times <= t0)[0]
    if len(idx) == 0:
        raise ValueError(f"No valid forecast time <= t0 ({t0}) in selected AROME file.")
    return int(idx[-1])


def build_time_window_indices(
    valid_times: pd.DatetimeIndex,
    t0: pd.Timestamp,
    window_hours: int = 21,
):
    """
    Select the latest valid time <= t0, then all valid times from that start
    through start + window_hours.

    This generalizes the old "8 steps" logic:
      - for 3-hourly products: 8 samples over 0,3,...,21 h
      - for hourly products: 22 samples over 0,1,...,21 h
    """
    idx0 = latest_time_index_leq(valid_times, t0)
    start_time = valid_times[idx0]
    end_time = start_time + pd.Timedelta(hours=window_hours)

    sel_idx = np.where((valid_times >= start_time) & (valid_times <= end_time))[0]
    sel_idx = sel_idx[sel_idx >= idx0]

    if len(sel_idx) == 0:
        raise ValueError(f"No forecast steps selected for t0={t0}.")

    return idx0, sel_idx, valid_times[sel_idx]


# ---------------------------------------------------------------------
# CRS / georeferencing helpers
# ---------------------------------------------------------------------

def _affine_from_xy(x: np.ndarray, y: np.ndarray) -> Affine:
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return Affine(
        dx, 0.0, float(x[0] - dx / 2.0),
        0.0, dy, float(y[0] - dy / 2.0),
    )


def attach_geo(da: xr.DataArray, crs: pyproj.CRS, x_dim="x", y_dim="y") -> xr.DataArray:
    x = da[x_dim].values
    y = da[y_dim].values
    transform = _affine_from_xy(x, y)

    return (
        da.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=False)
          .rio.write_crs(crs)
          .rio.write_transform(transform)
    )


def arome_crs_from_ds(ds: xr.Dataset) -> pyproj.CRS:
    """
    Infer AROME-Arctic Lambert CRS from CF metadata.
    """
    if "projection_lambert" not in ds.variables:
        raise KeyError("Dataset missing 'projection_lambert' variable.")

    gm = ds["projection_lambert"]

    try:
        return pyproj.CRS.from_cf(gm.attrs)
    except Exception:
        pass

    attrs = gm.attrs
    if "grid_mapping_name" not in attrs:
        raise ValueError(
            "Could not infer CRS from projection_lambert attrs. "
            f"Available attrs: {attrs}"
        )

    return pyproj.CRS.from_cf(attrs)


def meridian_convergence_gamma(da: xr.DataArray, crs: pyproj.CRS, x_dim="x", y_dim="y") -> np.ndarray:
    """
    Same sign convention as your validated CARRA script:
      gamma = - meridian_convergence
    """
    x = da[x_dim].values
    y = da[y_dim].values
    XX, YY = np.meshgrid(x, y)

    to_ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(XX, YY)

    proj = pyproj.Proj(crs)
    factors = proj.get_factors(lon, lat)
    gamma = -np.deg2rad(factors.meridian_convergence)
    return gamma


def reproject_scalar_to_sar_grid(
    da: xr.DataArray,
    target_crs,
    target_transform,
    H,
    W,
    source_crs
) -> np.ndarray:
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


# ---------------------------------------------------------------------
# AROME data helpers
# ---------------------------------------------------------------------

def get_valid_times(ds: xr.Dataset) -> pd.DatetimeIndex:
    """
    Return valid forecast times.

    For this archive, with decode_times=True, `time` is usually already decoded.
    If not, fall back to forecast_reference_time + time offset.
    """
    if "time" not in ds.variables and "time" not in ds.coords:
        raise KeyError("Dataset missing 'time' coordinate.")

    time_coord = ds["time"]

    try:
        vt = pd.to_datetime(time_coord.values)
        if np.issubdtype(vt.dtype, np.datetime64):
            return pd.DatetimeIndex(vt)
    except Exception:
        pass

    if "forecast_reference_time" not in ds.variables and "forecast_reference_time" not in ds.coords:
        raise ValueError("Could not derive valid times: no forecast_reference_time present.")

    frt = pd.to_datetime(ds["forecast_reference_time"].values)

    raw = time_coord.values
    if np.issubdtype(raw.dtype, np.timedelta64):
        return pd.DatetimeIndex(frt + pd.to_timedelta(raw))

    units = str(time_coord.attrs.get("units", "")).lower()
    if "hour" in units:
        return pd.DatetimeIndex(frt + pd.to_timedelta(raw, unit="h"))
    if "second" in units:
        return pd.DatetimeIndex(frt + pd.to_timedelta(raw, unit="s"))

    raise ValueError(
        f"Could not interpret AROME time coordinate. dtype={raw.dtype}, units='{units}'"
    )


def validate_dataset_against_request(
    ds: xr.Dataset,
    url: str,
    cycle_time: pd.Timestamp,
    valid_times: pd.DatetimeIndex,
):
    """
    Safety checks to catch stale / wrong OPeNDAP dataset behavior.
    """
    ds_source = ds.encoding.get("source", "UNKNOWN")

    logger.info(f"Requested URL: {url}")
    logger.info(f"Dataset source: {ds_source}")
    logger.info(
        f"Forecast coverage: first_valid={valid_times[0]}, last_valid={valid_times[-1]}"
    )

    if valid_times[0].year != cycle_time.year or valid_times[0].month != cycle_time.month:
        raise ValueError(
            f"Opened dataset appears inconsistent with requested cycle. "
            f"Requested cycle={cycle_time}, but first valid time is {valid_times[0]}."
        )

    if cycle_time.date() < valid_times[0].date() or cycle_time.date() > valid_times[-1].date():
        logger.warning(
            f"Requested cycle {cycle_time} is not bracketed by valid-time dates "
            f"({valid_times[0]} .. {valid_times[-1]})."
        )


def get_wind_dim_names(da: xr.DataArray):
    """
    Infer time, vertical, y, x dims for wind fields across extracted/det/pp layouts.
    """
    dims = da.dims

    time_dim = "time" if "time" in dims else None
    if time_dim is None:
        raise ValueError(f"Could not find time dim in {dims}")

    x_dim = None
    y_dim = None

    for cand in ("x", "x1", "projection_x_coordinate"):
        if cand in dims:
            x_dim = cand
            break

    for cand in ("y", "y1", "projection_y_coordinate"):
        if cand in dims:
            y_dim = cand
            break

    if x_dim is None or y_dim is None:
        raise ValueError(f"Could not infer x/y dims from {dims}")

    other_dims = [d for d in dims if d not in (time_dim, x_dim, y_dim)]

    if len(other_dims) > 1:
        raise ValueError(f"Unexpected extra dims in wind field: {dims}")

    vertical_dim = other_dims[0] if other_dims else None
    return time_dim, vertical_dim, y_dim, x_dim


def get_arome_wind_stack(ds: xr.Dataset, sel_idx: np.ndarray):
    """
    Extract x/y 10m wind stacks for selected time indices from either
    extracted, det, or pp archive layouts.
    Returns:
      u, v, time_dim, y_dim, x_dim
    """
    if "x_wind_10m" not in ds.variables or "y_wind_10m" not in ds.variables:
        raise KeyError("Expected variables 'x_wind_10m' and 'y_wind_10m' not found.")

    u_da = ds["x_wind_10m"]
    v_da = ds["y_wind_10m"]

    dims_u = get_wind_dim_names(u_da)
    dims_v = get_wind_dim_names(v_da)

    if dims_u != dims_v:
        raise ValueError(
            f"x_wind_10m and y_wind_10m dims do not match: {u_da.dims} vs {v_da.dims}"
        )

    time_dim, vertical_dim, y_dim, x_dim = dims_u

    indexers = {time_dim: sel_idx}
    if vertical_dim is not None:
        indexers[vertical_dim] = 0

    u = u_da.isel(indexers)
    v = v_da.isel(indexers)

    return u, v, time_dim, y_dim, x_dim


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def build_output_npz_path_from_row(row: dict, out_root: str = OUT_ROOT) -> str:
    """
    Mirror the CARRA path layout but under MEAN_AROME_ARCTIC_WIND_8steps.
    Uses region + t from the JSONL row.
    """
    region = row["region"]
    t0 = parse_index_time(row["t"])

    out_dir = os.path.join(
        out_root,
        region,
        f"{t0.year:04d}",
        f"{t0.month:02d}",
        f"{t0.day:02d}",
    )
    fname = f"{t0.strftime('%Y%m%dT%H%M')}_mean_wind_8steps_future.npz"
    return os.path.join(out_dir, fname)


def save_mean_wind_npz(ds_case: xr.Dataset, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        u10_mean=ds_case["u10_mean"].values.astype(np.float32),
        v10_mean=ds_case["v10_mean"].values.astype(np.float32),
        wspd_mean=ds_case["wspd_mean"].values.astype(np.float32),
        attrs=dict(ds_case.attrs),
    )


# ---------------------------------------------------------------------
# Core per-sample computation
# ---------------------------------------------------------------------

def compute_case_8step_mean_on_sar_arome(
    t0: pd.Timestamp,
    sar_path: str,
    window_hours: int = 21,
) -> xr.Dataset:
    """
    For one SAR timestamp:
      - choose latest 3-hourly cycle <= t0
      - open that AROME forecast file (with fallback)
      - find latest valid time <= t0
      - select all valid times from that start through start + window_hours
      - mean them
      - rotate x/y grid winds -> east/north
      - reproject to SAR grid
    """
    cycle_time = floor_to_3h_cycle(t0)

    ds = None
    url = None
    try:
        ds, url = open_arome_dataset_with_fallback(cycle_time)

        valid_times = get_valid_times(ds)
        validate_dataset_against_request(ds, url, cycle_time, valid_times)

        idx0, sel_idx, sel_times = build_time_window_indices(
            valid_times=valid_times,
            t0=t0,
            window_hours=window_hours,
        )

        logger.info(
            f"t0={t0}, cycle_time={cycle_time}, selected_start={valid_times[idx0]}, "
            f"selected_end={sel_times[-1]}, n_selected_steps={len(sel_idx)}, "
            f"window_hours={window_hours}"
        )

        u_stack, v_stack, time_dim, y_dim, x_dim = get_arome_wind_stack(ds, sel_idx)

        u_mean = u_stack.mean(time_dim)
        v_mean = v_stack.mean(time_dim)

        source_crs = arome_crs_from_ds(ds)

        u_mean = attach_geo(u_mean, source_crs, x_dim=x_dim, y_dim=y_dim)
        v_mean = attach_geo(v_mean, source_crs, x_dim=x_dim, y_dim=y_dim)

        # Rotate model-grid x/y winds -> earth-relative east/north
        gamma = meridian_convergence_gamma(u_mean, source_crs, x_dim=x_dim, y_dim=y_dim)
        cosg = np.cos(gamma)
        sing = np.sin(gamma)

        ug = u_mean.values
        vg = v_mean.values

        u_east = ug * cosg - vg * sing
        v_north = ug * sing + vg * cosg

        u_east_da = xr.DataArray(
            u_east,
            coords=u_mean.coords,
            dims=u_mean.dims,
            name="u_east",
        )
        v_north_da = xr.DataArray(
            v_north,
            coords=v_mean.coords,
            dims=v_mean.dims,
            name="v_north",
        )

        u_east_da = attach_geo(u_east_da, source_crs, x_dim=x_dim, y_dim=y_dim)
        v_north_da = attach_geo(v_north_da, source_crs, x_dim=x_dim, y_dim=y_dim)

        target_crs, target_transform, H, W = load_sar_grid(sar_path)

        u_sar = reproject_scalar_to_sar_grid(
            u_east_da, target_crs, target_transform, H, W, source_crs
        )
        v_sar = reproject_scalar_to_sar_grid(
            v_north_da, target_crs, target_transform, H, W, source_crs
        )

        # Keep same SAR image convention as your CARRA script
        u_sar_img = u_sar.astype(np.float32)
        v_sar_img = (-v_sar).astype(np.float32)

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
                arome_cycle_time=str(cycle_time),
                arome_source_url=url,
                arome_dataset_source=str(ds.encoding.get("source", "UNKNOWN")),
                arome_valid_time_nearest_before_t0=str(valid_times[idx0]),
                arome_time_window_start=str(sel_times[0]),
                arome_time_window_end=str(sel_times[-1]),
                n_selected_steps=int(len(sel_idx)),
                window_hours=int(window_hours),
                wind_var_u="x_wind_10m",
                wind_var_v="y_wind_10m",
                wind_mode="grid_rotated_to_earth",
                wind_time_dim=time_dim,
                wind_y_dim=y_dim,
                wind_x_dim=x_dim,
                sar_path=sar_path,
            ),
        )

        return ds_out

    finally:
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------

def process_index_jsonl(
    input_jsonl: str,
    output_jsonl: str,
    out_root: str = OUT_ROOT,
    window_hours: int = 21,
    overwrite_npz: bool = False,
):
    n_total = 0
    n_written = 0
    n_skipped_existing = 0
    n_failed = 0

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    with open(input_jsonl, "r") as fin, open(output_jsonl, "w") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            n_total += 1

            try:
                row = json.loads(line)

                if "t" not in row:
                    raise KeyError("Missing key 't' in JSONL row.")
                if "sar_t_path" not in row:
                    raise KeyError("Missing key 'sar_t_path' in JSONL row.")
                if "region" not in row:
                    raise KeyError("Missing key 'region' in JSONL row.")

                t0 = parse_index_time(row["t"])
                sar_path = row["sar_t_path"]
                out_path = build_output_npz_path_from_row(row, out_root=out_root)

                logger.info(
                    f"[{line_no}] Processing region={row['region']}, "
                    f"t0={t0}, sar_path={sar_path}"
                )

                if (not overwrite_npz) and os.path.exists(out_path):
                    logger.info(f"[{line_no}] Exists, skipping NPZ creation: {out_path}")
                    n_skipped_existing += 1
                else:
                    ds_case = compute_case_8step_mean_on_sar_arome(
                        t0=t0,
                        sar_path=sar_path,
                        window_hours=window_hours,
                    )
                    save_mean_wind_npz(ds_case, out_path)
                    logger.info(f"[{line_no}] Wrote {out_path}")

                row_out = dict(row)
                row_out["future_wind_path"] = out_path

                fout.write(json.dumps(row_out) + "\n")
                fout.flush()
                n_written += 1

            except Exception as e:
                n_failed += 1
                logger.exception(f"[{line_no}] Failed processing row: {e}")

    logger.info(
        f"Done. total={n_total}, written_rows={n_written}, "
        f"skipped_existing_npz={n_skipped_existing}, failed={n_failed}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create AROME-Arctic mean wind NPZs from an existing JSONL index and write a parallel AROME index JSONL."
    )
    parser.add_argument("--input-jsonl", type=str, required=True, help="Input JSONL index file.")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Output JSONL index file with updated future_wind_path.")
    parser.add_argument("--overwrite-npz", action="store_true", help="Overwrite existing NPZ files.")
    args = parser.parse_args()

    logger.info(f"Input JSONL : {args.input_jsonl}")
    logger.info(f"Output JSONL: {args.output_jsonl}")
    logger.info(f"Output root : {OUT_ROOT}")

    process_index_jsonl(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        out_root=OUT_ROOT,
        window_hours=21,
        overwrite_npz=args.overwrite_npz,
    )
