"""
Plot daily sea ice extent in a fixed region of interest for years 2014-2021.

Data source: Copernicus Marine ice charts  (icecharts_2011-2022.nc)
Region:      lat 81.641-83.913 N,  lon 27-46 E
Ice-covered: SIC >= 25  (values: 5, 25, 55, 80, 95, 100; 157 = fill/land)
Resolution:  1 km² per pixel  →  extent in km²

One line per year, colour-coded with a blue→red gradient.
X-axis: day of year (1-366).
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

TEXTWIDTH_PT = 418.25368
TEXTWIDTH_IN = TEXTWIDTH_PT / 72.27


def setup_pub_style(fontsize=9):
    mpl.rcParams.update({
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def fig_textwidth(height_ratio=0.4):
    return (TEXTWIDTH_IN, TEXTWIDTH_IN * height_ratio)

ICE_CHART_PATH = "/lustre/storeB/users/nicholsh/icecharts_2011-2022.nc"

LAT_MIN, LAT_MAX = 81.641, 83.913
LON_MIN, LON_MAX = 27.0,   46.0

SIC_THRESHOLD = 25      # SIC class >= 25 counts as ice-covered
FILL_VALUE    = 157     # land / no-data mask value

YEARS = list(range(2014, 2022))   # 2014 - 2021 inclusive

OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE  = os.path.join(OUTPUT_DIR, "sea_ice_extent_daily.npz")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sea_ice_extent_daily.png")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def compute_and_cache():
    """Compute daily extents from the ice chart file and save to CACHE_FILE."""
    print("Opening ice chart dataset...")
    ds = xr.open_dataset(ICE_CHART_PATH)

    lat = ds.lat.values   # (yc, xc)
    lon = ds.lon.values   # (yc, xc)

    # Static ROI mask — same for every timestep
    roi = (
        (lat >= LAT_MIN) & (lat <= LAT_MAX) &
        (lon >= LON_MIN) & (lon <= LON_MAX)
    )
    print(f"ROI pixels: {roi.sum():,}  (~{roi.sum():.0f} km²)")

    sic = ds["sic"]   # (time, yc, xc)  uint8, lazy

    # ── Compute daily extent for each year ────────────────────────────────
    # results[year] = (day_of_year array, extent_km2 array)
    results = {}

    for year in YEARS:
        print(f"Processing {year}...")
        t_sel = sic.sel(time=sic.time.dt.year == year)
        if len(t_sel.time) == 0:
            print(f"  no data for {year}, skipping")
            continue

        # Load all days at once: (n_days, yc, xc)
        arr = t_sel.values.astype(np.int16)
        times = t_sel.time.values

        doys = []
        extents = []
        for i, day_arr in enumerate(arr):
            ice_mask = (day_arr != FILL_VALUE) & (day_arr >= SIC_THRESHOLD) & roi
            doy = int((times[i].astype('datetime64[D]') -
                       np.datetime64(f'{year}-01-01', 'D')).astype(int)) + 1
            doys.append(doy)
            extents.append(ice_mask.sum())   # km²

        results[year] = (np.array(doys), np.array(extents, dtype=float))
        print(f"  {len(doys)} days of data")

    ds.close()

    # Save to cache
    save_dict = {}
    for year, (doys, extents) in results.items():
        save_dict[f"{year}_doy"]    = doys
        save_dict[f"{year}_extent"] = extents
    np.savez(CACHE_FILE, **save_dict)
    print(f"\nCached extents to: {CACHE_FILE}")
    return results


def load_cache():
    """Load previously computed extents from CACHE_FILE."""
    data = np.load(CACHE_FILE)
    results = {}
    years = sorted({int(k.split("_")[0]) for k in data.files})
    for year in years:
        results[year] = (data[f"{year}_doy"], data[f"{year}_extent"])
    print(f"Loaded cached extents from: {CACHE_FILE}  ({len(results)} years)")
    return results


def plot(results):
    setup_pub_style()
    fig, ax = plt.subplots(figsize=fig_textwidth())

    palette = sns.color_palette("colorblind", n_colors=len(results))

    # Month boundary day-of-year ticks (non-leap year)
    month_doys = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

    # October 1 = DOY 274 (non-leap year); 2014 had no data before Oct anyway
    YEAR_DOY_MIN = {2014: 274}

    for i, year in enumerate(sorted(results)):
        doys, extents = results[year]
        doy_min = YEAR_DOY_MIN.get(year, 1)
        if doy_min > 1:
            mask = doys >= doy_min
            doys, extents = doys[mask], extents[mask]
        ax.plot(doys, extents / 1e3,   # km² → 10³ km²
                color=palette[i], linewidth=0.8, label=str(year))

    ax.set_xticks(month_doys)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlim(1, 366)
    ax.set_xlabel("Month")
    ax.set_ylabel("Sea ice extent  [×10³ km²]")
    # # ax.set_title(
    #     f"Daily sea ice extent  (SIC ≥ {SIC_THRESHOLD}%)\n"
    #     f"Region: lat {LAT_MIN}–{LAT_MAX} N,  lon {LON_MIN}–{LON_MAX} E"
    # )
    ax.legend(title="Year", bbox_to_anchor=(1.01, 1), loc="upper left",
              frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_FILE}")


def main():
    if os.path.exists(CACHE_FILE):
        print(f"Cache found: {CACHE_FILE}")
        results = load_cache()
    else:
        print("No cache found — computing from source data...")
        results = compute_and_cache()
    plot(results)


if __name__ == "__main__":
    main()
