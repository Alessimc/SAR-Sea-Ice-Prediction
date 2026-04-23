"""
Compare CARRA wind distributions between 2020 (val split) and 2021 (test split).

2020 index: model_dev_main/index_files/min400PM_wind_drift_SAR_dataset/index_val.jsonl
2021 index: AROME_datahandling/index_test.jsonl

Saves: data_dist/carra_2020_vs_2021_u10.png, _v10.png, _wspd.png
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_2020 = os.path.join(REPO_ROOT, "model_dev_main", "index_files",
                          "min400PM_wind_drift_SAR_dataset", "index_val.jsonl")
INDEX_2021 = os.path.join(REPO_ROOT, "AROME_datahandling", "index_test.jsonl")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

PIXELS_PER_FILE = 500


def collect(index_path, label):
    """Sample pixel values from all wind NPZ files listed in a JSONL index."""
    with open(index_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    paths = [e["future_wind_path"] for e in entries]
    print(f"[{label}] {len(paths)} entries")

    rng = np.random.default_rng(42)
    u, v, spd = [], [], []

    for fpath in paths:
        try:
            f = np.load(fpath, allow_pickle=True)
            uu = f["u10_mean"].ravel()
            vv = f["v10_mean"].ravel()
            ss = f["wspd_mean"].ravel()
        except Exception as e:
            print(f"  skip {fpath}: {e}")
            continue

        valid = np.isfinite(uu) & np.isfinite(vv) & np.isfinite(ss)
        uu, vv, ss = uu[valid], vv[valid], ss[valid]
        if len(uu) == 0:
            continue

        idx = rng.choice(len(uu), size=min(PIXELS_PER_FILE, len(uu)), replace=False)
        u.append(uu[idx])
        v.append(vv[idx])
        spd.append(ss[idx])

    print(f"  loaded {len(u)} files successfully")
    return np.concatenate(u), np.concatenate(v), np.concatenate(spd)


def plot(vals_2020, vals_2021, varname, xlabel, outname):
    bins = np.linspace(
        min(vals_2020.min(), vals_2021.min()),
        max(vals_2020.max(), vals_2021.max()),
        80,
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals_2020, bins=bins, density=True, alpha=0.6, color="#1f77b4",
            label=f"CARRA 2020  (n={len(vals_2020):,})")
    ax.hist(vals_2021, bins=bins, density=True, alpha=0.6, color="#d62728",
            label=f"CARRA 2021  (n={len(vals_2021):,})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"CARRA distribution of {varname}  —  2020 vs 2021")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, outname)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}"
          f"  |  2020: {vals_2020.mean():+.2f}±{vals_2020.std():.2f}"
          f"  2021: {vals_2021.mean():+.2f}±{vals_2021.std():.2f}")


def main():
    print("Loading 2020 CARRA...")
    u20, v20, spd20 = collect(INDEX_2020, "CARRA 2020")

    print("\nLoading 2021 CARRA...")
    u21, v21, spd21 = collect(INDEX_2021, "CARRA 2021")

    print("\nPlotting...")
    plot(u20,   u21,   "U10 (eastward)",  "u10  [m s⁻¹]",       "carra_2020_vs_2021_u10.png")
    plot(v20,   v21,   "V10 (northward)", "v10  [m s⁻¹]",       "carra_2020_vs_2021_v10.png")
    plot(spd20, spd21, "Wind speed",      "wind speed  [m s⁻¹]", "carra_2020_vs_2021_wspd.png")


if __name__ == "__main__":
    main()
