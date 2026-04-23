"""
Plot CARRA vs AROME-Arctic wind distributions using paths from JSONL index files.

CARRA paths:  AROME_datahandling/index_test.jsonl       (field: future_wind_path)
AROME paths:  AROME_datahandling/index_test_AROME.jsonl (field: future_wind_path)

Saves: data_dist/2021_wind_dist.pdf  (and a cache NPZ for fast re-runs)
"""

import json
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARRA_INDEX = os.path.join(REPO_ROOT, "AROME_datahandling", "index_test.jsonl")
AROME_INDEX = os.path.join(REPO_ROOT, "AROME_datahandling", "index_test_AROME.jsonl")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(OUTPUT_DIR, "2021_wind_dist_cache.npz")

PIXELS_PER_FILE = 500

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


def load_paired_paths(carra_index, arome_index):
    """Return (carra_paths, arome_paths) for entries present in both indexes
    where both NPZ files exist on disk."""
    def read_index(path):
        with open(path) as f:
            return {json.loads(line)["id"]: json.loads(line)
                    for line in f if line.strip()}

    carra_entries = read_index(carra_index)
    arome_entries = read_index(arome_index)

    shared_ids = sorted(set(carra_entries) & set(arome_entries))
    print(f"Shared entries in both indexes: {len(shared_ids)}")

    carra_paths, arome_paths = [], []
    skipped = 0
    for eid in shared_ids:
        cp = carra_entries[eid]["future_wind_path"]
        ap = arome_entries[eid]["future_wind_path"]
        if os.path.exists(cp) and os.path.exists(ap):
            carra_paths.append(cp)
            arome_paths.append(ap)
        else:
            skipped += 1

    print(f"Pairs with both files on disk: {len(carra_paths)}  (skipped {skipped})")
    return carra_paths, arome_paths


def collect(paths, label):
    """Sample pixel values from a list of NPZ file paths."""
    print(f"[{label}] loading {len(paths)} files")
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


def load_or_compute():
    """Load sampled arrays from cache if available, otherwise compute and save."""
    if os.path.exists(CACHE_PATH):
        print(f"Loading from cache: {CACHE_PATH}")
        cache = np.load(CACHE_PATH)
        return (cache["cu"], cache["cv"], cache["cspd"],
                cache["au"], cache["av"], cache["aspd"])

    carra_paths, arome_paths = load_paired_paths(CARRA_INDEX, AROME_INDEX)

    print("Loading CARRA...")
    cu, cv, cspd = collect(carra_paths, "CARRA")

    print("\nLoading AROME-Arctic...")
    au, av, aspd = collect(arome_paths, "AROME")

    np.savez(CACHE_PATH, cu=cu, cv=cv, cspd=cspd, au=au, av=av, aspd=aspd)
    print(f"Cache saved: {CACHE_PATH}")

    return cu, cv, cspd, au, av, aspd


def plot_combined(cu, cv, cspd, au, av, aspd):
    setup_pub_style(fontsize=9)

    panels = [
        (cu, au, r"$u_{10}$  [m s$^{-1}$]"),
        (cv, av, r"$v_{10}$  [m s$^{-1}$]"),
        (cspd, aspd, r"wind speed  [m s$^{-1}$]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=fig_textwidth(height_ratio=0.4), sharey=True)

    for ax, (cvals, avals, xlabel) in zip(axes, panels):
        bins = np.linspace(
            min(cvals.min(), avals.min()),
            max(cvals.max(), avals.max()),
            80,
        )
        ax.hist(cvals, bins=bins, density=True, alpha=0.6, color="#1f77b4", label="CARRA")
        ax.hist(avals, bins=bins, density=True, alpha=0.6, color="#d62728", label="AROME-Arctic")
        ax.set_xlabel(xlabel)

    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=True)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "2021_wind_dist.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    cu, cv, cspd, au, av, aspd = load_or_compute()
    print("\nPlotting...")
    plot_combined(cu, cv, cspd, au, av, aspd)


if __name__ == "__main__":
    main()
