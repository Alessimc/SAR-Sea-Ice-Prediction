# SAR Sea Ice Drift Prediction

A U-Net model for predicting 24-hour Arctic sea ice drift velocity fields from Sentinel-1 SAR imagery and CARRA reanalysis wind. Developed as part of a master's thesis at the University of Oslo / Met Norway.

**Inputs:** past SAR-derived drift (24 h backward context), 24 h mean future wind (CARRA or AROME-Arctic), SAR backscatter (HH, HV, incidence angle).  
**Output:** predicted 24 h sea ice drift velocity field (u, v) on a 100 m EPSG:4326 grid.

---

## Repository Structure

| Directory | Description |
|---|---|
| `src/data_creation/` | Scripts to download Sentinel-1 scenes, compute SAR pairs/triplets, run FT+PM drift, and extract CARRA wind |
| `src/sea_ice_drift/` | Feature tracking + pattern matching code adapted from [nansencenter/sea_ice_drift](https://github.com/nansencenter/sea_ice_drift/) |
| `AROME_datahandling/` | Alternative wind source: 24 h mean AROME-Arctic wind via OPeNDAP |
| `model_dev_main/` | Dataset index files, PyTorch dataloader, U-Net model, training and inference scripts, experiment configs |
| `test_baselines/` | Persistence baseline and scalar wind-drift baseline (α, θ) |
| `variation_in_wind_drift_relation/` | Wind-drift relationship analysis grouped by year and season |
| `buoy_dataset/` | IABP buoy tracks and drift validation results |
| `create_thesis_figures/` | Notebooks for thesis figures (not part of the data pipeline) |
| `configs/` | Data path config and Copernicus OAuth credentials template |

---

## Data Pipeline

The full pipeline goes from raw Sentinel-1 downloads to a trained model:

1. **Download** Sentinel-1 EW GeoTIFFs from Copernicus Dataspace (`src/data_creation/download_SAR_samples.py`)
2. **Pair scenes** ~24 h apart and form triplets for backward drift context (`src/data_creation/data_creation_v3/`)
3. **Compute drift** via Feature Tracking + Pattern Matching (`src/sea_ice_drift/adapted_dualPol.py`)
4. **Convert** pixel drift to velocity in m/s
5. **Extract wind** — 24 h mean from CARRA (`src/data_creation/data_creation_v3/create_future_mean_wind.py`) or AROME-Arctic (`AROME_datahandling/`)
6. **Validate drift** against IABP buoys (`buoy_dataset/`)
7. **Build JSONL index files** linking wind, drift, and SAR paths per sample (`model_dev_main/index_files/`)
8. **Compute normalisation statistics** over the training set
9. **Train** the U-Net (`model_dev_main/src/train/train_wind_drift_SAR.py`)
10. **Evaluate** on validation/test sets (`model_dev_main/src/inference/`)

---

## License

This repository is licensed under the **GNU General Public License v3.0** (see [LICENSE](LICENSE)).

The sea ice drift computation code (`src/sea_ice_drift/`) is adapted from
[nansencenter/sea_ice_drift](https://github.com/nansencenter/sea_ice_drift/) (GPL-3.0),
which requires derivative works to carry the same license.

---

