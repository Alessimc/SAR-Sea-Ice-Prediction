# SAR Sea Ice Drift Prediction

A U-Net model for predicting 24-hour Arctic sea ice drift velocity fields from Sentinel-1 SAR imagery and CARRA reanalysis wind. Developed as part of a master's thesis at the University of Oslo / Met Norway.

**Inputs:** past SAR-derived drift (24 h backward context), 24 h mean future wind (CARRA or AROME-Arctic), SAR backscatter (HH, HV, incidence angle).  
**Output:** predicted 24 h sea ice drift velocity field (u, v) in m/s on a 100 m EPSG:4326 grid.

---

## Repository Structure

```
SAR-Sea-Ice-Prediction/
├── src/
│   ├── data_creation/
│   │   ├── download_SAR_samples.py          # Step 1 – download Sentinel-1 GeoTIFFs
│   │   └── data_creation_v3/
│   │       ├── create_pairs_per_region.py   # Step 2 – group scenes into 24 h pairs → CSV
│   │       ├── create_triplets_per_region.py# Step 2 – group into triplets for past and future drift pairs
│   │       ├── create_sea_ice_drift_dataset.py          # Step 3 – FT+PM drift → NPZ
│   │       ├── create_sea_ice_drift_dataset_wBackwardPastDrift.py  # Step 3 – with backward drift
│   │       ├── convert_pixel_drift_to_velocity.py       # Step 4 – pixel → m/s
│   │       ├── create_future_mean_wind.py   # Step 5a – CARRA 24 h mean wind → NPZ
│   │       └── validate_SAR_drift.py        # (optional) validate against IABP buoys
│   ├── sea_ice_drift/
│   │   ├── adapted.py                       # FT+PM (single-pol, HV)
│   │   └── adapted_dualPol.py               # FT+PM (dual-pol HV+HH) – adapted from GPL-3.0 code
│   └── utils.py                             # Shared logging utilities
│
├── AROME_datahandling/
│   └── create_arome_arctic_future_mean_wind_from_index.py  # Step 5b – AROME-Arctic wind (OPeNDAP)
│
├── model_dev_main/
│   ├── index_files/
│   │   ├── create_index_files.py            # Step 7 – build train/val/test JSONL (no SAR)
│   │   ├── create_index_files_with_SAR.py   # Step 7 – same + SAR path
│   │   ├── create_min400PM_HV_HH_index_files_with_SAR.py  # Step 7 – min 400 PM vectors filter
│   │   └── */norm_stats_train.yaml          # Pre-computed normalisation statistics
│   ├── configs/
│   │   ├── exploratory/                     # Early hyperparameter sweep configs
│   │   └── min400PM_experiments/            # Main experiment configs
│   ├── scripts/                             # SGE job scripts (HPC cluster, excluded from repo)
│   └── src/
│       ├── dataloader/
│       │   ├── DriftWindSARDataset.py       # Main PyTorch Dataset
│       │   ├── DriftWindDataset.py          # Legacy dataset (drift+wind only)
│       │   ├── compute_channelwise_stats.py # Step 8 – normalisation stats (no SAR)
│       │   ├── compute_channelwise_stats_wSAR.py  # Step 8 – with SAR channels
│       │   └── compute_sar_hist.py          # SAR dB histogram analysis
│       ├── models/
│           └── Unet.py                      # U-Net and ResU-Net (GroupNorm, AvgPool)
│       ├── train/
│       │   ├── train_utils.py               # Shared: logging, checkpointing, train/eval loops
│       │   ├── train_wind_drift_SAR.py      # Step 10 – main training script
│       │   ├── train_wind_drift_SAR_respred.py      # residual prediction variant
│       │   ├── train_wind_drift_SAR_wDivLoss.py     # divergence-regularised loss
│       │   └── train_wind_drift.py          # legacy no-SAR training
│       └── inference/
│           ├── eval_best_test_metrics.py    # Step 11 – metrics on test set → JSON
│           ├── eval_best_val_metrics.py     # Step 11 – metrics on val set → JSON
│           ├── infer_and_plot_quiver_SAR.py # quiver plots (SAR model)
│           └── infer_and_plot_divergence.py # divergence field plots
│
├── test_baselines/
│   ├── find_alpha_theta.py          # fit wind baseline (α, θ) on training data
│   └── compute_baselines.py         # evaluate persistence + wind baselines on test set
│
├── variation_in_wind_drift_relation/
│   └── find_alpha_theta_yearly_seasonally.py  # seasonal/yearly wind-drift analysis
│
├── buoy_dataset/
│   ├── IABP_buoys.csv                         # raw IABP buoy tracks (2014–2020)
│   ├── region_daily_presence.csv              # which regions had buoys on each date
│   ├── region_date_tiff_and_buoys.csv         # SAR TIFFs matched to buoy presence
│   ├── tiff_buoy_closest_observations.csv     # closest buoy observation per TIFF
│   ├── valid_buoy_drift_pairs_dt6h_to_dt26h.csv  # input to validate_SAR_drift.py
│   ├── valid_HV_24h_tol2_drift_pairs_with_errors.csv   # validation results (HV)
│   ├── valid_HH_24h_tol2_drift_pairs_with_errors.csv   # validation results (HH)
│   ├── valid_HV_HH_24h_tol2_drift_pairs_with_errors.csv  # validation results (HV+HH)
│   ├── explore_buoys.ipynb            # buoy matching and exploratory analysis
│   ├── validation_plots_HV.ipynb      # scatter/error plots for HV results
│   ├── validation_plots_HH.ipynb      # scatter/error plots for HH results
│   └── validation_plots_HV_HH.ipynb   # scatter/error plots for HV+HH results
│
├── create_thesis_figures/           # notebooks for thesis figures (not part of pipeline)
│
├── configs/
│   ├── data_paths.yaml              # add your own
│   ├── copernicus_OAuth.yaml        # add your own as in the example below
│   └── copernicus_OAuth.yaml.example  # template
│
├── LICENSE                          # GPL-3.0
├── requirements.txt
└── pyproject.toml
```

---

## Data Pipeline

The full pipeline from raw satellite data to a trained model has 10 steps.

**Step 1 — Download Sentinel-1 SAR scenes**
```bash
python -m src.data_creation.download_SAR_samples \
  --start 2014-01-01T00:00 --end 2014-12-31T23:59 \
  --bbox 27.0,82.0,36.0,84.0 \
  --client_nr 1 \
  --oauth_config configs/copernicus_OAuth.yaml \
  --data_paths_config configs/data_paths.yaml
```
Downloads Sentinel-1 EW GeoTIFFs to `SAR_sea_ice_dataset/region-*/YYYY/MM/DD/`.

**Step 2 — Create SAR pairs and triplets**
```bash
python -m src.data_creation.data_creation_v3.create_pairs_per_region \
  --data_paths_config configs/data_paths.yaml

python -m src.data_creation.data_creation_v3.create_triplets_per_region \
  --data_paths_config configs/data_paths.yaml
```

**Step 3 — Compute sea ice drift (FT + PM)**
```bash
python -m src.data_creation.data_creation_v3.create_sea_ice_drift_dataset_wBackwardPastDrift \
  --polarization_mode HV+HH \
  --triplet_csv /path/to/triplets_region-XX.csv \
  --data_paths_config configs/data_paths.yaml
```

**Step 4 — Convert pixel drift to velocity (m/s)**
```bash
python -m src.data_creation.data_creation_v3.convert_pixel_drift_to_velocity \
  --in-root /path/to/VECTOR_FIELDS_24h_pairs_wBackwardPastDrift/HV_HH \
  --out-root /path/to/VECTOR_FIELDS_24h_pairs_wBackwardPastDrift_velocity/HV_HH
```

**Step 5a — Create future mean wind from CARRA**
```bash
python -m src.data_creation.data_creation_v3.create_future_mean_wind \
  --csv-path /path/to/paired_HV_HH_min400PM.csv \
  --out-root /path/to/MEAN_CARRA_WIND_8steps \
  --carra-dir /path/to/CARRA
```
> CARRA data (3-hourly, Lambert Conformal, monthly NetCDF) is available from the
> [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-carra-single-levels).

**Step 5b — AROME-Arctic wind via OPeNDAP**
```bash
AROME_WIND_OUT_ROOT=/path/to/MEAN_AROME_ARCTIC_WIND_8steps \
python -m AROME_datahandling.create_arome_arctic_future_mean_wind_from_index \
  --input-jsonl /path/to/index_train.jsonl \
  --output-jsonl /path/to/index_train_arome.jsonl
```

**Step 6 — Validate drift against IABP buoys**

The `buoy_dataset/` directory contains pre-computed IABP buoy tracks matched to SAR acquisition pairs. To rerun the validation from scratch, first run `explore_buoys.ipynb` (setting the `SAR_DATASET_ROOT` environment variable to your SAR data root) to regenerate the matching CSVs, then:

```bash
python -m src.data_creation.data_creation_v3.validate_SAR_drift \
  --polarization_mode HV+HH \
  --buoy_pairs_csv buoy_dataset/valid_buoy_drift_pairs_dt6h_to_dt26h.csv
```
Output is written to `buoy_dataset/valid_HV_HH_24h_tol2_drift_pairs_with_errors.csv`. Use the `--output-dir` flag to redirect elsewhere.

Validation scatter plots are in `buoy_dataset/validation_plots_HV_HH.ipynb`.

**Step 7 — Build JSONL index files**
```bash
WIND_ROOT=/path/to/MEAN_CARRA_WIND_8steps \
PAIRS_CSV=/path/to/paired_HV_HH_min400PM.csv \
python -m model_dev_main.index_files.create_min400PM_HV_HH_index_files_with_SAR
```

**Step 8 — Compute normalisation statistics**
```bash
python -m model_dev_main.src.dataloader.compute_channelwise_stats_wSAR \
  --index model_dev_main/index_files/min400PM_wind_drift_SAR_dataset/index_train.jsonl \
  --out   model_dev_main/index_files/min400PM_wind_drift_SAR_dataset/norm_stats_train.yaml
```

**Step 9 — Train the U-Net**
```bash
python -m model_dev_main.src.train.train_wind_drift_SAR \
  --config model_dev_main/configs/min400PM_experiments/unet_4layers_drift_wFutureWind_SAR_all_base16_lr1e-4_bs16.yaml
```
Checkpoints and loss CSV are saved to `model_dev_main/runs/<experiment_name>/`.

**Step 10 — Evaluate**
```bash
python -m model_dev_main.src.inference.eval_best_test_metrics \
  --run_dir model_dev_main/runs/unet_4layers_drift_wFutureWind_SAR_all_base16_lr1e-4_bs16 \
  --index_file model_dev_main/index_files/min400PM_wind_drift_SAR_dataset/index_test.jsonl
```

---

## Baselines

```bash
# Fit wind baseline on training data
python -m test_baselines.find_alpha_theta \
  --config model_dev_main/runs/EXPERIMENT/config_used.yaml \
  --out test_baselines/wind_baseline_fit.json

# Evaluate persistence + wind baselines on test set
python -m test_baselines.compute_baselines \
  --config model_dev_main/runs/EXPERIMENT/config_used.yaml \
  --index_file model_dev_main/index_files/min400PM_wind_drift_SAR_dataset/index_test.jsonl \
  --baseline persistence --out test_baselines/persistence_test_metrics.json
```

---

## Installation

```bash
git clone https://github.com/YourUsername/SAR-Sea-Ice-Prediction.git
cd SAR-Sea-Ice-Prediction
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

---

## Configuration

1. **Data paths:** Copy `configs/data_paths.yaml` and fill in your local paths to the SAR dataset, ice chart NetCDF, and CARRA directory.
2. **Credentials:** Copy `configs/copernicus_OAuth.yaml.example` → `configs/copernicus_OAuth.yaml` and add your [Copernicus Dataspace](https://dataspace.copernicus.eu/) OAuth credentials.


---


## License

This repository is licensed under the **GNU General Public License v3.0** (see [LICENSE](LICENSE)).

The sea ice drift computation code (`src/sea_ice_drift/`) is adapted from
[nansencenter/sea_ice_drift](https://github.com/nansencenter/sea_ice_drift/) (GPL-3.0),
which requires derivative works to carry the same license.

---

