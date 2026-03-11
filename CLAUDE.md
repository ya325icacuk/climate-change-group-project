# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Imperial College ELEC70127 "ML for Tackling Climate Change" group project (40% of grade). The task is to apply ML to the **TropiCycloneNet Dataset (TCND)** — a multimodal tropical cyclone dataset covering 1950–2023 across six ocean basins (EP, NA, NI, SI, SP, WP).

**Deadlines:** Week 1 deliverable (1-page description) March 12; Presentation + repo submission March 27.

## Dataset Structure

Data lives in `data/tropicyclonenet/` (gitignored). Download via `gdown` as shown in the starter notebook.

- **Test subset:** ~3.34 GB (`TCND_test.zip`)
- **Full dataset:** ~25.7 GB (`TCND_full.zip`)

After extraction, three modalities organized by basin and year:

```
TCND_test/TCND_test/
├── Data1D/{basin}/test/     — Tab-separated .txt files per storm (1D time series)
│   Columns: index, category(?), lat_offset, lon_offset, wind_norm, pressure_norm, datetime(YYYYMMDDhh), name
├── Data3D/{basin}/{year}/{storm}/  — NetCDF (.nc) files per 6-hour timestep
│   Variables: sst (2D), u/v/z (4D: time × pressure_level × lat × lon)
│   Pressure levels: 200, 500, 850, 925 hPa; Grid: 20°×20° at 0.25° resolution
└── Env-Data/{basin}/{year}/{storm}/  — .npy files per 6-hour timestep
    Structured environmental features
```

## Development Environment

- Python with conda (environment `torch_5070` exists on this machine)
- Key packages: `gdown netCDF4 pandas numpy matplotlib cartopy seaborn xarray scipy`
- Install: `pip install -q gdown netCDF4 pandas numpy matplotlib cartopy seaborn xarray scipy`
- Primary development is in Jupyter notebooks (`.ipynb`)

## Running the Notebook

```bash
jupyter notebook "Starter Notebook.ipynb"
```

The starter notebook handles: data download via gdown, extraction, inspection of all three modalities, track/intensity visualization with cartopy, storm-centered field plots with xarray, and GIF animation generation.

## Suggested Project Directions (from starter notebook)

- **A — Track vs intensity forecasting:** different input modalities for each task
- **B — Basin generalization:** train on one basin, test transfer to another
- **C — Rapid intensification precursors:** use Env-Data + Data3D for sharp wind speed increases
- **D — Multimodal fusion ablation:** compare Data1D-only, Data3D-only, Env-Data-only, and fused models
