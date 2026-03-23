# Cross-Basin Generalisation for Tropical Cyclone Forecasting using TropiCycloneNet

> **Note:** This README is a working draft and will need to be revisited before final submission.

**Team:** Yasmin Akhmedova, Sam, Loïc

**Module:** ELEC70127 — ML for Tackling Climate Change, Imperial College London

**Date:** March 2026

---

## Overview

This project investigates whether deep learning models trained on **Western Pacific (WP)** tropical cyclone data can generalise to the **South Pacific (SP)** — a data-scarce basin in the opposite hemisphere. We evaluate two architecture families (U-Net and Fourier Neural Operator) with baseline and temporally-conditioned variants, plus a hybrid U-FNO, across a three-stage pipeline: zero-shot transfer → fine-tuning → transfer gap analysis.

## Repository Structure

```
cross-basin-cyclone-forecasting/
├── README.md                                        ← You are here
├── main-analysis.ipynb                              ← Main report notebook (start here)
├── requirements.txt                                 ← Python dependencies
├── checkpoints/                                     ← Trained model weights (.pt files)
├── data/                                            ← Dataset (see Data section below)
├── figures/                                         ← Generated plots and visualisations
└── supplementary-notebooks/
    ├── preprocessing/
    │   ├── data-preprocessing-pipeline.ipynb         ← Shared data preprocessing
    │   └── temporal-feature-extraction.py            ← Temporal feature extraction script
    ├── models/
    │   ├── unet-baseline.ipynb                       ← Baseline U-Net
    │   ├── unet-film-temporal.ipynb                  ← U-Net + FiLM time conditioning
    │   ├── fno-baseline.ipynb                        ← Baseline FNO
    │   ├── fno-v2-padded-film.ipynb                  ← FNO v2 (padding + 3 layers + FiLM)
    │   └── ufno-hybrid.ipynb                         ← U-FNO hybrid gated architecture
    └── supplementary-analysis/
        ├── feature-ablation-and-shap.ipynb           ← Ablation study + SHAP explainability
        └── model-comparison.ipynb                    ← 6-model comparison & visualisations
```

## Getting Started

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Data

**Pre-processed, model-ready data is included** in `data/processed/` (~2.1 GB). All model training, evaluation, and analysis notebooks load directly from this folder — no additional download or preprocessing is required to run them.

The raw TCND dataset (~8.7 GB) is **not** included. It is only needed if you want to reproduce the preprocessing from scratch. See `data/DATA_GUIDE.md` for full details on what is included, what is not, and how to regenerate everything from raw data if needed.

### 3. Running Order

Since the processed data and trained checkpoints are already included, you can jump straight to any step:

1. **Report:** Open `main-analysis.ipynb` for the full narrative with code excerpts and results — **start here**
2. **Training:** Run individual model notebooks in `supplementary-notebooks/models/` (can be run in parallel)
3. **Analysis:** Run notebooks in `supplementary-notebooks/supplementary-analysis/`

To reproduce everything from scratch (optional):
1. Download raw TCND data — see `supplementary-notebooks/preprocessing/data-preprocessing-pipeline.ipynb`
2. Run the preprocessing notebook to regenerate `data/processed/`
3. Run `supplementary-notebooks/preprocessing/temporal-feature-extraction.py` to regenerate temporal features

### 4. Checkpoints

Pre-trained model weights are in `checkpoints/`. Available checkpoints:

| Checkpoint | Model | Description |
| --- | --- | --- |
| `unet_best_wp.pt` | U-Net | WP-trained baseline |
| `unet_best_ft.pt` | U-Net | SP fine-tuned baseline |
| `fno_best_wp.pt` | FNO | WP-trained baseline |
| `fno_best_ft.pt` | FNO | SP fine-tuned baseline |
| `fno_v2_best_wp.pt` | FNO v2 | WP-trained (padded + FiLM) |
| `fno_v2_best_ft.pt` | FNO v2 | SP fine-tuned (padded + FiLM) |

> Checkpoints for U-Net+FiLM and U-FNO will be added once training completes.

## Models

| Model | Notebook | Time-aware | Key Idea |
| --- | --- | --- | --- |
| U-Net (baseline) | `unet-baseline.ipynb` | No | SE attention, residual blocks, DropPath |
| U-Net + FiLM | `unet-film-temporal.ipynb` | Yes | FiLM temporal conditioning on baseline U-Net |
| FNO (baseline) | `fno-baseline.ipynb` | No | 2 spectral layers, frequency-domain learning |
| FNO v2 | `fno-v2-padded-film.ipynb` | Yes | Reflect padding, 3 layers, FiLM |
| U-FNO | `ufno-hybrid.ipynb` | Yes | 3-branch gated hybrid (spectral + U-Net + residual) |

## References

- Huang et al. (2025) — TropiCycloneNet benchmark dataset, *Nature Communications*
- Li et al. (2021) — Fourier Neural Operator, *ICLR*
- Perez et al. (2018) — FiLM conditioning, *AAAI*
- Wen et al. (2022) — U-FNO, *Advances in Water Resources*
