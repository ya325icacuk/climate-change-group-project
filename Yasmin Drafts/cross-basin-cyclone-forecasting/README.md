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
    │   ├── model-1a-unet.ipynb                       ← Model 1a: U-Net (baseline spatial)
    │   ├── model-1b-unet-film.ipynb                  ← Model 1b: U-Net + FiLM (final spatial)
    │   ├── model-2a-fno.ipynb                        ← Model 2a: FNO (baseline spectral)
    │   ├── model-2b-fno-film.ipynb                   ← Model 2b: FNO + FiLM (padding + temporal)
    │   └── model-2c-ufno.ipynb                       ← Model 2c: U-FNO (final spectral hybrid)
    └── supplementary-analysis/
        ├── feature-ablation-and-shap.ipynb           ← Ablation study + SHAP explainability
        ├── ablation-and-shap-comparison.ipynb        ← Ablation + SHAP on both final models
        └── model-comparison.ipynb                    ← 6-model comparison & visualisations
```

## Getting Started

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Data

**Pre-processed, model-ready data is included** in `data/processed-data/` (~1.4 GB). All model training, evaluation, and analysis notebooks load directly from this folder — no additional download or preprocessing is required to run them.

The raw TCND dataset (~5.4 GB) is **not** included. It is only needed if you want to reproduce the preprocessing from scratch. See `data/DATA_GUIDE.md` for full details on what is included, what is not, and how to regenerate everything from raw data if needed.

### 3. Running Order

Since the processed data and trained checkpoints are already included, you can jump straight to any step:

1. **Report:** Open `main-analysis.ipynb` for the full narrative with code excerpts and results — **start here**
2. **Training:** Run individual model notebooks in `supplementary-notebooks/models/` (can be run in parallel)
3. **Analysis:** Run notebooks in `supplementary-notebooks/supplementary-analysis/`

To reproduce everything from scratch (optional):
1. Download raw TCND data — see `supplementary-notebooks/preprocessing/data-preprocessing-pipeline.ipynb`
2. Run the preprocessing notebook to regenerate `data/processed-data/`
3. Run `supplementary-notebooks/preprocessing/temporal-feature-extraction.ipynb` to regenerate temporal features

### 4. Checkpoints

Pre-trained model weights are in `checkpoints/`. Available checkpoints:

| Checkpoint | Model | Description |
| --- | --- | --- |
| `unet_best_wp.pt` | Model 1a: U-Net | WP-trained baseline |
| `unet_best_ft.pt` | Model 1a: U-Net | SP fine-tuned baseline |
| `unet_film_best_wp.pt` | Model 1b: U-Net + FiLM | WP-trained (final spatial) |
| `unet_film_best_ft.pt` | Model 1b: U-Net + FiLM | SP fine-tuned (final spatial) |
| `fno_best_wp.pt` | Model 2a: FNO | WP-trained baseline |
| `fno_best_ft.pt` | Model 2a: FNO | SP fine-tuned baseline |
| `fno_v2_best_wp.pt` | Model 2b: FNO + FiLM | WP-trained (padded + FiLM) |
| `fno_v2_best_ft.pt` | Model 2b: FNO + FiLM | SP fine-tuned (padded + FiLM) |
| `ufno_best_wp.pt` | Model 2c: U-FNO | WP-trained (final spectral) |
| `ufno_best_ft.pt` | Model 2c: U-FNO | SP fine-tuned (final spectral) |

## Models

| ID | Short Name | Notebook | FiLM | Key Idea |
| --- | --- | --- | --- | --- |
| **1a** | U-Net | `model-1a-unet.ipynb` | No | SE attention, residual blocks, DropPath |
| **1b** | U-Net + FiLM | `model-1b-unet-film.ipynb` | Yes | Temporal conditioning on baseline U-Net |
| **2a** | FNO | `model-2a-fno.ipynb` | No | 2 spectral layers, frequency-domain learning |
| **2b** | FNO + FiLM | `model-2b-fno-film.ipynb` | Yes | Reflect padding, 3 layers, FiLM |
| **2c** | U-FNO | `model-2c-ufno.ipynb` | Yes | 3-branch gated hybrid (spectral + U-Net + residual) |

## References

- Huang et al. (2025) — TropiCycloneNet benchmark dataset, *Nature Communications*
- Li et al. (2021) — Fourier Neural Operator, *ICLR*
- Perez et al. (2018) — FiLM conditioning, *AAAI*
- Wen et al. (2022) — U-FNO, *Advances in Water Resources*
