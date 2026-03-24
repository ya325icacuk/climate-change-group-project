# Experiment Results Summary

## Overview

**Task:** Cross-basin generalisation for tropical cyclone forecasting (WP → SP).
**Dual-head classification** at each 6-hourly timestep:
- **Direction** (8 classes): E, SE, S, SW, W, NW, N, NE
- **Intensity change** (4 classes): Weakening, Steady, Slow-intensification, Rapid-intensification

**Dataset:** TropiCycloneNet (TCND) — WP train (3,252 samples, 105 storms), WP val (730, 26 storms), SP test (367, 15 storms), SP fine-tune (354 train, 81 val).

**Training:** 20 Optuna HPO trials (50 epochs each) → 300-epoch full training with best config. All 5 models trained in parallel on RTX 5090 (32GB VRAM, ~30GB peak usage).

---

## Optimised Hyperparameters (Optuna, 20 trials)

| Model | Key Config | Params | HPO Best |
|-------|-----------|-------:|--------:|
| U-Net | base_ch=32, 5 levels, head=256, drop_path=0.065, bs=32 | 39.1M | 61.4% |
| U-Net+FiLM | base_ch=48, 3 levels, head=512, time_emb=64, bs=32 | 5.9M | 61.4% |
| FNO | hidden=64, modes=15, 4 layers, dropout=0.11, bs=64 | 7.4M | 60.7% |
| FNO v2 | hidden=48, modes=20, 5 layers, padding=9, time_emb=64, bs=64 | 9.3M | 61.9% |
| U-FNO | hidden=48, modes=16, 2 layers, padding=13, dropout=0.11, bs=64 | 2.5M | 63.2% |

---

## Model Comparison (300-Epoch Full Training)

### In-Basin Performance (WP Validation)

| Model | Params | Dir Acc | Dir F1 | Int Acc | Int F1 | Epochs |
|-------|-------:|--------:|-------:|--------:|-------:|-------:|
| **U-Net** | 39.1M | **0.633** | **0.464** | 0.581 | 0.471 | 134/300 |
| U-FNO | 2.5M | 0.610 | 0.463 | 0.585 | 0.463 | 100/300 |
| FNO v2 | 9.3M | 0.597 | 0.374 | 0.590 | 0.407 | 135/300 |
| U-Net+FiLM | 5.9M | 0.592 | 0.426 | **0.603** | **0.492** | 120/300 |
| FNO | 7.4M | 0.579 | 0.401 | 0.641 | 0.471 | 114/300 |

### Zero-Shot Cross-Basin Transfer (SP Test)

| Model | Dir Acc | Dir F1 | Int Acc | Int F1 | Dir Gap |
|-------|--------:|-------:|--------:|-------:|--------:|
| **U-FNO** | **0.379** | **0.295** | **0.523** | **0.409** | -0.231 |
| FNO v2 | 0.368 | 0.221 | 0.485 | 0.333 | -0.229 |
| U-Net+FiLM | 0.335 | 0.270 | 0.493 | 0.396 | -0.257 |
| FNO | 0.330 | 0.245 | 0.485 | 0.338 | -0.249 |
| U-Net | 0.311 | 0.222 | 0.411 | 0.340 | **-0.322** |

### Fine-Tuned Performance (SP Test, 354 samples)

| Model | Dir Acc | Dir F1 | Int Acc | Int F1 | Dir Recovery |
|-------|--------:|-------:|--------:|-------:|-------------:|
| **U-Net+FiLM** | **0.409** | 0.281 | 0.466 | 0.395 | +0.074 |
| FNO | 0.354 | 0.273 | 0.520 | **0.421** | +0.024 |
| U-Net | 0.349 | 0.252 | 0.469 | 0.387 | +0.038 |
| FNO v2 | 0.327 | 0.241 | **0.542** | 0.399 | -0.041 |
| U-FNO | 0.316 | **0.306** | 0.490 | 0.422 | -0.063 |

---

## Key Findings

### Direction Forecasting
- **U-Net leads in-basin (63.3%)** with 5-level encoder-decoder (39.1M params), but suffers the worst transfer gap (-32.2pp).
- **U-FNO leads zero-shot transfer (37.9%)** with only 2.5M params — spectral+spatial fusion creates transferable features.
- **U-Net+FiLM leads after fine-tuning (40.9%)** — temporal conditioning enables effective adaptation to new basins.

### Intensity Classification
- **FNO achieves best WP intensity (64.1%)** — spectral convolutions excel at global thermodynamic patterns.
- **U-Net+FiLM achieves best intensity F1 (49.2%)** — temporal features improve per-class balance.
- **U-FNO achieves best zero-shot intensity (52.3%)** — again, the best transferer.

### The Efficiency-Generalisation Trade-off
| Scenario | Best Model | Why |
|----------|-----------|-----|
| In-basin deployment | U-Net (63.3%) | Largest model memorises WP patterns best |
| Cross-basin (no local data) | U-FNO (37.9%) | Compact spectral+spatial features transfer best |
| Cross-basin (with fine-tuning) | U-Net+FiLM (40.9%) | Temporal conditioning enables effective adaptation |
| Intensity forecasting | FNO (64.1% WP) | Spectral methods capture global thermodynamic state |
| Parameter efficiency | U-FNO (61.0% with 2.5M) | 16x smaller than U-Net with only 2.3pp less accuracy |

### HPO Insights
- All models converged to LR ~ 2-5e-4, confirming the data distribution dominates optimisation
- U-FNO's optimal 2-layer architecture was unexpected — depth hurts when branches provide sufficient expressivity
- U-Net preferred depth (5 levels) over width, while U-Net+FiLM preferred width (48ch) over depth (3 levels)
- FNO v2 benefited from 20 Fourier modes (vs 12 baseline) — higher spectral resolution improves atmospheric feature extraction

---

## Ablation Study (U-Net, WP Validation — from prior experiments)

### Grid Channel Importance (Leave-One-Out)
| Channel | Dir Drop | Int Drop |
|---------|--------:|--------:|
| **u_wind** | **+17.5pp** | -1.1pp |
| **v_wind** | **+11.6pp** | +4.3pp |
| wind_shear | +1.0pp | +3.4pp |
| **SST** | -0.4pp | **+4.8pp** |

### Modality Ablation
- **Grid essential for direction** (31pp drop without). **Env essential for intensity** (12pp drop without).
- **1D track data redundant** — removing it slightly improves performance.

---

## Training Infrastructure

- **GPU:** NVIDIA RTX 5090 (32GB VRAM)
- **Peak VRAM:** ~30GB (all 5 models training in parallel)
- **Total time:** ~7 hours (HPO + 300-epoch training)
- **Framework:** PyTorch + Optuna + OneCycleLR + EMA
- **Augmentation:** CutOut (16x16), Gaussian noise (σ=0.05), channel dropout (15%), label smoothing (0.05)
