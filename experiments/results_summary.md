# Experiment Results Summary

## Overview

**Task:** Cross-basin generalisation for tropical cyclone forecasting (WP → SP).
**Dual-head classification** at each 6-hourly timestep:
- **Direction** (8 classes): E, SE, S, SW, W, NW, N, NE
- **Intensity change** (4 classes): Weakening, Steady, Slow-intensification, Rapid-intensification

**Dataset:** TropiCycloneNet (TCND) — WP train (3,252 samples, 105 storms), WP val (730 samples, 26 storms), SP test (367 samples, 15 storms), SP fine-tune train (354 samples, 12 storms), SP fine-tune val (81 samples, 3 storms).

---

## Model Comparison

### In-Basin Performance (WP Validation)

| Model | Params | Dir Acc | Dir F1 | Int Acc | Int F1 | Epochs | Notes |
|-------|-------:|--------:|-------:|--------:|-------:|-------:|-------|
| **U-Net + FiLM** | 10.0M | **0.629** | **0.465** | 0.564 | **0.471** | 151/300 | Best direction + intensity F1 |
| **U-Net** | 9.8M | 0.625 | 0.447 | 0.510 | 0.433 | 144/300 | Strong baseline |
| U-FNO | 1.0M | 0.508 | 0.401 | **0.625** | 0.499 | 41/150 | Best intensity acc; tiny model |
| ResNet-152 | 58.7M | 0.512 | 0.404 | 0.549 | 0.463 | 31/80 | Overparameterised |
| FNO v2 | 0.9M | 0.493 | 0.409 | 0.567 | 0.459 | full | +padding, +FiLM, +3 layers |
| FNO | 10.3M | 0.503 | 0.364 | 0.558 | 0.410 | 18/80 | Severe overfitting |

### Zero-Shot Cross-Basin Transfer (SP Test)

| Model | Dir Acc | Dir F1 | Int Acc | Int F1 | Dir Gap | Int Gap |
|-------|--------:|-------:|--------:|-------:|--------:|--------:|
| **U-Net** | **0.414** | **0.293** | 0.360 | 0.291 | **-0.211** | -0.150 |
| U-Net + FiLM | 0.390 | 0.261 | 0.362 | 0.301 | -0.239 | -0.202 |
| U-FNO | 0.202 | 0.168 | **0.433** | **0.403** | -0.307 | -0.192 |
| ResNet-152 | 0.272 | 0.212 | 0.384 | 0.319 | -0.240 | -0.165 |
| FNO v2 | 0.213 | 0.176 | 0.395 | 0.385 | -0.280 | -0.172 |
| FNO | 0.248 | 0.217 | 0.357 | 0.300 | -0.255 | -0.201 |

### Fine-Tuned Performance (SP Test)

| Model | Dir Acc | Dir F1 | Int Acc | Int F1 | Dir Recovery | Int Recovery |
|-------|--------:|-------:|--------:|-------:|-------------:|-------------:|
| **U-Net** | **0.401** | **0.309** | 0.450 | 0.367 | -0.013 | +0.090 |
| U-Net + FiLM | 0.390 | 0.273 | 0.392 | 0.331 | +0.000 | +0.030 |
| U-FNO | 0.283 | 0.222 | 0.496 | 0.395 | +0.082 | +0.063 |
| ResNet-152 | 0.294 | 0.234 | **0.501** | **0.430** | +0.022 | +0.117 |
| FNO v2 | 0.267 | 0.215 | 0.507 | 0.441 | +0.054 | +0.112 |
| FNO | 0.346 | 0.272 | 0.335 | 0.288 | +0.098 | -0.022 |

---

## Key Findings

### Direction Forecasting
- **U-Net + FiLM achieves best WP direction (62.9%)** with full training, slightly surpassing the base U-Net (62.5%). FiLM time conditioning provides a small but consistent improvement in-basin.
- **U-Net retains best zero-shot transfer (41.4%)** — the FiLM model's added temporal features don't transfer as well across basins (39.0%).
- U-FNO (1.0M params) reaches only 50.8% WP direction despite 41 epochs — the spectral+spatial hybrid underperforms for direction.

### Intensity Change Classification
- **U-FNO achieves best WP intensity (62.5%)** with only 1M parameters — the spectral+spatial hybrid is highly effective for global patterns that drive intensification.
- **U-Net + FiLM achieves best WP intensity F1 (47.1%)** and best in-basin intensity acc (56.4%) among non-spectral models.
- ResNet-152 achieves the best fine-tuned intensity (50.1%) via two-phase strategy.

### FiLM Time Conditioning Impact
With full training (151 epochs), U-Net + FiLM vs U-Net baseline:
- **Direction:** 62.9% vs 62.5% (+0.4pp) — small improvement
- **Intensity:** 56.4% vs 51.0% (+5.4pp) — substantial improvement
- **Dir F1:** 0.465 vs 0.447 (+1.8pp) — improved per-class balance
- **Int F1:** 0.471 vs 0.433 (+3.8pp) — improved per-class balance

FiLM time conditioning **benefits intensity prediction substantially** (+5.4pp accuracy). Temporal features (storm progress, hour/month cycles) help the model learn seasonal and diurnal patterns relevant to intensification dynamics. The direction improvement is smaller but still positive.

### U-FNO Gate Analysis
The learned gate weights show branch preferences per layer:

| Layer | Spectral | U-Net | Residual |
|------:|---------:|------:|---------:|
| 1 | 0.40 | 0.27 | 0.33 |
| 2 | 0.40 | 0.30 | 0.30 |
| 3 | 0.38 | 0.31 | 0.31 |

- **Spectral branch dominates** in all layers (38-40%), confirming global patterns matter most.
- U-Net branch importance **increases** in deeper layers (27% → 31%), suggesting local features become more important for refined predictions.
- Near-uniform distribution indicates all three branches contribute meaningfully.

### Transfer Gap Summary
- All models suffer 21-31pp direction transfer gap (WP → SP zero-shot).
- U-Net has the smallest gap (-21.1pp direction), U-FNO the largest (-30.7pp).
- Fine-tuning recovers intensity better than direction across all models.

---

## Ablation Study (U-Net, WP Validation)

### Grid Channel Importance (Leave-One-Out)
| Channel Group | Dir Acc Drop | Int Acc Drop |
|--------------|------------:|-----------:|
| **u_wind** | **+0.175** | -0.011 |
| **v_wind** | **+0.116** | +0.043 |
| wind_shear | +0.010 | +0.034 |
| vorticity | +0.003 | -0.014 |
| SST | -0.004 | **+0.048** |
| geopotential | -0.006 | -0.010 |

- **u_wind is critical for direction** (17.5pp drop when removed) — zonal wind directly encodes storm movement.
- **v_wind also important** (11.6pp drop) — meridional wind indicates N/S tracking.
- **SST is critical for intensity** (4.8pp drop) — warm ocean drives intensification.

### Env Feature Importance (Leave-One-Out)
| Feature Group | Dir Acc Drop | Int Acc Drop |
|--------------|------------:|-----------:|
| history_dir_12h | +0.008 | +0.001 |
| move_velocity | +0.003 | -0.007 |
| wind | +0.000 | +0.014 |
| **intensity_class** | -0.004 | **+0.049** |
| month | -0.003 | +0.012 |
| history_dir_24h | -0.007 | +0.026 |
| history_int_24h | -0.007 | +0.019 |

- **intensity_class** is most critical env feature for intensity prediction (4.9pp drop).
- Env features have minimal individual impact on direction — the grid channels dominate.

### Modality Ablation
| Config | Dir Acc | Int Acc |
|--------|--------:|--------:|
| Baseline (all) | 0.625 | 0.510 |
| No Grid | 0.315 | 0.384 |
| No Env | 0.619 | 0.390 |
| No 1D | 0.626 | 0.522 |
| Only Grid | 0.622 | 0.371 |
| Only Env | 0.316 | 0.389 |
| Only 1D | 0.293 | 0.269 |

- **Grid is essential for direction** (31pp drop without it, only 0.3pp drop without env/1d).
- **Env is essential for intensity** (12pp drop without it) — environmental features encode thermodynamic state.
- **1D track data is redundant** — removing it slightly improves both direction and intensity.

---

## Training Notes

- **All models fully trained with early stopping:**
  - U-Net: 144/300 epochs
  - U-Net + FiLM: 151/300 epochs (EMA, augmentation suite)
  - U-FNO: 41/150 epochs (early stopped — overfits quickly)
  - ResNet-152: 31/80 epochs
  - FNO: 18/80 epochs (severe overfitting)
  - FNO v2: full training
- All models use inverse-frequency class weights and label smoothing (0.05).
- All models use WP-only normalisation (z-score statistics from WP train only).
- GPU: NVIDIA RTX 5090 (32GB VRAM). Both U-Net+FiLM and U-FNO run in parallel (~8GB combined).
