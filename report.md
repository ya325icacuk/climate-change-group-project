# Cross-Basin Generalisation for Tropical Cyclone Forecasting — Experiment Report

## 1. Problem Statement

We investigate whether deep learning models trained on **Western Pacific (WP)** tropical cyclone data can generalise to the **South Pacific (SP)** — a data-scarce basin in the opposite hemisphere. This is the "Cross-Basin Generalisation" task from the TropiCycloneNet (TCND) dataset.

Each model performs **dual-head classification** at every 6-hourly timestep:

- **Direction** (8 classes): E, SE, S, SW, W, NW, N, NE — the compass direction the storm will move in the next 24 hours.
- **Intensity change** (4 classes): Weakening, Steady, Slow-intensification, Rapid-intensification — how the storm's wind speed will change over the next 24 hours.

### Why WP → SP?

- **Data scarcity**: SP has ~5x fewer storms than WP (30 vs 131), making it impractical to train a reliable model on SP alone.
- **Hemisphere flip**: WP is Northern Hemisphere, SP is Southern Hemisphere. The Coriolis reversal introduces a structural distributional shift — WP storms predominantly track NW/W, while SP storms track SE/W/SW. A naive WP-trained model will systematically mispredict SP directions.
- **Shared physics**: Underlying atmospheric predictors (SST, vertical wind shear) overlap substantially between basins, suggesting physics-informed features should transfer even if direction labels do not.

### What We Are Testing

We evaluate **three architectures** — ResNet-152, U-Net, and Fourier Neural Operator (FNO) — on a three-stage pipeline:

1. **Zero-shot transfer**: Train on WP, apply hemisphere-aware preprocessing, evaluate on SP with no fine-tuning.
2. **Fine-tuning**: Fine-tune on a small SP training set (12 storms, 354–402 samples) and re-evaluate.
3. **Transfer gap analysis**: Quantify the drop in performance from in-basin to cross-basin and how much fine-tuning recovers.

### Key Assumptions

- **Storm-level splitting**: All data splits are at the storm level (not timestep level) to prevent temporal data leakage between correlated 6-hourly observations within the same storm.
- **Hemisphere-aware direction labels**: SP direction labels are reflected along the N/S axis before training/evaluation, so that WP and SP share the same label semantics (e.g., "poleward" maps to the same class in both hemispheres).
- **No basin-leaking features**: We remove 54 of the 94 environmental features that encode basin identity (area one-hot, longitude bins, latitude bins, raw coordinates), retaining only the 40 physics-relevant dimensions.
- **WP-only normalisation**: Per-channel z-score statistics are computed exclusively from WP training data and applied to all splits, preventing information leakage from SP into the normalisation.
- **Class imbalance handling**: All models use inverse-frequency class weights in the cross-entropy loss to counteract severe imbalance (e.g., Weakening ~46%, Steady ~6%).

---

## 2. Data Preprocessing

The shared preprocessing pipeline (`experiments/pre-processing.ipynb`) transforms raw TCND data into model-ready tensors.

### 2.1 Raw Data Sources

Three modalities per 6-hourly timestep, for WP and SP:

| Modality | Format | Content |
|----------|--------|---------|
| **Data3D** | NetCDF (.nc) | 81×81 gridded atmospheric fields: SST (1ch), u/v wind and geopotential height at 4 pressure levels (200, 500, 850, 925 hPa) = 13 base channels |
| **Env-Data** | NumPy (.npy) | 94-dim structured environmental features per timestep, including pre-computed 24h direction and intensity-change labels |
| **Data1D** | TSV (.txt) | Per-storm track files with lat/lon offsets, normalised wind and pressure (4 features retained) |

### 2.2 Key Processing Steps

1. **Basin-leaking feature removal**: 54 dimensions stripped from Env-Data (area one-hot, longitude/latitude bins, raw coordinates) → 40-dim clean vector: wind, move velocity, intensity class (6), month (12), history direction 12h/24h (8+8), history intensity change 24h (4).

2. **Hemisphere-aware transforms (SP only)**:
   - Direction reflection: N/S mirror map applied to direction labels and history direction features (SE↔NE, S↔N, SW↔NW).
   - Meridional wind flip: `v = -v` at all pressure levels so that "poleward" wind has consistent sign across hemispheres.

3. **Derived channels**: Two physics-motivated channels appended to the 13 base channels:
   - **Wind shear magnitude** (ch 13): `√((u₂₀₀ − u₈₅₀)² + (v₂₀₀ − v₈₅₀)²)`
   - **850 hPa relative vorticity** (ch 14): `∂v₈₅₀/∂x − ∂u₈₅₀/∂y` via centred finite differences.
   - Total: **15 channels** per grid sample.

4. **Normalisation**: Per-channel z-score (mean=0, std=1) computed from WP training split only (Welford's online algorithm). Applied to all splits. NaN/fill values replaced with 0.0 before normalisation.

5. **Missing data handling**: NetCDF fill values (>1e10) and non-finite values replaced with 0.0 (primarily SST over land). Sentinel labels (−1) for timesteps without valid 24h futures are preserved but excluded during training.

### 2.3 Data Splits

| Split | Storms | Timesteps | Valid Labels | Purpose |
|-------|--------|-----------|-------------|---------|
| WP train | 105 | 3,716 | 3,252 | Source-domain training |
| WP val | 26 | 846 | 730 | Source-domain validation |
| SP test | 15 | 427 | 367 | Zero-shot transfer evaluation |
| SP fine-tune train | 12 | 402 | 354 | Target-domain fine-tuning |
| SP fine-tune val | 3 | 93 | 81 | Fine-tuning validation |

### 2.4 Class Distribution

**Direction** (WP train): NW (30.5%) and W (28.2%) dominate; E is rare (6.7%).

**Intensity** (WP train): Weakening (~46%), Slow-intensification (~34%), Rapid-intensification (~13%), Steady (~6%).

All models apply inverse-frequency class weighting and/or label smoothing to mitigate this imbalance.

---

## 3. Model Architectures

### 3.1 ResNet-152

- **Backbone**: ResNet-152 (no pretrained weights), modified with 3×3 stride-1 `conv1` (instead of 7×7 stride-2) to preserve spatial resolution for 81×81 inputs.
- **Fusion**: Late fusion — ResNet extracts 2048-d spatial features via global average pooling, concatenated with env (40-d) and 1D (4-d) before two MLP classification heads.
- **Parameters**: 58.7M
- **Regularisation**: Dropout (0.25), label smoothing (0.05), vertical flip augmentation.
- **Training**: AdamW + OneCycleLR, max 80 epochs, early stopping (patience 15). Stopped at epoch 31.

### 3.2 U-Net

- **Architecture**: 4-level encoder-decoder (32→64→128→256, bottleneck 512) with skip connections, residual blocks, SE channel attention, and DropPath.
- **Fusion**: Global average pooling of decoder output (32-d) concatenated with env (40-d) and 1D (4-d) = 76-d fused vector.
- **Parameters**: 9.8M
- **Regularisation**: Dropout2d (0.2), Mixup (α=0.2), CutOut (2×16×16), Gaussian noise (σ=0.05), channel dropout (p=0.15), EMA weights.
- **HPO**: Optuna (30 trials × 30 epochs). Finding: model is data-limited, not capacity-limited (bc=32 > bc=64).
- **Training**: AdamW + OneCycleLR, max 300 epochs, early stopping (patience 50). Stopped at epoch 144.

### 3.3 Fourier Neural Operator (FNO)

- **Architecture**: Lifting layer → 2 spectral blocks (SpectralConv2d with truncated Fourier modes + skip connections + BatchNorm) → projection → global average pooling.
- **Fusion**: Same late-fusion as above — pooled grid features concatenated with env and 1D.
- **Parameters**: 10.3M (hidden=80, modes=20, layers=2)
- **Regularisation**: Dropout2d (0.05).
- **HPO**: Optuna (60 trials × 30 epochs). Best optimizer: Muon (fell back to AdamW).
- **Training**: AdamW + cosine annealing, max 80 epochs, early stopping (patience 15). Stopped at epoch 18.

---

## 4. Results

### 4.1 In-Basin Performance (WP Validation)

| Metric | ResNet-152 | U-Net | FNO |
|--------|-----------|-------|-----|
| **Direction Accuracy** | 0.512 | **0.625** | 0.503 |
| Direction F1 (macro) | 0.404 | **0.447** | 0.364 |
| **Intensity Accuracy** | 0.549 | 0.510 | **0.558** |
| Intensity F1 (macro) | **0.463** | 0.433 | 0.410 |

**U-Net leads on direction** by a wide margin (+11pp over ResNet, +12pp over FNO). Its encoder-decoder architecture with skip connections preserves fine-grained spatial patterns that indicate storm movement direction.

**ResNet and FNO are marginally better on intensity**, likely because intensity change depends more on aggregate features (overall SST, shear magnitude) than spatial structure, playing to the strengths of deep feature extractors and spectral methods.

### 4.2 Zero-Shot Cross-Basin Transfer (SP Test)

| Metric | ResNet-152 | U-Net | FNO |
|--------|-----------|-------|-----|
| **Direction Accuracy** | 0.272 | **0.414** | 0.248 |
| Direction F1 (macro) | 0.212 | **0.293** | 0.217 |
| **Intensity Accuracy** | **0.384** | 0.360 | 0.357 |
| Intensity F1 (macro) | **0.319** | 0.291 | 0.300 |

| Transfer Gap (WP→SP) | ResNet-152 | U-Net | FNO |
|-----------------------|-----------|-------|-----|
| Direction Accuracy | −0.240 | **−0.211** | −0.255 |
| Intensity Accuracy | −0.165 | **−0.150** | −0.201 |

All models suffer a substantial transfer gap (20–25pp on direction, 15–20pp on intensity). **U-Net retains the best zero-shot direction accuracy (0.414) and has the smallest transfer gap**, suggesting its learned spatial features generalise best across basins.

### 4.3 Fine-Tuned Performance (SP Test)

| Metric | ResNet-152 | U-Net | FNO |
|--------|-----------|-------|-----|
| **Direction Accuracy** | 0.294 | **0.401** | 0.346 |
| Direction F1 (macro) | 0.234 | **0.309** | 0.272 |
| **Intensity Accuracy** | **0.501** | 0.450 | 0.335 |
| Intensity F1 (macro) | **0.430** | 0.367 | 0.288 |

| Fine-tune Recovery (vs zero-shot) | ResNet-152 | U-Net | FNO |
|-----------------------------------|-----------|-------|-----|
| Direction Accuracy | +0.022 | −0.013 | +0.098 |
| Intensity Accuracy | +0.117 | +0.090 | −0.022 |

Fine-tuning results are mixed:

- **Intensity improves substantially** for ResNet (+11.7pp) and U-Net (+9.0pp), suggesting intensity-relevant features are adaptable with limited data.
- **Direction is harder to recover**: U-Net's direction actually degrades slightly (−1.3pp) after fine-tuning, likely due to overfitting on only 354 SP samples. FNO gains the most on direction (+9.8pp) but from a lower baseline.
- **ResNet achieves the best fine-tuned intensity (0.501)**, benefiting from its two-phase strategy (freeze backbone → unfreeze last layer).

### 4.4 Overfitting Analysis

With only ~3,250 training samples, overfitting is the dominant challenge:

| Model | Params | Params-per-sample | Epochs before early stop | Severity |
|-------|--------|-------------------|--------------------------|----------|
| ResNet-152 | 58.7M | 18,050:1 | 31/80 | High |
| FNO | 10.3M | 3,170:1 | 18/80 | Severe |
| U-Net | 9.8M | 3,015:1 | 144/300 | Moderate |

**FNO overfits fastest** despite similar parameter count to U-Net — spectral convolutions have high expressivity but lack the inductive biases (locality, skip connections) that help U-Net regularise. **ResNet is massively overparameterised** for this dataset. **U-Net trained longest** thanks to aggressive augmentation (Mixup, CutOut, channel dropout, EMA).

### 4.5 Model-Specific Insights

- **U-Net**: Shallow encoder features (level 0, 32ch) have the strongest activations, while deep features (level 3, 256ch) are underutilised — consistent with the data-limited regime.
- **FNO**: Fourier mode analysis reveals the model predominantly learns meridional (N-S) patterns (top modes all have width-mode=0), suggesting large-scale atmospheric structure matters more than fine spatial details.
- **ResNet**: Layer activation magnitudes drop sharply at layer4 (0.03 vs ~1.0–1.6 in earlier layers), indicating the deepest layers are barely contributing — further evidence of overparameterisation.

---

## 5. Summary

| | Best In-Basin Direction | Best In-Basin Intensity | Best Zero-Shot Transfer | Best Fine-Tuned Overall |
|-|------------------------|------------------------|------------------------|------------------------|
| **Winner** | U-Net (0.625) | FNO (0.558) | U-Net (0.414 dir) | U-Net (dir) / ResNet (int) |

The **U-Net is the strongest overall model** for this task. It achieves the highest direction accuracy both in-basin and cross-basin, has the smallest transfer gap, and maintains the best balance between capacity and regularisation. However, no model achieves strong absolute performance on cross-basin transfer — even the best zero-shot direction accuracy (0.414) is far from operational utility.

---

## 6. Next Steps

1. **Transformer architecture**: Implement a spatiotemporal transformer with cross-attention fusion of Data3D, Env-Data, and Data1D. Supervisor feedback suggests this is a strong candidate — non-iterative multi-horizon prediction avoids error accumulation, and careful multimodal fusion (cross-attention or diffusion-style conditioning) could outperform the late-fusion strategy used by all current models.

2. **Physics-Informed GAN**: Implement a GAN with physical constraints (vorticity conservation, potential vorticity, cyclogenesis thermodynamics) as an adversarial regulariser. This could improve transfer by enforcing universal physical laws rather than learning basin-specific patterns.

3. **Improved fine-tuning strategies**: The current fine-tuning results are underwhelming, especially for direction. Options to explore:
   - Larger SP fine-tuning fractions (25%, 50% of SP data instead of the current fixed split).
   - Progressive unfreezing with discriminative learning rates.
   - Coriolis-aware conditioning: encode latitude as sin(φ) as a scalar input, letting the model learn hemisphere effects rather than relying on label reflection.

4. **Data augmentation for transfer**: Apply basin-agnostic augmentations during WP training (random rotations, spatial jitter) to encourage learning features that are invariant to storm orientation.

5. **Ensemble methods**: Combine predictions from U-Net (best direction) and ResNet (best intensity) to exploit complementary strengths.

6. **Lead-time analysis**: Evaluate accuracy as a function of forecast lead time (6h, 12h, 24h) — supervisor feedback identified this as a key evaluation dimension for operational relevance.

7. **Extension to other basins**: Test whether the best model(s) generalise to other data-scarce basins (NI, SI) without basin-specific retraining, as originally planned in the project proposal.
