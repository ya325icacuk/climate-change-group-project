# Implementation Report — New Architectures & Analysis (Post-Hadrian Meeting, March 18)

This document summarizes all new code added after the supervisor meeting. The main goals were: (1) add temporal awareness via FiLM conditioning, (2) fix FNO issues, (3) explore hybrid U-FNO, and (4) add explainability tools.

---

## 1. Temporal Feature Extraction

**File:** `experiments/add_time_features.py`

Standalone script that reads `data/processed/split_index.csv` and computes a 6-dimensional temporal feature vector per timestep:

| Feature | Description |
|---------|-------------|
| `storm_progress` | Normalized 0 to 1 across the storm lifetime |
| `hour_sin`, `hour_cos` | Cyclical encoding of hour-of-day (period = 24h) |
| `month_sin`, `month_cos` | Cyclical encoding of month (period = 12) |
| `timestep_idx_normalized` | Same as storm_progress (useful for ablation) |

Outputs saved to `data/processed/time/{split}_time.pt` as `dict[storm_id -> Tensor(N_t, 6)]` for all 5 splits (wp_train, wp_val, sp_test, sp_ft_train, sp_ft_val). Storm keys and timestep counts match the existing grid tensors exactly.

**To regenerate:**
```bash
cd experiments && python add_time_features.py
```

---

## 2. U-Net + FiLM Time Conditioning

**File:** `experiments/unet_film.ipynb`

Built on top of the baseline U-Net (SE attention, residual connections, DropPath). Key additions:

- **`FiLMLayer`**: Projects time embedding to per-channel scale (gamma) and shift (beta). Identity-initialized (gamma=1, beta=0) so the model starts equivalent to the baseline.
- **`FiLMConvBlock`**: FiLM is applied after the second BatchNorm, before GELU activation, in every encoder and decoder block.
- **Time MLP**: `(6,) -> 64 -> 64` with GELU, producing a conditioning vector broadcast to all FiLM layers.
- **Training**: Same hyperparameters as baseline (BASE_CHANNELS=32, N_LEVELS=4, HEAD_DIM=256, EPOCHS=300, PATIENCE=50, LR=5e-4). Includes EMA, CutOut, Gaussian noise, channel dropout, mixup augmentations.
- **Pipeline**: WP training -> WP eval -> SP zero-shot -> SP fine-tuning -> confusion matrices -> summary.
- **Checkpoints**: `unet_film_best_wp.pt`, `unet_film_best_ft.pt`

---

## 3. FNO v2 (Padded Spectral Conv + 3 Layers + FiLM)

**File:** `experiments/fno_v2.ipynb`

Three improvements over the baseline FNO:

1. **Spatial padding**: `F.pad(x, [9,9,9,9], mode='reflect')` before `rfft2`, then crop back after `irfft2`. This reduces spectral leakage and boundary artifacts that were polluting the Fourier signal.
2. **3 spectral layers** (baseline had 2): More capacity for learning fine-grained frequency patterns. FNO can exhibit "spectral leaps" (sudden accuracy jumps after long plateaus), so more layers + patience is warranted.
3. **FiLM conditioning**: Same mechanism as U-Net+FiLM, injected after BatchNorm in each spectral block.

Configuration: `HIDDEN_CHANNELS=32, N_MODES=12, N_LAYERS=3, PADDING=9, EPOCHS=150, PATIENCE=30`

Analysis sections include Fourier mode importance visualization (which frequency modes are most active per layer) and per-storm prediction timeline.

**Checkpoints**: `fno_v2_best_wp.pt`, `fno_v2_best_ft.pt`

---

## 4. U-FNO (Hybrid Gated Architecture)

**File:** `experiments/ufno.ipynb`

Inspired by Google's U-FNO paper. Each `UFNOBlock` has 3 parallel branches combined with learnable softmax-gated weights:

| Branch | Role |
|--------|------|
| `PaddedSpectralConv2d` | Global frequency-domain features (with reflect padding) |
| `UNetBranch` | Local spatial features (lightweight single-level encode-decode) |
| `nn.Conv2d(ch, ch, 1)` | Residual / identity pathway |

- **Gating**: `self.gate = nn.Parameter(torch.ones(3)/3)` combined via `F.softmax`. The model learns which branch matters most at each layer.
- **FiLM**: Applied after BatchNorm in each block, same as other models.
- **Config**: `HIDDEN_CH=32, N_MODES=12, N_LAYERS=3, PADDING=9, TIME_EMB_DIM=64`
- **Analysis**: Gate weight visualization showing learned branch importance per layer.
- **Checkpoints**: `ufno_best_wp.pt`, `ufno_best_ft.pt`

---

## 5. Ablation Study + SHAP Explainability

**File:** `experiments/ablation_shap.ipynb`

Uses the baseline U-Net checkpoint (`unet_best_wp.pt`) to analyze feature importance.

### Ablation approaches

- **Leave-one-group-out (grid)**: Zero out one channel group at a time (SST, u_wind, v_wind, geopotential, wind_shear, vorticity) and measure accuracy drop.
- **Leave-one-group-out (env)**: Same for env feature groups (wind, move_velocity, intensity_class, month, history_dir_12h/24h, history_int_24h).
- **Modality ablation**: 6 configs — No Grid / No Env / No 1D / Only Grid / Only Env / Only 1D.
- **Add-one-in**: Start from all-zeros baseline, add one feature group at a time to measure individual contribution.

### Explainability

- **SHAP GradientExplainer** on env features: `EnvWrapper` fixes grid/d1d to dataset means, feeds only env through the model. 100 background samples, 200 explanation samples. Outputs beeswarm plot + grouped mean |SHAP| bar chart.
- **Gradient-based grid attribution**: Computes `mean(|d_loss/d_input|)` per channel as a tractable alternative to full SHAP on (15, 81, 81) tensors. Per-channel and per-group bar charts.

---

## 6. Comparison Notebook

**File:** `experiments/comparison.ipynb`

Updated from 3 models to 6 models:

| # | Model | Time-aware | Architecture |
|---|-------|-----------|-------------|
| 1 | FNO (baseline) | No | 2 spectral layers, no padding |
| 2 | ResNet-152 (baseline) | No | Standard ResNet backbone |
| 3 | U-Net (baseline) | No | SE attention, residual, DropPath |
| 4 | U-Net + FiLM | Yes | Baseline U-Net + FiLM time conditioning |
| 5 | FNO v2 | Yes | 3 layers + padding + FiLM |
| 6 | U-FNO | Yes | 3-branch gated hybrid + FiLM |

Changes made:
- **CycloneDataset** returns 6-tuple `(grid, env, d1d, time_feat, dir_lbl, int_lbl)` and loads time features from `data/processed/time/`.
- **`evaluate()`** takes a `uses_time` flag — old models receive `(grid, env, d1d)`, new models receive `(grid, env, d1d, time_feat)`.
- **Checkpoint loading** loads 12 checkpoints (6 WP-trained + 6 SP fine-tuned) with correct hyperparameters for each architecture.
- **All visualizations** scale to 6 models: bar charts (rotated x-labels), confusion matrices (2x3 grid), radar chart, waterfall chart, per-class F1, and summary table.

---

## How to Run

### Prerequisites
```bash
conda activate climate
pip install shap  # needed for ablation_shap.ipynb only
```

### Training order
1. Run `experiments/add_time_features.py` (already done — outputs exist in `data/processed/time/`)
2. Train new models (can be run in parallel on separate GPUs):
   - `experiments/unet_film.ipynb`
   - `experiments/fno_v2.ipynb`
   - `experiments/ufno.ipynb`
3. Run `experiments/ablation_shap.ipynb` (uses existing `unet_best_wp.pt`)
4. Run `experiments/comparison.ipynb` (needs all 6 model checkpoints)

### Expected checkpoints
After training, the following files should exist in `experiments/`:
```
unet_film_best_wp.pt    unet_film_best_ft.pt
fno_v2_best_wp.pt       fno_v2_best_ft.pt
ufno_best_wp.pt         ufno_best_ft.pt
```

---

## File Overview

| File | Description | Status |
|------|-------------|--------|
| `experiments/add_time_features.py` | Temporal feature extraction script | Done |
| `experiments/unet_film.ipynb` | U-Net + FiLM training notebook | Ready to train |
| `experiments/fno_v2.ipynb` | FNO v2 training notebook | Ready to train |
| `experiments/ufno.ipynb` | U-FNO training notebook | Ready to train |
| `experiments/ablation_shap.ipynb` | Ablation + SHAP analysis | Ready to run |
| `experiments/comparison.ipynb` | 6-model comparison | Needs all checkpoints |
| `experiments/unet.ipynb` | Baseline U-Net (unchanged) | Already trained |
| `experiments/fno.ipynb` | Baseline FNO (unchanged) | Already trained |
| `experiments/resnet.ipynb` | Baseline ResNet-152 (unchanged) | Already trained |
| `experiments/pre-processing.ipynb` | Data preprocessing (unchanged) | Already run |
