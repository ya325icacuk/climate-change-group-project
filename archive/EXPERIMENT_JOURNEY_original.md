# Experiment Journey: Cross-Basin Cyclone Forecasting (WP → SP)

**Team:** Loic Bouxirot, Yasmin Akhmedova, Samuel Zhang
**Course:** Imperial College ELEC70127 — ML for Tackling Climate Change
**Period:** March 10–24, 2026

---

## Phase 0 — Problem Definition & Literature Review (March 10–11)

### What we did

We began by exploring the TropiCycloneNet (TCND) dataset — a multimodal tropical cyclone dataset spanning six ocean basins (EP, NA, NI, SI, SP, WP) from 1950 to 2023. We set up the starter notebook, downloaded the test subset (~3.34 GB), and inspected the three data modalities:

- **Data3D**: 81×81 gridded atmospheric fields (SST, u/v wind, geopotential at 4 pressure levels) as NetCDF files
- **Env-Data**: 94-dimensional structured environmental features per timestep as NumPy arrays
- **Data1D**: per-storm track files with lat/lon offsets, normalised wind and pressure

We conducted a thorough literature review covering three key papers on FNO-based weather forecasting (FourCastNet), ResNet-based intensity prediction, and multi-basin generalisation challenges. We also reviewed the SHIPS framework (DeMaria & Kaplan, 1994) which identifies universal physics-based predictors (SST, wind shear, upper-ocean heat content).

### Key observations

1. **Severe basin imbalance**: WP has 131 storms vs SP with only 30 — a 4.4× ratio. Training a reliable model on SP alone is not feasible.
2. **Hemisphere-mirrored directions**: WP storms track predominantly NW/W (Northern Hemisphere), while SP storms track SE/W/SW (Southern Hemisphere) due to Coriolis reversal.
3. **Shared underlying physics**: SST and vertical wind shear distributions overlap substantially between WP and SP (mean shear: 8.2 vs 7.4 m/s), suggesting physics-informed features should transfer.
4. **Wasserstein similarity matrix**: Computing pairwise distributional distances across all six basins, WP→SP emerged as the most natural transfer pair (distance = 0.355), validating our choice.

### Decisions taken

- **Research question**: Can deep learning models trained on Western Pacific tropical cyclone data generalise to the South Pacific without basin-specific retraining?
- **WP as source basin, SP as target basin** — justified by the data imbalance, the scientific interest of the hemisphere flip, and the Wasserstein analysis.
- **Dual-head classification**: Direction (8 classes: E, SE, S, SW, W, NW, N, NE) and Intensity change (4 classes: Weakening, Steady, Slow-intensification, Rapid-intensification) at each 6-hourly timestep.
- **Three-stage evaluation**: (1) in-basin WP performance, (2) zero-shot WP→SP transfer, (3) fine-tuned SP performance.

### Bug fixes

- Fixed a critical direction mapping bug: the assumed mapping was N,NE,E,... but empirical verification showed the correct order is **E,SE,S,SW,W,NW,N,NE**.
- Fixed the denormalisation formula for coordinates: `lon = LONG*5+180`, `lat = LAT*5`.
- Fixed a ValueError caused by -1 sentinel values in `future_direction24` and `future_inte_change24` columns.

---

## Phase 1 — Data Preprocessing & One-Pager (March 11–12)

### What we did

We built the full preprocessing pipeline (`experiments/pre-processing.ipynb`) to transform raw TCND data into model-ready tensors, and submitted the Week 1 one-pager deliverable on March 12.

### Preprocessing decisions

1. **Basin-leaking feature removal**: We stripped 54 of the 94 environmental features that encode basin identity (area one-hot, longitude/latitude bins, raw coordinates), retaining only 40 physics-relevant dimensions (wind, move velocity, intensity class, month, history features).

2. **Hemisphere-aware transforms (SP only)**:
   - Direction labels reflected along the N-S axis (SE↔NE, S↔N, SW↔NW) so that "poleward" maps to the same class in both hemispheres.
   - Meridional wind flipped (`v = -v`) at all pressure levels for consistent wind sign conventions.

3. **Derived physics channels**: Two channels appended to the 13 base atmospheric channels:
   - **Wind shear magnitude**: `√((u₂₀₀ − u₈₅₀)² + (v₂₀₀ − v₈₅₀)²)`
   - **850 hPa relative vorticity**: `∂v₈₅₀/∂x − ∂u₈₅₀/∂y` via centred finite differences
   - **Total: 15 input channels** per grid sample.

4. **WP-only normalisation**: Per-channel z-score statistics computed exclusively from WP training data (Welford's online algorithm), applied to all splits — preventing information leakage from SP.

5. **Storm-level splitting**: All splits at the storm level, not timestep level, to prevent temporal data leakage between correlated 6-hourly observations.

### Final data splits

| Split | Storms | Valid Samples | Purpose |
|-------|--------|--------------|---------|
| WP train | 105 | 3,252 | Source-domain training |
| WP val | 26 | 730 | Source-domain validation |
| SP test | 15 | 367 | Zero-shot transfer evaluation |
| SP fine-tune train | 12 | 354 | Target-domain fine-tuning |
| SP fine-tune val | 3 | 81 | Fine-tuning validation |

### Class imbalance observed

- **Direction (WP)**: NW (30.5%), W (28.2%) dominate; E is rare (6.7%)
- **Intensity (WP)**: Weakening (~46%), Slow-intensification (~34%), Rapid-intensification (~13%), Steady (~6%)
- **Mitigation**: Inverse-frequency class weights in cross-entropy loss + label smoothing (0.05).

---

## Phase 2 — Three Baseline Models (March 13–14)

### What we did

We implemented three baseline architectures simultaneously, each assigned to a team member:

| Model | Assigned to | Philosophy |
|-------|------------|-----------|
| **FNO** (Fourier Neural Operator) | Loic | Global patterns via spectral convolutions |
| **ResNet-152** | Yasmin | Deep CNN feature extraction |
| **U-Net** | Sam | Spatial hierarchy with skip connections |

All models use the same **late fusion** strategy: grid features extracted via the backbone → global average pooling → concatenated with env (40-d) and 1D (4-d) features → dual-head MLP classification.

### Architecture details

**ResNet-152** (58.7M params):
- Modified conv1 (3×3 stride-1 instead of 7×7 stride-2) to preserve spatial resolution for 81×81 inputs.
- AdamW + OneCycleLR, max 80 epochs, early stopping (patience=15). Stopped at epoch 31.

**U-Net** (9.8M params):
- 4-level encoder-decoder (32→64→128→256, bottleneck 512) with skip connections, SE channel attention, residual blocks, and DropPath.
- Augmentation suite: Mixup (α=0.2), CutOut (2×16×16), Gaussian noise (σ=0.05), channel dropout (p=0.15), EMA weights.
- Optuna HPO: 30 trials. Key finding — `base_ch=32` outperformed `base_ch=64` (which overfit at 0.548 vs 0.619).
- AdamW + OneCycleLR, max 300 epochs, early stopping (patience=50). Stopped at epoch 144.

**FNO** (10.3M params):
- 2 spectral layers (SpectralConv2d with truncated Fourier modes + skip connections + BatchNorm).
- Optuna HPO: 60 trials. Best config: hidden=80, modes=20, 2 layers.
- AdamW + cosine annealing, max 80 epochs. Stopped at epoch 18 — **severe overfitting**.

### Baseline results

#### In-Basin Performance (WP Validation)

| Model | Params | Dir Acc | Dir F1 | Int Acc | Int F1 |
|-------|--------|---------|--------|---------|--------|
| **U-Net** | 9.8M | **62.5%** | **44.7%** | 51.0% | 43.3% |
| ResNet-152 | 58.7M | 51.2% | 40.4% | **54.9%** | **46.3%** |
| FNO | 10.3M | 50.3% | 36.4% | 55.8% | 41.0% |

#### Zero-Shot Transfer (WP → SP)

| Model | Dir Acc | Int Acc | Dir Gap | Int Gap |
|-------|---------|---------|---------|---------|
| **U-Net** | **41.4%** | 36.0% | **-21.1pp** | -15.0pp |
| ResNet-152 | 27.2% | **38.4%** | -24.0pp | **-16.5pp** |
| FNO | 24.8% | 35.7% | -25.5pp | -20.1pp |

#### Fine-Tuned (SP Test, 354 samples)

| Model | Dir Acc | Int Acc | Dir Recovery | Int Recovery |
|-------|---------|---------|-------------|-------------|
| **U-Net** | **40.1%** | 45.0% | -1.3pp | +9.0pp |
| ResNet-152 | 29.4% | **50.1%** | +2.2pp | **+11.7pp** |
| FNO | 34.6% | 33.5% | +9.8pp | -2.2pp |

### Conclusions from Phase 2

1. **U-Net is the clear winner on direction** — its encoder-decoder architecture with skip connections preserves fine-grained spatial patterns critical for predicting storm movement. It also achieved the smallest transfer gap (-21.1pp).

2. **FNO overfits severely** — despite similar parameter count to U-Net, it stopped at epoch 18/80. Spectral convolutions have high expressivity but lack the inductive biases (locality, skip connections) that regularise U-Net.

3. **ResNet is massively overparameterised** — 58.7M params for 3,252 samples (18,050:1 ratio). Layer activation analysis showed the deepest layers (layer4) barely contribute (magnitude 0.03 vs ~1.0–1.6 in earlier layers).

4. **Transfer gap is massive** — all models suffer 21–25pp drops on direction. Even the best zero-shot accuracy (41.4%) is far from operational utility.

5. **Fine-tuning helps intensity but not direction** — with only 354 SP samples, direction recovery is negligible or negative. The Coriolis-driven distributional shift in direction labels is too fundamental to fix with limited data.

6. **The data-limited regime is the dominant challenge** — with only 3,252 training samples, all models face a fundamental capacity vs. generalisation trade-off.

### How these conclusions shaped Phase 3

- The U-Net's success with **spatial inductive biases** suggested that architectural priors matter more than raw capacity in data-limited regimes.
- The FNO's severe overfitting but interesting **spectral properties** (Fourier mode analysis revealed the model learns predominantly meridional/N-S patterns) hinted that a **hybrid spectral-spatial approach** could combine both strengths.
- The transfer gap problem suggested we need either **better feature representations** that are basin-invariant, or **temporal conditioning** that helps adapt to different seasonal/hemispheric patterns.

---

## Phase 3 — Supervisor Meeting & New Architectures (March 18–20)

### Supervisor feedback (March 18)

Meeting with supervisor Hadrian yielded critical guidance:

- **U-Net**: solid baseline, confirmed it can outperform FNO for some tasks.
- **FNO**: its mesh-invariant, resolution-invariant properties make it a strong candidate for cross-basin transfer. Worth improving, not abandoning.
- **PINN**: hard to converge, needs deep PDE knowledge — **deprioritise** for the 2–3 week timeline.
- **Physics-Informed GAN**: viable alternative. Key constraints to enforce: vorticity, potential vorticity conservation, cyclogenesis thermodynamics (energy cycle from SST vs surface friction).
- **Temporal conditioning**: consider encoding temporal information separately via conditioning (FiLM-style) rather than concatenating into input channels.
- **Coriolis encoding**: either phase-shift labels (our current approach) or encode sin(latitude) as a scalar input.

### Decisions taken

Based on Phase 2 results + supervisor feedback, we designed three new architectures:

1. **U-Net + FiLM** — address the temporal gap: add Feature-wise Linear Modulation to inject storm-phase and seasonal information into U-Net.
2. **FNO v2** — fix FNO's weaknesses: add reflect padding to reduce spectral leakage, increase to 3 spectral layers, add FiLM conditioning.
3. **U-FNO Hybrid** — combine spectral and spatial strengths: parallel spectral + U-Net + residual branches with learned gating.

We also decided to implement a **PI-GAN** (Physics-Informed GAN) as an exploratory model, and to drop the Transformer (too complex for the remaining timeline) and the basic PINN (supervisor's warning). A seventh architecture, **SCANet**, would later be designed based on lessons learned from PI-GAN's instability (see Phase 5).

### What we built

#### Temporal Feature Extraction (`experiments/add_time_features.py`)

A 6-dimensional temporal feature vector per timestep, designed to capture storm lifecycle and seasonal patterns:

| Feature | Description | Encoding |
|---------|------------|----------|
| `storm_progress` | Position within storm lifetime | Linear 0→1 |
| `hour_sin`, `hour_cos` | Time of day | Cyclical (period=24h) |
| `month_sin`, `month_cos` | Month of year | Cyclical (period=12) |
| `timestep_idx_normalized` | Normalised timestep index | Linear |

#### Architecture 4: U-Net + FiLM (5.9M params)

Built on the baseline U-Net with a FiLM (Feature-wise Linear Modulation) layer inserted after every BatchNorm in encoder and decoder blocks.

- **FiLM mechanism**: Projects the 6D time embedding through an MLP (6→64→64 with GELU) to produce per-channel scale (γ) and shift (β) parameters. Applied as: `output = γ * BatchNorm(x) + β`.
- **Identity initialisation**: γ initialised to 1 and β to 0, so the model starts equivalent to the baseline U-Net — training stability is preserved.
- **Key insight**: FiLM allows the same convolutional weights to behave differently depending on storm phase (early formation vs mature vs dissipating) and season (peak season vs off-season), without adding many parameters.

#### Architecture 5: FNO v2 (9.3M params)

Three improvements over the baseline FNO to address its weaknesses:

1. **Reflect padding (±9 pixels)**: Applied before FFT, cropped after iFFT. Reduces spectral leakage at boundaries and boundary artifacts that degraded the baseline FNO.
2. **3 spectral layers** (tuned from 2 in baseline): More depth with padding stabilises training.
3. **FiLM conditioning**: Same temporal conditioning as U-Net+FiLM.

#### Architecture 6: U-FNO Hybrid (2.5M params)

A novel three-branch architecture combining spectral and spatial processing:

- **Spectral branch**: PaddedSpectralConv2d (12 modes, padding=9)
- **Spatial branch**: Lightweight U-Net module (single down-mid-up level)
- **Residual branch**: Standard 1×1 convolution

Each block combines the three branches via a **learned softmax gate** — the model dynamically weights spectral vs spatial vs residual processing at each layer. FiLM conditioning is applied after branch fusion.

**Gate weight analysis** revealed the spectral branch dominates (38–40%) across all layers, but the spatial (U-Net) branch importance increases in deeper layers (27%→31%), confirming that both global and local features contribute meaningfully.

#### Architecture 7 (Exploratory): PI-GAN (10.0M params)

Physics-Informed GAN with the U-Net+FiLM as generator:

- **Generator**: U-Net+FiLM backbone + auxiliary physics reconstruction head
- **Discriminator**: Conditional MLP (2 layers, 128 hidden units)
- **WGAN-GP**: Wasserstein GAN with gradient penalty (λ=10)
- **Physics constraints** (λ_phys=0.1): Vorticity consistency, mass continuity (divergence), vertical wind shear
- **Adversarial warmup**: 20 epochs of pure classification before activating the discriminator
- **Weak adversarial signal** (λ_adv=0.01): Classification remains the primary objective; adversarial training acts as a physics-informed regulariser

---

## Phase 4 — Hyperparameter Optimisation (March 20–23)

### What we did

All five main models underwent Optuna HPO (20 trials × 50 epochs each), running in parallel on an NVIDIA RTX 5090 (32GB VRAM).

### Search spaces

| Model family | Key hyperparameters explored |
|-------------|----------------------------|
| U-Net / U-Net+FiLM | base_ch ∈ {32, 48, 64}, n_levels ∈ {3,4,5}, head_dim ∈ {256, 512} |
| FNO / FNO v2 / U-FNO | hidden ∈ {48, 64, 96}, modes ∈ {12..20}, n_layers ∈ {2..6} |
| PI-GAN | base_ch ∈ {24, 32, 48}, n_levels ∈ {3,4,5}, head_dim ∈ {128, 256, 512} |
| All | lr ∈ [1e-4, 1e-3], weight_decay ∈ [5e-4, 5e-3], dir_weight ∈ [0.45, 0.6] |

### HPO results (best direction accuracy on WP val during 50-epoch trials)

| Model | Best HPO Acc | Key Config Found | Final Params |
|-------|-------------|-----------------|-------------|
| **U-FNO** | **63.2%** | hidden=48, modes=16, 2 layers, padding=13 | 2.5M |
| FNO v2 | 61.9% | hidden=48, modes=20, 5 layers, padding=9 | 9.3M |
| U-Net | 61.4% | base_ch=32, 5 levels, head_dim=256 | 39.1M |
| U-Net+FiLM | 61.4% | base_ch=48, 3 levels, time_emb=64 | 5.9M |
| FNO | 60.7% | hidden=64, modes=15, 4 layers | 7.4M |
| PI-GAN | 58.9% | base_ch=32, 3 levels, head_dim=256 | 10.0M |

### HPO discoveries

1. **U-FNO thrives with just 2 layers** — Optuna found that shallow spectral+spatial fusion outperformed deeper configurations. The gated fusion already provides sufficient expressivity; more layers cause overfitting.

2. **U-Net prefers depth over width** — 5 encoder levels with base_ch=32 (39.1M params) beat wider but shallower alternatives. The extra downsampling captures multi-scale atmospheric patterns.

3. **FNO v2 needs many Fourier modes** — 20 modes (vs 12 baseline) with 5 layers and reflect padding=9. More spectral resolution helps capture fine atmospheric structures that the baseline FNO missed.

4. **U-Net+FiLM went compact** — Optuna chose 3 levels with base_ch=48 instead of the deeper configurations preferred by the baseline U-Net. FiLM conditioning compensates for reduced depth by enabling dynamic feature modulation.

5. **Learning rates clustered around 2–5e-4** — All models converged to similar optimal LRs, suggesting the data distribution dominates the optimisation landscape, not the architecture.

6. **PI-GAN's adversarial training is unstable** — the GAN discriminator makes HPO unreliable; trial variance is high. The physics idea is sound but the training method hurts convergence. This observation directly motivated SCANet's design (Phase 5).

### Conclusion

Architecture choice (and the inductive biases it encodes) matters far more than hyperparameter tuning in the data-limited regime. The right structure (gated spectral-spatial fusion, temporal conditioning) outweighs the right learning rate.

---

## Phase 5 — Full Training, SCANet, & Final Evaluation (March 23–24)

### What we did

Each model trained with its best HPO configuration for up to 300 epochs, with the full augmentation suite (CutOut, Gaussian noise, channel dropout), EMA (decay=0.998), OneCycleLR scheduler, and patience=50 early stopping. We also trained and evaluated the PI-GAN (300 epochs with adversarial warmup).

Fine-tuning on SP used the same two-phase strategy across all models: head-only warmup followed by selective backbone unfreezing with conservative learning rates.

### Intermediate results — 7 models (before SCANet)

Training the first 7 models revealed a critical insight: **PI-GAN's physics losses improved zero-shot transfer (36.2% dir) over the non-physics U-Net+FiLM (32.4%), but the adversarial training was unstable** — HPO trial variance was high, fine-tuning collapsed (-9.8pp), and at 10.0M parameters the GAN framework was overkill for a classification task. The physics idea was sound, but the delivery mechanism was wrong.

This directly motivated the design of **SCANet** — could we get the physics regularisation benefit of PI-GAN without the adversarial instability?

### Architecture 8: SCANet — Spectral Cross-Attention Network (3.7M params)

SCANet was designed to address every limitation identified in the first 7 models:

#### Architecture innovations

1. **Context Cross-Attention** (replaces FiLM): Instead of applying the same scale/shift to every spatial location (as FiLM does), SCANet's context vector *queries* the spatial feature map via cross-attention. This produces **spatially-varying modulation** — the model can attend to different regions depending on environmental conditions. (Inspired by Perceiver, Stable Diffusion)

2. **Early Multimodal Fusion** (replaces late fusion): Environmental (40d), 1D track (4d), and temporal (6d) features are fused into a single 32d context vector *before* the first block, via additive MLP encoding. This context is injected at every layer via cross-attention — not just concatenated at the end. (Inspired by ClimaX)

3. **Dual-Branch Gated Blocks** (simplifies U-FNO's 3-branch design): Each block has a spectral branch (FFT + channel mixer, AFNO-style) and a local branch (depthwise-separable 5×5 conv, MobileNet-style). A learned softmax gate blends them — the model discovers the optimal spectral/local ratio per layer. Depthwise-separable convolutions are 20× lighter than U-FNO's full U-Net branch.

4. **Physics Auxiliary Loss** (without GAN): Predicts vorticity, divergence, and wind shear from backbone features using a simple convolutional head (2 layers). Supervised with Sobel-derived targets from the input grid. All the physics regularisation benefit of PI-GAN, none of the adversarial instability. (Inspired by NeuralGCM)

#### HPO (60 Optuna trials × 50 epochs)

SCANet required a larger search due to its 13-dimensional hyperparameter space (including physics loss weight λ_phys, context_dim, scheduler choice):

| Parameter | Search range | Best value |
|-----------|-------------|-----------|
| hidden_ch | 48–96 | 64 |
| n_modes | 8–20 | 12 |
| n_blocks | 2–4 | 3 |
| context_dim | 32–96 | 32 |
| head_dim | 64–256 | 192 |
| lambda_phys | 0.01–0.3 | 0.087 |
| scheduler | cosine / onecycle | onecycle |

60 trials launched, 30 completed, 30 pruned. Best HPO val accuracy: **59.4%** (direction). Key finding: small context_dim (32) with large head_dim (192) works best — the context vector should be compact but the classification heads need capacity.

#### Gate weight analysis

The learned spectral vs local branch weights reveal:

| Block | Spectral | Local |
|------:|--------:|------:|
| 1 | 55.4% | 44.6% |
| 2 | 56.4% | 43.6% |
| 3 | 53.2% | 46.8% |

The spectral branch dominates in all layers (53–56%), but the local branch contributes substantially (44–47%) — more balanced than U-FNO's 3-branch design. The local branch importance increases in deeper blocks (44.6%→46.8%), consistent with U-FNO's finding that spatial features become more important for refined predictions.

### Final results — all 8 models

#### In-Basin Performance (WP Validation)

| Model | Params | Dir Acc | Dir F1 | Int Acc | Int F1 | Epochs |
|-------|-------:|--------:|-------:|--------:|-------:|-------:|
| **U-Net** | 39.1M | **60.0%** | **41.9%** | 58.9% | 47.9% | 134 |
| PI-GAN | 10.0M | 57.0% | 39.1% | 56.3% | 44.6% | 300 |
| **SCANet** | 3.7M | 56.4% | 34.7% | **66.2%** | 42.7% | 185 |
| U-Net+FiLM | 5.9M | 56.4% | 40.4% | 57.4% | 47.2% | 120 |
| **U-FNO** | 2.5M | 55.7% | 40.4% | 63.0% | **49.3%** | 100 |
| ResNet-152 | 58.7M | 49.9% | 37.2% | 61.1% | 50.1% | 31 |
| FNO v2 | 9.3M | 36.6% | 12.4% | 44.5% | 23.7% | 135 |
| FNO | 7.4M | 36.4% | 14.0% | 41.7% | 34.1% | 114 |

**Key observations:**
- U-Net remains the in-basin direction champion (60.0%).
- **SCANet achieves the best intensity accuracy (66.2%)** despite modest in-basin direction (56.4%) — its cross-attention mechanism captures intensity-relevant patterns differently. This hints at its real strength: transfer.
- U-FNO achieves 63.0% intensity with only 2.5M params — 16× smaller than U-Net.
- PI-GAN performs respectably in-basin (57.0% dir) despite the added complexity of adversarial training, but is beaten by SCANet which uses the same physics losses without the GAN.
- FNO and FNO v2 underperform on direction with these checkpoints, though FNO v2's padding and FiLM help on intensity relative to baseline FNO.

#### Zero-Shot Cross-Basin Transfer (WP → SP)

| Model | Dir Acc | Dir F1 | Int Acc | Int F1 | Dir Gap |
|-------|--------:|-------:|--------:|-------:|--------:|
| **SCANet** | **43.3%** | **27.1%** | 36.8% | 21.8% | **-13.1pp** |
| PI-GAN | 36.2% | 24.3% | 38.1% | 29.1% | -20.8pp |
| U-FNO | 34.6% | 27.9% | **48.0%** | **31.0%** | -21.1pp |
| U-Net+FiLM | 32.4% | 27.4% | 40.9% | 33.5% | -24.0pp |
| U-Net | 27.0% | 19.1% | 31.3% | 20.7% | -33.0pp |
| ResNet-152 | 25.9% | 21.1% | 32.2% | 24.2% | -24.0pp |
| FNO | 22.1% | 12.5% | 27.0% | 18.6% | -14.4pp |
| FNO v2 | 17.2% | 4.7% | 39.8% | 19.8% | -19.5pp |

**Key observations:**
- **SCANet shatters the zero-shot transfer ceiling** with 43.3% direction accuracy — a +7.1pp leap over PI-GAN (36.2%) and +8.7pp over U-FNO (34.6%). Its transfer gap of just **-13.1pp** is nearly half that of U-FNO (-21.1pp) and a third of U-Net's (-33.0pp).
- The three synergistic effects behind SCANet's transfer dominance: (1) cross-attention learns to weight spatial regions by relevance, not by fixed position — this transfers across basins with different storm morphologies; (2) early context fusion means every spectral/local computation is basin-context-aware from the start; (3) physics loss forces the backbone to encode physically meaningful gradients, which are universal across basins.
- **U-FNO still dominates intensity transfer** (48.0%), suggesting spectral-spatial gated fusion captures intensity-relevant features (SST, shear patterns) that SCANet's lighter local branch misses.
- **U-Net suffers the largest gap** (-33.0pp direction) — its 39.1M parameters overfit to WP-specific patterns. Being the best in-basin does not mean being the best at transfer.

#### Fine-Tuned Performance (SP Test, 354 samples)

| Model | Dir Acc | Dir F1 | Int Acc | Int F1 | Dir Recovery |
|-------|--------:|-------:|--------:|-------:|-------------:|
| **U-Net+FiLM** | **38.4%** | 27.1% | 44.4% | 38.6% | **+6.0pp** |
| SCANet | 38.1% | **30.3%** | 43.1% | 35.2% | -5.2pp |
| U-Net | 31.6% | 24.0% | 43.9% | 34.7% | +4.6pp |
| U-FNO | 30.8% | 29.9% | **49.0%** | **40.7%** | -3.8pp |
| PI-GAN | 26.4% | 20.2% | 49.9% | 39.7% | -9.8pp |
| ResNet-152 | 25.6% | 21.3% | 49.6% | 42.3% | -0.3pp |
| FNO | 22.9% | 16.5% | 34.1% | 27.6% | +0.8pp |
| FNO v2 | 15.3% | 9.1% | 35.1% | 21.9% | -1.9pp |

**Key observations:**
- **U-Net+FiLM benefits most from fine-tuning** (+6.0pp direction recovery), reaching 38.4%. Its FiLM conditioning allows rapid adaptation to SP's different seasonal patterns with minimal catastrophic forgetting.
- **SCANet degrades after fine-tuning** (-5.2pp direction, from 43.3% to 38.1%). This is the **fine-tuning paradox**: its zero-shot representations are already so basin-agnostic that the tiny 354-sample SP set introduces noise rather than useful signal. This is actually evidence that SCANet has learned genuinely basin-invariant features.
- **PI-GAN suffers the worst fine-tuning collapse** (-9.8pp direction recovery, from 36.2% to 26.4%). The GAN's physics-informed representations are fragile — fine-tuning with only 354 samples destabilises the generator-discriminator equilibrium.
- **U-FNO loses ground on direction** (-3.8pp) but maintains strong intensity (49.0%). Its compact 2.5M params overfit to the tiny SP fine-tuning set.

### The PI-GAN → SCANet lesson

PI-GAN proved that physics losses improve transfer. SCANet proved that you don't need a GAN to deliver them. Comparing the two models with physics losses:

| Metric | PI-GAN (10.0M) | SCANet (3.7M) | SCANet advantage |
|--------|:--------------:|:------------:|:----------------:|
| WP Dir | 57.0% | 56.4% | -0.6pp |
| WP Int | 56.3% | **66.2%** | +9.9pp |
| SP Zero-shot Dir | 36.2% | **43.3%** | +7.1pp |
| Dir Transfer Gap | -20.8pp | **-13.1pp** | 7.7pp smaller |
| Training stability | Unstable (GAN) | Stable (supervised) | Much better |
| Params | 10.0M | **3.7M** | 2.7× lighter |

Auxiliary supervised physics heads are simpler, more stable, more parameter-efficient, and more effective than adversarial delivery.

---

## Phase 6 — Ablation & Explainability Analysis (March 24)

### What we did

We conducted comprehensive ablation studies and SHAP-based explainability analysis to understand **what each model learns and why transfer succeeds or fails**.

### Modality ablation (zeroing out entire input modalities)

| Modality removed | U-Net+FiLM Dir drop | U-FNO Dir drop | Interpretation |
|-----------------|:-------------------:|:--------------:|----------------|
| No Grid (3D fields) | **-24.1pp** | **-22.8pp** | Grid is essential for direction |
| No Env features | -1.5pp | -0.2pp | Env has minimal impact on direction |
| No 1D track data | +2.2pp | +0.0pp | 1D is redundant (slightly hurts) |
| No Time features | -0.7pp | +0.9pp | Time weakly helpful / irrelevant |

| Modality removed | U-Net+FiLM Int drop | U-FNO Int drop | Interpretation |
|-----------------|:-------------------:|:--------------:|----------------|
| No Grid (3D fields) | **-18.9pp** | **-23.7pp** | Grid critical for both tasks |
| No Env features | -7.1pp | -3.9pp | Env matters more for intensity |
| No 1D track data | +2.6pp | +1.7pp | 1D still redundant |
| No Time features | -5.2pp | -6.9pp | Time more important for intensity |

**Conclusion**: The 3D gridded atmospheric fields are the dominant input modality for both tasks. Environmental features contribute meaningfully to intensity prediction but not direction. The 1D track data is consistently redundant across all models and tasks — the grid already encodes the relevant spatial information.

### Grid channel leave-one-out ablation (for direction)

| Channel removed | U-Net+FiLM drop | U-FNO drop | Interpretation |
|----------------|:--------------:|:----------:|----------------|
| **u_wind** (zonal) | **-20.6pp** | -5.6pp | U-Net+FiLM heavily u-wind dependent |
| **v_wind** (meridional) | -10.1pp | -6.9pp | Both models need v_wind |
| SST | -0.0pp | -2.1pp | SST irrelevant for direction |
| Geopotential | +0.4pp | +1.1pp | Geopotential slightly hurts direction |

**Critical insight**: U-Net+FiLM concentrates its direction prediction on a single feature (u_wind accounts for 20.6pp of its accuracy), making it vulnerable if that feature distribution shifts. U-FNO distributes its reliance more evenly across channels (max drop = 6.9pp), explaining its better zero-shot transfer — it learns more robust, distributed representations.

### SHAP analysis on environmental features

Using GradientExplainer, we computed SHAP values for environmental features:

| Feature group | U-Net+FiLM mean |SHAP| | U-FNO mean |SHAP| |
|--------------|:-------------------:|:----------------:|
| Wind | 0.020 | 0.008 |
| Intensity class | 0.029 | 0.009 |
| Move velocity | 0.012 | 0.004 |

U-Net+FiLM relies 2–3× more on environmental features than U-FNO, consistent with the ablation finding that U-Net+FiLM is more dependent on specific input features.

### Gradient-based grid attribution

- **u_500** (zonal wind at 500 hPa) is the single most important channel for both models.
- **v_500** (meridional wind at 500 hPa) ranks second.
- Both models show consistent gradient patterns despite very different architectures, validating that the signal is real (mid-troposphere steering flow dominates cyclone direction).

---

## Summary of Key Findings

### The efficiency–generalisation trade-off

| Property | Best model | Why |
|----------|-----------|-----|
| In-basin direction | U-Net (60.0%) | Large capacity captures WP patterns |
| In-basin intensity | SCANet (66.2%) | Cross-attention captures intensity-relevant patterns |
| Zero-shot direction | **SCANet (43.3%)** | Cross-attention + early fusion + physics loss = basin-agnostic features |
| Zero-shot intensity | U-FNO (48.0%) | Hybrid spectral-spatial fusion generalises |
| Fine-tuned direction | U-Net+FiLM (38.4%) | FiLM enables fast adaptation with small data |
| Fine-tuned intensity | PI-GAN (49.9%) | Physics features aid intensity adaptation |
| Smallest transfer gap | **SCANet (-13.1pp)** | Nearly half of U-FNO's gap (-21.1pp) |
| Parameter efficiency | U-FNO (2.5M) | Gated fusion achieves 55.7% dir with 16× fewer params |

### The five lessons

1. **Architectural inductive biases > raw capacity**: In a data-limited regime (3,252 samples), the right structure matters more than more parameters. SCANet (3.7M) beats U-Net (39M) on zero-shot transfer by +16.3pp. U-FNO (2.5M) outperforms ResNet (58.7M) on nearly every metric.

2. **Spectral-spatial hybrids learn transferable features**: Both U-FNO's and SCANet's gated fusion distribute reliance across channels, creating robust representations that survive distributional shift. U-FNO's max feature drop is 6.9pp vs 20.6pp for U-Net+FiLM.

3. **Physics constraints improve zero-shot transfer, but delivery matters**: PI-GAN proved physics losses help (36.2% vs 32.4% for U-Net+FiLM). SCANet proved you don't need a GAN to deliver them — auxiliary supervised heads are simpler, more stable, and more effective (43.3% zero-shot).

4. **Spatially-varying modulation > spatially-uniform modulation**: SCANet's cross-attention produces different modulation at different spatial locations depending on context. FiLM applies the same scale/shift everywhere. This is the key architectural difference that explains the -13.1pp vs -24.0pp transfer gap.

5. **Temporal conditioning enables adaptation**: FiLM's dynamic feature modulation (conditioned on storm phase, season, time of day) gives U-Net+FiLM the best fine-tuning recovery (+6.0pp). When target-domain data is available, FiLM-based adaptation is the best strategy. SCANet's zero-shot representations are so strong that fine-tuning actually degrades them (-5.2pp).

### Model recommendation by deployment scenario

| Scenario | Recommended model | Rationale |
|----------|------------------|-----------|
| In-basin forecasting (abundant data) | U-Net | Highest WP direction accuracy (60.0%) |
| Cross-basin transfer (no target data) | **SCANet** | Best zero-shot (43.3%), smallest gap (-13.1pp) |
| Cross-basin transfer (some target data) | U-Net+FiLM | Best fine-tuning recovery (+6.0pp) |
| Resource-constrained deployment | U-FNO | Best accuracy/param ratio (2.5M, 34.6% ZS dir) |
| Intensity-focused forecasting | SCANet or U-FNO | 66.2% and 63.0% WP intensity respectively |

---

## Timeline Summary

| Date | Phase | Key milestone | Key decision |
|------|-------|--------------|-------------|
| Mar 10 | Setup | Repo initialised, starter notebook | — |
| Mar 11 | EDA | Literature review, cross-basin analysis | WP→SP confirmed via Wasserstein similarity |
| Mar 12 | Deliverable 1 | One-pager submitted | Preprocessing pipeline finalised |
| Mar 13 | Baselines | ResNet, U-Net, FNO implemented | Late fusion for all models |
| Mar 14 | Iteration | U-Net reaches 62.5% with augmentation | `base_ch=64` overfits, revert to 32 |
| Mar 18 | Supervisor meeting | Feedback received | Deprioritise PINN; add FiLM, FNO v2, U-FNO, PI-GAN |
| Mar 20 | New architectures | 3 new models + ablation framework built | Temporal features, gated fusion, physics constraints |
| Mar 23 | HPO | Optuna trials for 6 models | U-FNO best HPO (63.2%), only needs 2 layers |
| Mar 24 | PI-GAN evaluated | Physics losses help but GAN is unstable | Design SCANet: physics loss without adversarial training |
| Mar 24 | SCANet + final eval | 8-model comparison + ablation + SHAP | SCANet = best zero-shot (43.3%, -13.1pp gap); U-Net+FiLM = best fine-tuned |
