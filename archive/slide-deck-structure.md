# Slide Deck Structure

## Slide 1: Central Narrative + Motivation (2 min)

### Central Narrative

- Research question: can we transfer TC forecasting models trained on the data-rich WP to the data-scarce SP, despite the hemisphere flip?
- Hypothesis: intensity prediction should transfer well (shared thermodynamic physics like SST and wind shear) while direction prediction should not (movement patterns governed by hemisphere-dependent Coriolis force)
- Key finding preview: no single architecture excels at both tasks. Spatial models (U-Net+FiLM) capture local wind patterns best for direction, spectral models (U-FNO) capture global thermodynamic patterns best for intensity
- Structure: EDA builds the transfer hypothesis, method tests it with two model families, results confirm the asymmetry, explainability reveals why

### Section 1: Motivation and Research Question

- Data scarcity problem: SP has only [30] storms / [922] timesteps vs WP's [117] storms / [3,621] timesteps in TCND. Too few to train reliable deep learning models on SP alone, especially for 8-class direction and 4-class intensity classification
- Transfer learning as the solution: Qu et al. (2025) fine-tuned a WNP model on ENP and outperformed operational forecasting systems, showing cross-basin transfer is viable within the same hemisphere
- Why WP-to-SP is the hardest and most interesting transfer pair: WP is in the Northern Hemisphere, SP in the Southern Hemisphere, so the Coriolis force reverses. Storms curve NW/W in WP but SE/W in SP. This is the first cross-hemisphere transfer test on TCND
- The hemisphere flip as a natural experiment to separate transferable from non-transferable knowledge: intensity physics (SST drives energy, wind shear inhibits organisation) operates identically in both hemispheres, but storm movement is governed by hemisphere-specific steering flows and Coriolis deflection
- State research question: can we transfer WP-trained models to SP, and does intensity prediction transfer better than direction prediction as the shared-physics argument would suggest?

---

## Slide 2: EDA and Transfer Hypothesis (2.5 min)

- WP and SP are the most statistically similar cross-hemisphere basin pair, with the lowest average Wasserstein distance across physical variables (SST, wind shear, pressure, wind speed). This makes them the strongest candidate for cross-hemisphere transfer and the fairest test of our hypothesis
- Direction distributions are near-mirror images across hemispheres: WP storms move predominantly NW/W, SP storms predominantly SE/W/SW. A WP-trained direction classifier will systematically predict the wrong quadrant for SP storms unless corrected
- Intensity-related variables (SST, vertical wind shear) have substantially overlapping distributions between WP and SP. The thermodynamic environment that drives intensification looks similar in both basins. DeMaria and Kaplan (1994) identified SST and shear as the two strongest intensity predictors
- Three data modalities, introduced through the lens of what transfers and what doesn't:
  - **Data3D** ([15]-channel spatial grids, [81x81] pixels): u/v wind at 4 pressure levels (200, 500, 850, 925 hPa), SST, geopotential height, plus two derived channels (wind shear magnitude and 850 hPa vorticity computed during preprocessing). Wind U/V components reverse sign across hemispheres, but SST and shear magnitude patterns are basin-agnostic
  - **Env-Data** ([40] dimensions after cleaning): originally [94] features, but we removed [54] that directly leak basin identity (area one-hot encodings, coordinate bins, basin flags). Remaining features: wind components, movement velocity, intensity class, direction history, shear magnitude, steering flow. Still heavily hemisphere-specific due to direction/movement encoding
  - **Data1D** ([4] dims): lat/lon offset from storm centre, normalized wind speed, normalized pressure. Minimal track snapshot
  - **Temporal features** ([6] dims, extracted separately): storm lifecycle progress (0-1), cyclical sin/cos hour-of-day, cyclical sin/cos month-of-year, normalized storm duration. Used to condition models on where the storm is in its lifecycle via FiLM
- Hemisphere-aware preprocessing applied before any modelling: SP direction labels reflected (N/S swapped), meridional wind components flipped in sign, all z-score normalization statistics computed from WP training data only (no data leakage from SP)
- Clear prediction to test: intensity prediction should transfer well because the underlying physics is shared; direction prediction should transfer poorly because movement patterns are fundamentally hemisphere-dependent

---

## Slide 3: Method (2.5 min)

### Two-stage experimental pipeline

- **Stage 1, Zero-shot:** Train each model exclusively on WP data ([1,863] train / [518] val timesteps). Apply hemisphere-aware corrections before evaluating on SP test set ([354] timesteps, [12] storms). No SP data seen during training. This measures how much knowledge transfers for free based on shared physics alone
- **Stage 2, Fine-tuning:** Take the best WP-pretrained models and fine-tune on the SP training split ([354] timesteps, [12] storms) with reduced learning rate. Evaluate on SP validation ([532] timesteps, [18] storms). This measures how much target-basin data is needed to close the transfer gap, and whether WP pretraining gives a head start vs training from scratch on SP

### Five architectures in two model families

**Spatial family** (local pattern learners, hypothesised to be better for direction):

- **U-Net baseline (1a):** 4-level encoder-decoder with 32, 64, 128, 256, 512 channel progression, [9.8M] params. SE (squeeze-and-excitation) channel attention learns which of the [15] input channels matter most. Residual connections and DropPath (stochastic depth) for regularisation. Global average pooling of grid features fused with env + 1D features before two classification heads (direction + intensity)
- **U-Net + FiLM (1b):** Same U-Net backbone + Feature-wise Linear Modulation conditioned on temporal features. 6-dim temporal input embedded to 64-dim, then generates per-level affine scale and shift parameters (gamma, beta) applied at every encoder and decoder level. [10.0M] params. FiLM adds [+5.4pp] intensity accuracy over baseline by encoding storm lifecycle stage. A storm early in its life behaves differently from one at peak intensity

**Spectral family** (global pattern learners, hypothesised to be better for intensity):

- **FNO baseline (2a):** 4 spectral convolution layers that learn in frequency domain via 2D FFT. Multiply by learnable mode weights in Fourier space, then inverse FFT back. [10.3M] params. Motivation: spectral representations are not tied to a specific grid location or hemisphere, so learned patterns might be more basin-agnostic. Problem: severely overfits with only [1,816] training samples (early stops at epoch [18/80], params-per-sample ratio [5,672:1])
- **FNO v2 + FiLM (2b):** Redesigned to fix overfitting. Reduced to 3 spectral layers, only [0.9M] params ([496:1] ratio). Added 9-pixel reflect padding around the grid to avoid FFT wraparound boundary artifacts. Added FiLM temporal conditioning at each spectral layer. Trains to full [150] epochs without overfitting
- **U-FNO hybrid (2c):** 3-branch gated architecture that combines spectral path (FFT-based), U-Net path (local convolution), and residual path (identity skip) at each layer. Branches are weighted by learned softmax gates. Model learns to allocate [40%] spectral, [30%] U-Net, [30%] residual, revealing that it prefers frequency-domain features. [1.0M] params. Best intensity model overall

Also benchmarked: ResNet-152 as a standard CNN baseline for comparison. Included in the model comparison but not a focus of the narrative.

### Key design insight

Spatial convolutions see local wind patterns and gradients (what determines storm movement direction); spectral methods see large-scale thermodynamic structure across the full grid (what determines intensification). FiLM temporal conditioning helps intensity more than direction because lifecycle stage encodes intensification likelihood but not directional information.

---

## Slide 4: Results (3.5 min) + Takeaways (1 min)

### Section 4: Results and Analysis

**WP in-basin performance** (validating that models work before testing transfer):

- U-Net+FiLM: best direction model, [62.9%] accuracy, [0.465] macro F1 on 8-class direction
- U-FNO: best intensity model, [62.5%] accuracy, [0.499] macro F1 on 4-class intensity
- No single model wins both tasks. The architecture-task trade-off is real and consistent across all models including ResNet-152 baseline. Spatial convolutions learn local wind field patterns that determine storm heading; spectral methods learn global SST/shear patterns that determine intensification

**Zero-shot cross-basin transfer** (WP model applied directly to SP, no fine-tuning):

- Direction transfer gap is large: [-23.9pp] for U-Net+FiLM ([62.9%] to [39.0%]), [-30.7pp] for U-FNO ([50.8%] to [20.2%]). Hemisphere flip breaks direction predictions as expected
- Intensity transfer gap is consistently smaller: [-19.2pp] for U-FNO ([62.5%] to [43.3%]), [-20.2pp] for U-Net+FiLM ([56.4%] to [36.2%]). Shared physics partially preserved
- Hypothesis confirmed: intensity transfers better than direction across all architectures
- U-Net+FiLM retains best zero-shot direction ([39.0%]), U-FNO retains best zero-shot intensity ([43.3%]). The architecture-task specialisation holds even under domain shift

**Fine-tuning with SP data** ([354] timesteps, [12] storms):

- Direction barely recovers: U-Net+FiLM stays flat at [39.0%] (no gain), U-FNO gains [+8.2pp] to [28.3%]. Direction patterns are fundamentally different across hemispheres. Small amounts of SP data cannot overcome Coriolis reversal
- Intensity improves meaningfully: U-FNO reaches [49.6%] ([+6.3pp] over zero-shot), U-Net+FiLM reaches [39.2%] ([+3.0pp]). Physics-based intensity representations recalibrate efficiently on new basin data
- Implication: WP pretraining provides a useful initialisation for intensity, but direction requires either much more SP data or fundamentally different approaches (e.g., domain adaptation)

**Explainability** (why the architecture split exists):

- Modality ablation on U-Net baseline: removing the grid causes [31pp] direction accuracy drop ([62.5%] to [31.5%]) but only [12.6pp] intensity drop. Removing env causes only [0.6pp] direction drop but [12.0pp] intensity drop. 1D track features are nearly redundant (~0pp change). Grid drives direction, env drives intensity
- SHAP GradientExplainer on environmental features: wind shear magnitude and SST-related features have highest Shapley values for intensity classification. Historical direction and movement velocity dominate direction classification
- Gradient attribution on grid channels: u_wind and v_wind at 850 hPa have highest mean absolute gradients. The low-level wind field is what the spatial models learn to read for direction
- These findings explain the architecture trade-off: U-Net's local convolutions are ideal for reading wind field spatial structure (direction); U-FNO's spectral layers are ideal for capturing basin-wide thermodynamic conditions (intensity)

**Overfitting management** (important given extreme data scarcity):

- Only [1,816] WP training samples for models with millions of parameters. Params-per-sample ratio is the key metric
- FNO baseline ([10.3M] params, [5,672:1] ratio): catastrophic overfitting, early stops at epoch [18/80]
- U-FNO ([1.0M] params, [551:1] ratio): moderate overfitting, early stops at epoch [41/150]. 10x parameter reduction via gating and smaller layers
- FNO v2 ([0.9M] params, [496:1] ratio): fully resolved, trains all [150] epochs. Reflect padding + FiLM + smaller architecture
- Lesson: in extreme low-data regimes, architecture design (gating, padding, parameter efficiency) matters as much as model expressivity

### Section 5: Takeaways

- Hypothesis confirmed: intensity transfers across hemispheres because SST and wind shear physics are universal; direction does not transfer because Coriolis-driven movement patterns are hemisphere-specific. The gap is consistent across all architectures
- No single architecture excels at both tasks: spatial models (U-Net+FiLM) specialise in direction via local convolutions, spectral models (U-FNO) specialise in intensity via frequency-domain learning. This is a fundamental result, not a tuning artifact
- FiLM temporal conditioning and spectral methods disproportionately help intensity. Storm lifecycle stage predicts intensification likelihood but not heading. Direction remains a local spatial problem
- Practical implication: transfer learning from a data-rich basin is viable for intensity forecasting in data-scarce basins with minimal fine-tuning. Direction forecasting requires either substantially more target-basin data or explicit domain adaptation to handle the hemisphere flip
- Future work: extension to NI/SI basins to test generality, domain adaptation techniques (e.g., adversarial alignment) for direction transfer, ensemble of spatial + spectral models to get best of both worlds
