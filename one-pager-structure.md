# 1-Pager Structure: Cross-Basin Generalization for TC Forecasting

**Deliverable**: 1-page description - problem context, data statistics, visualisation
**Due**: March 12, 11:59 pm (5 points)

---

## 1. Problem Context (~1/3 of page)

**Title**: *Cross-Basin Generalization for Tropical Cyclone Forecasting using TropiCycloneNet*

- **What**: Train TC forecasting models on the Western Pacific (WP) and evaluate zero-shot transfer to the South Pacific (SP) across five architectures. Then fine-tune the best performers to measure accuracy and computational trade-offs.
  - *Ref: Qu et al. (2025) demonstrated WNP-trained TIFNet fine-tuned on ENP outperforms operational models, showing cross-basin transfer is viable.*
- **Why WP to SP**:
  - *Data scarcity* — SP has 5x fewer storms than WP (30 vs 131), so you can't train a reliable model on SP alone
    - *Ref: Huang et al. (2025) show WP-only models (TCN_M^WP) fail on Southern Hemisphere basins, with 61–73% worse track performance than multi-basin models.*
  - *Hemisphere flip* — WP is Northern Hemisphere, SP is Southern Hemisphere. This introduces a structural distributional shift (Coriolis reversal) beyond just sample size, making it the most scientifically interesting transfer pair.
  - *Climate adaptation* — TC activity may shift to under-recorded basins as oceans warm
- **The core challenge**: WP storms track NW/W, SP storms track SE/W/SW due to the Coriolis reversal. A naive model trained on WP will systematically mispredict SP directions. But underlying atmospheric physics (SST, vertical wind shear) is shared across both basins, so physics-informed features should transfer.
  - *Ref: DeMaria & Kaplan (1994) SHIPS identifies physics-based predictors (SST, shear, upper-ocean heat content) that should be universal across basins.*

### Proposed Experiments

We apply **minimal data preprocessing** — specifically a phase/direction shift to account for the Coriolis reversal — and then evaluate **zero-shot transfer** of five model architectures trained on WP data applied directly to SP:

### Supervisor Meeting Feedback (March 2026)

**On architectures:**
- **U-Net**: Can outperform FNO in some tasks (reconstruction, segmentation). A solid baseline — not to be underestimated.
- **FNO**: Mesh-invariant and resolution-invariant. Strong candidate for cross-basin transfer because it maps between functions, not fixed input/output grids.
- **PINN**: Supervisor is "quite concerning" — hard to converge, needs deep domain knowledge of PDEs. Risky for 2–3 week timeline. **Deprioritize.**
- **Physics-Informed GAN**: Viable. Key physics constraints to enforce: vorticity, potential vorticity (must conserve), cyclogenesis thermodynamics (energy cycle from SST vs surface friction). Need "equation of state + something extra."
- **Transformer**: Supervisor's PhD topic. Key tips:
  - Fuse 3D data, 1D data, and env-data carefully via cross-attention or diffusion-style conditioning
  - Consider encoding spatial+temporal into input channels vs conditioning temporal separately
  - Use GeLU activation, Muon optimizer (faster than AdamW for transformers), check flash attention support on GPU

**On methodology:**
- **Lead time**: Key concept — nowcasting (short-term), 3–7 day mid-range, long-range (climatology). Report accuracy vs lead time as a key evaluation dimension.
- **Ensemble comparison**: Operational forecasts use ~50 ensemble models (physics-based, slow). Beating them with 1 ML model = publishable result.
- **Coriolis encoding**: Two approaches — (1) phase shift/relabeling, or (2) encode latitude as sin(φ) and let the model learn. The cyclogenesis equation has a term scaled by sin(latitude). Could condition this as a scalar input.
- **Fine-tuning strategy**: Data-centric first (preprocessing, label adaptation), then model-centric fine-tuning. Extension: hyperparameter tuning if time permits.
- **Feasibility**: All 5 architectures are feasible to implement (not hard to code), but tuning is the hard part. PINN is the riskiest.

**On deliverables:**
- 1-pager format: up to the team, should fit in 1 page including figures
- Final deliverable: GitHub repo (confirm with Goga via email)
- Presentation level of detail: ask Goga via email

| Importance   | Architecture                               | Why include                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Key reference                                  | Assignment |
| --- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- | ---------- |
| 2   | **U-Net**                                  | Strong spatial feature extraction for gridded Data3D fields; widely used baseline for weather prediction                                                                                                                                                                                                                                                                                                                                                                                                                       | —                                              | Sam          |
| 3   | **ResNet**                                 | ResNet50, 180?                                                                                                                                                                                                                                                                                                                                                                                                                       | —                                              | Yasmin          |
| 1   | **FNO** (Fourier Neural Operator)          | Spectral approach is inherently basin-agnostic; proven on weather forecasting (FourCastNet)                                                                                                                                                                                                                                                                                                                                                                                                                                    | Chen et al. (2022)                             | Loïc          |
| -   | **PINN** (Physics-Informed Neural Network) | Encodes conservation laws (momentum, thermodynamics) as hard constraints; should transfer universal physics                                                                                                                                                                                                                                                                                                                                                                                                                    | Beucler et al. (2021); Kashinath et al. (2021) | —          |
| 4   | **Physics-Informed GAN**                   | Combines adversarial generation with physical constraints; tested for TC track prediction (conserve: u wind v wind vorticity, potential vorticity, thermodynamics: cyclongenesis of tropical cyclones, energy cycle find the thermodynamic constraints EoS and something extra)                                                                                                                                                                                                                                                | Ruttgers et al. (2019)                         | —          |
| 5   | **Transformer** (spatiotemporal)           | Non-iterative multi-horizon prediction avoids error accumulation; proven cross-basin on ENP (carefully fuse the features (1. 3D data, 1D data and environmental data, scaler etc. How do we condition scalar quantities differently, condition and fuse to each attention head, diffusion etc to fuse properly. 2. Do some architectual reseacrch on encoding temporal and spacial together into input channel 3. GeLu optimise muon, optional argument for weight decay. Check which does 5090 support flash attention? ...)) | Qu et al. (2025) — TIFNet                      | —          |

**Three-stage pipeline**:
1. **Zero-shot evaluation** — Train each model on WP, apply Coriolis phase shift, evaluate on SP with no fine-tuning. Rank models by transfer gap (WP in-basin accuracy minus SP accuracy).
2. **Fine-tuning** — Select the top-performing zero-shot models, fine-tune on small fractions (10%, 25%, 50%) of SP data. Measure accuracy vs. computational cost trade-off.
3. **Extension** — Investigate whether the best-performing model(s) generalize to other data-scarce basins (NI, SI) without basin-specific retraining.

---

## 2. Data Statistics (~1/3 of page)

- **Dataset**: TropiCycloneNet (TCND) — 6 basins, 1950–2023, 3 modalities: *Ref: Huang et al. (2025), Nature Communications*
  - `Data1D`: cyclone-center attributes (lon, lat, pressure, wind) — *derived from IBTrACS (Knapp et al., 2010), note inter-agency wind averaging differences (2-min vs 1-min) across basins*
  - `Data3D`: 81×81 gridded fields (SST, u/v wind, geopotential height) at 0.25° resolution
  - `Env-Data`: structured environmental features (shear, subtropical high, direction/intensity labels)
- **Temporal resolution**: 6-hourly
- **WP vs SP comparison**:

|                 | WP (train)        | SP (test)         | Ratio    |
| --------------- | ----------------- | ----------------- | -------- |
| Storms          | 131               | 30                | 4.4x     |
| Timesteps       | 4,562             | 922               | 5.0x     |
| Top directions  | NW (30%), W (28%) | SE (25%), W (19%) | Mirrored |
| Mean wind shear | 8.2 m/s           | 7.4 m/s           | Similar  |

*Note: storm counts from Env-Data. The storms_per_basin.png figure uses Data3D counts (e.g. WP=145) — pick one source and be consistent. Shear from notebook n=500 per basin, the sst_shear_by_basin.png figure uses n=80.*

- **What transfers**: Vertical wind shear and SST distributions overlap substantially between WP and SP — these are the universal intensity predictors identified by DeMaria & Kaplan (1994).
- **What doesn't**: Direction labels are hemisphere-mirrored due to Coriolis reversal. Our minimal preprocessing applies a phase shift to remap WP directions to SP-equivalent directions before zero-shot evaluation.
  - *Ref: Huang et al. (2025) explicitly show WP-trained models cannot generalise to SH without hemisphere-aware training.*

---

## 3. Key Visualisations (~1/3 of page)

*Note: choose the best 4 for the final submission.*

| Figure                             | Shows                                      | Why include                                                   |
| ---------------------------------- | ------------------------------------------ | ------------------------------------------------------------- |
| `storms_per_basin.png`             | Bar chart: data imbalance across 6 basins  | Motivates transfer learning — SP is data-scarce               |
| `track_map.png`                    | WP vs SP storm tracks on a map             | Geographic separation + hemisphere flip, visually striking    |
| `basin_similarity_matrix.png`      | Wasserstein distance heatmap across basins | Quantifies WP–SP similarity relative to other basin pairs     |
| `direction_by_basin.png`           | Polar roses for all 6 basins               | Directly shows the WP vs SP directional shift                 |
| `sst_shear_by_basin.png`           | SST + shear distributions for all 6 basins | The positive case for WP–SP transfer — physics is shared      |
| `confusion_direction_wp_vs_sp.png` | Direction confusion matrix WP vs SP        | Shows exactly which directions are mirrored and which overlap |
| `transfer_gap_bar_chart.png`       | Transfer gap quantification                | Directly motivates the zero-shot → fine-tune pipeline         |

---

## References (for the 1-pager)

1. Huang, C., et al. "Benchmark dataset and deep learning method for global tropical cyclone forecasting." *Nature Communications*, 2025.
2. Qu, W., et al. "Accurate tropical cyclone intensity forecasts using a non-iterative spatiotemporal transformer model." *npj Climate and Atmospheric Science*, 2025.
3. DeMaria, M. & Kaplan, J. "A Statistical Hurricane Intensity Prediction Scheme (SHIPS)." *Weather and Forecasting*, 1994.
4. Knapp, K.R., et al. "The International Best Track Archive for Climate Stewardship (IBTrACS)." *Bull. Amer. Meteor. Soc.*, 2010.
5. Chen, L., et al. "FourCastNet: A Global Data-driven High-resolution Weather Forecasting Model using Adaptive Fourier Neural Operators." *arXiv:2202.11214*, 2022.
6. Beucler, T., et al. "Enforcing Analytic Constraints in Neural Networks Emulating Physical Systems." *Physical Review Letters*, 2021.
7. Kashinath, K., et al. "Physics-informed machine learning: case studies for weather and climate modelling." *Phil. Trans. R. Soc. A*, 2021.
8. Ruttgers, M., et al. "Prediction of a typhoon track using a generative adversarial network and satellite images." *Scientific Reports*, 2019.
9. Guo, L., et al. "FuXi-TC: A generative framework integrating deep learning and physics-based models for improved tropical cyclone forecasts." *arXiv*, 2025.

---

## Layout Sketch

```
┌──────────────────────────────────────────────────┐
│  TITLE                                           │
│  Cross-Basin Generalization for TC Forecasting   │
│  using TropiCycloneNet (WP → SP)                 │
├──────────────────────────────────────────────────┤
│                                                  │
│  PROBLEM CONTEXT (2-3 short paragraphs)          │
│  - Why WP → SP (data scarcity + hemisphere)      │
│  - The challenge (direction shift vs shared      │
│    physics)                                      │
│  - Our approach: 5 architectures, zero-shot      │
│    then fine-tune, minimal preprocessing         │
│                                                  │
├──────────────────────────────────────────────────┤
│                                                  │
│  EXPERIMENTAL PLAN (compact)                     │
│  - 5 models: U-Net, FNO, PINN, PI-GAN, Transf.   │
│  - Pipeline: zero-shot → select best → fine-tune │
│  - Extension: generalize to NI/SI                │
│                                                  │
├──────────────────────────────────────────────────┤
│                                                  │
│  DATA STATISTICS                                 │
│  - WP vs SP comparison table                     │
│  - What transfers / what doesn't                 │
│                                                  │
├────────────────┬─────────────┬───────────────────┤
│   FIGURE 1     │  FIGURE 2   │   FIGURE 3        │
│   Basin        │  Track map  │   Direction       │
│   imbalance    │  WP vs SP   │   roses / SST     │
│                │             │                   │
│  (motivates    │ (shows the  │ (shows transfer   │
│   the problem) │  challenge) │  potential)       │
└────────────────┴─────────────┴───────────────────┘
```
