# Basin Generalization for Tropical Cyclone Forecasting

## Background

Tropical cyclone (TC) forecasting has been transformed by deep learning, with models like TropiCycloneNet (TCN_M), TIFNet, and FuXi-TC achieving operational-level accuracy for track and intensity prediction. However, these models are typically trained and evaluated within a single ocean basin — the Western North Pacific (WNP) in most cases — raising a fundamental question: **do these models learn universal cyclone physics, or do they memorize basin-specific statistical patterns?**

The six major TC basins (NA, EP, WP, NI, SI, SP) differ substantially in:

- **Thermodynamic environment**: WP SSTs reach 28-32C vs. NA at 26-30C
- **Steering dynamics**: Atlantic storms are steered by subtropical high + westerlies; Indian Ocean storms are monsoon-dominated
- **Data availability**: WP has ~26 TCs/year with good satellite coverage; NI has ~5 TCs/year with fewer monitoring agencies
- **Hemisphere physics**: Southern Hemisphere TCs rotate clockwise (opposite to NH), creating a fundamental symmetry that models must handle
- **Measurement conventions**: Different agencies use different wind-averaging periods (1-min vs. 2-min vs. 10-min MSW), introducing systematic biases

Basin generalization — training on one basin, evaluating on another — directly tests whether a model has learned transferable physics or brittle correlations. This is both a scientific question (what aspects of TC dynamics are universal?) and a practical one (can we forecast cyclones in data-sparse basins using models trained on data-rich ones?).

### The TCND Dataset

The Tropical Cyclone Nature Disaster (TCND) dataset from the TropiCycloneNet project provides an ideal testbed:

- **Coverage**: 3,630 TCs across 6 basins, 1950-2023 (70+ years)
- **Multimodal data**:
  - `Data_1d`: Tabular features per time step (position, intensity, movement)
  - `Data_3d`: 81x81 spatial patches at 0.25 deg resolution (SST, wind U/V, geopotential height at 200/500/850/925 hPa)
  - `Env-Data`: Environmental context including basin identifier (`area` one-hot), subtropical high region, movement history
- **Basin label**: The `area` field provides an explicit domain label — it can be used as a training signal for domain adaptation or deliberately masked to prevent leakage

---

## Aim

**Primary goal**: Quantify how well TC forecasting models generalize across ocean basins and develop methods to improve cross-basin transfer.

**Specific objectives**:

1. **Establish the transfer gap**: Measure the performance drop when a model trained on basin A is evaluated on basin B, for all meaningful basin pairs
2. **Identify what transfers and what doesn't**: Use feature attribution (Grad-CAM, SHAP) to determine whether models rely on universal physical predictors (SST, vertical wind shear, Coriolis) or basin-specific proxies (absolute position, local GPH patterns)
3. **Close the gap**: Apply domain adaptation techniques (adversarial training, physics-informed features, multi-basin pretraining) to improve cross-basin performance
4. **Determine minimum data requirements**: Find the smallest fraction of target-basin data needed to achieve competitive performance via few-shot fine-tuning

**Success criteria**:
- Zero-shot transfer achieves >70% of in-basin performance on track prediction
- Few-shot fine-tuning on 10-20% of target basin data closes >80% of the gap
- Feature attribution confirms reliance on physically meaningful features

---

## Experimental Design

### Phase 1: Baseline Establishment

**Approach 1 — Naive Transfer (No Adaptation)**

Train on all WP data (the largest basin), evaluate directly on NA, NI, SI, SP with no modifications.

| Train | Test | Task |
|---|---|---|
| WP | WP (holdout) | In-basin reference |
| WP | NA | Cross-basin (similar hemisphere) |
| WP | NI | Cross-basin (monsoon-dominated, data-sparse) |
| WP | SI | Cross-basin (opposite hemisphere) |
| WP | SP | Cross-basin (opposite hemisphere) |

**Architecture**: TropiCycloneNet baseline (CNN on Data_3d + MLP on Env-Data) or a simplified variant.

**Metrics**:
- Track: Accuracy on 8-class direction prediction (`future_direction24`), top-2 accuracy
- Intensity: Accuracy on intensity change class (`future_inte_change24`), MAE of wind speed
- Transfer gap: (In-basin metric) - (Cross-basin metric)

**Expected outcome**: Reasonable transfer to EP (similar Pacific dynamics), poor transfer to NI (monsoon steering) and SI/SP (hemisphere flip).

### Phase 2: Feature Engineering

**Approach 3 — Physics-Informed Universal Features**

Replace basin-specific features with physically universal ones:

| Basin-Specific (remove/transform) | Universal Replacement |
|---|---|
| Absolute longitude/latitude | Distance to nearest coastline, relative position within basin |
| Subtropical high GPH (absolute) | Relative ridge position (angle/distance from TC to GPH max) |
| `area` one-hot | Remove entirely |
| `location_long`/`location_lat` one-hot | Relative position encoding or remove |

Compute from Data_3d:
- **Vertical wind shear**: `|V_200 - V_850|` area-averaged over inner 5-degree box
- **SST anomaly**: Relative to 26.5C genesis threshold
- **Maximum potential intensity (MPI)**: Derived from SST and outflow temperature
- **Translational speed**: From `move_velocity`

### Phase 3: Domain Adaptation

Choose one or both based on Phase 1-2 results:

**Approach 2 — Domain-Adversarial Neural Network (DANN)**

- Shared feature extractor (CNN + MLP)
- Task head: predicts intensity/track
- Domain discriminator: classifies source basin (gradient reversal layer)
- Loss: `L_task - lambda * L_domain` with lambda annealed 0 to 1
- Critical: Exclude `area`, mask `location_long`/`location_lat` to prevent trivial domain identification

**Approach 4 — Multi-Basin Pretraining + Few-Shot Fine-Tuning**

- Pretrain on WP + NA + EP (three largest basins)
- Fine-tune on 10%, 25%, 50% of NI or SI data
- Variants: freeze-and-probe, full fine-tune (reduced LR), adapter layers
- Compare against: training from scratch on small target data, zero-shot transfer
- Directly inspired by TIFNet's two-stage pretrain/fine-tune paradigm, which successfully transferred WNP models to ENP

### Phase 4: Advanced Techniques

**Approach 5 — Basin-Conditional Normalization**
- Shared CNN/transformer backbone
- Per-basin BatchNorm statistics or FiLM (Feature-wise Linear Modulation) layers
- At test time on unseen basin: compute running stats from a few samples or use closest-matching basin's stats

**Approach 6 — Hemisphere-Aware Augmentation**
- Mirror Data_3d patches along equatorial axis
- Flip V-component sign
- Reverse Coriolis-dependent direction labels
- This directly addresses the NH→SH transfer failure documented in the TropiCycloneNet paper (61-73% track improvement when SH data is included)

---

## Evaluation Framework

### Metrics

| Metric | Track Prediction | Intensity Prediction |
|---|---|---|
| **Primary** | Accuracy of 8-class direction (`future_direction24`) | Accuracy of intensity change class (`future_inte_change24`) |
| **Secondary** | Top-2 accuracy (adjacent directions) | MAE of wind speed (denormalize: WND * 100 m/s) |
| **Transfer gap** | In-basin accuracy minus cross-basin accuracy | Same |

### Full Experimental Matrix

| Experiment | Train | Test | Method | Phase |
|---|---|---|---|---|
| In-basin baseline | WP | WP (holdout) | Standard split | 1 |
| Zero-shot WP to NA | WP | NA | Naive transfer | 1 |
| Zero-shot WP to NI | WP | NI | Naive transfer | 1 |
| Zero-shot WP to SI | WP | SI | Naive transfer | 1 |
| Physics features WP to NI | WP | NI | Universal features | 2 |
| DANN WP to NI | WP + NI (unlabeled) | NI | Adversarial adaptation | 3 |
| Multi-basin pretrain | WP+NA+EP | NI (few-shot) | Pretrain + fine-tune | 3 |
| Conditional BN | All basins | NI (few-shot) | Basin-conditional norm | 4 |
| Hemisphere augment | WP (augmented) | SI | Augmented transfer | 4 |

---

## Potential Directions

### High Priority

1. **Wind speed normalization**: TIFNet explicitly identifies cross-basin wind-averaging differences (1-min vs 2-min MSW) as a key barrier. Before any cross-basin experiment, normalize intensity labels to a common convention using IBTrACS conversion factors.

2. **Feature leakage audit**: The `area` one-hot, `location_long`, and `location_lat` features in Env-Data directly encode basin identity. These must be removed or transformed before any meaningful transfer experiment. The `subtropical_high` GPH field also has basin-specific structure.

3. **Hemisphere handling**: The TropiCycloneNet paper provides direct evidence that WP-only models fail in SH basins (61-73% track degradation). Hemisphere-aware augmentation (flipping spatial fields, reversing V-wind, adjusting Coriolis labels) should be implemented early.

4. **Start with the TIFNet transfer result as a reference point**: TIFNet's WNP-to-ENP transfer (Figure S8) is currently the strongest published evidence of cross-basin TC model transfer. Try to replicate a similar setup with TCND data as validation.

### Medium Priority

5. **Synthetic data for data-sparse basins**: FuXi-TC's approach of using WRF-generated data to train diffusion models could be adapted — generate synthetic TC scenarios for NI/SI to augment the limited real data.

6. **Multi-task learning across basins**: Train a single model on all basins simultaneously but with basin-conditional normalization layers. This is how GraphCast and Pangu-Weather implicitly solve basin generalization — they train globally.

7. **Temporal transfer**: Beyond spatial (basin) transfer, test temporal transfer — train on 1980-2010, test on 2011-2023. Climate change is shifting TC characteristics, and a model that doesn't transfer across time is unlikely to transfer across space.

8. **Grad-CAM / SHAP feature attribution**: For every trained model, generate attribution maps showing which input features drive predictions. A good cross-basin model should attend to SST gradients and vertical shear rather than absolute position.

### Exploratory

9. **Graph Neural Networks on the sphere**: GraphCast shows that GNNs naturally avoid basin-specific bias by operating on a spherical mesh. A GNN-based TC model could be inherently more transferable than CNN-based approaches.

10. **Foundation model fine-tuning**: Use a pretrained global weather model (Pangu-Weather, GraphCast, FourCastNet) as a feature extractor, then fine-tune a TC-specific head. The global pretraining may already encode basin-invariant dynamics.

11. **Causal discovery**: Use causal inference methods to identify which environmental variables have causal (not merely correlational) relationships with TC intensity change. Causal features are more likely to transfer across basins than correlational ones.

12. **Ensemble across basin-specific models**: Instead of one transferable model, train basin-specific models and combine predictions via ensemble weighting when a test TC's basin is known. Compare this "specialist ensemble" against a single generalist model.

---

## Suggested Timeline

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Data exploration + naive baseline | Per-basin performance table, transfer gap quantification |
| 2 | Physics-informed features | Comparison: raw features vs. universal features for cross-basin transfer |
| 3 | Domain adaptation (DANN or multi-basin pretrain) | Transfer gap reduction results |
| 4 | Best-method refinement + visualization | Grad-CAM maps, per-basin performance figures, final comparison table |

---

## Tools & Libraries

- **PyTorch** for model training (recommended for DANN gradient reversal)
- **xarray** for Data_3d NetCDF files
- **pandas** for Data_1d CSV handling
- **cartopy** / **matplotlib** for geographic visualizations
- **captum** (PyTorch) for Grad-CAM / SHAP feature attribution
- **wandb** or **tensorboard** for experiment tracking across basin pairs

---

## Key References

See [recommended-literature.md](recommended-literature.md) for the full annotated bibliography. The three most important papers for this project:

1. **Huang et al. (2025)** — TropiCycloneNet/TCND: Provides the dataset and baseline model; directly quantifies multi-basin vs. single-basin performance gaps
2. **Qu et al. (2025)** — TIFNet: Demonstrates successful WNP-to-ENP cross-basin transfer; identifies wind speed normalization as a critical preprocessing step
3. **Ganin et al. (2016)** — DANN: The foundational domain adaptation method; gradient reversal layer is directly applicable to basin-invariant feature learning

For detailed approach descriptions, see [approaches-and-methods.md](approaches-and-methods.md).
For basin generalization background, see [basin-generalization-overview.md](basin-generalization-overview.md).
