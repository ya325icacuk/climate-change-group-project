# Basin Generalization — Overview

## What Is Basin Generalization?

Basin generalization is a **transfer learning** problem: you train a tropical cyclone (TC) forecasting model on data from one ocean basin and evaluate how well it predicts cyclone behavior in a *different* basin it has never seen during training.

In the context of the TropiCycloneNet project and the TCND dataset, the specific transfer pairs proposed are:

| Train Basin | Test Basin | Rationale |
|---|---|---|
| **North Atlantic (NA)** | **Western Pacific (WP)** | Atlantic storms are well-documented (HURDAT2); WP is the most active basin globally with the strongest typhoons. Tests whether Atlantic-learned dynamics generalize to a higher-energy environment. |
| **Western Pacific (WP)** | **Indian Ocean (NI/SI)** | WP has the largest sample size in TCND; Indian Ocean basins have fewer storms and distinct monsoon-driven steering patterns. Tests generalization to a data-sparse regime. |

## Why It Matters

1. **Data scarcity**: Some basins (North Indian, South Pacific) have far fewer labeled TC events. A model that transfers well eliminates the need for basin-specific retraining.
2. **Climate change adaptation**: As ocean temperatures shift, TC activity may emerge or intensify in basins with limited historical records. Cross-basin models provide early forecasting capability.
3. **Universal physics vs. local dynamics**: Basin generalization directly tests whether a model has learned *fundamental cyclone physics* (Carnot heat engine, Coriolis deflection, vertical wind shear sensitivity) or merely memorized basin-specific statistical patterns (e.g., Atlantic recurvature tracks).

## Key Differences Between Basins

| Factor | Atlantic (NA) | Western Pacific (WP) | Indian Ocean (NI/SI) |
|---|---|---|---|
| **Average annual TCs** | ~12 | ~26 | ~12 (combined) |
| **Peak intensity** | Cat 5 rare | Frequent super typhoons (>70 m/s) | Strong cyclones but fewer Cat 5 |
| **SST range** | 26-30C | 28-32C | 27-31C |
| **Steering flow** | Subtropical high + westerlies | Monsoon trough + subtropical ridge | Monsoon reversal dominates |
| **Coriolis effect** | Mid-latitude recurvature common | Varies; some straight-running storms | Cross-equatorial genesis possible (SI) |
| **Vertical wind shear** | High shear from Saharan Air Layer | Lower average shear | Highly seasonal (monsoon-dependent) |
| **Data quality (IBTrACS)** | Excellent (aircraft recon) | Good (satellite-era) | Moderate (fewer agencies) |

## Connection to TCND Dataset

The TCND dataset (from the TropiCycloneNet PDF) is well-suited for this task because:

- It covers **6 major basins** (NA, EP, WP, NI, SI, SP) over **70 years** (1950-2023)
- The `area` field in **Env-Data** is a one-hot encoded vector of shape `(6,)` identifying the basin — this can be used as a domain label or deliberately masked during training
- The **Data_3d** spatial patches (81x81 grids at 0.25 deg resolution) capture basin-agnostic atmospheric structure (SST, wind components, geopotential height at 200/500/850/925 hPa)
- The **Env-Data** "Golden Branch" contains physics-informed features (subtropical high region from 500 hPa GPH, movement history) that may encode basin-specific vs. universal signals

## What Success Looks Like

A successful basin generalization experiment would show:

- **Zero-shot transfer**: A model trained only on WP achieves competitive MAE/RMSE on NA or NI test data without fine-tuning
- **Few-shot adaptation**: Fine-tuning on a small fraction (~10-20%) of the target basin's data closes most of the performance gap
- **Feature attribution**: Grad-CAM or SHAP analysis reveals the model relies on physically meaningful features (SST gradients, vertical shear patterns) rather than basin-specific location biases
