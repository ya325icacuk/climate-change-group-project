# Approaches & Methods — Basin Generalization Experiments

## Overview

This document outlines concrete experimental approaches for the basin generalization task using the TCND dataset. The goal: train on one basin, test on another, and systematically improve cross-basin transfer.

---

## Approach 1: Naive Baseline (No Adaptation)

**Idea**: Train a model on Basin A, evaluate directly on Basin B with no modifications.

**Setup**:
- Train on all WP (Western Pacific) data from TCND (largest basin, ~26 TCs/year)
- Evaluate on NA, NI, SI, SP without any fine-tuning
- Use both intensity prediction (`future_inte_change24`) and track prediction (`future_direction24`) as tasks

**Architecture**: Use the TropiCycloneNet baseline or a simple CNN on Data_3d + MLP on Env-Data.

**What to measure**:
- Per-basin accuracy/MAE compared to within-basin performance
- Which basins transfer best/worst and why

**Expected result**: Decent transfer to EP and SP (similar tropical dynamics), poor transfer to NI (monsoon-dominated steering).

---

## Approach 2: Domain-Adversarial Neural Network (DANN)

**Idea**: Use adversarial training to learn basin-invariant features.

**Method**:
1. Shared feature extractor (CNN for Data_3d, MLP for Env-Data)
2. Task head: predicts intensity change or track direction
3. Domain discriminator: tries to classify which basin the sample came from
4. Gradient reversal layer between feature extractor and domain discriminator

**Key implementation detail**: The `area` one-hot vector in Env-Data directly provides the domain label. During DANN training, **exclude the `area` feature from model input** so the model cannot trivially identify the basin.

```
Loss = L_task - lambda * L_domain
```

Where `lambda` is annealed from 0 to 1 during training (following Ganin et al., 2016).

**TCND-specific considerations**:
- The `location_long` and `location_lat` one-hot encodings will leak basin identity — consider masking these or replacing them with relative position encodings
- The `subtropical_high` GPH field has basin-specific structure — the DANN should learn to extract the *relative* steering pattern rather than absolute GPH values

---

## Approach 3: Physics-Informed Feature Engineering

**Idea**: Replace basin-specific features with physically universal ones that should transfer naturally.

**Universal features** (should transfer across basins):
- SST anomaly relative to 26.5C threshold (TC genesis threshold)
- Vertical wind shear: |V_200 - V_850| computed from Data_3d U/V components
- Mid-level relative humidity (proxy from geopotential height gradients)
- Translational speed of the TC (from `move_velocity`)
- Maximum potential intensity (MPI) derived from SST and outflow temperature

**Basin-specific features to handle carefully**:
- Absolute longitude/latitude → convert to **distance from TC to nearest coastline** or **relative position within basin**
- Subtropical high → extract **relative ridge position** (angle and distance from TC center to nearest GPH maximum)
- Month → keep as-is (seasonality is universal, though shifted by hemisphere)

**Implementation with TCND**:
```python
# Compute vertical wind shear from Data_3d
u_200 = data_3d['u_component'][level='200']  # 81x81 grid
u_850 = data_3d['u_component'][level='850']
v_200 = data_3d['v_component'][level='200']
v_850 = data_3d['v_component'][level='850']

# Area-averaged shear over inner 5-degree box (21x21 center pixels)
shear_u = (u_200[30:51, 30:51] - u_850[30:51, 30:51]).mean()
shear_v = (v_200[30:51, 30:51] - v_850[30:51, 30:51]).mean()
vertical_shear = np.sqrt(shear_u**2 + shear_v**2)
```

---

## Approach 4: Multi-Basin Pre-training + Few-Shot Fine-Tuning

**Idea**: Pre-train on all available basins, then fine-tune on a small subset of the target basin.

**Setup**:
1. Pre-train on WP + NA + EP (the three largest basins in TCND)
2. Fine-tune on 10%, 25%, 50% of NI or SI data
3. Compare against: (a) training from scratch on the small target basin data, (b) zero-shot transfer from step 1

**Variants**:
- **Freeze-and-probe**: Freeze the pre-trained feature extractor, only train the final classification/regression head
- **Full fine-tuning**: Update all weights with a reduced learning rate
- **Adapter layers**: Insert small trainable adapter modules between frozen layers (parameter-efficient)

**This tests**: Whether multi-basin pre-training learns a better initialization than single-basin training, and how much target data is needed to close the gap.

---

## Approach 5: Basin-Conditional Normalization

**Idea**: Share most model weights across basins but use basin-specific batch normalization or FiLM (Feature-wise Linear Modulation) layers.

**Method**:
- The main CNN/transformer backbone is shared
- Each basin gets its own BatchNorm statistics (mean/variance) or FiLM parameters (scale/shift)
- At test time on a new basin, compute running statistics from a few samples or use the closest-matching basin's statistics

**Why this works**: Different basins have different feature distributions (e.g., WP SSTs are systematically warmer than NA), but the *relationships* between features are similar. Basin-conditional normalization handles the distribution shift while sharing the learned relationships.

---

## Approach 6: Hemisphere-Aware Augmentation

**Idea**: Southern hemisphere TCs rotate clockwise (opposite to NH). Use data augmentation to double the effective training set.

**Augmentation strategy**:
- Mirror Data_3d spatial patches along the equatorial axis
- Flip the sign of the V-component of wind
- Reverse the Coriolis-dependent direction labels
- Adjust `location_lat` accordingly

**Benefit**: Allows NA/WP (NH) training data to augment SI/SP (SH) transfer, and vice versa.

---

## Evaluation Framework

For all approaches, use a consistent evaluation:

| Metric | Track Prediction | Intensity Prediction |
|---|---|---|
| **Primary** | Accuracy of 8-class direction (future_direction24) | Accuracy of intensity change class (future_inte_change24) |
| **Secondary** | Top-2 accuracy (adjacent directions) | MAE of wind speed (denormalize: WND * 100 m/s) |
| **Transfer gap** | (In-basin accuracy) - (Cross-basin accuracy) | Same, for intensity |

### Experimental Matrix

| Experiment | Train | Test | Method |
|---|---|---|---|
| Baseline (in-basin) | WP | WP (holdout) | Standard split |
| Zero-shot NA→WP | NA | WP | Approach 1 |
| Zero-shot WP→NI | WP | NI | Approach 1 |
| DANN WP→NI | WP + NI (unlabeled) | NI | Approach 2 |
| Physics features | WP | NI | Approach 3 |
| Multi-basin pretrain | WP+NA+EP | NI (few-shot) | Approach 4 |
| Conditional BN | All basins | NI (few-shot) | Approach 5 |
| Hemisphere augment | WP (augmented) | SI | Approach 6 |

---

## Recommended Starting Point

**Start simple, add complexity incrementally:**

1. **Week 1**: Implement Approach 1 (naive baseline) for WP→NA and WP→NI. This establishes the transfer gap you're trying to close.
2. **Week 2**: Implement Approach 3 (physics-informed features). Compute vertical wind shear and SST anomalies from Data_3d. See if universal features alone reduce the gap.
3. **Week 3**: Implement Approach 2 (DANN) or Approach 4 (multi-basin pre-training), depending on which shows more promise from initial experiments.
4. **Week 4**: Combine the best-performing elements and prepare visualizations (per-basin performance maps, feature attribution with Grad-CAM).

## Tools & Libraries

- **PyTorch** or **TensorFlow** for model training
- **xarray** for reading Data_3d NetCDF files
- **pandas** for Data_1d CSV handling
- **cartopy** or **matplotlib** with basemap for geographic visualizations
- **captum** (PyTorch) or **tf-explain** for Grad-CAM / SHAP feature attribution
- **wandb** or **tensorboard** for experiment tracking across basin pairs
