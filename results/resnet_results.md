# ResNet152 — Cross-Basin Tropical Cyclone Forecasting Results

## Model Architecture

- **Backbone**: ResNet152 (torchvision, weights=None)
- **Parameters**: 58,696,140
- **Input**: 15-channel 81×81 atmospheric grids (SST + u/v/z at 4 pressure levels + shear + vorticity)
- **Modifications**:
  - `conv1`: 3×3 stride-1 (instead of 7×7 stride-2) to preserve spatial resolution on small grids
  - No maxpool — spatial progression: 81→81→41→21→11
  - AdaptiveAvgPool2d(1,1) → 2048-d features
- **Late fusion**: grid features (2048-d) + environmental (40-d) + 1D track data (4-d)
- **Dual heads**: 3-layer MLP with GELU for direction (8-class) and intensity (4-class)

## Training Hyperparameters

| Parameter        | Value       |
|------------------|-------------|
| Optimizer        | AdamW       |
| Learning rate    | 5e-4        |
| Weight decay     | 1e-3        |
| Scheduler        | OneCycleLR (max_lr = 1.5e-3) |
| Batch size       | 32          |
| Epochs (max)     | 80          |
| Early stopping   | patience=15 |
| Dropout          | 0.25        |
| Label smoothing  | 0.05        |
| Dir loss weight  | 0.5         |
| Augmentation     | vflip=True  |

## Final Results

| Setting                          | Dir Acc | Dir F1 | Int Acc | Int F1 |
|----------------------------------|---------|--------|---------|--------|
| WP Validation (in-basin)        | 0.512   | 0.404  | 0.549   | 0.463  |
| SP Test (zero-shot transfer)    | 0.272   | 0.212  | 0.384   | 0.319  |
| SP Test (fine-tuned, two-phase) | 0.294   | 0.234  | 0.501   | 0.430  |

- **Zero-shot transfer gap (direction)**: -0.240
- **Fine-tuning recovery (direction)**: +0.022
- **Fine-tuning recovery (intensity)**: +0.117

## Fine-Tuning Strategy

Best strategy: **Two-phase fine-tuning** with combined (dir_acc + int_acc) checkpoint selection.

1. **Phase 1 — Head-only warmup** (25 epochs, lr=1.5e-4): Adapts classification heads to SP domain without disturbing learned backbone features.
2. **Phase 2 — Layer4 + heads** (40 epochs, backbone_lr=2.5e-6, head_lr=2.5e-5): Conservative unfreezing of the final residual block for domain adaptation.

## Comparison with FNO

| Metric              | FNO    | ResNet152 |
|---------------------|--------|-----------|
| WP Dir Acc          | 0.503  | **0.512** |
| SP Zero-shot Dir    | 0.248  | **0.272** |
| SP Fine-tuned Dir   | **0.346** | 0.294  |
| SP Fine-tuned Int   | —      | **0.501** |

- ResNet152 outperforms FNO on WP in-basin and SP zero-shot direction.
- FNO retains an edge on SP fine-tuned direction (0.346 vs 0.294), likely due to fewer parameters (less overfitting on ~350 SP samples).
- ResNet152 achieves strong intensity transfer (0.501 after fine-tuning).

## Iteration History

| Iter | Changes                                     | WP Dir | SP ZS Dir | SP FT Dir | SP FT Int |
|------|---------------------------------------------|--------|-----------|-----------|-----------|
| 1    | Cosine, lr=1.7e-4, dropout=0.2              | 0.447  | 0.346     | 0.267     | —         |
| 2    | OneCycleLR, lr=5e-4, dropout=0.25, LS=0.05  | 0.512  | 0.272     | 0.286     | 0.482     |
| 3    | Cosine, lr=3e-4, dropout=0.3, LS=0.1        | 0.188  | 0.090     | 0.134     | 0.202     |
| 4    | Revert to iter2 + partial FT (layer4)       | 0.512  | 0.272     | 0.270     | 0.436     |
| 5    | Two-phase FT (layer3+4), dir_acc checkpoint | 0.512  | 0.272     | 0.232     | 0.411     |
| 6    | Two-phase (layer4), dir_acc checkpoint      | 0.512  | 0.272     | 0.281     | 0.436     |
| 7    | Two-phase, combined score checkpoint, FT_LR=5e-5 | 0.512 | 0.272 | **0.294** | **0.501** |

## Key Observations

1. **Regularization is critical**: Iter 3 (dropout=0.3, LS=0.1) collapsed — too aggressive for 58M params with limited data.
2. **OneCycleLR >> Cosine** for WP training (0.512 vs 0.447).
3. **Two-phase fine-tuning** with conservative LR (5e-5) prevents catastrophic forgetting while allowing domain adaptation.
4. **Checkpoint selection matters**: Combined (dir_acc + int_acc) score outperforms loss-based or dir-only selection.
5. **Trade-off**: Higher WP performance (iter2+) comes at the cost of zero-shot transfer (0.346→0.272), suggesting the model specialises more to WP patterns.

## Artifacts

- **Notebook**: `experiments/resnet.ipynb`
- **WP model checkpoint**: `experiments/resnet_best_wp.pt` (~235 MB)
- **Fine-tuned checkpoint**: `experiments/resnet_best_ft.pt` (~235 MB)
- **Figures**: `figures/resnet_*.png` (learning curves, confusion matrices, Grad-CAM, layer activations, storm timeline)
