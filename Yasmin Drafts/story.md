# Training Journey: Cross-Basin Cyclone Forecasting

## The Quest

We set out to find the best deep learning architecture for predicting tropical cyclone direction and intensity changes across ocean basins. Our battlefield: the TropiCycloneNet dataset — Western Pacific storms training our models, South Pacific storms testing their generalization.

Five architectures entered the arena. Each faced a rigorous 20-trial Optuna hyperparameter search, then endured the full 300-epoch gauntlet with their optimal configurations.

---

## Chapter 1: The Contestants

| Model | Philosophy | HPO Range | Final Params |
|-------|-----------|-----------|-------------|
| **U-Net** | Spatial hierarchy + SE attention + skip connections | 9.8M–39M | 39.1M |
| **U-Net + FiLM** | U-Net + temporal awareness via Feature-wise Linear Modulation | 5.9M–36M | 5.9M |
| **FNO** | Global patterns via Fourier spectral convolutions | 1.2M–15M | 7.4M |
| **FNO v2** | FNO + reflect padding + FiLM + tuned depth | 0.9M–12M | 9.3M |
| **U-FNO** | Hybrid: spectral + spatial + residual with gated fusion | 1.0M–10M | 2.5M |

**The arena:** NVIDIA RTX 5090 (32GB VRAM). All 5 models trained simultaneously (~15-30GB combined).

**The dataset:** 3,252 WP training samples (105 storms), 730 WP validation (26 storms), 367 SP test (15 storms). A brutally data-limited regime.

---

## Chapter 2: Hyperparameter Search

*20 Optuna trials x 50 epochs per model, all 5 running in parallel.*

### Search Space (Scaled Up for 32GB VRAM)
We expanded the search to include larger architectures:
- **U-Net/FiLM:** base_ch ∈ {32, 48, 64}, n_levels ∈ {3,4,5}, head_dim ∈ {256, 512}
- **FNO/FNO v2/U-FNO:** hidden ∈ {48, 64, 96}, modes ∈ {12..20}, n_layers ∈ {2..6}
- **All:** lr ∈ [1e-4, 1e-3], weight_decay ∈ [5e-4, 5e-3], dir_weight ∈ [0.45, 0.6]

### HPO Results (Best Direction Accuracy on WP Val)

| Model | Best HPO Acc | Best Trial | HPO Time |
|-------|-------------|-----------|----------|
| **U-FNO** | **63.2%** | Trial 19 | 3h 0m |
| FNO v2 | 61.9% | Trial 13 | 2h 16m |
| U-Net | 61.4% | Trial 4 | 2h 54m |
| U-Net + FiLM | 61.4% | Trial 15 | 3h 0m |
| FNO | 60.7% | Trial 12 | 2h 3m |

### Key HPO Discoveries

1. **U-FNO thrives with just 2 layers** — Optuna found that shallow spectral+spatial fusion (hidden=48, modes=16, 2 layers, padding=13) outperformed deeper configs. The gated fusion already provides enough expressivity.

2. **U-Net prefers depth over width** — Optuna chose 5 levels with base_ch=32 (39M params) over wider but shallower alternatives. The extra downsampling captures multi-scale atmospheric patterns.

3. **FNO v2 likes many modes** — 20 Fourier modes (vs 12 baseline) with 5 layers and reflect padding=9. More spectral resolution helps capture fine atmospheric structures.

4. **U-Net + FiLM went compact** — 3 levels, base_ch=48, head_dim=512. Temporal conditioning via FiLM lets the model be smaller while maintaining performance through dynamic feature modulation.

5. **Learning rates clustered around 2-5e-4** — all models found similar optimal LRs, suggesting the data distribution, not the architecture, dominates the optimization landscape.

---

## Chapter 3: The 300-Epoch Gauntlet

Each model trained with its best HPO config, full augmentation (CutOut, noise, channel dropout), EMA (decay=0.998), OneCycleLR scheduler, and patience=50 early stopping.

### In-Basin Results (WP Validation)

| Model | Params | Dir Acc | Dir F1 | Int Acc | Int F1 | Epochs |
|-------|-------:|--------:|-------:|--------:|-------:|-------:|
| **U-Net** | 39.1M | **63.3%** | **46.4%** | 58.1% | 47.1% | 134 |
| U-FNO | 2.5M | 61.0% | 46.3% | 58.5% | 46.3% | 100 |
| FNO v2 | 9.3M | 59.7% | 37.4% | 59.0% | 40.7% | 135 |
| U-Net+FiLM | 5.9M | 59.2% | 42.6% | **60.3%** | **49.2%** | 120 |
| FNO | 7.4M | 57.9% | 40.1% | 64.1% | 47.1% | 114 |

**U-Net takes the WP direction crown at 63.3%.** But the real surprise is U-FNO: with just 2.5M parameters (16x smaller than U-Net), it achieves 61.0% direction with the best F1 balance (46.3% dir + 46.3% int). U-Net+FiLM leads on intensity (60.3%) with the best intensity F1 (49.2%).

---

## Chapter 4: Cross-Basin Transfer

### Zero-Shot (WP → SP, no fine-tuning)

| Model | Dir Acc | Dir F1 | Int Acc | Int F1 | Dir Gap |
|-------|--------:|-------:|--------:|-------:|--------:|
| **U-FNO** | **37.9%** | **29.5%** | **52.3%** | **40.9%** | -23.1pp |
| FNO v2 | 36.8% | 22.1% | 48.5% | 33.3% | -22.9pp |
| U-Net+FiLM | 33.5% | 27.0% | 49.3% | 39.6% | -25.7pp |
| FNO | 33.0% | 24.5% | 48.5% | 33.8% | -24.9pp |
| U-Net | 31.1% | 22.2% | 41.1% | 34.0% | -32.2pp |

**U-FNO dominates zero-shot transfer** with the best direction (37.9%), best intensity (52.3%), and smallest transfer gap (-23.1pp). The spectral+spatial hybrid learns features that generalize best across basins. U-Net, despite being best in-basin, suffers the largest transfer gap (-32.2pp) — its 39M parameters overfit to WP-specific patterns.

### Fine-Tuned (354 SP samples)

| Model | Dir Acc | Dir F1 | Int Acc | Int F1 | Recovery |
|-------|--------:|-------:|--------:|-------:|---------:|
| **U-Net+FiLM** | **40.9%** | 28.1% | 46.6% | 39.5% | +7.4pp dir |
| FNO | 35.4% | 27.3% | **52.0%** | **42.1%** | +2.4pp dir |
| U-Net | 34.9% | 25.2% | 46.9% | 38.7% | +3.8pp dir |
| FNO v2 | 32.7% | 24.1% | 54.2% | 39.9% | -4.1pp dir |
| U-FNO | 31.6% | **30.6%** | 49.0% | 42.2% | -6.3pp dir |

Fine-tuning is a mixed bag. **U-Net+FiLM benefits most from fine-tuning** (+7.4pp direction), likely because its temporal conditioning can adapt to SP's different seasonal patterns. Paradoxically, **U-FNO loses ground after fine-tuning** — its compact 2.5M params may overfit to the tiny 354-sample SP training set.

---

## Chapter 5: Findings & Lessons Learned

### The Efficiency Champion: U-FNO
With only 2.5M parameters, U-FNO achieves the best zero-shot transfer (37.9% dir, 52.3% int) and competitive in-basin performance (61.0% dir). Its gated fusion of spectral, spatial, and residual branches creates robust features that transfer across basins. The Optuna-discovered 2-layer architecture with padding=13 is surprisingly effective.

### The In-Basin King: U-Net
At 39M parameters with 5 encoder levels, U-Net squeezes the most from WP training data (63.3% dir). But this comes at the cost of generalization — it has the largest transfer gap (-32.2pp).

### The Temporal Advantage: U-Net + FiLM
FiLM conditioning gives the best fine-tuning recovery (+7.4pp direction after SP fine-tuning) and the best intensity F1 (49.2%). Temporal features (storm progress, hour/month cycles) help adaptation to new basins with different seasonal patterns.

### The Data Dilemma
With only 3,252 training samples, all models face a fundamental tension:
- Larger models (U-Net 39M) memorize WP patterns but fail to generalize
- Smaller models (U-FNO 2.5M) learn transferable features but plateau on in-basin accuracy
- The sweet spot depends on the deployment scenario: **in-basin → U-Net**, **cross-basin → U-FNO**, **adaptive → U-Net+FiLM**

### HPO Insight
Optuna consistently preferred moderate learning rates (2-5e-4), moderate weight decay, and direction-weighted losses (0.5-0.57). The architecture choices were more impactful than training hyperparameters — the right model structure matters more than the right learning rate.

---

## Final Scoreboard

| | Best In-Basin Dir | Best In-Basin Int | Best Zero-Shot | Best Fine-Tuned |
|-|-------------------|-------------------|----------------|-----------------|
| **Winner** | U-Net (63.3%) | FNO (64.1%) | U-FNO (37.9% dir) | U-Net+FiLM (40.9% dir) |
| **Runner-up** | U-FNO (61.0%) | U-Net+FiLM (60.3%) | FNO v2 (36.8% dir) | FNO (35.4% dir) |

**Total training time:** ~7 hours (HPO + 300-epoch runs, 5 models in parallel on RTX 5090)
**Total VRAM peak:** ~30GB / 32GB
