# Project Summary — Cross-Basin Generalisation for Tropical Cyclone Forecasting

**Course:** Imperial College ELEC70127 — ML for Tackling Climate Change (40% of grade)
**Deadlines:** 1-page description March 12 | Presentation + repo submission March 27
**Dataset:** TropiCycloneNet (TCND) — multimodal tropical cyclone data, 1950–2023, six ocean basins

---

## Core Research Question

Can deep learning models trained on **Western Pacific (WP)** tropical cyclone data generalise to the **South Pacific (SP)** — a data-scarce basin in the opposite hemisphere?

- **Why WP → SP?** SP has ~5× fewer storms (30 vs 131), and the hemisphere flip (Coriolis reversal) introduces structural distributional shift in storm direction.
- **Dual-head classification** at each 6-hourly timestep:
  - **Direction** (8 classes): compass direction of storm movement over next 24h
  - **Intensity change** (4 classes): weakening, steady, slow-intensification, rapid-intensification
- **Three-stage pipeline**: zero-shot transfer → fine-tuning on small SP set → transfer gap analysis

---

## Models (6 total)

| # | Model | Time-aware | Notebook | Key Details |
|---|-------|:----------:|----------|-------------|
| 1 | ResNet-152 | No | `experiments/resnet.ipynb` | Standard ResNet backbone (baseline) |
| 2 | U-Net | No | `experiments/unet.ipynb` | SE attention, residual connections, DropPath (baseline) |
| 3 | FNO | No | `experiments/fno.ipynb` | 2 spectral layers, no padding (baseline) |
| 4 | U-Net + FiLM | Yes | `experiments/unet_film.ipynb` | Baseline U-Net + FiLM time conditioning |
| 5 | FNO v2 | Yes | `experiments/fno_v2.ipynb` | 3 layers + reflect padding + FiLM |
| 6 | U-FNO | Yes | `experiments/ufno.ipynb` | 3-branch gated hybrid (spectral + U-Net + residual) + FiLM |

All models save WP-trained and SP fine-tuned checkpoints (`.pt` files) in `experiments/`.

---

## Repository Structure

```
├── experiments/
│   ├── pre-processing.ipynb          # Central data preprocessing (all models depend on this)
│   ├── add_time_features.py          # 6D temporal feature extraction script
│   ├── resnet.ipynb                  # Baseline ResNet-152
│   ├── unet.ipynb                    # Baseline U-Net
│   ├── fno.ipynb                     # Baseline FNO
│   ├── unet_film.ipynb              # U-Net + FiLM (post-meeting)
│   ├── fno_v2.ipynb                 # FNO v2 (post-meeting)
│   ├── ufno.ipynb                   # U-FNO hybrid (post-meeting)
│   ├── ablation_shap.ipynb          # Ablation study + SHAP explainability
│   └── comparison.ipynb             # 6-model comparison (bar charts, confusion matrices, radar, etc.)
├── figures/                          # ~60+ generated plots (training curves, confusion matrices, GradCAM, etc.)
├── src/
│   └── data_analysis.py             # Data analysis utilities
├── literature-review/
│   ├── recommended-literature.md
│   ├── approaches-and-methods.md
│   ├── basin-generalization-overview.md
│   └── Core Literature/             # Key reference PDFs
├── report/
│   └── 1page/report.tex             # Submitted Week 1 one-pager (LaTeX + compiled PDF)
├── report.md                         # Main experiment report (full pipeline results)
├── one-pager-structure.md            # Planning doc for 1-pager deliverable
├── README_implementation.md          # ⬅ GO-TO DOC FOR RECENT UPDATES (see below)
├── Starter Notebook.ipynb            # Initial data exploration & visualisation
├── Starter Notebook Sam.ipynb        # Sam's exploration notebook
├── data/                             # TCND dataset (gitignored)
└── CLAUDE.md                         # Claude Code project instructions
```

---

## Recent Updates — README_implementation.md

**`README_implementation.md` is the go-to document for all recent project updates.** It covers everything added after the TA meeting with Hadrian on March 18, 2026:

1. **Temporal Feature Extraction** — 6D vectors (storm progress, cyclical hour/month encodings) saved per split
2. **U-Net + FiLM** — FiLM conditioning layers injected into baseline U-Net for temporal awareness
3. **FNO v2** — Reflect padding (reduces spectral leakage), 3 spectral layers, FiLM conditioning
4. **U-FNO** — 3-branch gated hybrid with learnable softmax gating across spectral, U-Net, and residual branches
5. **Ablation + SHAP** — Leave-one-group-out, modality ablation, add-one-in, gradient attribution, SHAP beeswarm plots
6. **Updated Comparison** — Expanded from 3 to 6 models with time-aware evaluation, full visualization suite

It also includes training order, prerequisite commands, expected checkpoint file list, and a status table for every file.

---

## Key Data Pipeline

1. Download TCND via `gdown` (starter notebook)
2. Run `experiments/pre-processing.ipynb` — produces processed splits in `data/processed/`
3. Run `experiments/add_time_features.py` — produces temporal features in `data/processed/time/`
4. Train models (can run in parallel on separate GPUs)
5. Run `experiments/ablation_shap.ipynb` and `experiments/comparison.ipynb` for analysis

---

## Literature & References

The `literature-review/` directory contains:
- Recommended reading list and approach summaries
- Basin generalisation overview with referenced papers
- Core papers: TropiCycloneNet benchmark, FuXi-TC, TIFNet intensity forecasting
