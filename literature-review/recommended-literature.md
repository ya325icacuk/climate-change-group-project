# Recommended Literature — Basin Generalization for TC Forecasting

## Core References (from TropiCycloneNet)

These are directly cited in the TropiCycloneNet presentation and form the foundation:

1. **Huang, C., et al. "Benchmark dataset and deep learning method for global tropical cyclone forecasting." (Nature Communications, 2025)**
   - The original TCND/TropiCycloneNet paper. Describes the dataset structure (3630 TCs, 6 basins, 1950–2023), baseline models, and multi-basin evaluation.
   - PubMed: https://pubmed.ncbi.nlm.nih.gov/40595595/
   - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12219760/
   - **Key takeaways for basin generalization:**
     - **Direct cross-basin evidence**: TCN_M trained on all basins vs. TCN_M^WP (WP-only) shows 61.3–72.6% track improvement and 2.8–25.4% intensity improvement when evaluated on Southern Hemisphere basins. WP-only models fail to learn SH rotation patterns.
     - **Per-basin performance gaps**: NI consistently has the highest forecast errors across all models; WP has the lowest. This quantifies the transfer difficulty hierarchy.
     - **Env-T-Net matters**: Ablation (Table 4) shows the environment temporal network contributes 3.30% track / 2.00% intensity improvement — the temporal evolution of steering flow is critical, not just snapshots.
     - **GC-Net for OOD handling**: The Generator Chooser Network uses Bayesian selection among multiple generators, providing a natural mechanism for handling out-of-distribution basin characteristics.
     - **Hemisphere asymmetry is real**: The paper explicitly shows WP-trained models cannot generalize to SH without hemisphere-aware training, validating Approach 6 (hemisphere augmentation).

2. **Guo, L., et al. "FuXi-TC: A generative framework integrating deep learning and physics-based models for improved tropical cyclone forecasts." (arXiv, 2025)**
   - DDPM diffusion model that corrects FuXi global weather model biases using WRF-generated training data. Achieves 1000× speedup over WRF (2s on 1 GPU vs 83 min on 32 CPUs).
   - arXiv: https://arxiv.org/html/2508.16168v1
   - **Key takeaways for basin generalization:**
     - **WNP-only training**: Currently trained exclusively on Western North Pacific. The paper explicitly identifies "improving model generalization across ocean basins" as key future work — confirming basin generalization is an open problem.
     - **NWP-derived synthetic data**: Uses WRF simulations to generate high-resolution training data for the diffusion model. This approach could address data scarcity in NI/SI — generate synthetic WRF data for under-sampled basins to augment DL training.
     - **Multi-variable conditioning**: Conditions on Z500, MSLP, T2M, U/V winds at 200/850 hPa, and SST. These are largely basin-agnostic variables, suggesting the conditioning framework could transfer.
     - **Diffusion for uncertainty**: DDPM naturally provides ensemble forecasts via multiple sampling passes — useful for quantifying transfer uncertainty when applying to unseen basins.

3. **Qu, W., et al. "Accurate tropical cyclone intensity forecasts using a non-iterative spatiotemporal transformer model." (npj Climate and Atmospheric Science, 2025 — TIFNet)**
   - Non-iterative transformer encoder-decoder for 0–120h intensity prediction in a single forward pass, avoiding cumulative error from iterative rollout.
   - ResearchGate: https://www.researchgate.net/publication/398411677
   - **Key takeaways for basin generalization:**
     - **Direct cross-basin transfer demonstrated**: WNP-trained TIFNet fine-tuned on ENP using IBTrACS 1-min MSW outperforms operational models for 26 ENP tropical cyclones (Figure S8). This is the strongest empirical evidence that cross-basin transfer works.
     - **Wind speed definition mismatch**: Explicitly identifies "cross-basin differences in wind speed definitions (2-min vs 1-min MSW)" as a key challenge. Different agencies use different averaging periods — this must be normalized before cross-basin experiments.
     - **Two-stage training paradigm**: Pretrain on ERA5 reanalysis (1990–2020), fine-tune on operational forecasts (IFS+CMA, 2020–2022). This pretrain→fine-tune pipeline directly maps to Approach 4 (multi-basin pretrain + few-shot fine-tune).
     - **Non-iterative advantage for transfer**: Single forward pass means errors don't compound across forecast hours. This architectural choice is especially important for cross-basin transfer where small per-step biases would accumulate.
     - **57% MAE reduction vs IFS at 24h**: Sets a strong benchmark. Cross-basin experiments should compare against this within-basin performance ceiling.

## Domain Adaptation & Transfer Learning in Weather/Climate

4. **Lam, R., et al. "Learning skillful medium-range global weather forecasting." (GraphCast, 2023)**
   - Google DeepMind's GraphCast demonstrates that a single global model can generalize across all regions. Key insight: graph neural networks on the sphere naturally avoid basin-specific bias.
   - Science, 2023. DOI: 10.1126/science.adi2336

5. **Bi, K., et al. "Accurate medium-range global weather forecasting with 3D neural networks." (Pangu-Weather, 2023)**
   - Pangu-Weather uses 3D Earth-specific transformer blocks. Trained globally, it implicitly solves the basin generalization problem by learning on all basins simultaneously.
   - Nature, 2023. DOI: 10.1038/s41586-023-06185-3

6. **Chen, L., et al. "FourCastNet: A Global Data-driven High-resolution Weather Forecasting Model using Adaptive Fourier Neural Operators." (2022)**
   - Fourier neural operators for weather prediction. The spectral approach is inherently basin-agnostic.
   - arXiv: 2202.11214

## Tropical Cyclone Track & Intensity Prediction (ML-based)

7. **Ruttgers, M., et al. "Prediction of a typhoon track using a generative adversarial network and satellite images." (2019)**
   - GAN-based track prediction using satellite imagery. Trained on WP data; discussion of generalizability.
   - Scientific Reports, 2019. DOI: 10.1038/s41598-019-42339-y

8. **Alemany, S., et al. "Predicting Hurricane Trajectories Using a Recurrent Neural Network." (2019)**
   - LSTM-based track prediction. Trained on Atlantic; good baseline for single-basin models.
   - AAAI, 2019.

9. **Chen, R., et al. "A Hybrid CNN-LSTM Model for Typhoon Formation Forecasting." (2019)**
   - CNN-LSTM fusion for WP typhoon genesis prediction. Directly relevant to cross-basin genesis transfer.
   - GeoInformatica, 2019.

10. **DeMaria, M. & Kaplan, J. "A Statistical Hurricane Intensity Prediction Scheme (SHIPS)." (1994/updated)**
    - The classical statistical-dynamical intensity model. Essential reading to understand the physics-based predictors (SST, shear, upper-ocean heat content) that should transfer across basins.
    - Weather and Forecasting, 1994.

## Transfer Learning & Domain Adaptation Theory

11. **Pan, S. J. & Yang, Q. "A Survey on Transfer Learning." (2010)**
    - Foundational survey. Defines domain adaptation, covariate shift, and transfer learning taxonomy.
    - IEEE TKDE, 2010. DOI: 10.1109/TKDE.2009.191

12. **Ganin, Y., et al. "Domain-Adversarial Training of Neural Networks." (2016)**
    - Introduces the gradient reversal layer for unsupervised domain adaptation. Directly applicable: train a feature extractor that cannot distinguish source basin from target basin.
    - JMLR, 2016.

13. **Long, M., et al. "Deep Transfer Learning with Joint Adaptation Networks." (2017)**
    - Joint MMD-based domain adaptation. Could be applied to align feature distributions between Atlantic and Pacific cyclone representations.
    - ICML, 2017.

14. **Wang, M. & Deng, W. "Deep Visual Domain Adaptation: A Survey." (2018)**
    - Comprehensive survey of deep domain adaptation methods for visual data — relevant since Data_3d spatial patches are essentially multi-channel images.
    - Neurocomputing, 2018.

## Physics-Informed Machine Learning for Geoscience

15. **Beucler, T., et al. "Enforcing Analytic Constraints in Neural Networks Emulating Physical Systems." (2021)**
    - How to encode physical conservation laws as hard constraints in neural networks. Critical for ensuring cross-basin models respect thermodynamic and momentum conservation.
    - Physical Review Letters, 2021. DOI: 10.1103/PhysRevLett.126.098302

16. **Kashinath, K., et al. "Physics-informed machine learning: case studies for weather and climate modelling." (2021)**
    - Survey of physics-informed ML in climate science. Section on tropical cyclones is directly relevant.
    - Phil. Trans. R. Soc. A, 2021. DOI: 10.1098/rsta.2020.0093

## IBTrACS & Observational Data

17. **Knapp, K. R., et al. "The International Best Track Archive for Climate Stewardship (IBTrACS)." (2010)**
    - The underlying observational dataset that TCND's Data_1d is derived from. Understanding inter-agency differences is crucial for basin generalization since different agencies use different wind-averaging periods.
    - Bull. Amer. Meteor. Soc., 2010. DOI: 10.1175/2009BAMS2755.1

## Suggested Reading Order

For someone starting on the basin generalization project:

1. Start with **[1]** (the TropiCycloneNet paper) to understand the dataset
2. Read **[17]** (IBTrACS) to understand data provenance and inter-basin measurement inconsistencies
3. Read **[10]** (SHIPS) for the physical predictors that should be universal
4. Read **[11]** and **[12]** for transfer learning and domain adaptation foundations
5. Read **[4]** or **[5]** (GraphCast/Pangu-Weather) to see how global models handle multi-region generalization
6. Then dive into **[15]** and **[16]** for physics-informed approaches to improve transferability
