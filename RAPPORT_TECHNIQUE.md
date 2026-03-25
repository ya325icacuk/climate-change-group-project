# Rapport Technique : Prévision Inter-Bassins des Cyclones Tropicaux par Deep Learning

**Équipe :** Loic Bouxirot, Yasmin Akhmedova, Samuel Zhang
**Cours :** Imperial College ELEC70127 — ML for Tackling Climate Change
**Période :** 10–24 mars 2026

---

## Table des matières

1. [Contexte et objectifs](#1--contexte-et-objectifs)
2. [Le jeu de données TropiCycloneNet](#2--le-jeu-de-données-tropicyclonenet)
3. [Prétraitement et ingénierie des données](#3--prétraitement-et-ingénierie-des-données)
4. [Architectures de modèles — description détaillée](#4--architectures-de-modèles--description-détaillée)
5. [Techniques d'entraînement et régularisation](#5--techniques-dentraînement-et-régularisation)
6. [Optimisation des hyperparamètres (Optuna)](#6--optimisation-des-hyperparamètres-optuna)
7. [Résultats finaux](#7--résultats-finaux)
8. [Études d'ablation](#8--études-dablation)
9. [Analyse SHAP et explicabilité](#9--analyse-shap-et-explicabilité)
10. [Leçons apprises et recommandations](#10--leçons-apprises-et-recommandations)

---

## 1 — Contexte et objectifs

### Problématique

Les cyclones tropicaux constituent l'un des phénomènes météorologiques les plus dévastateurs. Leur prévision repose historiquement sur des modèles numériques coûteux en calcul. L'apprentissage profond offre une alternative prometteuse, mais la rareté des données (quelques milliers d'échantillons par bassin) et la variabilité inter-bassins (effet de Coriolis, régimes de SST différents) posent des défis majeurs.

### Question de recherche

> *Un modèle entraîné sur les cyclones du Pacifique Ouest (WP) peut-il généraliser au Pacifique Sud (SP) sans ré-entraînement spécifique ?*

### Justification du choix WP → SP

- **Déséquilibre des données** : WP possède 131 tempêtes contre seulement 30 pour SP (ratio 4.4×).
- **Intérêt scientifique** : le passage d'hémisphère nord à sud implique une inversion de Coriolis — les directions de déplacement se reflètent selon l'axe N-S.
- **Distance de Wasserstein** : parmi les 6 bassins, WP→SP présente la plus faible distance distributionnelle (0.355), validant ce choix comme le transfert inter-bassins le plus naturel.

### Protocole d'évaluation en trois étapes

1. **Performance intra-bassin** (WP validation) — capacité du modèle sur le bassin source
2. **Transfert zero-shot** (WP → SP test) — généralisation sans aucune donnée SP
3. **Performance après fine-tuning** (SP test) — adaptation avec un petit jeu SP (354 échantillons)

### Tâche de classification duale

Chaque prédiction comprend deux têtes de classification :
- **Direction** (8 classes) : E, SE, S, SW, W, NW, N, NE
- **Changement d'intensité** (4 classes) : Affaiblissement, Stable, Intensification lente, Intensification rapide

---

## 2 — Le jeu de données TropiCycloneNet

Le dataset TCND couvre 1950–2023 sur six bassins océaniques et fournit trois modalités par pas de temps de 6 heures :

| Modalité | Format | Contenu | Dimensions |
|----------|--------|---------|------------|
| **Data3D** | NetCDF (.nc) | Champs atmosphériques sur grille 81×81 | SST (2D), u/v/z à 4 niveaux de pression (200, 500, 850, 925 hPa) |
| **Env-Data** | NumPy (.npy) | Caractéristiques environnementales structurées | 94 dimensions brutes → 40 après filtrage |
| **Data1D** | Texte (.txt) | Séries temporelles par tempête | lat/lon offsets, vent normalisé, pression normalisée |

### Splits de données

| Split | Tempêtes | Échantillons | Usage |
|-------|----------|-------------|-------|
| WP train | 105 | 3 252 | Entraînement domaine source |
| WP val | 26 | 730 | Validation domaine source |
| SP test | 15 | 367 | Évaluation transfert zero-shot |
| SP fine-tune train | 12 | 354 | Fine-tuning domaine cible |
| SP fine-tune val | 3 | 81 | Validation fine-tuning |

---

## 3 — Prétraitement et ingénierie des données

### 3.1 Suppression des features révélant le bassin

Sur les 94 dimensions environnementales, 54 encodent l'identité du bassin (one-hot de zone, bins de longitude/latitude, coordonnées brutes). Ces features sont supprimées pour ne conserver que **40 dimensions physiquement pertinentes** : vent, vitesse de déplacement, classe d'intensité, mois, et features d'historique.

### 3.2 Transformations hémisphériques (SP uniquement)

Pour que « vers le pôle » corresponde à la même classe dans les deux hémisphères :

- **Réflexion des labels de direction** selon l'axe N-S : SE↔NE, S↔N, SW↔NW
- **Inversion du vent méridional** (`v = -v`) à tous les niveaux de pression

### 3.3 Canaux physiques dérivés

Deux canaux sont ajoutés aux 13 canaux atmosphériques de base :

| Canal | Formule | Motivation |
|-------|---------|------------|
| **Cisaillement de vent** | `√((u₂₀₀ − u₈₅₀)² + (v₂₀₀ − v₈₅₀)²)` | Facteur clé d'intensification (SHIPS) |
| **Vorticité relative à 850 hPa** | `∂v₈₅₀/∂x − ∂u₈₅₀/∂y` (différences finies centrées) | Structure rotationnelle du cyclone |

**Total : 15 canaux d'entrée** par échantillon grillé.

### 3.4 Normalisation WP-only

Statistiques z-score (moyenne, écart-type) calculées **exclusivement** sur les données WP train (algorithme de Welford), puis appliquées à tous les splits — aucune fuite d'information depuis SP.

### 3.5 Features temporelles

Un vecteur de 6 dimensions par pas de temps, conçu pour capturer le cycle de vie et la saisonnalité :

| Feature | Description | Encodage |
|---------|------------|----------|
| `storm_progress` | Position dans la vie de la tempête | Linéaire 0→1 |
| `hour_sin`, `hour_cos` | Heure du jour | Cyclique (période 24h) |
| `month_sin`, `month_cos` | Mois de l'année | Cyclique (période 12) |
| `timestep_idx_normalized` | Index temporel normalisé | Linéaire |

### 3.6 Découpage au niveau des tempêtes

Tous les splits sont effectués au niveau des tempêtes, pas des pas de temps — cela empêche la fuite temporelle entre observations corrélées à 6 heures d'intervalle.

---

## 4 — Architectures de modèles — description détaillée

Huit architectures ont été conçues, implémentées et évaluées. Toutes utilisent la même stratégie de **fusion tardive** (sauf SCANet) : les features de la grille sont extraites par le backbone → pooling moyen global → concaténation avec env (40d), 1D (4d) et time (6d) → classification par deux têtes MLP.

---

### 4.1 U-Net (Modèle 1a) — 9.8M paramètres

#### Philosophie
Architecture encodeur-décodeur avec connexions résiduelles (skip connections) pour préserver les patterns spatiaux multi-échelles. Conçu comme baseline spatiale robuste.

#### Architecture

```
Entrée (15, 81, 81)
  ↓ Lifting Conv 15→32
  ↓ EncoderBlock₁ (32→64)  ──skip──→  DecoderBlock₁ (64+64→64)
  ↓ EncoderBlock₂ (64→128) ──skip──→  DecoderBlock₂ (128+128→128)
  ↓ EncoderBlock₃ (128→256) ─skip──→  DecoderBlock₃ (256+256→256)
  ↓ Bottleneck (256→512→256)
  ↓ GAP → concat(env, 1D, time) → MLP
  ↓
  Direction (8 classes)  |  Intensité (4 classes)
```

#### Mécanismes clés

**Squeeze-and-Excitation (SE) Attention** : après chaque bloc convolutif, le SE block recalibre les canaux :
1. *Squeeze* : pooling moyen global → vecteur de C dimensions
2. *Excitation* : FC(C → C/r) → ReLU → FC(C/r → C) → Sigmoid (r=4)
3. Le vecteur de poids résultant pondère chaque canal — le modèle apprend quels canaux atmosphériques sont les plus pertinents.

**Skip Connections** : les features de chaque niveau d'encodeur sont concaténées avec le niveau correspondant du décodeur, permettant de combiner les patterns globaux (basse résolution) avec les détails locaux (haute résolution).

**Blocs résiduels** : chaque ConvBlock contient une connexion résiduelle `output = conv(x) + x`, facilitant le flux de gradient dans les architectures profondes.

---

### 4.2 U-Net + FiLM (Modèle 1b) — 10.0M paramètres

#### Philosophie
Ajouter une conscience temporelle au U-Net via FiLM (Feature-wise Linear Modulation). L'hypothèse : les mêmes poids convolutifs doivent pouvoir se comporter différemment selon la phase de vie de la tempête.

#### Le mécanisme FiLM en détail

FiLM (*Feature-wise Linear Modulation*, Perez et al., 2018) module les activations de chaque couche en fonction d'un vecteur de conditionnement :

```
time_feat (6d) → MLP(6→64→64, GELU) → time_emb (64d)
Pour chaque bloc :
  time_emb → Linear(64 → 2×C) → γ (scale), β (shift)
  sortie = γ ⊗ BatchNorm(x) + β
```

**Initialisation identité** : γ initialisé à 1, β à 0. Au départ, le modèle se comporte exactement comme le U-Net de base — la stabilité d'entraînement est préservée.

**Différence avec la concaténation** : plutôt que de concaténer les features temporelles à l'entrée (où elles seraient noyées par les 15 canaux de grille), FiLM les injecte à chaque couche, modulant le comportement de chaque feature map.

**Limitation** : FiLM applique les mêmes γ et β à toutes les positions spatiales — la modulation est **spatialement uniforme**. Un cyclone en formation au nord de la grille reçoit la même modulation temporelle qu'un pixel de mer ouverte au sud.

---

### 4.3 FNO — Fourier Neural Operator (Modèle 2a) — 10.3M paramètres

#### Philosophie
Exploiter les propriétés globales des convolutions spectrales. Dans le domaine de Fourier, une multiplication ponctuelle correspond à une convolution globale dans l'espace physique — le modèle capture les patterns à grande échelle en une seule opération.

#### La convolution spectrale (SpectralConv2d)

Le cœur du FNO est la couche de convolution spectrale :

```python
# 1. Transformée de Fourier 2D réelle
xf = rfft2(x)                    # (B, C_in, H, W//2+1) complexe

# 2. Multiplication par poids appris (tronqués à N modes)
#    Basses fréquences :
out_f[:, :modes, :modes] = einsum("bixy,ioxy->boxy", xf[:, :modes, :modes], W₁)
#    Hautes fréquences :
out_f[:, -modes:, :modes] = einsum("bixy,ioxy->boxy", xf[:, -modes:, :modes], W₂)

# 3. Transformée de Fourier inverse
out = irfft2(out_f)              # retour dans l'espace physique
```

Les poids W₁, W₂ sont des tenseurs complexes appris de forme `(C_in × C_out × modes × modes)`. Seuls les N premiers modes de Fourier sont conservés — les hautes fréquences (bruit) sont éliminées naturellement.

#### Propriétés clés

- **Invariance au maillage** : le FNO peut en théorie être entraîné sur une résolution et évalué sur une autre
- **Réceptivité globale** : chaque couche spectrale voit l'intégralité de la grille 81×81
- **Skip résiduel** : une convolution 1×1 classique est ajoutée en parallèle pour préserver les informations locales

#### Problème observé
Le FNO de base souffre d'**overfitting sévère** — il s'arrête à l'époque 53/300. La haute expressivité des convolutions spectrales, sans les biais inductifs de localité du U-Net, conduit à une mémorisation rapide.

---

### 4.4 FNO v2 + FiLM (Modèle 2b) — 9.3M paramètres

#### Philosophie
Corriger les faiblesses du FNO de base par trois améliorations ciblées.

#### Amélioration 1 : Reflect Padding

```python
x = F.pad(x, [padding]*4, mode='reflect')  # padding=9 pixels
# ... FFT / multiplication / iFFT ...
x = x[:, :, padding:-padding, padding:-padding]  # recadrage
```

Le padding par réflexion réduit la **fuite spectrale** (spectral leakage) aux bords de la grille. Sans padding, la FFT suppose une périodicité qui crée des artefacts de bord.

#### Amélioration 2 : Profondeur accrue
5 couches spectrales (contre 2 dans le FNO de base). Le padding stabilise suffisamment l'entraînement pour supporter plus de profondeur.

#### Amélioration 3 : Conditionnement FiLM
Même mécanisme temporel que le U-Net+FiLM, injecté après chaque couche spectrale.

---

### 4.5 U-FNO — Hybrid (Modèle 2c) — 1.0M paramètres

#### Philosophie
Combiner le meilleur du spectral (couverture globale) et du spatial (sensibilité locale) dans une architecture hybride légère avec fusion adaptative.

#### Architecture à trois branches par bloc

```
Entrée x
  ├── Branche spectrale : SpectralConv2d (12 modes, padding=9)
  ├── Branche spatiale : Mini-U-Net (down → mid → up, skip connection)
  ├── Branche résiduelle : Conv 1×1
  ↓
  Fusion gated : out = softmax(gate)[0]·spectral + [1]·spatial + [2]·résiduel
  ↓
  BatchNorm → FiLM(time_emb) → GELU → Dropout → connexion résiduelle
```

#### Le mécanisme de gating appris

Chaque bloc possède un paramètre `gate = nn.Parameter(torch.ones(3)/3)` — trois poids initialisés uniformément. Lors de l'entraînement, le modèle apprend la pondération optimale de chaque branche via softmax :

| Bloc | Spectral | Spatial (U-Net) | Résiduel |
|-----:|--------:|:------:|:--------:|
| 1 | 38% | 27% | 35% |
| 2 | 40% | 29% | 31% |
| 3 | 39% | 31% | 30% |

La branche spectrale domine (38–40%), mais la branche spatiale gagne en importance dans les couches profondes (27%→31%), confirmant que les features locales deviennent plus pertinentes pour les prédictions raffinées.

#### Efficacité paramétrique
La branche spatiale utilise un Mini-U-Net à un seul niveau (descente + montée), environ **20× plus léger** qu'un U-Net complet. Résultat : seulement 1.0M paramètres pour des performances compétitives.

---

### 4.6 ResNet-152 — CNN profond (58.7M paramètres)

#### Philosophie
Utiliser un backbone pré-conçu (torchvision) comme extracteur de features.

#### Modifications par rapport au ResNet standard
- **Conv1 modifiée** : 3×3 stride-1 au lieu de 7×7 stride-2, pour préserver la résolution spatiale sur les grilles 81×81 (plus petites que ImageNet 224×224)
- **Pas de pré-entraînement** ImageNet (les 15 canaux atmosphériques n'ont rien à voir avec des images naturelles)

#### Problème constaté
Avec 58.7M paramètres pour 3 252 échantillons (ratio 18 050:1 paramètres/échantillon), le modèle est **massivement surparamétré**. L'analyse des activations montre que les couches les plus profondes (layer4) contribuent à peine (magnitude 0.03 vs ~1.0–1.6 dans les couches précoces).

---

### 4.7 PI-GAN — Physics-Informed GAN (10.0M paramètres)

#### Philosophie
Utiliser un entraînement adversarial avec des contraintes physiques pour forcer le modèle à apprendre des représentations physiquement cohérentes, plus transférables entre bassins.

#### Architecture

**Générateur** : U-Net+FiLM + tête auxiliaire de reconstruction physique
- Même backbone que le modèle 1b
- Tête physique supplémentaire : 2 couches conv → 4 canaux de sortie (vorticité, divergence, cisaillement u, cisaillement v)

**Discriminateur** : MLP conditionnel
- Entrée : features fusionnées + embeddings de labels (direction + intensité)
- 2 couches cachées (128 unités), LayerNorm, LeakyReLU(0.2), Dropout(0.3)
- Sortie : score réel/faux (scalaire)

#### Pertes physiques

Les cibles physiques sont calculées analytiquement à partir de la grille d'entrée via des filtres de Sobel :

| Quantité | Formule | Canaux utilisés |
|----------|---------|-----------------|
| **Vorticité** | `∂v₈₅₀/∂x − ∂u₈₅₀/∂y` | u₈₅₀, v₈₅₀ |
| **Divergence** | `∂u₈₅₀/∂x + ∂v₈₅₀/∂y` | u₈₅₀, v₈₅₀ |
| **Cisaillement u** | `u₂₀₀ − u₈₅₀` | u₂₀₀, u₈₅₀ |
| **Cisaillement v** | `v₂₀₀ − v₈₅₀` | v₂₀₀, v₈₅₀ |

#### Entraînement WGAN-GP

```
L_discriminateur = E[D(faux)] − E[D(réel)] + λ_gp · pénalité_gradient
L_générateur    = L_classification + λ_adv · (−E[D(faux)]) + λ_phys · L_physique
```

| Hyperparamètre | Valeur | Rôle |
|----------------|--------|------|
| λ_adv | 0.01 | Signal adversarial faible — la classification reste l'objectif principal |
| λ_phys | 0.1 | Régularisation physique significative |
| λ_gp | 10.0 | Pénalité de gradient (stabilisation WGAN) |
| Warmup adversarial | 20 époques | Classification seule d'abord, puis activation du discriminateur |

#### Résultat clé
La perte physique améliore le transfert zero-shot (36.2% dir vs 33.8% pour U-Net+FiLM sans physique), mais l'entraînement GAN est **instable** — forte variance entre les essais HPO, effondrement lors du fine-tuning (-9.8pp). Cette observation a directement motivé la conception de SCANet.

---

### 4.8 SCANet — Spectral Cross-Attention Network (3.7M paramètres)

#### Philosophie
Adresser chaque limitation identifiée dans les 7 premiers modèles : remplacer FiLM par une attention croisée spatialement variable, fusionner les modalités tôt plutôt que tardivement, conserver les pertes physiques mais sans GAN.

#### Innovation 1 : Fusion multimodale précoce

Contrairement aux 7 autres modèles qui concatènent env/1D/time à la fin (fusion tardive), SCANet crée un **vecteur de contexte unifié** avant le premier bloc :

```python
context = time_mlp(time_feat)              # (B, 32)
context = context + env_mlp(env)           # fusion additive
context = context + d1d_mlp(d1d)           # fusion additive
# Ce vecteur de 32 dimensions est injecté à chaque bloc via cross-attention
```

Chaque MLP projette sa modalité vers `context_dim=32`, puis les projections sont sommées. L'information contextuelle est ainsi disponible dès la première couche, pas seulement après le pooling final.

#### Innovation 2 : Cross-Attention contextuelle (remplace FiLM)

```python
class ContextCrossAttention:
    q = q_proj(context)           # (B, 1, C) — requête depuis le contexte
    k = k_conv(x).reshape(B,C,HW) # (B, C, HW) — clés depuis les features spatiales
    v = v_conv(x).reshape(B,C,HW) # (B, C, HW) — valeurs depuis les features spatiales

    attn = softmax(q @ k^T) @ v   # (B, 1, C) — attention croisée
    gate = attn → (B, C, 1, 1)    # modulation par canal
    return x * gate                # gating du feature map
```

**Différence fondamentale avec FiLM** : le FiLM applique les mêmes γ/β à chaque position spatiale. Le cross-attention de SCANet produit des poids de modulation qui dépendent du *contenu spatial* — le modèle peut prêter attention à différentes régions selon les conditions environnementales. C'est cette modulation **spatialement variable** qui explique la supériorité de SCANet en transfert zero-shot (-13.1pp de gap vs -22.5pp pour FiLM).

#### Innovation 3 : Blocs à double branche avec gating

Simplification du design à 3 branches du U-FNO :

| Branche | Mécanisme | Inspiration |
|---------|-----------|-------------|
| **Spectrale** | SpectralConv2d + mixeur de canaux 1×1 | AFNO (Adaptive FNO) |
| **Locale** | Conv séparable en profondeur 5×5 | MobileNet |

```
Bloc SCANet :
  x → spectral(x) ← softmax(gate)[0]
  x → local(x)    ← softmax(gate)[1]
  fusion = pondération gated → cross_attention(fusion, context) → résiduel
```

La convolution séparable en profondeur (*depthwise-separable*) est **20× plus légère** qu'une convolution classique : d'abord une convolution par canal (groups=C, noyau 5×5), puis une convolution pointwise 1×1.

#### Innovation 4 : Perte physique auxiliaire (sans GAN)

Même formulation que PI-GAN (vorticité, divergence, cisaillement), mais livrée comme **perte supervisée auxiliaire** plutôt que via un discriminateur :

```
L_total = L_classification + λ_phys · L_physique
```

Tous les bénéfices de la régularisation physique (features transférables, cohérence géophysique), aucune instabilité adversariale.

#### Analyse des poids de gating

| Bloc | Spectral | Local |
|-----:|--------:|------:|
| 1 | 55.4% | 44.6% |
| 2 | 56.4% | 43.6% |
| 3 | 53.2% | 46.8% |

Plus équilibré que U-FNO : la branche locale contribue à 44–47% (contre 27–31% pour le Mini-U-Net du U-FNO), grâce à l'efficacité des convolutions séparables en profondeur.

---

## 5 — Techniques d'entraînement et régularisation

### 5.1 Augmentation de données

| Technique | Paramètres | Effet |
|-----------|-----------|-------|
| **Mixup** | α = 0.2 | Mélange linéaire d'échantillons et labels → régularisation douce |
| **CutOut** | 2 trous de 16×16 | Masquage aléatoire de régions → robustesse spatiale |
| **Bruit gaussien** | σ = 0.05 | Perturbation des valeurs → robustesse au bruit |
| **Channel Dropout** | p = 0.15 | Mise à zéro de canaux entiers → évite la dépendance à un seul canal |

### 5.2 Stratégies d'optimisation

| Stratégie | Détails |
|-----------|---------|
| **Optimiseur** | AdamW (ou Muon pour certains modèles FNO) |
| **Scheduler** | OneCycleLR (max_lr = 3× base_lr) ou CosineAnnealing |
| **Early stopping** | Patience = 50 époques sur la précision direction (WP val) |
| **EMA** | Moyenne mobile exponentielle des poids (decay = 0.998), utilisée pour l'évaluation finale |
| **Label smoothing** | ε = 0.05, redistribue 5% de la probabilité cible vers les autres classes |
| **Poids de classe** | Pondération inverse de la fréquence dans la cross-entropie, pour compenser le déséquilibre (NW/W dominent la direction, Affaiblissement domine l'intensité) |

### 5.3 Stratégie de fine-tuning (SP)

Protocole en deux phases pour l'adaptation au Pacifique Sud :

1. **Phase head-only** : seules les têtes de classification sont entraînées (backbone gelé), taux d'apprentissage conservateur
2. **Phase full** : dégel sélectif du backbone avec taux d'apprentissage réduit

Cette approche évite l'effondrement catastrophique des représentations apprises sur WP lors de l'adaptation avec seulement 354 échantillons SP.

---

## 6 — Optimisation des hyperparamètres (Optuna)

### Protocole

- **20 essais × 50 époques** pour les 6 premiers modèles
- **60 essais × 50 époques** pour SCANet (espace de recherche à 13 dimensions)
- **Médiane Pruner** : abandon anticipé des essais sous-performants
- GPU : NVIDIA RTX 5090 (32 GB VRAM)

### Espaces de recherche

| Famille | Hyperparamètres clés explorés |
|---------|------------------------------|
| U-Net / U-Net+FiLM | base_ch ∈ {32, 48, 64}, n_levels ∈ {3,4,5}, head_dim ∈ {256, 512} |
| FNO / FNO v2 / U-FNO | hidden ∈ {48, 64, 96}, modes ∈ {12..20}, n_layers ∈ {2..6} |
| PI-GAN | base_ch ∈ {24, 32, 48}, n_levels ∈ {3,4,5}, head_dim ∈ {128, 256, 512} |
| SCANet | hidden_ch ∈ {48..96}, n_modes ∈ {8..20}, n_blocks ∈ {2..4}, context_dim ∈ {32..96}, λ_phys ∈ [0.01, 0.3] |

### Résultats HPO (meilleure précision direction sur WP val)

| Modèle | Acc HPO | Config clé | Params | Temps HPO |
|--------|---------|------------|--------|-----------|
| **U-FNO** | **63.2%** | hidden=48, modes=16, 2 couches | 2.5M | 3h 00m |
| FNO v2 | 61.9% | hidden=48, modes=20, 5 couches | 9.3M | 2h 16m |
| U-Net | 61.4% | base_ch=32, 5 niveaux | 39.1M | 2h 54m |
| U-Net+FiLM | 61.4% | base_ch=48, 3 niveaux, time_emb=64 | 5.9M | 3h 00m |
| FNO | 60.7% | hidden=64, modes=15, 4 couches | 7.4M | 2h 03m |
| PI-GAN | 58.9% | base_ch=32, 3 niveaux | 10.0M | 3h 30m |
| SCANet | 56.4% | hidden=64, modes=12, 3 blocs, context=32 | 3.7M | 5h 10m |

### Découvertes clés

1. **U-FNO prospère avec seulement 2 couches** — la fusion gated spectral+spatial fournit assez d'expressivité ; plus de couches cause de l'overfitting.
2. **U-Net préfère la profondeur à la largeur** — 5 niveaux avec base_ch=32 battent les configurations plus larges mais moins profondes.
3. **U-Net+FiLM devient compact** — FiLM compense la profondeur réduite (3 niveaux au lieu de 5) par la modulation dynamique.
4. **Les taux d'apprentissage convergent** vers 2–5×10⁻⁴ pour tous les modèles — la distribution des données domine le paysage d'optimisation.

---

## 7 — Résultats finaux

Chaque modèle entraîné avec sa meilleure configuration HPO pendant 300 époques maximum (early stopping, patience=50).

### 7.1 Performance intra-bassin (WP Validation)

| Modèle | Params | Dir Acc | Dir F1 | Int Acc | Int F1 | Époques |
|--------|-------:|--------:|-------:|--------:|-------:|--------:|
| PI-GAN | 10.0M | **57.0%** | **39.1%** | 56.3% | 44.6% | 300 |
| SCANet | 3.7M | 56.4% | 34.7% | **66.2%** | 42.7% | 185 |
| U-Net+FiLM | 10.0M | 56.3% | 37.1% | 58.5% | 46.3% | 189 |
| U-Net | 9.8M | 55.9% | 34.1% | 54.2% | 44.8% | 155 |
| ResNet-152 | 58.7M | 49.9% | 37.2% | 61.1% | 50.1% | 31 |
| U-FNO | 1.0M | 47.9% | 38.0% | 60.0% | 50.1% | 66 |
| FNO | 10.3M | 47.5% | 32.3% | 60.6% | 44.8% | 53 |
| FNO v2 | 9.3M | 46.9% | 31.2% | 58.7% | 46.0% | 59 |

### 7.2 Transfert zero-shot (WP → SP)

| Modèle | Dir Acc | Dir F1 | Int Acc | Int F1 | Gap Dir |
|--------|--------:|-------:|--------:|-------:|--------:|
| **SCANet** | **43.3%** | **27.1%** | 36.8% | 21.8% | **-13.1pp** |
| U-Net | 36.8% | 24.2% | **45.8%** | **36.8%** | -19.1pp |
| PI-GAN | 36.2% | 24.3% | 38.1% | 29.1% | -20.8pp |
| FNO | 34.3% | 25.7% | 39.8% | 33.3% | -13.2pp |
| U-Net+FiLM | 33.8% | 23.4% | 39.5% | 28.3% | -22.5pp |
| U-FNO | 32.2% | 23.9% | 42.5% | 28.0% | -15.7pp |
| FNO v2 | 26.2% | 18.0% | 43.6% | 32.8% | -20.8pp |
| ResNet-152 | 25.9% | 21.1% | 32.2% | 24.2% | -24.0pp |

### 7.3 Après fine-tuning (SP Test, 354 échantillons)

| Modèle | Dir Acc | Dir F1 | Int Acc | Int F1 | Récupération Dir |
|--------|--------:|-------:|--------:|-------:|----------------:|
| **U-Net** | **39.8%** | **33.2%** | **49.9%** | **42.9%** | **+3.0pp** |
| SCANet | 38.1% | 30.3% | 43.1% | 35.2% | -5.2pp |
| FNO | 35.4% | 26.9% | 31.9% | 21.2% | +1.1pp |
| U-Net+FiLM | 34.6% | 24.9% | 45.8% | 38.2% | +0.8pp |
| FNO v2 | 27.0% | 20.6% | 39.8% | 30.4% | +0.8pp |
| PI-GAN | 26.4% | 20.2% | 49.9% | 39.7% | -9.8pp |
| ResNet-152 | 25.6% | 21.3% | 49.6% | 42.3% | -0.3pp |
| U-FNO | 22.6% | 18.7% | 44.1% | 37.9% | -9.6pp |

### 7.4 PI-GAN vs SCANet : la leçon de la livraison physique

| Métrique | PI-GAN (10.0M) | SCANet (3.7M) | Avantage SCANet |
|----------|:--------------:|:------------:|:---------------:|
| WP Dir | 57.0% | 56.4% | -0.6pp |
| WP Int | 56.3% | **66.2%** | +9.9pp |
| SP Zero-shot Dir | 36.2% | **43.3%** | +7.1pp |
| Gap transfert Dir | -20.8pp | **-13.1pp** | 7.7pp plus petit |
| Stabilité | Instable (GAN) | Stable (supervisé) | Nettement meilleur |
| Paramètres | 10.0M | **3.7M** | 2.7× plus léger |

**Conclusion** : les têtes physiques supervisées auxiliaires sont plus simples, plus stables, plus efficaces en paramètres, et plus performantes que la livraison adversariale.

---

## 8 — Études d'ablation

### 8.1 Ablation par modalité (mise à zéro de modalités entières)

Deux modèles analysés : U-Net+FiLM et U-FNO.

#### Impact sur la direction

| Modalité retirée | U-Net+FiLM Δ Dir | U-FNO Δ Dir | Interprétation |
|-----------------|:-----------------:|:----------:|----------------|
| Pas de grille 3D | **-24.1pp** | **-22.8pp** | La grille est essentielle |
| Pas d'Env | -1.5pp | -0.2pp | Impact minimal sur la direction |
| Pas de 1D | +2.2pp | +0.0pp | 1D est redondant (nuit légèrement) |
| Pas de Time | -0.7pp | +0.9pp | Faiblement utile / non pertinent |

#### Impact sur l'intensité

| Modalité retirée | U-Net+FiLM Δ Int | U-FNO Δ Int | Interprétation |
|-----------------|:-----------------:|:----------:|----------------|
| Pas de grille 3D | **-18.9pp** | **-23.7pp** | Critique pour les deux tâches |
| Pas d'Env | -7.1pp | -3.9pp | Plus important pour l'intensité |
| Pas de 1D | +2.6pp | +1.7pp | Toujours redondant |
| Pas de Time | -5.2pp | -6.9pp | Plus important pour l'intensité |

**Conclusion** : les champs atmosphériques 3D grillés sont la modalité dominante. Les features environnementales contribuent significativement à l'intensité mais pas à la direction. Les données 1D de trajectoire sont systématiquement redondantes — la grille encode déjà l'information spatiale pertinente.

### 8.2 Ablation des canaux de grille (leave-one-out, direction)

| Canal retiré | U-Net+FiLM Δ | U-FNO Δ | Interprétation |
|-------------|:----------:|:------:|----------------|
| **u_wind** (zonal, 4 niveaux) | **-20.6pp** | -5.6pp | U-Net+FiLM très dépendant de u |
| **v_wind** (méridional) | -10.1pp | -6.9pp | Les deux modèles ont besoin de v |
| SST | -0.0pp | -2.1pp | Non pertinent pour la direction |
| Géopotentiel | +0.4pp | +1.1pp | Nuit légèrement à la direction |

**Insight critique** : U-Net+FiLM concentre sa prédiction de direction sur une seule feature (u_wind = 20.6pp de précision), le rendant vulnérable si cette distribution change. U-FNO distribue sa dépendance plus uniformément (max = 6.9pp), expliquant son meilleur transfert zero-shot — il apprend des représentations plus **robustes et distribuées**.

### 8.3 Qu'est-ce que l'ablation nous apprend ?

L'ablation révèle la **stratégie interne** de chaque modèle :

- **U-Net+FiLM** = spécialiste : il mise tout sur le vent zonal → fragile en transfert
- **U-FNO** = généraliste : il utilise tous les canaux de manière équilibrée → robuste en transfert
- **La grille domine** : retirer les 15 canaux grillés cause une chute de ~24pp, alors que retirer les 40 features env ne cause qu'une chute de ~1.5pp
- **Les données 1D sont inutiles** : les retirer améliore parfois la performance (la grille 81×81 encode déjà la position et le mouvement)

---

## 9 — Analyse SHAP et explicabilité

### 9.1 Méthode

Nous avons utilisé **SHAP GradientExplainer** pour attribuer l'importance de chaque feature environnementale. Le protocole :

1. Wrapper du modèle : la grille et les données 1D/time sont fixées à la moyenne du dataset, seules les features env varient
2. 100 échantillons de fond (background) + 200 échantillons expliqués
3. Calcul des valeurs SHAP pour chaque feature env → importance = mean(|SHAP|)

### 9.2 Résultats SHAP (features environnementales)

| Groupe de features | U-Net+FiLM mean |SHAP| | U-FNO mean |SHAP| |
|--------------------|:---------------------:|:------------------:|
| Vent | 0.020 | 0.008 |
| Classe d'intensité | 0.029 | 0.009 |
| Vitesse de déplacement | 0.012 | 0.004 |

U-Net+FiLM s'appuie **2–3× plus** sur les features environnementales que U-FNO, cohérent avec les résultats d'ablation montrant une dépendance plus concentrée.

### 9.3 Attribution par gradient sur les canaux de grille

L'analyse des gradients moyens absolus par canal (`mean(|∂L/∂x|)`) confirme les résultats d'ablation :

- **u₅₀₀** (vent zonal à 500 hPa) est le canal le plus important pour les deux modèles
- **v₅₀₀** (vent méridional à 500 hPa) arrive en deuxième position
- Les deux modèles montrent des patterns de gradient cohérents malgré des architectures très différentes
- **Interprétation physique** : le flux directeur de la mi-troposphère (500 hPa) est le principal déterminant de la direction des cyclones tropicaux — un résultat bien connu en météorologie opérationnelle

### 9.4 Qu'est-ce que SHAP ?

**SHAP** (*SHapley Additive exPlanations*) est une méthode d'explicabilité issue de la théorie des jeux coopératifs. Pour chaque prédiction, SHAP attribue à chaque feature une valeur de Shapley — sa contribution marginale moyenne à la prédiction, en considérant toutes les combinaisons possibles de features.

- **Valeur SHAP positive** → la feature pousse la prédiction vers cette classe
- **Valeur SHAP négative** → la feature pousse la prédiction en dehors de cette classe
- **|SHAP| élevé** → la feature est importante pour cette prédiction

Le **GradientExplainer** approxime les valeurs SHAP par des gradients intégrés — plus rapide que l'estimateur exact pour les modèles de deep learning, tout en respectant les propriétés axiomatiques de Shapley (efficience, symétrie, linéarité, nullité).

---

## 10 — Leçons apprises et recommandations

### Les cinq leçons

1. **Les biais inductifs architecturaux > la capacité brute** : dans un régime limité en données (3 252 échantillons), la bonne structure compte plus que plus de paramètres. SCANet (3.7M) bat U-Net (9.8M) en transfert zero-shot de +6.5pp. U-FNO (1.0M) atteint une intensité compétitive (60.0%) avec 10× moins de paramètres.

2. **Les hybrides spectral-spatial apprennent des features transférables** : la fusion gated du U-FNO et de SCANet distribue la dépendance entre les canaux, créant des représentations robustes qui survivent au changement de distribution. Chute max de U-FNO = 6.9pp contre 20.6pp pour U-Net+FiLM.

3. **Les contraintes physiques améliorent le transfert, mais le mode de livraison compte** : PI-GAN a prouvé que les pertes physiques aident. SCANet a prouvé qu'on n'a pas besoin d'un GAN pour les livrer — les têtes supervisées auxiliaires sont plus simples, stables et efficaces.

4. **Modulation spatialement variable > modulation uniforme** : le cross-attention de SCANet produit une modulation différente à chaque position spatiale selon le contexte. FiLM applique les mêmes γ/β partout. C'est la différence architecturale clé qui explique -13.1pp vs -22.5pp de gap de transfert.

5. **Le fine-tuning est une arme à double tranchant** : U-Net bénéficie le plus du fine-tuning (+3.0pp), tandis que SCANet et U-FNO se dégradent (-5.2pp et -9.6pp). Les modèles avec de fortes représentations zero-shot peuvent être détériorés par le fine-tuning sur de très petits datasets (354 échantillons).

### Recommandation par scénario de déploiement

| Scénario | Modèle recommandé | Justification |
|----------|-------------------|---------------|
| Prévision intra-bassin (données abondantes) | PI-GAN ou SCANet | Meilleure précision WP (57.0% / 56.4% dir) |
| Transfert inter-bassins (aucune donnée cible) | **SCANet** | Meilleur zero-shot (43.3%), plus petit gap (-13.1pp) |
| Transfert inter-bassins (quelques données cible) | U-Net | Meilleur fine-tuné (39.8% dir, 49.9% int) |
| Déploiement à ressources limitées | U-FNO | Meilleur ratio précision/paramètres (1.0M) |
| Prévision d'intensité | SCANet ou FNO | 66.2% et 60.6% intensité WP respectivement |

### Tableau de bord final

| Catégorie | Gagnant | Performance |
|-----------|---------|-------------|
| Meilleure dir intra-bassin | PI-GAN | 57.0% |
| Meilleure int intra-bassin | SCANet | 66.2% |
| Meilleur transfert zero-shot | **SCANet** | 43.3% dir, gap -13.1pp |
| Meilleur après fine-tuning | U-Net | 39.8% dir, 49.9% int |
| Plus petit gap de transfert | **SCANet** | -13.1pp |
| Meilleure efficacité paramétrique | U-FNO | 1.0M params, 47.9% dir |

---

## Chronologie complète

| Date | Phase | Jalon | Décision clé |
|------|-------|-------|-------------|
| 10 mars | Setup | Repo initialisé, starter notebook | — |
| 11 mars | EDA | Revue littérature, analyse inter-bassins | WP→SP confirmé via Wasserstein |
| 12 mars | Livrable 1 | One-pager soumis | Pipeline de prétraitement finalisé |
| 13 mars | Baselines | ResNet, U-Net, FNO implémentés | Fusion tardive pour tous |
| 14 mars | Itération | U-Net atteint 62.5% avec augmentation | base_ch=64 overfitte, retour à 32 |
| 18 mars | Réunion superviseur | Feedback reçu | Déprioritiser PINN ; ajouter FiLM, FNO v2, U-FNO, PI-GAN |
| 20 mars | Nouvelles architectures | 3 nouveaux modèles construits | Features temporelles, fusion gated, contraintes physiques |
| 23 mars | HPO | Essais Optuna pour tous les modèles | U-FNO meilleur HPO (63.2%) |
| 24 mars | PI-GAN évalué | Pertes physiques aident mais GAN instable | Conception de SCANet |
| 24 mars | SCANet + éval finale | Comparaison 8 modèles + ablation + SHAP | SCANet = meilleur zero-shot ; U-Net = meilleur fine-tuné |

**Temps total d'entraînement :** ~12 heures (HPO + runs 300 époques, 8 modèles sur RTX 5090)
**VRAM pic :** ~30 GB / 32 GB
