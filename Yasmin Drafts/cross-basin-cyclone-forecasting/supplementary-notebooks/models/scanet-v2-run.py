#!/usr/bin/env python
# coding: utf-8

# # SCANet v2 — Improved Spectral Cross-Attention Network
#
# **Improvements over SCANet v1:**
# 1. True spatially-varying multi-head cross-attention (per-position output, not global gate)
# 2. Multi-scale local branch (parallel 3×3, 5×5, 7×7 depthwise convs with learned fusion)
# 3. Task-specific context separation (dir_context vs int_context)
# 4. Extended physics loss (+potential vorticity, +thermal wind balance → 7 channels)
# 5. Backbone freezing strategy for fine-tuning (freeze spectral+local, adapt heads+cross-attn)
# 6. Context-conditioned adaptive gating (per-sample gate via MLP instead of static parameter)
#
# **Base architecture:** SCANet v1 (spectral cross-attention + depthwise-sep + physics aux loss)
# **Starting config:** Optuna-tuned v1 params (hidden_ch=64, n_modes=12, n_blocks=3, etc.)

# ## Section 0: Setup & Configuration

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, accuracy_score
)
from pathlib import Path
from collections import Counter
from copy import deepcopy
from tqdm.auto import tqdm
import warnings, os, multiprocessing

warnings.filterwarnings("ignore")
multiprocessing.set_start_method("fork", force=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

PROJECT_ROOT = Path("../..").resolve()
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR  = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
(PROJECT_ROOT / "checkpoints").mkdir(exist_ok=True)

# ── Architecture (from SCANet v1 Optuna best) ──
IN_CHANNELS   = 15
HIDDEN_CH     = 64
N_MODES       = 12
N_BLOCKS      = 3
PADDING       = 13
CONTEXT_DIM   = 32
TIME_DIM      = 6
ENV_DIM       = 40
D1D_DIM       = 4
HEAD_DIM      = 192

# ── v2 additions ──
N_ATTN_HEADS  = 4       # Multi-head cross-attention
LOCAL_KERNELS = [3, 5, 7]  # Multi-scale local branch

# ── Regularisation ──
DROPOUT       = 0.2
LABEL_SMOOTH  = 0.0

# ── Training ──
BATCH_SIZE    = 64
LR            = 2.3e-4
WEIGHT_DECAY  = 4.8e-3
EPOCHS        = 300
PATIENCE      = 50
DIR_WEIGHT    = 0.5

# ── Physics loss ──
LAMBDA_PHYS   = 0.087
N_PHYSICS_CH  = 7  # v2: 7 channels (was 4)

# ── Augmentation ──
USE_MIXUP      = True
USE_CUTOUT     = True
CUTOUT_SIZE    = 16
CUTOUT_N       = 2
USE_NOISE      = True
NOISE_STD      = 0.05
USE_CHAN_DROP   = True
CHAN_DROP_PROB  = 0.15

USE_ENV = True
USE_1D  = True

FT_LR       = 1e-4
FT_EPOCHS   = 80
FT_PATIENCE = 15

N_DIR_CLASSES = 8
N_INT_CLASSES = 4
DIR_LABELS  = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
INTE_LABELS = ["Weakening", "Steady", "Slow-intens.", "Rapid-intens."]

# ── Grid channel mapping (for physics losses) ──
CH_U = [1, 4, 7, 10]    # u wind at 200, 500, 850, 925 hPa
CH_V = [2, 5, 8, 11]    # v wind
CH_Z = [3, 6, 9, 12]    # geopotential
CH_SST   = 0
CH_SHEAR = 13
CH_VORT  = 14
IDX_200, IDX_500, IDX_850, IDX_925 = 0, 1, 2, 3

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})")
print(f"Data dir: {DATA_DIR}")


# ## Section 1: Data Loading & Dataset

SPLITS = {
    "wp_train":    {"reflected": False},
    "wp_val":      {"reflected": False},
    "sp_test":     {"reflected": True},
    "sp_ft_train": {"reflected": True},
    "sp_ft_val":   {"reflected": True},
}

raw = {}
for split in SPLITS:
    raw[split] = {
        "grids":  torch.load(DATA_DIR / "grids"  / f"{split}_grids.pt",  weights_only=False),
        "env":    torch.load(DATA_DIR / "env"    / f"{split}_env.pt",    weights_only=False),
        "data1d": torch.load(DATA_DIR / "data1d" / f"{split}_1d.pt",    weights_only=False),
        "labels": torch.load(DATA_DIR / "labels" / f"{split}_labels.pt", weights_only=False),
        "time":   torch.load(DATA_DIR / "time"   / f"{split}_time.pt",  weights_only=False),
    }

for s, d in raw.items():
    n = sum(v.shape[0] for v in d["grids"].values())
    print(f"  {s:15s}: {len(d['grids']):3d} storms, {n:5d} timesteps")


class CycloneDataset(Dataset):
    """Timestep-level samples with time features. Returns 6-tuple."""
    def __init__(self, grids, env, data1d, labels, time_feats,
                 use_reflected=False, d1d_mean=None, d1d_std=None):
        self.samples = []
        dir_key = "direction_reflected" if use_reflected else "direction"
        for storm_id in grids:
            g, e, d, t = grids[storm_id], env[storm_id], data1d[storm_id], time_feats[storm_id]
            d_lbl, i_lbl = labels[storm_id][dir_key], labels[storm_id]["intensity"]
            for idx in range(g.shape[0]):
                if d_lbl[idx].item() == -1 or i_lbl[idx].item() == -1:
                    continue
                self.samples.append((g[idx], e[idx], d[idx], t[idx],
                                     d_lbl[idx].long(), i_lbl[idx].long()))
        if d1d_mean is None:
            all_1d = torch.stack([s[2] for s in self.samples])
            self.d1d_mean = all_1d.mean(0)
            self.d1d_std  = all_1d.std(0).clamp(min=1e-6)
        else:
            self.d1d_mean, self.d1d_std = d1d_mean, d1d_std

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        grid, env, d1d, tf, dl, il = self.samples[idx]
        return grid, env, (d1d - self.d1d_mean) / self.d1d_std, tf, dl, il


datasets = {}
datasets["wp_train"] = CycloneDataset(
    raw["wp_train"]["grids"], raw["wp_train"]["env"],
    raw["wp_train"]["data1d"], raw["wp_train"]["labels"], raw["wp_train"]["time"])
d1d_mean, d1d_std = datasets["wp_train"].d1d_mean, datasets["wp_train"].d1d_std

for split, cfg in SPLITS.items():
    if split == "wp_train": continue
    datasets[split] = CycloneDataset(
        raw[split]["grids"], raw[split]["env"], raw[split]["data1d"],
        raw[split]["labels"], raw[split]["time"],
        use_reflected=cfg["reflected"], d1d_mean=d1d_mean, d1d_std=d1d_std)

loaders = {s: DataLoader(datasets[s], batch_size=BATCH_SIZE,
                         shuffle=(s == "wp_train"), num_workers=4,
                         pin_memory=True, persistent_workers=True)
           for s in SPLITS}

for s, ds in datasets.items():
    print(f"  {s:15s}: {len(ds):5d} samples")

# ── Class weights ──
dir_counts, int_counts = Counter(), Counter()
for *_, dl, il in datasets["wp_train"].samples:
    dir_counts[dl.item()] += 1
    int_counts[il.item()] += 1

def inv_freq(counts, n):
    total = sum(counts.values())
    return torch.tensor([total / max(counts.get(c, 1), 1) / n for c in range(n)])

dir_weights = inv_freq(dir_counts, N_DIR_CLASSES)
int_weights = inv_freq(int_counts, N_INT_CLASSES)
print("Dir weights:", dir_weights.numpy().round(3))
print("Int weights:", int_weights.numpy().round(3))


# ## Section 2: Model Architecture — SCANet v2

# ── Spectral Conv (unchanged from v1) ──

class SpectralConv2d(nn.Module):
    """2D Fourier spectral conv with reflect padding."""
    def __init__(self, in_ch, out_ch, modes1, modes2, padding=9):
        super().__init__()
        self.modes1, self.modes2, self.out_channels, self.padding = modes1, modes2, out_ch, padding
        s = (2 / (in_ch + out_ch)) ** 0.5
        self.w1 = nn.Parameter(s * (torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat) - 0.5))
        self.w2 = nn.Parameter(s * (torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat) - 0.5))

    def forward(self, x):
        B, C, H, W = x.shape
        if self.padding > 0:
            x = F.pad(x, [self.padding] * 4, mode='reflect')
        Hp, Wp = x.shape[-2], x.shape[-1]
        xf = torch.fft.rfft2(x)
        of = torch.zeros(B, self.out_channels, Hp, Wp // 2 + 1, dtype=torch.cfloat, device=x.device)
        of[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, :self.modes1, :self.modes2], self.w1)
        of[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, -self.modes1:, :self.modes2], self.w2)
        x = torch.fft.irfft2(of, s=(Hp, Wp))
        if self.padding > 0:
            x = x[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return x


# ── [IMPROVEMENT 1] Multi-Head Spatially-Varying Cross-Attention ──

class MultiHeadSpatialCrossAttention(nn.Module):
    """Multi-head cross-attention that produces per-position modulation.

    Unlike v1's ContextCrossAttention which outputs a global (B, C, 1, 1) gate,
    this produces a spatially-varying (B, C, H, W) modulation map.

    The context vector is projected into multiple query heads, which attend to
    different spatial positions of the feature map. The result is reshaped back
    to (B, C, H, W), giving each spatial location a different modulation.

    Reference: Perceiver (Jaegle et al., 2021), Stable Diffusion cross-attention
    """
    def __init__(self, context_dim, channels, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_ch = channels // n_heads
        assert channels % n_heads == 0, f"channels ({channels}) must be divisible by n_heads ({n_heads})"

        # Context → queries: project context to (n_heads * head_ch) = channels
        # Use multiple query tokens (n_query) to produce spatial output
        self.n_query = 16  # sqrt(16)=4, will reshape to spatial grid then upsample
        self.q_proj = nn.Linear(context_dim, self.n_query * channels)

        # Spatial features → keys/values
        self.k_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)

        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, 1)

        self.scale = self.head_ch ** -0.5
        self.norm = nn.LayerNorm(channels)

        # Initialize near-identity
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, context):
        """x: (B, C, H, W), context: (B, D) -> spatially-modulated x: (B, C, H, W)"""
        B, C, H, W = x.shape
        nH = self.n_heads
        dH = self.head_ch

        # Query from context: (B, n_query, C) -> (B, nH, n_query, dH)
        q = self.q_proj(context).reshape(B, self.n_query, nH, dH).permute(0, 2, 1, 3)

        # Key/Value from spatial features: (B, C, H, W) -> (B, nH, HW, dH)
        k = self.k_proj(x).reshape(B, nH, dH, H * W).permute(0, 1, 3, 2)  # (B, nH, HW, dH)
        v = self.v_proj(x).reshape(B, nH, dH, H * W).permute(0, 1, 3, 2)  # (B, nH, HW, dH)

        # Attention: (B, nH, n_query, dH) @ (B, nH, dH, HW) -> (B, nH, n_query, HW)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Aggregate: (B, nH, n_query, HW) @ (B, nH, HW, dH) -> (B, nH, n_query, dH)
        out = torch.matmul(attn, v)  # (B, nH, n_query, dH)

        # Reshape to spatial: (B, nH, n_query, dH) -> (B, C, sqrt(n_query), sqrt(n_query))
        sq = int(self.n_query ** 0.5)
        out = out.permute(0, 2, 1, 3).reshape(B, sq, sq, C).permute(0, 3, 1, 2)  # (B, C, sq, sq)

        # Upsample to match spatial dims
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        # Apply as residual modulation (sigmoid gate for stability)
        gate = torch.sigmoid(self.out_proj(out))  # (B, C, H, W) in [0, 1]
        return x * gate


# ── [IMPROVEMENT 2] Multi-Scale Local Branch ──

class MultiScaleDepthwiseSepConv(nn.Module):
    """Parallel depthwise-separable convolutions at 3×3, 5×5, 7×7 scales.

    Captures local patterns at multiple spatial scales simultaneously.
    A learned 1×1 conv fuses the multi-scale features.

    Reference: Inception (Szegedy et al., 2015) meets MobileNet (Howard et al., 2017)
    """
    def __init__(self, channels, kernels=(3, 5, 7)):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            self.branches.append(nn.Sequential(
                nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            ))
        # Fuse multi-scale features
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(kernels), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(outs, dim=1))


# ── [IMPROVEMENT 6] Context-Conditioned Adaptive Gating ──

class AdaptiveGate(nn.Module):
    """Per-sample gating conditioned on context, replacing static nn.Parameter gate.

    Instead of a fixed spectral/local ratio, the gate weights are predicted
    from the context vector, allowing the model to dynamically adjust the
    spectral vs local balance based on the current atmospheric conditions.
    """
    def __init__(self, context_dim, n_branches=2):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, n_branches),
        )
        # Initialize to uniform
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(self, context):
        """context: (B, D) -> gate weights: (B, n_branches)"""
        return F.softmax(self.gate_mlp(context), dim=-1)


# ── SCANet v2 Block ──

class SCANetV2Block(nn.Module):
    """Improved SCANet block with:
    - Multi-head spatial cross-attention (improvement 1)
    - Multi-scale local branch (improvement 2)
    - Context-conditioned adaptive gate (improvement 6)
    """
    def __init__(self, channels, modes, padding, context_dim, n_heads=4,
                 local_kernels=(3, 5, 7), dropout=0.1):
        super().__init__()
        # Spectral branch (AFNO-style: spectral + channel mixer)
        self.spectral = SpectralConv2d(channels, channels, modes, modes, padding)
        self.channel_mixer = nn.Conv2d(channels, channels, 1)
        # Multi-scale local branch (improvement 2)
        self.local_conv = MultiScaleDepthwiseSepConv(channels, local_kernels)
        # Context-conditioned adaptive gate (improvement 6)
        self.gate = AdaptiveGate(context_dim, n_branches=2)
        # Multi-head spatial cross-attention (improvement 1)
        self.cross_attn = MultiHeadSpatialCrossAttention(context_dim, channels, n_heads)
        # Normalization + regularization
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x, context):
        # Context-conditioned gate (per-sample)
        g = self.gate(context)  # (B, 2)

        spectral_out = self.channel_mixer(self.spectral(x))
        local_out = self.local_conv(x)

        # Per-sample weighted fusion
        g_s = g[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        g_l = g[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out = g_s * spectral_out + g_l * local_out

        # Multi-head spatial cross-attention
        out = self.cross_attn(out, context)
        out = self.norm(out)
        out = self.dropout(F.gelu(out))
        return out + x  # residual


# ── SCANet v2 Model ──

class SCANetV2(nn.Module):
    """SCANet v2 — Improved Spectral Cross-Attention Network.

    Key changes from v1:
    - Task-specific context: separate dir_context and int_context (improvement 3)
    - Multi-head spatial cross-attention (improvement 1)
    - Multi-scale local branch (improvement 2)
    - Context-conditioned adaptive gating (improvement 6)
    - Extended physics head: 7 channels (improvement 4)
    """
    def __init__(self, in_channels=15, hidden_ch=64, n_modes=12, n_blocks=3,
                 padding=13, n_dir_classes=8, n_int_classes=4,
                 env_dim=40, d1d_dim=4, time_dim=6, context_dim=32,
                 use_env=True, use_1d=True,
                 head_dim=192, dropout=0.2,
                 n_attn_heads=4, local_kernels=(3, 5, 7),
                 n_physics_ch=7):
        super().__init__()
        self.use_env, self.use_1d = use_env, use_1d

        # Stage 0: Context encoders
        # Shared base context (for backbone blocks)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, context_dim), nn.GELU(),
            nn.Linear(context_dim, context_dim))
        self.env_mlp = nn.Sequential(
            nn.Linear(env_dim, context_dim), nn.GELU(),
            nn.Linear(context_dim, context_dim))
        self.d1d_mlp = nn.Sequential(
            nn.Linear(d1d_dim, context_dim), nn.GELU(),
            nn.Linear(context_dim, context_dim))

        # [IMPROVEMENT 3] Task-specific context projections
        # Shared context → task-specific context for each head
        self.dir_context_proj = nn.Sequential(
            nn.Linear(context_dim, context_dim), nn.GELU(),
            nn.Linear(context_dim, context_dim))
        self.int_context_proj = nn.Sequential(
            nn.Linear(context_dim, context_dim), nn.GELU(),
            nn.Linear(context_dim, context_dim))

        # Stage 1: Lifting
        self.lifting = nn.Sequential(nn.Conv2d(in_channels, hidden_ch, 1), nn.GELU())

        # Stage 2: SCANet v2 blocks (use shared context for backbone)
        self.blocks = nn.ModuleList([
            SCANetV2Block(hidden_ch, n_modes, padding, context_dim,
                          n_heads=n_attn_heads, local_kernels=local_kernels,
                          dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Stage 3: Projection + pooling
        self.projection = nn.Sequential(nn.Conv2d(hidden_ch, hidden_ch, 1), nn.GELU())

        # Stage 4: Classification heads with task-specific context (improvement 3)
        fuse_dim = hidden_ch + context_dim
        self.head_dir = nn.Sequential(
            nn.Linear(fuse_dim, head_dim), nn.GELU(), nn.Dropout(dropout * 2),
            nn.Linear(head_dim, head_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_dim // 2, n_dir_classes))
        self.head_int = nn.Sequential(
            nn.Linear(fuse_dim, head_dim), nn.GELU(), nn.Dropout(dropout * 2),
            nn.Linear(head_dim, head_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_dim // 2, n_int_classes))

        # Stage 5: Extended physics reconstruction head (improvement 4)
        # 7 channels: vorticity_850, divergence_850, u_shear, v_shear,
        #             potential_vorticity_850, thermal_wind_u, thermal_wind_v
        self.physics_head = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch // 2), nn.GELU(),
            nn.Conv2d(hidden_ch // 2, n_physics_ch, 1))

    def forward(self, grid, env=None, d1d=None, time_feat=None):
        # Context encoding (additive fusion — shared backbone context)
        context = self.time_mlp(time_feat)
        if self.use_env and env is not None:
            context = context + self.env_mlp(env)
        if self.use_1d and d1d is not None:
            context = context + self.d1d_mlp(d1d)

        # [IMPROVEMENT 3] Task-specific context projections
        dir_context = self.dir_context_proj(context)
        int_context = self.int_context_proj(context)

        # Lifting
        x = self.lifting(grid)

        # SCANet v2 blocks (shared context)
        for blk in self.blocks:
            x = blk(x, context)

        # Projection
        feat_map = self.projection(x)

        # Physics prediction (from spatial feature map)
        physics_pred = self.physics_head(feat_map)

        # Global average pooling
        pooled = feat_map.mean(dim=(-2, -1))  # (B, hidden_ch)

        # Task-specific context fusion (improvement 3)
        dir_fused = torch.cat([pooled, dir_context], dim=-1)
        int_fused = torch.cat([pooled, int_context], dim=-1)

        # Classification with task-specific features
        dir_logits = self.head_dir(dir_fused)
        int_logits = self.head_int(int_fused)

        return dir_logits, int_logits, physics_pred


# ── [IMPROVEMENT 4] Extended Physics Loss ──

class ExtendedPhysicsLoss(nn.Module):
    """Extended physics loss with 7 target channels (was 4 in v1).

    Original targets (v1):
      [vorticity_850, divergence_850, u_shear_200-850, v_shear_200-850]

    New targets (v2):
      + potential_vorticity_850: PV ≈ (f + ζ) * dθ/dp, approximated as vort * (z_500 - z_850)
      + thermal_wind_u: u_500 - u_850 (mid-level shear, captures steering flow)
      + thermal_wind_v: v_500 - v_850

    These additional physics quantities are basin-invariant and should improve
    cross-basin transfer by forcing the backbone to encode more universal features.

    References:
      - Hoskins et al. (1985): PV thinking for atmospheric dynamics
      - DeMaria & Kaplan (1994): mid-level shear as TC intensity predictor
    """
    def __init__(self, dx=0.25):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float32) / (8.0 * dx)
        sobel_y = torch.tensor([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]], dtype=torch.float32) / (8.0 * dx)
        self.register_buffer('kx', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('ky', sobel_y.view(1, 1, 3, 3))

    def _grad_x(self, field):
        return F.conv2d(field, self.kx, padding=1)

    def _grad_y(self, field):
        return F.conv2d(field, self.ky, padding=1)

    def compute_targets(self, grid):
        """Compute 7 physics targets from input grid channels.
        Returns (B, 7, H, W):
          [0] vorticity_850
          [1] divergence_850
          [2] u_shear (200-850)
          [3] v_shear (200-850)
          [4] potential_vorticity_850 (approx)
          [5] thermal_wind_u (500-850)
          [6] thermal_wind_v (500-850)
        """
        u_200 = grid[:, CH_U[IDX_200]:CH_U[IDX_200] + 1]
        u_500 = grid[:, CH_U[IDX_500]:CH_U[IDX_500] + 1]
        u_850 = grid[:, CH_U[IDX_850]:CH_U[IDX_850] + 1]
        v_200 = grid[:, CH_V[IDX_200]:CH_V[IDX_200] + 1]
        v_500 = grid[:, CH_V[IDX_500]:CH_V[IDX_500] + 1]
        v_850 = grid[:, CH_V[IDX_850]:CH_V[IDX_850] + 1]
        z_500 = grid[:, CH_Z[IDX_500]:CH_Z[IDX_500] + 1]
        z_850 = grid[:, CH_Z[IDX_850]:CH_Z[IDX_850] + 1]

        # Original 4 targets
        vort = self._grad_x(v_850) - self._grad_y(u_850)
        div  = self._grad_x(u_850) + self._grad_y(v_850)
        u_shear = u_200 - u_850
        v_shear = v_200 - v_850

        # New target 5: Approximate potential vorticity at 850 hPa
        # PV ≈ (f + ζ) * ∂θ/∂p ≈ vort * stability
        # We approximate stability as (z_500 - z_850) which is proportional to layer thickness
        # and hence static stability. Normalize to keep scale manageable.
        stability = z_500 - z_850  # thickness ∝ mean virtual temperature
        pv_approx = vort * stability

        # New targets 6-7: Thermal wind (mid-level shear 500-850)
        # This captures the steering flow that determines TC track direction
        tw_u = u_500 - u_850
        tw_v = v_500 - v_850

        return torch.cat([vort, div, u_shear, v_shear, pv_approx, tw_u, tw_v], dim=1)

    def forward(self, grid, physics_pred):
        targets = self.compute_targets(grid).detach()
        return F.mse_loss(physics_pred, targets)


# ── Sanity check ──
_test_model = SCANetV2(
    in_channels=IN_CHANNELS, hidden_ch=HIDDEN_CH, n_modes=N_MODES,
    n_blocks=N_BLOCKS, padding=PADDING, context_dim=CONTEXT_DIM,
    head_dim=HEAD_DIM, n_attn_heads=N_ATTN_HEADS,
    local_kernels=tuple(LOCAL_KERNELS), n_physics_ch=N_PHYSICS_CH
).to(DEVICE)
n_params_v2 = sum(p.numel() for p in _test_model.parameters())
print(f"\nSCANet v2: {n_params_v2:,} params")
with torch.no_grad():
    d, i, p = _test_model(
        torch.randn(2, 15, 81, 81, device=DEVICE),
        torch.randn(2, 40, device=DEVICE),
        torch.randn(2, 4, device=DEVICE),
        torch.randn(2, 6, device=DEVICE))
    print(f"Output: dir={d.shape}, int={i.shape}, physics={p.shape}")
del _test_model; torch.cuda.empty_cache()


# ## Section 2b: Optuna Hyperparameter Optimisation

import optuna, gc
from optuna.trial import TrialState

_loader_train = loaders["wp_train"]
_loader_val   = loaders["wp_val"]
_dir_weights  = dir_weights.to(DEVICE)
_int_weights  = int_weights.to(DEVICE)
_physics_fn   = ExtendedPhysicsLoss().to(DEVICE)

OPTUNA_EPOCHS   = 30
OPTUNA_PATIENCE = 8
N_TRIALS        = 60


def _cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def objective(trial):
    # ── Architecture ──
    hidden_ch   = trial.suggest_int("hidden_ch", 48, 96, step=16)
    n_modes     = trial.suggest_int("n_modes", 8, 20, step=4)
    n_blocks    = trial.suggest_int("n_blocks", 2, 4)
    padding     = trial.suggest_int("padding", 5, 13, step=4)
    context_dim = trial.suggest_int("context_dim", 32, 96, step=32)
    head_dim    = trial.suggest_int("head_dim", 64, 256, step=64)

    # ── v2 specific ──
    n_attn_heads = trial.suggest_int("n_attn_heads", 2, 8, step=2)

    # ── Regularisation ──
    dropout      = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
    label_smooth = trial.suggest_float("label_smoothing", 0.0, 0.15, step=0.05)
    dir_weight   = trial.suggest_float("dir_weight", 0.3, 0.7, step=0.1)
    lambda_phys  = trial.suggest_float("lambda_phys", 0.01, 0.3, log=True)

    # ── Optimiser ──
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-4, 5e-3, log=True)

    # ── Scheduler ──
    sched_name = trial.suggest_categorical("scheduler", ["cosine", "onecycle"])

    # ── Ensure hidden_ch divisible by n_attn_heads ──
    if hidden_ch % n_attn_heads != 0:
        raise optuna.TrialPruned()

    # ── Build model ──
    torch.manual_seed(SEED)
    m = SCANetV2(
        in_channels=IN_CHANNELS, hidden_ch=hidden_ch, n_modes=n_modes,
        n_blocks=n_blocks, padding=padding, context_dim=context_dim,
        env_dim=ENV_DIM, d1d_dim=D1D_DIM, time_dim=TIME_DIM,
        use_env=True, use_1d=True, head_dim=head_dim, dropout=dropout,
        n_attn_heads=n_attn_heads, local_kernels=tuple(LOCAL_KERNELS),
        n_physics_ch=N_PHYSICS_CH,
    )

    try:
        m = m.to(DEVICE)
    except torch.cuda.OutOfMemoryError:
        del m; _cleanup_gpu()
        raise optuna.TrialPruned()

    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)

    if sched_name == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr * 3, epochs=OPTUNA_EPOCHS,
            steps_per_epoch=len(_loader_train))
        step_per_batch = True
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=OPTUNA_EPOCHS, eta_min=1e-6)
        step_per_batch = False

    l_dir_fn = nn.CrossEntropyLoss(weight=_dir_weights, label_smoothing=label_smooth)
    l_int_fn = nn.CrossEntropyLoss(weight=_int_weights, label_smoothing=label_smooth)

    best_val_acc = 0.0
    patience_ctr = 0

    for epoch in range(OPTUNA_EPOCHS):
        # ── Train ──
        m.train()
        for grid, env, d1d, time_feat, dl, il in _loader_train:
            grid = grid.to(DEVICE); env = env.to(DEVICE)
            d1d = d1d.to(DEVICE); time_feat = time_feat.to(DEVICE)
            dl, il = dl.to(DEVICE), il.to(DEVICE)

            d_out, i_out, phys_pred = m(grid, env, d1d, time_feat)
            cls_loss = dir_weight * l_dir_fn(d_out, dl) + (1 - dir_weight) * l_int_fn(i_out, il)
            phys_loss = _physics_fn(grid, phys_pred)
            loss = cls_loss + lambda_phys * phys_loss

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if step_per_batch: sched.step()

        if not step_per_batch: sched.step()

        # ── Validate ──
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for grid, env, d1d, time_feat, dl, il in _loader_val:
                grid = grid.to(DEVICE); env = env.to(DEVICE)
                d1d = d1d.to(DEVICE); time_feat = time_feat.to(DEVICE)
                dl = dl.to(DEVICE)
                d_out, _, _ = m(grid, env, d1d, time_feat)
                correct += (d_out.argmax(1) == dl).sum().item()
                total += dl.size(0)

        val_acc = correct / total
        trial.report(val_acc, epoch)
        if trial.should_prune():
            del m, opt, sched; _cleanup_gpu()
            raise optuna.TrialPruned()

        if val_acc > best_val_acc:
            best_val_acc = val_acc; patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= OPTUNA_PATIENCE: break

    del m, opt, sched; _cleanup_gpu()
    return best_val_acc


# ── Run the study ──
pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
study = optuna.create_study(direction="maximize", pruner=pruner, study_name="scanetv2_hpo")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ── Summary ──
completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
pruned    = [t for t in study.trials if t.state == TrialState.PRUNED]
print(f"\nStudy complete: {len(completed)} completed, {len(pruned)} pruned, "
      f"{len(study.trials)} total")
print(f"Best trial #{study.best_trial.number}: val_dir_acc = {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# ── Optuna results visualization ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

vals = [t.value for t in completed]
best_so_far = [max(vals[:i+1]) for i in range(len(vals))]
axes[0].scatter(range(len(vals)), vals, alpha=0.4, s=20, label="Trial value")
axes[0].plot(best_so_far, color="red", linewidth=2, label="Best so far")
axes[0].set_xlabel("Trial"); axes[0].set_ylabel("Val Direction Accuracy")
axes[0].set_title("SCANet v2 — Optimisation History"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

importances = optuna.importance.get_param_importances(study)
top_params = list(importances.keys())[:8]
top_vals = [importances[k] for k in top_params]
axes[1].barh(top_params[::-1], top_vals[::-1], color="steelblue")
axes[1].set_xlabel("Importance"); axes[1].set_title("Hyperparameter Importance")
axes[1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(FIG_DIR / "scanetv2_optuna_results.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nTop 5 trials:")
top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
for t in top5:
    p = t.params
    print(f"  Trial {t.number:3d} | acc={t.value:.4f} | "
          f"hid={p['hidden_ch']} modes={p['n_modes']} blocks={p['n_blocks']} "
          f"pad={p['padding']} ctx={p['context_dim']} heads={p['n_attn_heads']} "
          f"lr={p['lr']:.4f} sched={p['scheduler']}")

# Save HPO results
import json as _json
hpo_results = {
    "best_params": study.best_params,
    "best_value": study.best_value,
    "n_trials": len(study.trials),
    "n_completed": len(completed),
    "n_pruned": len(pruned),
}
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
with open(PROJECT_ROOT / "logs" / "scanetv2_hpo.json", "w") as f:
    _json.dump(hpo_results, f, indent=2)
print("HPO results saved to logs/scanetv2_hpo.json")


# ── Apply best hyperparameters ──
bp = study.best_params

HIDDEN_CH      = bp["hidden_ch"]
N_MODES        = bp["n_modes"]
N_BLOCKS       = bp["n_blocks"]
PADDING        = bp["padding"]
CONTEXT_DIM    = bp["context_dim"]
HEAD_DIM       = bp["head_dim"]
N_ATTN_HEADS   = bp["n_attn_heads"]
DROPOUT        = bp["dropout"]
LABEL_SMOOTH   = bp["label_smoothing"]
DIR_WEIGHT     = bp["dir_weight"]
LAMBDA_PHYS    = bp["lambda_phys"]
LR             = bp["lr"]
WEIGHT_DECAY   = bp["weight_decay"]
BEST_SCHEDULER = bp["scheduler"]

print("\nBest hyperparameters applied:")
print(f"  Architecture : hid={HIDDEN_CH}, modes={N_MODES}, blocks={N_BLOCKS}, "
      f"pad={PADDING}, ctx={CONTEXT_DIM}, head={HEAD_DIM}, attn_heads={N_ATTN_HEADS}")
print(f"  Optimiser    : AdamW, lr={LR:.5f}, wd={WEIGHT_DECAY:.5f}")
print(f"  Scheduler    : {BEST_SCHEDULER}")
print(f"  Regularisation: dropout={DROPOUT}, label_smooth={LABEL_SMOOTH}, "
      f"dir_weight={DIR_WEIGHT}, lambda_phys={LAMBDA_PHYS:.4f}")

# ── Rebuild model with best params ──
torch.manual_seed(SEED)
model = SCANetV2(
    in_channels=IN_CHANNELS, hidden_ch=HIDDEN_CH, n_modes=N_MODES,
    n_blocks=N_BLOCKS, padding=PADDING, context_dim=CONTEXT_DIM,
    env_dim=ENV_DIM, d1d_dim=D1D_DIM, time_dim=TIME_DIM,
    use_env=USE_ENV, use_1d=USE_1D,
    head_dim=HEAD_DIM, dropout=DROPOUT,
    n_attn_heads=N_ATTN_HEADS, local_kernels=tuple(LOCAL_KERNELS),
    n_physics_ch=N_PHYSICS_CH).to(DEVICE)

physics_loss_fn = ExtendedPhysicsLoss().to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nSCANet v2 (tuned): {n_params:,} params")
print(f"Best Optuna val accuracy: {study.best_value:.4f}")

loss_dir_fn = nn.CrossEntropyLoss(weight=dir_weights.to(DEVICE), label_smoothing=LABEL_SMOOTH)
loss_int_fn = nn.CrossEntropyLoss(weight=int_weights.to(DEVICE), label_smoothing=LABEL_SMOOTH)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
if BEST_SCHEDULER == "onecycle":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 3, epochs=EPOCHS, steps_per_epoch=len(loaders["wp_train"]))
    SCHED_PER_BATCH = True
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)
    SCHED_PER_BATCH = False

print(f"Optimizer: AdamW (lr={LR:.5f}, wd={WEIGHT_DECAY:.5f})")
print(f"Scheduler: {BEST_SCHEDULER}")


# ## Section 3: Training

class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else: self.shadow[k] = v.clone()
    def apply(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
    def restore(self, model):
        model.load_state_dict(self.backup)


def cutout(grid, n_holes=2, hole_size=16):
    B, C, H, W = grid.shape
    mask = torch.ones_like(grid)
    for _ in range(n_holes):
        cy, cx = torch.randint(0, H, (B,)), torch.randint(0, W, (B,))
        for b in range(B):
            y1, y2 = max(0, cy[b] - hole_size // 2), min(H, cy[b] + hole_size // 2)
            x1, x2 = max(0, cx[b] - hole_size // 2), min(W, cx[b] + hole_size // 2)
            mask[b, :, y1:y2, x1:x2] = 0
    return grid * mask


def channel_dropout(grid, p=0.15):
    B, C, H, W = grid.shape
    mask = (torch.rand(B, C, 1, 1, device=grid.device) > p).float()
    if (mask.sum(1, keepdim=True) == 0).any():
        for b in range(B):
            if mask[b].sum() == 0: mask[b, torch.randint(0, C, (1,))] = 1.0
    return grid * mask


def mixup_data(x, y1, y2, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y1, y2, y1[idx], y2[idx], lam


def train_one_epoch(model, loader, optimizer, device, ema=None):
    model.train()
    m = {"loss": 0, "cls_loss": 0, "phys_loss": 0, "correct_d": 0, "correct_i": 0, "total": 0}

    for grid, env, d1d, tf, dl, il in loader:
        grid, env, d1d, tf = grid.to(device), env.to(device), d1d.to(device), tf.to(device)
        dl, il = dl.to(device), il.to(device)
        B = grid.size(0)

        # ── Augmentation ──
        if USE_CUTOUT and torch.rand(1).item() > 0.3:
            grid = cutout(grid, CUTOUT_N, CUTOUT_SIZE)
        if USE_NOISE:
            grid = grid + torch.randn_like(grid) * NOISE_STD
        if USE_CHAN_DROP and torch.rand(1).item() > 0.5:
            grid = channel_dropout(grid, CHAN_DROP_PROB)

        # ── Forward ──
        do_mixup = USE_MIXUP and torch.rand(1).item() > 0.5
        if do_mixup:
            grid_in, dl_a, il_a, dl_b, il_b, lam = mixup_data(grid, dl, il)
        else:
            grid_in = grid

        dir_logits, int_logits, phys_pred = model(grid_in, env, d1d, tf)

        # Classification loss
        if do_mixup:
            cls_loss = DIR_WEIGHT * (lam * loss_dir_fn(dir_logits, dl_a) +
                                     (1 - lam) * loss_dir_fn(dir_logits, dl_b)) + \
                       (1 - DIR_WEIGHT) * (lam * loss_int_fn(int_logits, il_a) +
                                           (1 - lam) * loss_int_fn(int_logits, il_b))
        else:
            cls_loss = DIR_WEIGHT * loss_dir_fn(dir_logits, dl) + \
                       (1 - DIR_WEIGHT) * loss_int_fn(int_logits, il)

        # Physics loss (use unmixed grid for clean targets)
        phys_loss = physics_loss_fn(grid, phys_pred)

        # Total loss
        loss = cls_loss + LAMBDA_PHYS * phys_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if SCHED_PER_BATCH: scheduler.step()
        if ema: ema.update(model)

        # Track metrics
        m["loss"]      += loss.item() * B
        m["cls_loss"]  += cls_loss.item() * B
        m["phys_loss"] += phys_loss.item() * B
        m["correct_d"] += (dir_logits.argmax(1) == dl).sum().item()
        m["correct_i"] += (int_logits.argmax(1) == il).sum().item()
        m["total"]     += B

    n = m["total"]
    return {
        "loss": m["loss"] / n, "cls_loss": m["cls_loss"] / n,
        "phys_loss": m["phys_loss"] / n,
        "dir_acc": m["correct_d"] / n, "int_acc": m["correct_i"] / n,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_loss = total = 0
    dp, dt, ip, it_ = [], [], [], []
    for grid, env, d1d, tf, dl, il in loader:
        grid, env, d1d, tf = grid.to(device), env.to(device), d1d.to(device), tf.to(device)
        dl, il = dl.to(device), il.to(device)
        outputs = model(grid, env, d1d, tf)
        do, io = outputs[0], outputs[1]
        loss = DIR_WEIGHT * loss_dir_fn(do, dl) + (1 - DIR_WEIGHT) * loss_int_fn(io, il)
        bs = grid.size(0)
        tot_loss += loss.item() * bs; total += bs
        dp.extend(do.argmax(1).cpu().tolist()); dt.extend(dl.cpu().tolist())
        ip.extend(io.argmax(1).cpu().tolist()); it_.extend(il.cpu().tolist())
    return {"loss": tot_loss / total,
            "dir_acc": accuracy_score(dt, dp), "int_acc": accuracy_score(it_, ip),
            "dir_f1": f1_score(dt, dp, average="macro", zero_division=0),
            "int_f1": f1_score(it_, ip, average="macro", zero_division=0),
            "dir_pred": dp, "dir_true": dt, "int_pred": ip, "int_true": it_}


# ── Training loop ──
history = {k: [] for k in ["train_loss", "val_loss", "train_dir_acc", "val_dir_acc",
                            "train_int_acc", "val_int_acc", "phys_loss"]}
best_val_dir_acc = 0.0
best_model_state = None
patience_counter = 0
ema = EMA(model, decay=0.998)

print(f"\n{'='*80}")
print(f" SCANet v2 Training — {EPOCHS} epochs, patience={PATIENCE}")
print(f" Physics loss: lambda={LAMBDA_PHYS} (7-channel extended)")
print(f" Improvements: spatial cross-attn, multi-scale local, adaptive gate,")
print(f"               task-specific context, extended physics, freeze-FT")
print(f"{'='*80}\n")

for epoch in range(1, EPOCHS + 1):
    tm = train_one_epoch(model, loaders["wp_train"], optimizer, DEVICE, ema=ema)
    if not SCHED_PER_BATCH: scheduler.step()
    ema.apply(model)
    vm = evaluate(model, loaders["wp_val"], DEVICE)
    ema.restore(model)

    history["train_loss"].append(tm["cls_loss"])
    history["val_loss"].append(vm["loss"])
    history["train_dir_acc"].append(tm["dir_acc"])
    history["val_dir_acc"].append(vm["dir_acc"])
    history["train_int_acc"].append(tm["int_acc"])
    history["val_int_acc"].append(vm["int_acc"])
    history["phys_loss"].append(tm["phys_loss"])

    if vm["dir_acc"] > best_val_dir_acc:
        best_val_dir_acc = vm["dir_acc"]
        best_model_state = deepcopy(dict(ema.shadow))
        patience_counter = 0; mk = " *"
    else:
        patience_counter += 1; mk = ""

    if epoch % 5 == 0 or epoch == 1 or mk:
        print(f"Ep {epoch:3d}/{EPOCHS} | cls={tm['cls_loss']:.4f} phys={tm['phys_loss']:.4f} | "
              f"V dir={vm['dir_acc']:.3f} int={vm['int_acc']:.3f}{mk}")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

model.load_state_dict(best_model_state)
torch.save(best_model_state, PROJECT_ROOT / "checkpoints" / "scanetv2_best_wp.pt")
print(f"\nBest val dir acc: {best_val_dir_acc:.4f}")


# ## Section 4: Training Curves

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (t, v, title) in zip(axes, [
    ("train_loss", "val_loss", "Loss"),
    ("train_dir_acc", "val_dir_acc", "Direction Accuracy"),
    ("train_int_acc", "val_int_acc", "Intensity Accuracy")]):
    ax.plot(history[t], label="Train"); ax.plot(history[v], label="Val")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
fig.suptitle("SCANet v2: Training Curves", fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "scanetv2_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# Physics loss curve
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history["phys_loss"])
ax.set_title("Extended Physics Reconstruction Loss (7-ch)"); ax.set_xlabel("Epoch")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "scanetv2_physics_loss.png", dpi=150, bbox_inches="tight")
plt.show()


# ## Section 5: Evaluation & Confusion Matrices

def plot_confusion_matrices(metrics, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, labels, cmap in [
        (axes[0], "dir", DIR_LABELS, "Blues"), (axes[1], "int", INTE_LABELS, "Oranges")]:
        cm = confusion_matrix(metrics[f"{key}_true"], metrics[f"{key}_pred"], labels=range(len(labels)))
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm_pct, annot=True, fmt=".0f", cmap=cmap,
                    xticklabels=labels, yticklabels=labels, cbar=False, ax=ax, vmin=0, vmax=100)
        ax.set_title(f"{title_prefix} — {key.title()} (acc={metrics[f'{key}_acc']:.1%})")
        ax.set_xlabel("Predicted")
    fig.tight_layout()
    return fig


wp_m = evaluate(model, loaders["wp_val"], DEVICE)
print(f"WP Val — dir={wp_m['dir_acc']:.3f} F1={wp_m['dir_f1']:.3f} | "
      f"int={wp_m['int_acc']:.3f} F1={wp_m['int_f1']:.3f}")
fig = plot_confusion_matrices(wp_m, "WP Val (SCANet v2)")
fig.savefig(FIG_DIR / "scanetv2_cm_wp.png", dpi=150, bbox_inches="tight")
plt.show()

sp_zs = evaluate(model, loaders["sp_test"], DEVICE)
print(f"SP Zero-Shot — dir={sp_zs['dir_acc']:.3f} F1={sp_zs['dir_f1']:.3f} | "
      f"int={sp_zs['int_acc']:.3f}")
print(f"Transfer gap (dir): {sp_zs['dir_acc'] - wp_m['dir_acc']:+.3f}")
fig = plot_confusion_matrices(sp_zs, "SP Zero-Shot (SCANet v2)")
fig.savefig(FIG_DIR / "scanetv2_cm_sp_zs.png", dpi=150, bbox_inches="tight")
plt.show()


# ## Section 6: Fine-Tuning (WP → SP)
# [IMPROVEMENT 5] Backbone freezing strategy

def finetune(model_state, ft_train_loader, ft_val_loader, freeze_backbone=False,
             lr=FT_LR, epochs=FT_EPOCHS, patience=FT_PATIENCE):
    """Fine-tune SCANet v2.

    Improvement 5: When freeze_backbone=True, freeze spectral+local branches
    but keep cross-attention Q projections and context projections trainable.
    This prevents the tiny SP set from corrupting basin-agnostic features
    while allowing task-specific adaptation.
    """
    ft = SCANetV2(
        in_channels=IN_CHANNELS, hidden_ch=HIDDEN_CH, n_modes=N_MODES,
        n_blocks=N_BLOCKS, padding=PADDING, context_dim=CONTEXT_DIM,
        env_dim=ENV_DIM, d1d_dim=D1D_DIM, time_dim=TIME_DIM,
        use_env=USE_ENV, use_1d=USE_1D,
        head_dim=HEAD_DIM, dropout=DROPOUT,
        n_attn_heads=N_ATTN_HEADS, local_kernels=tuple(LOCAL_KERNELS),
        n_physics_ch=N_PHYSICS_CH).to(DEVICE)
    ft.load_state_dict(model_state)

    if freeze_backbone:
        # Freeze everything first
        for p in ft.parameters():
            p.requires_grad = False
        # Unfreeze: classification heads
        for p in ft.head_dir.parameters():
            p.requires_grad = True
        for p in ft.head_int.parameters():
            p.requires_grad = True
        # Unfreeze: task-specific context projections (improvement 3)
        for p in ft.dir_context_proj.parameters():
            p.requires_grad = True
        for p in ft.int_context_proj.parameters():
            p.requires_grad = True
        # Unfreeze: cross-attention Q projections (allow spatial re-routing)
        for blk in ft.blocks:
            for p in blk.cross_attn.q_proj.parameters():
                p.requires_grad = True
            # Unfreeze: adaptive gate MLP (allow per-sample re-weighting)
            for p in blk.gate.parameters():
                p.requires_grad = True

        n_trainable = sum(p.numel() for p in ft.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in ft.parameters())
        print(f"  Freeze-FT: {n_trainable:,} / {n_total:,} params trainable "
              f"({100*n_trainable/n_total:.1f}%)")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, ft.parameters()),
                            lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_acc, best_st, pat = 0.0, None, 0
    hist = {"val_dir_acc": [], "val_loss": []}
    for ep in range(1, epochs + 1):
        ft.train()
        for grid, env, d1d, tf, dl, il in ft_train_loader:
            grid, env, d1d, tf, dl, il = [x.to(DEVICE) for x in [grid, env, d1d, tf, dl, il]]
            outputs = ft(grid, env, d1d, tf)
            do, io = outputs[0], outputs[1]
            loss = DIR_WEIGHT * loss_dir_fn(do, dl) + (1 - DIR_WEIGHT) * loss_int_fn(io, il)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ft.parameters(), 1.0); opt.step()
        sched.step()
        vm = evaluate(ft, ft_val_loader, DEVICE)
        hist["val_dir_acc"].append(vm["dir_acc"]); hist["val_loss"].append(vm["loss"])
        if vm["dir_acc"] > best_acc:
            best_acc = vm["dir_acc"]; best_st = deepcopy(ft.state_dict()); pat = 0
        else: pat += 1
        if ep % 10 == 0:
            print(f"  FT {ep:3d}/{epochs} | dir={vm['dir_acc']:.3f} int={vm['int_acc']:.3f}")
        if pat >= patience: print(f"  Early stop at ep {ep}"); break
    ft.load_state_dict(best_st)
    return ft, hist, best_acc


# Run both fine-tuning strategies
print("Strategy 1: Full fine-tuning")
ft_full, ft_full_h, ft_full_a = finetune(
    best_model_state, loaders["sp_ft_train"], loaders["sp_ft_val"])
print(f"  Best: {ft_full_a:.3f}")

print("\nStrategy 2: Backbone-frozen fine-tuning (improvement 5)")
ft_freeze, ft_freeze_h, ft_freeze_a = finetune(
    best_model_state, loaders["sp_ft_train"], loaders["sp_ft_val"], freeze_backbone=True)
print(f"  Best: {ft_freeze_a:.3f}")

print("\nStrategy 3: Head-only (+ cross-attn Q)")
ft_head, ft_head_h, ft_head_a = finetune(
    best_model_state, loaders["sp_ft_train"], loaders["sp_ft_val"], freeze_backbone=True)
print(f"  Best: {ft_head_a:.3f}")

# Select best FT strategy
ft_results = {
    "full": (ft_full, ft_full_a),
    "freeze-backbone": (ft_freeze, ft_freeze_a),
    "head-only": (ft_head, ft_head_a),
}
ft_strategy = max(ft_results, key=lambda k: ft_results[k][1])
ft_best = ft_results[ft_strategy][0]
torch.save(ft_best.state_dict(), PROJECT_ROOT / "checkpoints" / "scanetv2_best_ft.pt")

sp_ft = evaluate(ft_best, loaders["sp_test"], DEVICE)
print(f"\nSP Fine-Tuned ({ft_strategy}) — dir={sp_ft['dir_acc']:.3f} F1={sp_ft['dir_f1']:.3f} | "
      f"int={sp_ft['int_acc']:.3f}")
fig = plot_confusion_matrices(sp_ft, f"SP Fine-Tuned (SCANet v2, {ft_strategy})")
fig.savefig(FIG_DIR / "scanetv2_cm_sp_ft.png", dpi=150, bbox_inches="tight")
plt.show()


# ## Section 7: Gate Analysis (Adaptive Gate)

print("\n--- Adaptive Gate Analysis ---")
print("Per-sample gate weights vary with context. Showing mean gate weights on WP val:\n")

model.eval()
gate_accum = [[] for _ in range(N_BLOCKS)]

with torch.no_grad():
    for grid, env, d1d, tf, dl, il in loaders["wp_val"]:
        grid, env, d1d, tf = grid.to(DEVICE), env.to(DEVICE), d1d.to(DEVICE), tf.to(DEVICE)

        # Forward through context encoder
        context = model.time_mlp(tf)
        if model.use_env: context = context + model.env_mlp(env)
        if model.use_1d:  context = context + model.d1d_mlp(d1d)

        for i, blk in enumerate(model.blocks):
            g = blk.gate(context)  # (B, 2)
            gate_accum[i].append(g.cpu())

gate_means = []
gate_stds = []
for i in range(N_BLOCKS):
    all_gates = torch.cat(gate_accum[i], dim=0)  # (N, 2)
    mean_g = all_gates.mean(0).numpy()
    std_g = all_gates.std(0).numpy()
    gate_means.append(mean_g)
    gate_stds.append(std_g)
    print(f"  Layer {i+1}: Spectral={mean_g[0]:.3f}±{std_g[0]:.3f}, "
          f"Local={mean_g[1]:.3f}±{std_g[1]:.3f}")

gate_means = np.array(gate_means)
gate_stds = np.array(gate_stds)

fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(1, N_BLOCKS + 1)
w = 0.3
ax.bar(x_pos - w/2, gate_means[:, 0], w, yerr=gate_stds[:, 0],
       label="Spectral", color="#2196F3", capsize=4)
ax.bar(x_pos + w/2, gate_means[:, 1], w, yerr=gate_stds[:, 1],
       label="Local (Multi-Scale)", color="#4CAF50", capsize=4)
ax.set_xlabel("Layer"); ax.set_ylabel("Gate Weight (mean ± std)")
ax.set_title("SCANet v2 — Context-Conditioned Adaptive Gate Weights")
ax.set_xticks(x_pos); ax.legend(); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(FIG_DIR / "scanetv2_gate_weights.png", dpi=150, bbox_inches="tight")
plt.show()


# ## Section 8: Summary & Comparison with v1

print("\n" + "=" * 80)
print(" SCANet v2 — Final Results")
print("=" * 80)
print(f"{'Setting':20s} | {'Dir Acc':>8s} | {'Dir F1':>7s} | {'Int Acc':>8s} | {'Int F1':>7s}")
print("-" * 80)
for name, m_val in [("WP Val", wp_m), ("SP Zero-Shot", sp_zs),
                    (f"SP FT ({ft_strategy})", sp_ft)]:
    print(f"{name:20s} | {m_val['dir_acc']:>7.1%} | {m_val['dir_f1']:>6.1%} | "
          f"{m_val['int_acc']:>7.1%} | {m_val['int_f1']:>6.1%}")
print("=" * 80)
print(f"Parameters: {n_params:,}")
print(f"Transfer gap (dir): {sp_zs['dir_acc'] - wp_m['dir_acc']:+.3f}")
print(f"FT recovery (dir):  {sp_ft['dir_acc'] - sp_zs['dir_acc']:+.3f}")

print(f"\n--- Comparison with SCANet v1 ---")
print(f"{'Metric':25s} | {'v1':>10s} | {'v2':>10s} | {'Delta':>8s}")
print("-" * 60)
v1 = {"params": 3_704_726,
      "wp_dir": 0.5645, "wp_int": 0.6617, "wp_dir_f1": 0.3473,
      "sp_zs_dir": 0.4332, "sp_zs_int": 0.3678,
      "sp_ft_dir": 0.3815, "sp_ft_int": 0.4305,
      "transfer_gap": 0.4332 - 0.5645}
v2 = {"params": n_params,
      "wp_dir": wp_m["dir_acc"], "wp_int": wp_m["int_acc"], "wp_dir_f1": wp_m["dir_f1"],
      "sp_zs_dir": sp_zs["dir_acc"], "sp_zs_int": sp_zs["int_acc"],
      "sp_ft_dir": sp_ft["dir_acc"], "sp_ft_int": sp_ft["int_acc"],
      "transfer_gap": sp_zs["dir_acc"] - wp_m["dir_acc"]}

for label, k in [("Parameters", "params"), ("WP Dir Acc", "wp_dir"),
                 ("WP Dir F1", "wp_dir_f1"), ("WP Int Acc", "wp_int"),
                 ("SP ZS Dir Acc", "sp_zs_dir"), ("SP ZS Int Acc", "sp_zs_int"),
                 ("SP FT Dir Acc", "sp_ft_dir"), ("SP FT Int Acc", "sp_ft_int"),
                 ("Transfer Gap (dir)", "transfer_gap")]:
    v1v = v1[k]; v2v = v2[k]
    if k == "params":
        print(f"{label:25s} | {v1v:>10,} | {v2v:>10,} | {v2v-v1v:>+8,}")
    else:
        print(f"{label:25s} | {v1v:>9.1%} | {v2v:>9.1%} | {v2v-v1v:>+7.1%}")

# Save results
import json
results = {
    "model": "SCANet_v2",
    "params": n_params,
    "improvements": [
        "multi-head spatial cross-attention",
        "multi-scale local branch (3x3, 5x5, 7x7)",
        "task-specific context (dir vs int)",
        "extended physics loss (7 channels: +PV, +thermal wind)",
        "backbone freeze fine-tuning strategy",
        "context-conditioned adaptive gating",
    ],
    "wp_val": {"dir_acc": wp_m["dir_acc"], "dir_f1": wp_m["dir_f1"],
               "int_acc": wp_m["int_acc"], "int_f1": wp_m["int_f1"]},
    "sp_zeroshot": {"dir_acc": sp_zs["dir_acc"], "dir_f1": sp_zs["dir_f1"],
                    "int_acc": sp_zs["int_acc"], "int_f1": sp_zs["int_f1"]},
    "sp_finetuned": {"dir_acc": sp_ft["dir_acc"], "dir_f1": sp_ft["dir_f1"],
                     "int_acc": sp_ft["int_acc"], "int_f1": sp_ft["int_f1"]},
    "ft_strategy": ft_strategy,
    "ft_strategies_compared": {
        "full": ft_full_a,
        "freeze-backbone": ft_freeze_a,
        "head-only": ft_head_a,
    },
    "gate_weights_mean": gate_means.tolist(),
    "gate_weights_std": gate_stds.tolist(),
}
with open(PROJECT_ROOT / "logs" / "scanetv2_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to logs/scanetv2_results.json")
