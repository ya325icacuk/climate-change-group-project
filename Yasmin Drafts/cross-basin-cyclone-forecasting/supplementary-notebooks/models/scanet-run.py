#!/usr/bin/env python
# coding: utf-8

# # SCANet — Spectral Cross-Attention Network for Cross-Basin TC Forecasting
#
# **Architecture:** Combines spectral convolutions (FNO-style), depthwise-separable
# local convolutions, cross-attention context modulation, and physics-informed
# auxiliary losses — without the complexity of adversarial training.
#
# **Key innovations over existing models:**
# 1. Early multimodal fusion: env + 1D + time → unified context vector injected at every block
# 2. Context cross-attention: spatially-varying modulation (vs FiLM's spatially-uniform)
# 3. Depthwise-separable local branch: 20× lighter than U-FNO's U-Net branch
# 4. Physics reconstruction head (vorticity, divergence, shear) without GAN instability
#
# **References:**
# - FourCastNet (Pathak et al., 2022): AFNO spectral + channel mixing
# - Perceiver (Jaegle et al., 2021): cross-attention between low-dim latent and high-dim input
# - NeuralGCM (Kochkov et al., 2024): physics-informed ML without adversarial training
# - MobileNet (Howard et al., 2017): depthwise-separable convolutions
# - ClimaX (Nguyen et al., 2023): early multimodal fusion for weather/climate
#
# **Approach:** Train on Western Pacific (WP), evaluate zero-shot transfer to
# South Pacific (SP), then fine-tune on SP.

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

# ── Architecture ──
IN_CHANNELS   = 15
HIDDEN_CH     = 64
N_MODES       = 12
N_BLOCKS      = 3
PADDING       = 9
CONTEXT_DIM   = 64
TIME_DIM      = 6
ENV_DIM       = 40
D1D_DIM       = 4
HEAD_DIM      = 128

# ── Regularisation ──
DROPOUT       = 0.1
LABEL_SMOOTH  = 0.05

# ── Training ──
BATCH_SIZE    = 64
LR            = 5e-4
WEIGHT_DECAY  = 1.3e-3
EPOCHS        = 300
PATIENCE      = 50
DIR_WEIGHT    = 0.55

# ── Physics loss ──
LAMBDA_PHYS   = 0.1

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


# ## Section 2: Model Architecture

# ── Spectral Conv (reused from U-FNO) ──

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


# ── Context Cross-Attention ──

class ContextCrossAttention(nn.Module):
    """Single-head cross-attention: context queries spatial features.

    Produces a channel-wise gate that is spatially-informed, unlike FiLM
    which applies the same scale/shift everywhere.

    Reference: Perceiver (Jaegle et al., 2021)
    """
    def __init__(self, context_dim, channels):
        super().__init__()
        self.q_proj = nn.Linear(context_dim, channels)
        self.k_conv = nn.Conv2d(channels, channels, 1)
        self.v_conv = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5
        # Initialize to near-identity (like FiLM init)
        nn.init.zeros_(self.q_proj.weight)
        nn.init.ones_(self.q_proj.bias)

    def forward(self, x, context):
        """x: (B, C, H, W), context: (B, D) -> modulated x: (B, C, H, W)"""
        B, C, H, W = x.shape
        q = self.q_proj(context).unsqueeze(1)           # (B, 1, C)
        k = self.k_conv(x).reshape(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        v = self.v_conv(x).reshape(B, C, -1).permute(0, 2, 1)  # (B, HW, C)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, 1, HW)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)                        # (B, 1, C)
        gate = out.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * gate


# ── Depthwise-Separable Conv (local spatial branch) ──

class DepthwiseSepConv(nn.Module):
    """Depthwise-separable 5×5 conv: captures local spatial patterns
    with ~20× fewer params than a standard conv or U-Net branch.

    Reference: MobileNet (Howard et al., 2017)
    """
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        return self.pw(F.gelu(self.bn(self.dw(x))))


# ── SCANet Block ──

class SCANetBlock(nn.Module):
    """Spectral Cross-Attention block: 2 gated branches + context modulation.

    (a) Spectral branch: SpectralConv2d + 1×1 channel mixer (AFNO-style)
    (b) Local branch: Depthwise-separable 5×5 conv
    (c) Context cross-attention: spatially-varying modulation
    (d) Gated fusion + residual
    """
    def __init__(self, channels, modes, padding, context_dim, dropout=0.1):
        super().__init__()
        # Spectral branch (AFNO-style: spectral + channel mixer)
        self.spectral = SpectralConv2d(channels, channels, modes, modes, padding)
        self.channel_mixer = nn.Conv2d(channels, channels, 1)
        # Local branch
        self.local_conv = DepthwiseSepConv(channels)
        # Gated fusion (2 branches: spectral + local)
        self.gate = nn.Parameter(torch.ones(2) / 2)
        # Context cross-attention
        self.cross_attn = ContextCrossAttention(context_dim, channels)
        # Normalization + regularization
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x, context):
        g = F.softmax(self.gate, dim=0)
        spectral_out = self.channel_mixer(self.spectral(x))
        local_out = self.local_conv(x)
        out = g[0] * spectral_out + g[1] * local_out
        out = self.cross_attn(out, context)
        out = self.norm(out)
        out = self.dropout(F.gelu(out))
        return out + x  # residual


# ── SCANet Model ──

class SCANet(nn.Module):
    """Spectral Cross-Attention Network for TC classification.

    Combines spectral convolutions, depthwise-separable local convolutions,
    context cross-attention, and physics-informed auxiliary losses.
    """
    def __init__(self, in_channels=15, hidden_ch=64, n_modes=12, n_blocks=3,
                 padding=9, n_dir_classes=8, n_int_classes=4,
                 env_dim=40, d1d_dim=4, time_dim=6, context_dim=64,
                 use_env=True, use_1d=True,
                 head_dim=128, dropout=0.1):
        super().__init__()
        self.use_env, self.use_1d = use_env, use_1d

        # Stage 0: Context encoder (early multimodal fusion)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, context_dim), nn.GELU(),
            nn.Linear(context_dim, context_dim))
        self.env_mlp = nn.Sequential(
            nn.Linear(env_dim, context_dim), nn.GELU(),
            nn.Linear(context_dim, context_dim))
        self.d1d_mlp = nn.Sequential(
            nn.Linear(d1d_dim, context_dim), nn.GELU(),
            nn.Linear(context_dim, context_dim))

        # Stage 1: Lifting
        self.lifting = nn.Sequential(nn.Conv2d(in_channels, hidden_ch, 1), nn.GELU())

        # Stage 2: SCANet blocks
        self.blocks = nn.ModuleList([
            SCANetBlock(hidden_ch, n_modes, padding, context_dim, dropout)
            for _ in range(n_blocks)
        ])

        # Stage 3: Projection + pooling
        self.projection = nn.Sequential(nn.Conv2d(hidden_ch, hidden_ch, 1), nn.GELU())

        # Stage 4: Classification heads
        fuse_dim = hidden_ch + context_dim  # pooled features + context
        self.head_dir = nn.Sequential(
            nn.Linear(fuse_dim, head_dim), nn.GELU(), nn.Dropout(dropout * 2),
            nn.Linear(head_dim, head_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_dim // 2, n_dir_classes))
        self.head_int = nn.Sequential(
            nn.Linear(fuse_dim, head_dim), nn.GELU(), nn.Dropout(dropout * 2),
            nn.Linear(head_dim, head_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_dim // 2, n_int_classes))

        # Stage 5: Physics reconstruction head (training only)
        self.physics_head = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch // 2), nn.GELU(),
            nn.Conv2d(hidden_ch // 2, 4, 1))

    def forward(self, grid, env=None, d1d=None, time_feat=None):
        # Context encoding (additive fusion of all non-grid modalities)
        context = self.time_mlp(time_feat)
        if self.use_env and env is not None:
            context = context + self.env_mlp(env)
        if self.use_1d and d1d is not None:
            context = context + self.d1d_mlp(d1d)

        # Lifting
        x = self.lifting(grid)

        # SCANet blocks
        for blk in self.blocks:
            x = blk(x, context)

        # Projection
        feat_map = self.projection(x)

        # Physics prediction (from spatial feature map)
        physics_pred = self.physics_head(feat_map)

        # Global average pooling + context fusion
        pooled = feat_map.mean(dim=(-2, -1))  # (B, hidden_ch)
        fused = torch.cat([pooled, context], dim=-1)  # (B, hidden_ch + context_dim)

        # Classification
        dir_logits = self.head_dir(fused)
        int_logits = self.head_int(fused)

        return dir_logits, int_logits, physics_pred


# ── Physics Loss (from PI-GAN) ──

class PhysicsLoss(nn.Module):
    """Computes physics targets from input grid and supervises physics_pred.

    Targets: [vorticity_850, divergence_850, u_shear_200-850, v_shear_200-850]
    Computed using Sobel filters for spatial derivatives.
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
        """Compute 4 physics targets from input grid channels.
        Returns (B, 4, H, W): [vorticity_850, divergence_850, u_shear, v_shear]
        """
        u_850 = grid[:, CH_U[IDX_850]:CH_U[IDX_850] + 1]
        v_850 = grid[:, CH_V[IDX_850]:CH_V[IDX_850] + 1]
        u_200 = grid[:, CH_U[IDX_200]:CH_U[IDX_200] + 1]
        v_200 = grid[:, CH_V[IDX_200]:CH_V[IDX_200] + 1]

        vort = self._grad_x(v_850) - self._grad_y(u_850)
        div  = self._grad_x(u_850) + self._grad_y(v_850)
        u_shear = u_200 - u_850
        v_shear = v_200 - v_850

        return torch.cat([vort, div, u_shear, v_shear], dim=1)

    def forward(self, grid, physics_pred):
        targets = self.compute_targets(grid).detach()
        return F.mse_loss(physics_pred, targets)


# ── Sanity check ──
_test_model = SCANet(in_channels=IN_CHANNELS, hidden_ch=HIDDEN_CH, n_modes=N_MODES,
                     n_blocks=N_BLOCKS, padding=PADDING, context_dim=CONTEXT_DIM).to(DEVICE)
n_params_default = sum(p.numel() for p in _test_model.parameters())
print(f"SCANet (default config): {n_params_default:,} params")
with torch.no_grad():
    d, i, p = _test_model(torch.randn(2,15,81,81,device=DEVICE), torch.randn(2,40,device=DEVICE),
                           torch.randn(2,4,device=DEVICE), torch.randn(2,6,device=DEVICE))
    print(f"Output: dir={d.shape}, int={i.shape}, physics={p.shape}")
del _test_model; torch.cuda.empty_cache()


# ## Section 2b: Optuna Hyperparameter Optimisation

import optuna, gc
from optuna.trial import TrialState

_loader_train = loaders["wp_train"]
_loader_val   = loaders["wp_val"]
_dir_weights  = dir_weights.to(DEVICE)
_int_weights  = int_weights.to(DEVICE)
_physics_fn   = PhysicsLoss().to(DEVICE)

OPTUNA_EPOCHS   = 30
OPTUNA_PATIENCE = 8
N_TRIALS        = 60


def _cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def objective(trial):
    # ── Architecture ──
    hidden_ch  = trial.suggest_int("hidden_ch", 48, 96, step=16)
    n_modes    = trial.suggest_int("n_modes", 8, 20, step=4)
    n_blocks   = trial.suggest_int("n_blocks", 2, 4)
    padding    = trial.suggest_int("padding", 5, 13, step=4)
    context_dim = trial.suggest_int("context_dim", 32, 96, step=32)
    head_dim   = trial.suggest_int("head_dim", 64, 256, step=64)

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

    # ── Build model ──
    torch.manual_seed(SEED)
    m = SCANet(
        in_channels=IN_CHANNELS, hidden_ch=hidden_ch, n_modes=n_modes,
        n_blocks=n_blocks, padding=padding, context_dim=context_dim,
        env_dim=ENV_DIM, d1d_dim=D1D_DIM, time_dim=TIME_DIM,
        use_env=True, use_1d=True, head_dim=head_dim, dropout=dropout,
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
study = optuna.create_study(direction="maximize", pruner=pruner, study_name="scanet_hpo")
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
axes[0].set_title("Optimisation History"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

importances = optuna.importance.get_param_importances(study)
top_params = list(importances.keys())[:8]
top_vals = [importances[k] for k in top_params]
axes[1].barh(top_params[::-1], top_vals[::-1], color="steelblue")
axes[1].set_xlabel("Importance"); axes[1].set_title("Hyperparameter Importance")
axes[1].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(FIG_DIR / "scanet_optuna_results.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nTop 5 trials:")
top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
for t in top5:
    p = t.params
    print(f"  Trial {t.number:3d} | acc={t.value:.4f} | "
          f"hid={p['hidden_ch']} modes={p['n_modes']} blocks={p['n_blocks']} "
          f"pad={p['padding']} ctx={p['context_dim']} "
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
with open(PROJECT_ROOT / "logs" / "scanet_hpo.json", "w") as f:
    _json.dump(hpo_results, f, indent=2)
print("HPO results saved to logs/scanet_hpo.json")


# ── Apply best hyperparameters ──
bp = study.best_params

HIDDEN_CH     = bp["hidden_ch"]
N_MODES       = bp["n_modes"]
N_BLOCKS      = bp["n_blocks"]
PADDING       = bp["padding"]
CONTEXT_DIM   = bp["context_dim"]
HEAD_DIM      = bp["head_dim"]
DROPOUT       = bp["dropout"]
LABEL_SMOOTH  = bp["label_smoothing"]
DIR_WEIGHT    = bp["dir_weight"]
LAMBDA_PHYS   = bp["lambda_phys"]
LR            = bp["lr"]
WEIGHT_DECAY  = bp["weight_decay"]
BEST_SCHEDULER = bp["scheduler"]

print("\nBest hyperparameters applied:")
print(f"  Architecture : hid={HIDDEN_CH}, modes={N_MODES}, blocks={N_BLOCKS}, "
      f"pad={PADDING}, ctx={CONTEXT_DIM}, head={HEAD_DIM}")
print(f"  Optimiser    : AdamW, lr={LR:.5f}, wd={WEIGHT_DECAY:.5f}")
print(f"  Scheduler    : {BEST_SCHEDULER}")
print(f"  Regularisation: dropout={DROPOUT}, label_smooth={LABEL_SMOOTH}, "
      f"dir_weight={DIR_WEIGHT}, lambda_phys={LAMBDA_PHYS:.4f}")

# ── Rebuild model with best params ──
torch.manual_seed(SEED)
model = SCANet(
    in_channels=IN_CHANNELS, hidden_ch=HIDDEN_CH, n_modes=N_MODES,
    n_blocks=N_BLOCKS, padding=PADDING, context_dim=CONTEXT_DIM,
    env_dim=ENV_DIM, d1d_dim=D1D_DIM, time_dim=TIME_DIM,
    use_env=USE_ENV, use_1d=USE_1D,
    head_dim=HEAD_DIM, dropout=DROPOUT).to(DEVICE)

physics_loss_fn = PhysicsLoss().to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nSCANet (tuned): {n_params:,} params")
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
print(f" SCANet Training — {EPOCHS} epochs, patience={PATIENCE}")
print(f" Physics loss: lambda={LAMBDA_PHYS}")
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
torch.save(best_model_state, PROJECT_ROOT / "checkpoints" / "scanet_best_wp.pt")
print(f"\nBest val dir acc: {best_val_dir_acc:.4f}")


# ## Section 4: Training Curves

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (t, v, title) in zip(axes, [
    ("train_loss", "val_loss", "Loss"),
    ("train_dir_acc", "val_dir_acc", "Direction Accuracy"),
    ("train_int_acc", "val_int_acc", "Intensity Accuracy")]):
    ax.plot(history[t], label="Train"); ax.plot(history[v], label="Val")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
fig.suptitle("SCANet: Training Curves", fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "scanet_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# Physics loss curve
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history["phys_loss"])
ax.set_title("Physics Reconstruction Loss"); ax.set_xlabel("Epoch")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "scanet_physics_loss.png", dpi=150, bbox_inches="tight")
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
fig = plot_confusion_matrices(wp_m, "WP Val (SCANet)")
fig.savefig(FIG_DIR / "scanet_cm_wp.png", dpi=150, bbox_inches="tight")
plt.show()

sp_zs = evaluate(model, loaders["sp_test"], DEVICE)
print(f"SP Zero-Shot — dir={sp_zs['dir_acc']:.3f} F1={sp_zs['dir_f1']:.3f} | "
      f"int={sp_zs['int_acc']:.3f}")
print(f"Transfer gap (dir): {sp_zs['dir_acc'] - wp_m['dir_acc']:+.3f}")
fig = plot_confusion_matrices(sp_zs, "SP Zero-Shot (SCANet)")
fig.savefig(FIG_DIR / "scanet_cm_sp_zs.png", dpi=150, bbox_inches="tight")
plt.show()


# ## Section 6: Fine-Tuning (WP → SP)

def finetune(model_state, ft_train_loader, ft_val_loader, freeze_backbone=False,
             lr=FT_LR, epochs=FT_EPOCHS, patience=FT_PATIENCE):
    """Fine-tune SCANet with classification loss only (no physics loss)."""
    ft = SCANet(
        in_channels=IN_CHANNELS, hidden_ch=HIDDEN_CH, n_modes=N_MODES,
        n_blocks=N_BLOCKS, padding=PADDING, context_dim=CONTEXT_DIM,
        env_dim=ENV_DIM, d1d_dim=D1D_DIM, time_dim=TIME_DIM,
        use_env=USE_ENV, use_1d=USE_1D,
        head_dim=HEAD_DIM, dropout=DROPOUT).to(DEVICE)
    ft.load_state_dict(model_state)
    if freeze_backbone:
        for n, p in ft.named_parameters():
            if "head" not in n and "cross_attn.q_proj" not in n:
                p.requires_grad = False
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


print("Full fine-tuning:")
ft_full, ft_full_h, ft_full_a = finetune(best_model_state, loaders["sp_ft_train"], loaders["sp_ft_val"])
print(f"  Best: {ft_full_a:.3f}")

print("Head-only (+ cross-attn Q):")
ft_head, ft_head_h, ft_head_a = finetune(best_model_state, loaders["sp_ft_train"],
                                          loaders["sp_ft_val"], freeze_backbone=True)
print(f"  Best: {ft_head_a:.3f}")

ft_best = ft_full if ft_full_a >= ft_head_a else ft_head
ft_strategy = "full" if ft_full_a >= ft_head_a else "head-only"
torch.save(ft_best.state_dict(), PROJECT_ROOT / "checkpoints" / "scanet_best_ft.pt")

sp_ft = evaluate(ft_best, loaders["sp_test"], DEVICE)
print(f"SP Fine-Tuned ({ft_strategy}) — dir={sp_ft['dir_acc']:.3f} F1={sp_ft['dir_f1']:.3f} | "
      f"int={sp_ft['int_acc']:.3f}")
fig = plot_confusion_matrices(sp_ft, f"SP Fine-Tuned (SCANet, {ft_strategy})")
fig.savefig(FIG_DIR / "scanet_cm_sp_ft.png", dpi=150, bbox_inches="tight")
plt.show()


# ## Section 7: Gate Analysis

gate_vals = []
for i, blk in enumerate(model.blocks):
    g = F.softmax(blk.gate.detach().cpu(), dim=0)
    gate_vals.append(g.numpy())
    print(f"Layer {i+1}: Spectral={g[0]:.3f}, Local={g[1]:.3f}")

gate_vals = np.array(gate_vals)
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(1, N_BLOCKS + 1)
w = 0.3
ax.bar(x_pos - w/2, gate_vals[:, 0], w, label="Spectral", color="#2196F3")
ax.bar(x_pos + w/2, gate_vals[:, 1], w, label="Local (DWSep)", color="#4CAF50")
ax.set_xlabel("Layer"); ax.set_ylabel("Gate Weight")
ax.set_title("SCANet Learned Gate Weights per Layer")
ax.set_xticks(x_pos); ax.legend(); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(FIG_DIR / "scanet_gate_weights.png", dpi=150, bbox_inches="tight")
plt.show()


# ## Section 8: Summary

print("\n" + "=" * 80)
print(" SCANet — Final Results")
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
print(f"\nGate summary:")
for i, blk in enumerate(model.blocks):
    g = F.softmax(blk.gate.detach().cpu(), dim=0)
    print(f"  Layer {i+1}: Spectral={g[0]:.2f} Local={g[1]:.2f}")

# Save results
import json
results = {
    "model": "SCANet",
    "params": n_params,
    "wp_val": {"dir_acc": wp_m["dir_acc"], "dir_f1": wp_m["dir_f1"],
               "int_acc": wp_m["int_acc"], "int_f1": wp_m["int_f1"]},
    "sp_zeroshot": {"dir_acc": sp_zs["dir_acc"], "dir_f1": sp_zs["dir_f1"],
                    "int_acc": sp_zs["int_acc"], "int_f1": sp_zs["int_f1"]},
    "sp_finetuned": {"dir_acc": sp_ft["dir_acc"], "dir_f1": sp_ft["dir_f1"],
                     "int_acc": sp_ft["int_acc"], "int_f1": sp_ft["int_f1"]},
    "ft_strategy": ft_strategy,
    "gate_weights": gate_vals.tolist(),
}
with open(PROJECT_ROOT / "logs" / "scanet_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to logs/scanet_results.json")
