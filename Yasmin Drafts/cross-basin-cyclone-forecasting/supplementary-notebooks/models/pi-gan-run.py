#!/usr/bin/env python
# coding: utf-8

# # Physics-Informed GAN (PI-GAN) for Tropical Cyclone Forecasting
#
# **Task:** Predict cyclone movement direction (8 classes) and intensity change (4 classes)
# from multimodal inputs — 3D atmospheric grids, environmental features, and 1D track data.
#
# **Architecture:** Generator uses U-Net+FiLM backbone with an auxiliary physics
# reconstruction head. Discriminator is a conditional MLP on fused features + label
# embeddings. WGAN-GP for stable adversarial training.
#
# **Physics constraints:** Vorticity consistency, mass continuity (divergence),
# and vertical wind shear — computed from input grid channels and supervised via
# the generator's physics reconstruction head.
#
# **Reference:** Ruttgers et al. (2019), "Prediction of a typhoon track using a
# generative adversarial network and satellite images", Scientific Reports.
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

# ── Generator Architecture (U-Net+FiLM backbone) ──
N_LEVELS      = 4
BASE_CHANNELS = 32       # data-limited regime (~1800 samples) — large models overfit
HEAD_DIM      = 256
IN_CHANNELS   = 15
TIME_DIM      = 6
TIME_EMB_DIM  = 64

# ── Discriminator ──
DISC_HIDDEN   = 128
DISC_LAYERS   = 2
LABEL_EMB_DIM = 16

# ── Regularisation ──
DROPOUT       = 0.2
DROP_PATH     = 0.0
LABEL_SMOOTH  = 0.05

# ── Training ──
BATCH_SIZE    = 64
G_LR          = 5e-4
D_LR          = 2e-4
WEIGHT_DECAY  = 1.3e-3
EPOCHS        = 300
PATIENCE      = 50
DIR_WEIGHT    = 0.55

# ── GAN-specific ──
LAMBDA_ADV    = 0.01     # very small — adversarial is auxiliary, classification is primary
LAMBDA_PHYS   = 0.1      # physics regularization
GP_WEIGHT     = 10.0
ADV_WARMUP    = 20       # start adversarial loss after this many epochs

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
CH_V = [2, 5, 8, 11]    # v wind at 200, 500, 850, 925 hPa
CH_Z = [3, 6, 9, 12]    # geopotential at 200, 500, 850, 925 hPa
CH_SST   = 0
CH_SHEAR = 13
CH_VORT  = 14
# Indices into the pressure-level arrays: 200=0, 500=1, 850=2, 925=3
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

# ── Building blocks (shared with U-Net+FiLM) ──

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation. Identity-init (gamma=1, beta=0)."""
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.fc = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.fc.bias.data[:channels] = 1.0

    def forward(self, x, cond):
        gamma, beta = self.fc(cond).chunk(2, dim=1)
        return gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0.0: return x
        keep = torch.rand(x.size(0), 1, 1, 1, device=x.device) > self.p
        return x * keep / (1 - self.p)


class ConvBlock(nn.Module):
    """Double conv with residual + FiLM after second BN."""
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.drop2d = nn.Dropout2d(dropout)
        self.film = FiLMLayer(cond_dim, out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.drop_path = DropPath(drop_path)

    def forward(self, x, cond):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop2d(h)
        h = self.bn2(self.conv2(h))
        h = self.film(h, cond)
        h = self.act(h)
        return self.drop_path(h) + self.residual(x)


class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // r, 4)), nn.GELU(),
            nn.Linear(max(ch // r, 4), ch), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, cond_dim, dropout, drop_path)
        self.se = SEBlock(out_ch)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
    def forward(self, x, cond):
        skip = self.se(self.conv(x, cond))
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, cond_dim, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, cond_dim, dropout, drop_path)
    def forward(self, x, skip, cond):
        x = self.up(x)
        dh, dw = x.size(2) - skip.size(2), x.size(3) - skip.size(3)
        if dh > 0 or dw > 0:
            x = x[:, :, dh // 2:dh // 2 + skip.size(2), dw // 2:dw // 2 + skip.size(3)]
        elif dh < 0 or dw < 0:
            x = F.pad(x, [0, -dw, 0, -dh])
        return self.conv(torch.cat([x, skip], 1), cond)


# ── Generator ──

class PIGANGenerator(nn.Module):
    """U-Net+FiLM generator with auxiliary physics reconstruction head."""
    def __init__(self, in_channels=15, base_channels=32, n_levels=4,
                 n_dir_classes=8, n_int_classes=4,
                 env_dim=40, d1d_dim=4, time_dim=6, time_emb_dim=64,
                 use_env=True, use_1d=True,
                 dropout=0.0, head_dim=128, drop_path=0.0):
        super().__init__()
        self.use_env, self.use_1d = use_env, use_1d

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_emb_dim), nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim))

        dp = [drop_path * i / max(n_levels, 1) for i in range(n_levels + 1)]

        self.encoders = nn.ModuleList()
        ch = in_channels
        for i in range(n_levels):
            out = base_channels * (2 ** i)
            self.encoders.append(EncoderBlock(ch, out, time_emb_dim, dropout, dp[i]))
            ch = out

        bneck = base_channels * (2 ** n_levels)
        self.bottleneck = ConvBlock(ch, bneck, time_emb_dim, dropout, dp[n_levels])

        self.decoders = nn.ModuleList()
        ch = bneck
        for i in range(n_levels - 1, -1, -1):
            skip_ch = base_channels * (2 ** i)
            self.decoders.append(DecoderBlock(ch, skip_ch, skip_ch, time_emb_dim, dropout, dp[i]))
            ch = skip_ch

        self.gap = nn.AdaptiveAvgPool2d(1)
        fuse = base_channels + (env_dim if use_env else 0) + (d1d_dim if use_1d else 0)
        self.head_dir = nn.Sequential(
            nn.Linear(fuse, head_dim), nn.GELU(), nn.Dropout(dropout * 2),
            nn.Linear(head_dim, head_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_dim // 2, n_dir_classes))
        self.head_int = nn.Sequential(
            nn.Linear(fuse, head_dim), nn.GELU(), nn.Dropout(dropout * 2),
            nn.Linear(head_dim, head_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_dim // 2, n_int_classes))

        # Physics reconstruction head: predicts vorticity, divergence, u/v shear
        self.physics_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, 4, 1))

    def forward(self, grid, env=None, d1d=None, time_feat=None):
        t = self.time_mlp(time_feat)
        skips, x = [], grid
        for enc in self.encoders:
            skip, x = enc(x, t)
            skips.append(skip)
        x = self.bottleneck(x, t)
        for dec, skip_val in zip(self.decoders, reversed(skips)):
            x = dec(x, skip_val, t)

        feat_map = x  # (B, base_channels, 81, 81)
        physics_pred = self.physics_head(feat_map)  # (B, 4, 81, 81)

        x = self.gap(feat_map).flatten(1)  # (B, base_channels)
        parts = [x]
        if self.use_env and env is not None: parts.append(env)
        if self.use_1d and d1d is not None: parts.append(d1d)
        fused = torch.cat(parts, 1)  # (B, 76)

        dir_logits = self.head_dir(fused)
        int_logits = self.head_int(fused)
        return dir_logits, int_logits, fused, physics_pred


# ── Discriminator ──

class Discriminator(nn.Module):
    """Conditional MLP discriminator on fused features + label embeddings."""
    def __init__(self, feat_dim=76, n_dir=8, n_int=4,
                 label_emb_dim=32, hidden_dim=256, n_layers=3):
        super().__init__()
        self.dir_emb = nn.Embedding(n_dir, label_emb_dim)
        self.int_emb = nn.Embedding(n_int, label_emb_dim)

        input_dim = feat_dim + 2 * label_emb_dim
        layers = []
        ch = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(ch, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
            ])
            ch = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features, dir_labels, int_labels):
        d_emb = self.dir_emb(dir_labels)
        i_emb = self.int_emb(int_labels)
        x = torch.cat([features, d_emb, i_emb], dim=1)
        return self.net(x)


# ── Physics Loss ──

class PhysicsLoss(nn.Module):
    """Computes physics targets from input grid and supervises physics_pred."""
    def __init__(self, dx=0.25):
        super().__init__()
        # Sobel filters for spatial derivatives
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


# ── Instantiate models ──

fuse_dim = BASE_CHANNELS + (40 if USE_ENV else 0) + (4 if USE_1D else 0)

generator = PIGANGenerator(
    in_channels=IN_CHANNELS, base_channels=BASE_CHANNELS, n_levels=N_LEVELS,
    time_dim=TIME_DIM, time_emb_dim=TIME_EMB_DIM,
    use_env=USE_ENV, use_1d=USE_1D,
    dropout=DROPOUT, head_dim=HEAD_DIM, drop_path=DROP_PATH).to(DEVICE)

discriminator = Discriminator(
    feat_dim=fuse_dim, n_dir=N_DIR_CLASSES, n_int=N_INT_CLASSES,
    label_emb_dim=LABEL_EMB_DIM, hidden_dim=DISC_HIDDEN,
    n_layers=DISC_LAYERS).to(DEVICE)

physics_loss_fn = PhysicsLoss().to(DEVICE)

g_params = sum(p.numel() for p in generator.parameters())
d_params = sum(p.numel() for p in discriminator.parameters())
print(f"Generator:     {g_params:,} params")
print(f"Discriminator: {d_params:,} params")
print(f"Total:         {g_params + d_params:,} params")

loss_dir_fn = nn.CrossEntropyLoss(weight=dir_weights.to(DEVICE), label_smoothing=LABEL_SMOOTH)
loss_int_fn = nn.CrossEntropyLoss(weight=int_weights.to(DEVICE), label_smoothing=LABEL_SMOOTH)

opt_G = torch.optim.AdamW(generator.parameters(), lr=G_LR, weight_decay=WEIGHT_DECAY, betas=(0.5, 0.999))
opt_D = torch.optim.AdamW(discriminator.parameters(), lr=D_LR, weight_decay=0, betas=(0.5, 0.999))

sched_G = torch.optim.lr_scheduler.OneCycleLR(
    opt_G, max_lr=G_LR * 3, epochs=EPOCHS, steps_per_epoch=len(loaders["wp_train"]))

print(f"G optimizer: AdamW (lr={G_LR}, wd={WEIGHT_DECAY})")
print(f"D optimizer: AdamW (lr={D_LR}, wd=0)")
print(f"G scheduler: OneCycleLR (max_lr={G_LR * 3})")


# ## Section 3: Training

class EMA:
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


def gradient_penalty(disc, real_feat, fake_feat, dir_lbl, int_lbl, device):
    """WGAN-GP gradient penalty on interpolated features."""
    alpha = torch.rand(real_feat.size(0), 1, device=device)
    interp = (alpha * real_feat + (1 - alpha) * fake_feat).requires_grad_(True)
    d_interp = disc(interp, dir_lbl, int_lbl)
    grads = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def train_one_epoch_gan(gen, disc, loader, opt_g, opt_d, device,
                        sched_g=None, ema=None, epoch=1):
    gen.train(); disc.train()
    use_adv = epoch >= ADV_WARMUP  # warm up classification first
    m = {"g_loss": 0, "d_loss": 0, "cls_loss": 0, "adv_loss": 0,
         "phys_loss": 0, "correct_d": 0, "correct_i": 0, "total": 0}

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

        # ════════════════════════════════════════
        # (1) Update Discriminator (after warmup)
        # ════════════════════════════════════════
        d_loss_val = 0.0
        if use_adv:
            with torch.no_grad():
                dir_logits_d, int_logits_d, fused_d, _ = gen(grid, env, d1d, tf)
                pred_dir_d = dir_logits_d.argmax(1)
                pred_int_d = int_logits_d.argmax(1)

            d_real = disc(fused_d.detach(), dl, il)
            d_fake = disc(fused_d.detach(), pred_dir_d, pred_int_d)
            d_loss = d_fake.mean() - d_real.mean()
            gp = gradient_penalty(disc, fused_d.detach(), fused_d.detach(),
                                  dl, il, device)
            d_total = d_loss + GP_WEIGHT * gp

            opt_d.zero_grad()
            d_total.backward()
            opt_d.step()
            d_loss_val = d_loss.item()

        # ════════════════════════════════════════
        # (2) Update Generator
        # ════════════════════════════════════════
        do_mixup = USE_MIXUP and torch.rand(1).item() > 0.5
        if do_mixup:
            grid_in, dl_a, il_a, dl_b, il_b, lam = mixup_data(grid, dl, il)
        else:
            grid_in = grid

        dir_logits, int_logits, fused, phys_pred = gen(grid_in, env, d1d, tf)
        pred_dir = dir_logits.argmax(1)
        pred_int = int_logits.argmax(1)

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

        # Adversarial loss (after warmup)
        adv_loss_val = 0.0
        if use_adv:
            d_gen = disc(fused, pred_dir.detach(), pred_int.detach())
            adv_loss = -d_gen.mean()
            adv_loss_val = adv_loss.item()
            g_loss = cls_loss + LAMBDA_ADV * adv_loss + LAMBDA_PHYS * phys_loss
        else:
            g_loss = cls_loss + LAMBDA_PHYS * phys_loss

        opt_g.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt_g.step()
        if ema: ema.update(gen)
        if sched_g: sched_g.step()

        # ── Track metrics ──
        m["g_loss"]    += g_loss.item() * B
        m["d_loss"]    += d_loss_val * B
        m["cls_loss"]  += cls_loss.item() * B
        m["adv_loss"]  += adv_loss_val * B
        m["phys_loss"] += phys_loss.item() * B
        m["correct_d"] += (dir_logits.argmax(1) == dl).sum().item()
        m["correct_i"] += (int_logits.argmax(1) == il).sum().item()
        m["total"]     += B

    n = m["total"]
    return {
        "g_loss": m["g_loss"] / n, "d_loss": m["d_loss"] / n,
        "cls_loss": m["cls_loss"] / n, "adv_loss": m["adv_loss"] / n,
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
                            "train_int_acc", "val_int_acc",
                            "d_loss", "adv_loss", "phys_loss"]}
best_val_dir_acc = 0.0
best_model_state = None
patience_counter = 0
ema = EMA(generator, decay=0.998)

print(f"\n{'='*80}")
print(f" PI-GAN Training — {EPOCHS} epochs, patience={PATIENCE}")
print(f" Lambda: adv={LAMBDA_ADV}, phys={LAMBDA_PHYS}, GP={GP_WEIGHT}")
print(f"{'='*80}\n")

for epoch in range(1, EPOCHS + 1):
    tm = train_one_epoch_gan(generator, discriminator, loaders["wp_train"],
                              opt_G, opt_D, DEVICE, sched_g=sched_G, ema=ema,
                              epoch=epoch)
    ema.apply(generator)
    vm = evaluate(generator, loaders["wp_val"], DEVICE)
    ema.restore(generator)

    history["train_loss"].append(tm["cls_loss"])
    history["val_loss"].append(vm["loss"])
    history["train_dir_acc"].append(tm["dir_acc"])
    history["val_dir_acc"].append(vm["dir_acc"])
    history["train_int_acc"].append(tm["int_acc"])
    history["val_int_acc"].append(vm["int_acc"])
    history["d_loss"].append(tm["d_loss"])
    history["adv_loss"].append(tm["adv_loss"])
    history["phys_loss"].append(tm["phys_loss"])

    if vm["dir_acc"] > best_val_dir_acc:
        best_val_dir_acc = vm["dir_acc"]
        best_model_state = deepcopy(dict(ema.shadow))
        patience_counter = 0; mk = " *"
    else:
        patience_counter += 1; mk = ""

    if epoch % 5 == 0 or epoch == 1 or mk:
        print(f"Ep {epoch:3d}/{EPOCHS} | cls={tm['cls_loss']:.4f} adv={tm['adv_loss']:.4f} "
              f"phys={tm['phys_loss']:.4f} D={tm['d_loss']:.4f} | "
              f"V dir={vm['dir_acc']:.3f} int={vm['int_acc']:.3f}{mk}")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

generator.load_state_dict(best_model_state)
torch.save(best_model_state, PROJECT_ROOT / "checkpoints" / "pi_gan_best_wp.pt")
print(f"\nBest val dir acc: {best_val_dir_acc:.4f}")

# ## Section 4: Training Curves

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (t, v, title) in zip(axes, [
    ("train_loss", "val_loss", "Loss"),
    ("train_dir_acc", "val_dir_acc", "Direction Accuracy"),
    ("train_int_acc", "val_int_acc", "Intensity Accuracy")]):
    ax.plot(history[t], label="Train"); ax.plot(history[v], label="Val")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
fig.suptitle("PI-GAN: Training Curves", fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "pi_gan_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# GAN-specific dynamics plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, key, title in [
    (axes[0], "d_loss", "Discriminator Loss (WGAN)"),
    (axes[1], "adv_loss", "Generator Adversarial Loss"),
    (axes[2], "phys_loss", "Physics Reconstruction Loss")]:
    ax.plot(history[key])
    ax.set_title(title); ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
fig.suptitle("PI-GAN: Training Dynamics", fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "pi_gan_training_dynamics.png", dpi=150, bbox_inches="tight")
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


wp_m = evaluate(generator, loaders["wp_val"], DEVICE)
print(f"WP Val — dir={wp_m['dir_acc']:.3f} F1={wp_m['dir_f1']:.3f} | "
      f"int={wp_m['int_acc']:.3f} F1={wp_m['int_f1']:.3f}")
fig = plot_confusion_matrices(wp_m, "WP Val (PI-GAN)")
fig.savefig(FIG_DIR / "pi_gan_cm_wp.png", dpi=150, bbox_inches="tight")
plt.show()

sp_zs = evaluate(generator, loaders["sp_test"], DEVICE)
print(f"SP Zero-Shot — dir={sp_zs['dir_acc']:.3f} F1={sp_zs['dir_f1']:.3f} | "
      f"int={sp_zs['int_acc']:.3f}")
print(f"Transfer gap (dir): {sp_zs['dir_acc'] - wp_m['dir_acc']:+.3f}")
fig = plot_confusion_matrices(sp_zs, "SP Zero-Shot (PI-GAN)")
fig.savefig(FIG_DIR / "pi_gan_cm_sp_zs.png", dpi=150, bbox_inches="tight")
plt.show()

# ## Section 6: Fine-Tuning (WP → SP)

def finetune(model_state, ft_train_loader, ft_val_loader, freeze_backbone=False,
             lr=FT_LR, epochs=FT_EPOCHS, patience=FT_PATIENCE):
    """Fine-tune generator with classification loss only (no adversarial/physics)."""
    ft = PIGANGenerator(
        in_channels=IN_CHANNELS, base_channels=BASE_CHANNELS, n_levels=N_LEVELS,
        time_dim=TIME_DIM, time_emb_dim=TIME_EMB_DIM,
        use_env=USE_ENV, use_1d=USE_1D,
        dropout=DROPOUT, head_dim=HEAD_DIM, drop_path=DROP_PATH).to(DEVICE)
    ft.load_state_dict(model_state)
    if freeze_backbone:
        for n, p in ft.named_parameters():
            if "head" not in n: p.requires_grad = False
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

print("Head-only:")
ft_head, ft_head_h, ft_head_a = finetune(best_model_state, loaders["sp_ft_train"],
                                          loaders["sp_ft_val"], freeze_backbone=True)
print(f"  Best: {ft_head_a:.3f}")

ft_best = ft_full if ft_full_a >= ft_head_a else ft_head
ft_strategy = "full" if ft_full_a >= ft_head_a else "head-only"
torch.save(ft_best.state_dict(), PROJECT_ROOT / "checkpoints" / "pi_gan_best_ft.pt")

sp_ft = evaluate(ft_best, loaders["sp_test"], DEVICE)
print(f"SP Fine-Tuned ({ft_strategy}) — dir={sp_ft['dir_acc']:.3f} F1={sp_ft['dir_f1']:.3f} | "
      f"int={sp_ft['int_acc']:.3f}")
fig = plot_confusion_matrices(sp_ft, f"SP Fine-Tuned (PI-GAN, {ft_strategy})")
fig.savefig(FIG_DIR / "pi_gan_cm_sp_ft.png", dpi=150, bbox_inches="tight")
plt.show()

# ## Section 7: Summary

print("\n" + "=" * 80)
print(" PI-GAN — Final Results")
print("=" * 80)
print(f"{'Setting':20s} | {'Dir Acc':>8s} | {'Dir F1':>7s} | {'Int Acc':>8s} | {'Int F1':>7s}")
print("-" * 80)
for name, m_val in [("WP Val", wp_m), ("SP Zero-Shot", sp_zs),
                    (f"SP FT ({ft_strategy})", sp_ft)]:
    print(f"{name:20s} | {m_val['dir_acc']:>7.1%} | {m_val['dir_f1']:>6.1%} | "
          f"{m_val['int_acc']:>7.1%} | {m_val['int_f1']:>6.1%}")
print("=" * 80)
print(f"Generator params:     {g_params:,}")
print(f"Discriminator params: {d_params:,}")
print(f"Lambda: adv={LAMBDA_ADV}, phys={LAMBDA_PHYS}")
print(f"Transfer gap (dir):   {sp_zs['dir_acc'] - wp_m['dir_acc']:+.3f}")
print(f"FT recovery (dir):    {sp_ft['dir_acc'] - sp_zs['dir_acc']:+.3f}")
