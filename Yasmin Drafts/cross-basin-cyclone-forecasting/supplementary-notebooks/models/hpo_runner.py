"""
Hyperparameter Optimization Runner for Cross-Basin Cyclone Forecasting.
Runs Optuna HPO for a specified model, then trains the best config.

Usage:
    python hpo_runner.py --model unet --trials 20 --hpo-epochs 50 --final-epochs 300
    python hpo_runner.py --model unet_film --trials 20 --hpo-epochs 50
    python hpo_runner.py --model fno --trials 20 --hpo-epochs 50
    python hpo_runner.py --model fno_v2 --trials 20 --hpo-epochs 50
    python hpo_runner.py --model ufno --trials 20 --hpo-epochs 50
"""

import matplotlib
matplotlib.use('Agg')

import argparse, json, os, sys, time, warnings
from pathlib import Path
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
FIG_DIR = PROJECT_ROOT / "figures"
LOG_DIR = PROJECT_ROOT / "logs"
for d in [CKPT_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

N_DIR = 8; N_INT = 4
DIR_LABELS = ["E","SE","S","SW","W","NW","N","NE"]
INT_LABELS = ["Weakening","Steady","Slow-int.","Rapid-int."]

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

SPLITS = {"wp_train":False, "wp_val":False, "sp_test":True,
          "sp_ft_train":True, "sp_ft_val":True}

def load_data(use_time=False):
    raw = {}
    for s, ref in SPLITS.items():
        raw[s] = {
            "grids": torch.load(DATA_DIR/"grids"/f"{s}_grids.pt", weights_only=False),
            "env": torch.load(DATA_DIR/"env"/f"{s}_env.pt", weights_only=False),
            "data1d": torch.load(DATA_DIR/"data1d"/f"{s}_1d.pt", weights_only=False),
            "labels": torch.load(DATA_DIR/"labels"/f"{s}_labels.pt", weights_only=False),
        }
        if use_time:
            raw[s]["time"] = torch.load(DATA_DIR/"time"/f"{s}_time.pt", weights_only=False)
    return raw

class CycloneDataset(Dataset):
    def __init__(self, grids, env, d1d, labels, time_feats=None,
                 use_ref=False, d1d_mean=None, d1d_std=None):
        self.samples = []
        self.has_time = time_feats is not None
        dk = "direction_reflected" if use_ref else "direction"
        for sid in grids:
            g, e, d = grids[sid], env[sid], d1d[sid]
            t = time_feats[sid] if self.has_time else None
            dl, il = labels[sid][dk], labels[sid]["intensity"]
            for i in range(g.shape[0]):
                if dl[i].item() == -1 or il[i].item() == -1:
                    continue
                sample = [g[i], e[i], d[i]]
                if self.has_time:
                    sample.append(t[i])
                sample.extend([dl[i].long(), il[i].long()])
                self.samples.append(tuple(sample))
        if d1d_mean is None:
            a = torch.stack([s[2] for s in self.samples])
            self.d1d_mean, self.d1d_std = a.mean(0), a.std(0).clamp(min=1e-6)
        else:
            self.d1d_mean, self.d1d_std = d1d_mean, d1d_std

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        g, e, d = s[0], s[1], (s[2] - self.d1d_mean) / self.d1d_std
        if self.has_time:
            return g, e, d, s[3], s[4], s[5]
        return g, e, d, s[3], s[4]

def build_datasets(raw, use_time=False):
    ds = {}
    time_arg = lambda s: raw[s].get("time") if use_time else None
    ds["wp_train"] = CycloneDataset(raw["wp_train"]["grids"], raw["wp_train"]["env"],
        raw["wp_train"]["data1d"], raw["wp_train"]["labels"], time_arg("wp_train"))
    dm, ds_ = ds["wp_train"].d1d_mean, ds["wp_train"].d1d_std
    for s, ref in SPLITS.items():
        if s == "wp_train": continue
        ds[s] = CycloneDataset(raw[s]["grids"], raw[s]["env"], raw[s]["data1d"],
            raw[s]["labels"], time_arg(s), use_ref=ref, d1d_mean=dm, d1d_std=ds_)
    return ds

def build_loaders(datasets, batch_size=64):
    return {s: DataLoader(datasets[s], batch_size=batch_size,
            shuffle=(s=="wp_train"), num_workers=2, pin_memory=True,
            persistent_workers=True) for s in SPLITS}

def class_weights(dataset):
    dc, ic = Counter(), Counter()
    has_time = dataset.has_time
    for s in dataset.samples:
        dl_idx = 4 if has_time else 3
        il_idx = 5 if has_time else 4
        dc[s[dl_idx].item()] += 1
        ic[s[il_idx].item()] += 1
    n = len(dataset)
    dw = torch.tensor([n/(N_DIR*max(dc.get(c,1),1)) for c in range(N_DIR)])
    iw = torch.tensor([n/(N_INT*max(ic.get(c,1),1)) for c in range(N_INT)])
    return dw, iw

# ═══════════════════════════════════════════════════════════════
# AUGMENTATION
# ═══════════════════════════════════════════════════════════════

def cutout(grid, n_holes=2, hole_size=16):
    B, C, H, W = grid.shape
    mask = torch.ones_like(grid)
    for _ in range(n_holes):
        cy, cx = torch.randint(0, H, (B,)), torch.randint(0, W, (B,))
        for b in range(B):
            y1, y2 = max(0, cy[b]-hole_size//2), min(H, cy[b]+hole_size//2)
            x1, x2 = max(0, cx[b]-hole_size//2), min(W, cx[b]+hole_size//2)
            mask[b, :, y1:y2, x1:x2] = 0
    return grid * mask

def channel_dropout(grid, p=0.15):
    B, C, H, W = grid.shape
    mask = (torch.rand(B, C, 1, 1, device=grid.device) > p).float()
    if (mask.sum(1, keepdim=True) == 0).any():
        for b in range(B):
            if mask[b].sum() == 0: mask[b, torch.randint(0,C,(1,))] = 1.0
    return grid * mask

def augment_grid(grid, noise_std=0.05):
    if torch.rand(1).item() > 0.3:
        grid = cutout(grid)
    grid = grid + torch.randn_like(grid) * noise_std
    if torch.rand(1).item() > 0.5:
        grid = channel_dropout(grid)
    return grid

# ═══════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════

# --- U-Net ---
class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch//r,4)), nn.GELU(), nn.Linear(max(ch//r,4), ch), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)

class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0: return x
        keep = torch.rand(x.size(0),1,1,1,device=x.device) > self.p
        return x * keep / (1 - self.p)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.drop2d = nn.Dropout2d(dropout)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop2d(h)
        h = self.act(self.bn2(self.conv2(h)))
        return self.drop_path(h) + self.residual(x)

class UNet2d(nn.Module):
    def __init__(self, in_ch=15, base_ch=32, n_levels=4, n_dir=8, n_int=4,
                 env_dim=40, d1d_dim=4, dropout=0.2, head_dim=256, drop_path=0.1):
        super().__init__()
        dp = [drop_path * i / max(n_levels,1) for i in range(n_levels+1)]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        ch = in_ch
        for i in range(n_levels):
            out = base_ch * (2**i)
            self.encoders.append(ConvBlock(ch, out, dropout, dp[i]))
            self.se_blocks.append(SEBlock(out))
            self.pools.append(nn.MaxPool2d(2, ceil_mode=True))
            ch = out
        bneck = base_ch * (2**n_levels)
        self.bottleneck = ConvBlock(ch, bneck, dropout, dp[n_levels])
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        ch = bneck
        for i in range(n_levels-1, -1, -1):
            skip_ch = base_ch * (2**i)
            self.ups.append(nn.ConvTranspose2d(ch, ch, 2, stride=2))
            self.decoders.append(ConvBlock(ch + skip_ch, skip_ch, dropout, dp[i]))
            ch = skip_ch
        self.gap = nn.AdaptiveAvgPool2d(1)
        fuse = base_ch + env_dim + d1d_dim
        self.head_dir = nn.Sequential(nn.Linear(fuse,head_dim),nn.GELU(),nn.Dropout(dropout*2),
            nn.Linear(head_dim,head_dim//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(head_dim//2,n_dir))
        self.head_int = nn.Sequential(nn.Linear(fuse,head_dim),nn.GELU(),nn.Dropout(dropout*2),
            nn.Linear(head_dim,head_dim//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(head_dim//2,n_int))

    def forward(self, grid, env=None, d1d=None, **kw):
        skips = []
        x = grid
        for enc, se, pool in zip(self.encoders, self.se_blocks, self.pools):
            x = se(enc(x)); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            dh, dw = x.size(2)-skip.size(2), x.size(3)-skip.size(3)
            if dh > 0 or dw > 0: x = x[:,:,dh//2:dh//2+skip.size(2),dw//2:dw//2+skip.size(3)]
            elif dh < 0 or dw < 0: x = F.pad(x, [0,-dw,0,-dh])
            x = dec(torch.cat([x, skip], 1))
        x = self.gap(x).flatten(1)
        parts = [x]
        if env is not None: parts.append(env)
        if d1d is not None: parts.append(d1d)
        f = torch.cat(parts, 1)
        return self.head_dir(f), self.head_int(f)

# --- U-Net + FiLM ---
class FiLMLayer(nn.Module):
    def __init__(self, cond_dim, ch):
        super().__init__()
        self.fc = nn.Linear(cond_dim, ch*2)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)
        self.fc.bias.data[:ch] = 1.0
    def forward(self, x, c):
        g, b = self.fc(c).chunk(2, 1)
        return g.unsqueeze(-1).unsqueeze(-1) * x + b.unsqueeze(-1).unsqueeze(-1)

class FiLMConvBlock(nn.Module):
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

class UNetFiLM2d(nn.Module):
    def __init__(self, in_ch=15, base_ch=32, n_levels=4, n_dir=8, n_int=4,
                 env_dim=40, d1d_dim=4, time_dim=6, time_emb=64,
                 dropout=0.2, head_dim=256, drop_path=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_dim,time_emb),nn.GELU(),nn.Linear(time_emb,time_emb))
        dp = [drop_path*i/max(n_levels,1) for i in range(n_levels+1)]
        self.encoders = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for i in range(n_levels):
            out = base_ch*(2**i)
            self.encoders.append(FiLMConvBlock(ch, out, time_emb, dropout, dp[i]))
            self.se_blocks.append(SEBlock(out))
            self.pools.append(nn.MaxPool2d(2, ceil_mode=True))
            ch = out
        bneck = base_ch*(2**n_levels)
        self.bottleneck = FiLMConvBlock(ch, bneck, time_emb, dropout, dp[n_levels])
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        ch = bneck
        for i in range(n_levels-1,-1,-1):
            skip_ch = base_ch*(2**i)
            self.ups.append(nn.ConvTranspose2d(ch, ch, 2, stride=2))
            self.decoders.append(FiLMConvBlock(ch+skip_ch, skip_ch, time_emb, dropout, dp[i]))
            ch = skip_ch
        self.gap = nn.AdaptiveAvgPool2d(1)
        fuse = base_ch + env_dim + d1d_dim
        self.head_dir = nn.Sequential(nn.Linear(fuse,head_dim),nn.GELU(),nn.Dropout(dropout*2),
            nn.Linear(head_dim,head_dim//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(head_dim//2,n_dir))
        self.head_int = nn.Sequential(nn.Linear(fuse,head_dim),nn.GELU(),nn.Dropout(dropout*2),
            nn.Linear(head_dim,head_dim//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(head_dim//2,n_int))

    def forward(self, grid, env=None, d1d=None, time_feat=None, **kw):
        t = self.time_mlp(time_feat)
        skips = []; x = grid
        for enc, se, pool in zip(self.encoders, self.se_blocks, self.pools):
            x = se(enc(x, t)); skips.append(x); x = pool(x)
        x = self.bottleneck(x, t)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            dh, dw = x.size(2)-skip.size(2), x.size(3)-skip.size(3)
            if dh > 0 or dw > 0: x = x[:,:,dh//2:dh//2+skip.size(2),dw//2:dw//2+skip.size(3)]
            elif dh < 0 or dw < 0: x = F.pad(x, [0,-dw,0,-dh])
            x = dec(torch.cat([x, skip], 1), t)
        x = self.gap(x).flatten(1)
        f = torch.cat([x, env, d1d], 1)
        return self.head_dir(f), self.head_int(f)

# --- FNO ---
class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes, padding=0):
        super().__init__()
        self.modes, self.out_ch, self.padding = modes, out_ch, padding
        s = (2/(in_ch+out_ch))**0.5
        self.w1 = nn.Parameter(s*(torch.rand(in_ch,out_ch,modes,modes,dtype=torch.cfloat)-0.5))
        self.w2 = nn.Parameter(s*(torch.rand(in_ch,out_ch,modes,modes,dtype=torch.cfloat)-0.5))
    def forward(self, x):
        if self.padding > 0: x = F.pad(x, [self.padding]*4, mode='reflect')
        B,C,H,W = x.shape
        xf = torch.fft.rfft2(x)
        of = torch.zeros(B,self.out_ch,H,W//2+1,dtype=torch.cfloat,device=x.device)
        of[:,:,:self.modes,:self.modes] = torch.einsum("bixy,ioxy->boxy",xf[:,:,:self.modes,:self.modes],self.w1)
        of[:,:,-self.modes:,:self.modes] = torch.einsum("bixy,ioxy->boxy",xf[:,:,-self.modes:,:self.modes],self.w2)
        x = torch.fft.irfft2(of, s=(H,W))
        if self.padding > 0: x = x[:,:,self.padding:-self.padding,self.padding:-self.padding]
        return x

class FNO2d(nn.Module):
    def __init__(self, in_ch=15, hidden=32, modes=12, n_layers=4, padding=0,
                 n_dir=8, n_int=4, env_dim=40, d1d_dim=4, dropout=0.05):
        super().__init__()
        self.lifting = nn.Sequential(nn.Conv2d(in_ch, hidden, 1), nn.GELU())
        self.spectral = nn.ModuleList([SpectralConv2d(hidden, hidden, modes, padding) for _ in range(n_layers)])
        self.skip = nn.ModuleList([nn.Conv2d(hidden, hidden, 1) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden) for _ in range(n_layers)])
        self.drops = nn.ModuleList([nn.Dropout2d(dropout) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv2d(hidden, hidden, 1), nn.GELU())
        fuse = hidden + env_dim + d1d_dim
        self.head_dir = nn.Sequential(nn.Linear(fuse,128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,n_dir))
        self.head_int = nn.Sequential(nn.Linear(fuse,128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,n_int))

    def forward(self, grid, env=None, d1d=None, **kw):
        x = self.lifting(grid)
        for sp, sk, bn, dr in zip(self.spectral, self.skip, self.norms, self.drops):
            x = dr(F.gelu(bn(sp(x) + sk(x)))) + x
        x = self.proj(x).mean(dim=(-2,-1))
        f = torch.cat([x, env, d1d], 1)
        return self.head_dir(f), self.head_int(f)

# --- FNO v2 (with FiLM) ---
class FNOv2(nn.Module):
    def __init__(self, in_ch=15, hidden=32, modes=12, n_layers=3, padding=9,
                 n_dir=8, n_int=4, env_dim=40, d1d_dim=4, time_dim=6, time_emb=64, dropout=0.05):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_dim,time_emb),nn.GELU(),nn.Linear(time_emb,time_emb))
        self.lifting = nn.Sequential(nn.Conv2d(in_ch, hidden, 1), nn.GELU())
        self.spectral = nn.ModuleList([SpectralConv2d(hidden, hidden, modes, padding) for _ in range(n_layers)])
        self.skip = nn.ModuleList([nn.Conv2d(hidden, hidden, 1) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.BatchNorm2d(hidden) for _ in range(n_layers)])
        self.films = nn.ModuleList([FiLMLayer(time_emb, hidden) for _ in range(n_layers)])
        self.drops = nn.ModuleList([nn.Dropout2d(dropout) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv2d(hidden, hidden, 1), nn.GELU())
        fuse = hidden + env_dim + d1d_dim
        self.head_dir = nn.Sequential(nn.Linear(fuse,128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,n_dir))
        self.head_int = nn.Sequential(nn.Linear(fuse,128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,n_int))

    def forward(self, grid, env=None, d1d=None, time_feat=None, **kw):
        t = self.time_mlp(time_feat) if time_feat is not None else None
        x = self.lifting(grid)
        for sp, sk, bn, film, dr in zip(self.spectral, self.skip, self.norms, self.films, self.drops):
            h = bn(sp(x) + sk(x))
            if t is not None: h = film(h, t)
            x = dr(F.gelu(h)) + x
        x = self.proj(x).mean(dim=(-2,-1))
        f = torch.cat([x, env, d1d], 1)
        return self.head_dir(f), self.head_int(f)

# --- U-FNO ---
class UNetBranch(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(ch,ch,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(ch),nn.GELU())
        self.mid = nn.Sequential(nn.Conv2d(ch,ch,3,padding=1,bias=False),nn.BatchNorm2d(ch),nn.GELU())
        self.up = nn.ConvTranspose2d(ch,ch,2,stride=2)
        self.fuse = nn.Sequential(nn.Conv2d(ch*2,ch,1,bias=False),nn.BatchNorm2d(ch))
    def forward(self, x):
        skip = x; x = self.mid(self.down(x)); x = self.up(x)
        if x.shape != skip.shape: x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([x, skip], 1))

class UFNO2d(nn.Module):
    def __init__(self, in_ch=15, hidden=32, modes=12, n_layers=3, padding=9,
                 n_dir=8, n_int=4, env_dim=40, d1d_dim=4, time_dim=6, time_emb=64, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_dim,time_emb),nn.GELU(),nn.Linear(time_emb,time_emb))
        self.lifting = nn.Sequential(nn.Conv2d(in_ch,hidden,1),nn.GELU())
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "spectral": SpectralConv2d(hidden, hidden, modes, padding),
                "unet": UNetBranch(hidden),
                "residual": nn.Conv2d(hidden, hidden, 1),
                "gate": nn.ParameterDict({"w": nn.Parameter(torch.ones(3)/3)}),
                "norm": nn.BatchNorm2d(hidden),
                "film": FiLMLayer(time_emb, hidden),
                "drop": nn.Dropout2d(dropout),
            }))
        self.proj = nn.Sequential(nn.Conv2d(hidden,hidden,1),nn.GELU())
        fuse = hidden + env_dim + d1d_dim
        self.head_dir = nn.Sequential(nn.Linear(fuse,128),nn.GELU(),nn.Dropout(0.2),
            nn.Linear(128,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,n_dir))
        self.head_int = nn.Sequential(nn.Linear(fuse,128),nn.GELU(),nn.Dropout(0.2),
            nn.Linear(128,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,n_int))

    def forward(self, grid, env=None, d1d=None, time_feat=None, **kw):
        t = self.time_mlp(time_feat) if time_feat is not None else None
        x = self.lifting(grid)
        for blk in self.blocks:
            g = F.softmax(blk["gate"]["w"], dim=0)
            out = g[0]*blk["spectral"](x) + g[1]*blk["unet"](x) + g[2]*blk["residual"](x)
            out = blk["norm"](out)
            if t is not None: out = blk["film"](out, t)
            x = blk["drop"](F.gelu(out)) + x
        x = self.proj(x).mean(dim=(-2,-1))
        f = torch.cat([x, env, d1d], 1)
        return self.head_dir(f), self.head_int(f)

# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point(): self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
            else: self.shadow[k] = v.clone()
    def apply(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
    def restore(self, model):
        model.load_state_dict(self.backup)

def train_model(model, loaders, epochs, patience, lr, weight_decay, dir_weight,
                label_smooth=0.05, use_augment=True, use_ema=True, use_time=False):
    dw, iw = class_weights(loaders["wp_train"].dataset)
    loss_d = nn.CrossEntropyLoss(weight=dw.to(DEVICE), label_smoothing=label_smooth)
    loss_i = nn.CrossEntropyLoss(weight=iw.to(DEVICE), label_smoothing=label_smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr*3, epochs=epochs,
                                                  steps_per_epoch=len(loaders["wp_train"]))
    ema = EMA(model) if use_ema else None
    best_acc, best_state, pat, history = 0.0, None, 0, {"val_dir":[], "val_int":[], "train_loss":[], "val_loss":[]}

    for ep in range(1, epochs+1):
        model.train(); tl = tot = 0
        for batch in loaders["wp_train"]:
            if use_time:
                g, e, d, t, dl, il = [x.to(DEVICE) for x in batch]
            else:
                g, e, d, dl, il = [x.to(DEVICE) for x in batch]
                t = None
            if use_augment: g = augment_grid(g)
            do, io = model(g, e, d, time_feat=t)
            loss = dir_weight*loss_d(do,dl) + (1-dir_weight)*loss_i(io,il)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if ema: ema.update(model)
            sched.step()
            tl += loss.item()*g.size(0); tot += g.size(0)

        if ema: ema.apply(model)
        vm = evaluate(model, loaders["wp_val"], use_time=use_time)
        if ema: ema.restore(model)

        history["train_loss"].append(tl/tot)
        history["val_loss"].append(vm["loss"])
        history["val_dir"].append(vm["dir_acc"])
        history["val_int"].append(vm["int_acc"])

        if vm["dir_acc"] > best_acc:
            best_acc = vm["dir_acc"]
            best_state = deepcopy(dict(ema.shadow)) if ema else deepcopy(model.state_dict())
            pat = 0; mk = " *"
        else: pat += 1; mk = ""

        if ep % 10 == 0 or ep == 1 or mk:
            print(f"  Ep {ep:3d}/{epochs} | loss={tl/tot:.4f} | val dir={vm['dir_acc']:.3f} int={vm['int_acc']:.3f}{mk}")

        if pat >= patience:
            print(f"  Early stop at ep {ep}")
            break

    model.load_state_dict(best_state)
    return best_acc, history

@torch.no_grad()
def evaluate(model, loader, use_time=False):
    model.eval()
    tl=tot=0; dp,dt,ip,it_=[],[],[],[]
    dw, iw = class_weights(loader.dataset)
    loss_d = nn.CrossEntropyLoss(weight=dw.to(DEVICE), label_smoothing=0.05)
    loss_i = nn.CrossEntropyLoss(weight=iw.to(DEVICE), label_smoothing=0.05)
    for batch in loader:
        if use_time:
            g,e,d,t,dl,il = [x.to(DEVICE) for x in batch]
        else:
            g,e,d,dl,il = [x.to(DEVICE) for x in batch]
            t = None
        do, io = model(g, e, d, time_feat=t)
        loss = 0.5*loss_d(do,dl) + 0.5*loss_i(io,il)
        tl+=loss.item()*g.size(0); tot+=g.size(0)
        dp.extend(do.argmax(1).cpu().tolist()); dt.extend(dl.cpu().tolist())
        ip.extend(io.argmax(1).cpu().tolist()); it_.extend(il.cpu().tolist())
    return {"loss":tl/tot, "dir_acc":accuracy_score(dt,dp), "int_acc":accuracy_score(it_,ip),
            "dir_f1":f1_score(dt,dp,average="macro",zero_division=0),
            "int_f1":f1_score(it_,ip,average="macro",zero_division=0),
            "dir_pred":dp,"dir_true":dt,"int_pred":ip,"int_true":it_}

def finetune_model(model_state, model_class, model_kwargs, loaders, lr=1e-4,
                   epochs=50, patience=15, dir_weight=0.5, use_time=False):
    m = model_class(**model_kwargs).to(DEVICE)
    m.load_state_dict(model_state)
    dw, iw = class_weights(loaders["wp_train"].dataset)
    loss_d = nn.CrossEntropyLoss(weight=dw.to(DEVICE), label_smoothing=0.05)
    loss_i = nn.CrossEntropyLoss(weight=iw.to(DEVICE), label_smoothing=0.05)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_acc, best_st, pat = 0.0, None, 0
    for ep in range(1, epochs+1):
        m.train()
        for batch in loaders["sp_ft_train"]:
            if use_time:
                g,e,d,t,dl,il = [x.to(DEVICE) for x in batch]
            else:
                g,e,d,dl,il = [x.to(DEVICE) for x in batch]
                t = None
            do, io = m(g, e, d, time_feat=t)
            loss = dir_weight*loss_d(do,dl) + (1-dir_weight)*loss_i(io,il)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sched.step()
        vm = evaluate(m, loaders["sp_ft_val"], use_time=use_time)
        if vm["dir_acc"] > best_acc: best_acc = vm["dir_acc"]; best_st = deepcopy(m.state_dict()); pat = 0
        else: pat += 1
        if ep % 10 == 0: print(f"  FT {ep:3d} | dir={vm['dir_acc']:.3f} int={vm['int_acc']:.3f}")
        if pat >= patience: print(f"  FT early stop at {ep}"); break
    m.load_state_dict(best_st)
    return m

# ═══════════════════════════════════════════════════════════════
# MODEL CONFIGS
# ═══════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "unet": {"class": UNet2d, "use_time": False,
        "defaults": {"base_ch":48,"n_levels":4,"dropout":0.2,"head_dim":256,"drop_path":0.1},
        "hpo": lambda trial: {
            "base_ch": trial.suggest_categorical("base_ch", [32, 48, 64]),
            "n_levels": trial.suggest_int("n_levels", 3, 5),
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
            "head_dim": trial.suggest_categorical("head_dim", [256, 512]),
            "drop_path": trial.suggest_float("drop_path", 0.0, 0.2),
        }},
    "unet_film": {"class": UNetFiLM2d, "use_time": True,
        "defaults": {"base_ch":48,"n_levels":4,"dropout":0.2,"head_dim":256,"drop_path":0.1,"time_emb":64},
        "hpo": lambda trial: {
            "base_ch": trial.suggest_categorical("base_ch", [32, 48, 64]),
            "n_levels": trial.suggest_int("n_levels", 3, 5),
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
            "head_dim": trial.suggest_categorical("head_dim", [256, 512]),
            "time_emb": trial.suggest_categorical("time_emb", [64, 128]),
        }},
    "fno": {"class": FNO2d, "use_time": False,
        "defaults": {"hidden":64,"modes":16,"n_layers":4,"padding":0,"dropout":0.05},
        "hpo": lambda trial: {
            "hidden": trial.suggest_categorical("hidden", [48, 64, 96]),
            "modes": trial.suggest_int("modes", 12, 20),
            "n_layers": trial.suggest_int("n_layers", 3, 6),
            "dropout": trial.suggest_float("dropout", 0.02, 0.15),
        }},
    "fno_v2": {"class": FNOv2, "use_time": True,
        "defaults": {"hidden":64,"modes":16,"n_layers":3,"padding":9,"dropout":0.05,"time_emb":64},
        "hpo": lambda trial: {
            "hidden": trial.suggest_categorical("hidden", [48, 64, 96]),
            "modes": trial.suggest_int("modes", 12, 20),
            "n_layers": trial.suggest_int("n_layers", 2, 5),
            "padding": trial.suggest_categorical("padding", [5, 9, 13]),
            "time_emb": trial.suggest_categorical("time_emb", [64, 128]),
        }},
    "ufno": {"class": UFNO2d, "use_time": True,
        "defaults": {"hidden":64,"modes":16,"n_layers":3,"padding":9,"dropout":0.1,"time_emb":64},
        "hpo": lambda trial: {
            "hidden": trial.suggest_categorical("hidden", [48, 64, 96]),
            "modes": trial.suggest_int("modes", 12, 20),
            "n_layers": trial.suggest_int("n_layers", 2, 5),
            "padding": trial.suggest_categorical("padding", [5, 9, 13]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.2),
        }},
}

TRAIN_HPO = {
    "defaults": {"lr":5e-4, "weight_decay":1.3e-3, "dir_weight":0.55},
    "hpo": lambda trial: {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 5e-4, 5e-3, log=True),
        "dir_weight": trial.suggest_float("dir_weight", 0.45, 0.6),
    }
}

# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_training_curves(history, model_name, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train"); axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["val_dir"]); axes[1].set_title("Val Dir Acc")
    axes[2].plot(history["val_int"]); axes[2].set_title("Val Int Acc")
    fig.suptitle(f"{model_name} Training Curves", fontweight="bold"); fig.tight_layout()
    fig.savefig(fig_dir/f"{model_name}_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close('all')

def plot_confusion(metrics, title, fig_dir, prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, labels, cmap in [(axes[0],"dir",DIR_LABELS,"Blues"),(axes[1],"int",INT_LABELS,"Oranges")]:
        cm = confusion_matrix(metrics[f"{key}_true"], metrics[f"{key}_pred"], labels=range(len(labels)))
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm_pct, annot=True, fmt=".0f", cmap=cmap, xticklabels=labels,
                    yticklabels=labels, cbar=False, ax=ax, vmin=0, vmax=100)
        ax.set_title(f"{title} — {key.title()} ({metrics[f'{key}_acc']:.1%})")
    fig.tight_layout()
    fig.savefig(fig_dir/f"{prefix}.png", dpi=150, bbox_inches="tight")
    plt.close('all')

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_hpo(model_name, n_trials, hpo_epochs, hpo_patience):
    cfg = MODEL_REGISTRY[model_name]
    use_time = cfg["use_time"]
    print(f"\n{'='*60}")
    print(f" HPO: {model_name} ({n_trials} trials x {hpo_epochs} epochs)")
    print(f"{'='*60}")

    raw = load_data(use_time=use_time)
    datasets = build_datasets(raw, use_time=use_time)

    def objective(trial):
        model_params = cfg["hpo"](trial)
        train_params = TRAIN_HPO["hpo"](trial)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        loaders = build_loaders(datasets, batch_size)

        try:
            model = cfg["class"](**model_params).to(DEVICE)
            acc, _ = train_model(model, loaders, hpo_epochs, hpo_patience,
                                 train_params["lr"], train_params["weight_decay"],
                                 train_params["dir_weight"], use_time=use_time, use_ema=False)
            del model; torch.cuda.empty_cache()
            return acc
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return 0.0
            raise

    study = optuna.create_study(direction="maximize",
        study_name=f"{model_name}_hpo",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Save results
    result = {"model": model_name, "best_value": study.best_value,
              "best_params": study.best_params, "n_trials": n_trials}
    with open(LOG_DIR / f"{model_name}_hpo.json", "w") as f:
        json.dump(result, f, indent=2)

    return study.best_params, study.best_value

def run_full_training(model_name, params, epochs, patience):
    cfg = MODEL_REGISTRY[model_name]
    use_time = cfg["use_time"]
    print(f"\n{'='*60}")
    print(f" FULL TRAINING: {model_name} ({epochs} epochs)")
    print(f"{'='*60}")

    raw = load_data(use_time=use_time)
    datasets = build_datasets(raw, use_time=use_time)

    # Separate model params from training params
    model_param_keys = set(cfg["defaults"].keys())
    model_params = {k: params.get(k, cfg["defaults"][k]) for k in model_param_keys}
    lr = params.get("lr", TRAIN_HPO["defaults"]["lr"])
    wd = params.get("weight_decay", TRAIN_HPO["defaults"]["weight_decay"])
    dw = params.get("dir_weight", TRAIN_HPO["defaults"]["dir_weight"])
    bs = params.get("batch_size", 64)

    loaders = build_loaders(datasets, bs)
    model = cfg["class"](**model_params).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {model_name}: {n_params:,} params")
    print(f"  Config: {model_params}")
    print(f"  Training: lr={lr}, wd={wd}, dir_weight={dw}, bs={bs}")

    best_acc, history = train_model(model, loaders, epochs, patience, lr, wd, dw,
                                     use_time=use_time, use_ema=True)
    print(f"\n  Best WP dir acc: {best_acc:.4f}")

    # Plot training curves
    plot_training_curves(history, model_name, FIG_DIR)

    # Evaluate
    wp_m = evaluate(model, loaders["wp_val"], use_time=use_time)
    sp_zs = evaluate(model, loaders["sp_test"], use_time=use_time)
    print(f"  WP Val  — dir={wp_m['dir_acc']:.3f} F1={wp_m['dir_f1']:.3f} | int={wp_m['int_acc']:.3f} F1={wp_m['int_f1']:.3f}")
    print(f"  SP Zero — dir={sp_zs['dir_acc']:.3f} F1={sp_zs['dir_f1']:.3f} | int={sp_zs['int_acc']:.3f} F1={sp_zs['int_f1']:.3f}")

    # Save WP checkpoint
    torch.save(model.state_dict(), CKPT_DIR / f"{model_name}_best_wp.pt")

    # Plot confusion matrices
    plot_confusion(wp_m, f"{model_name} WP Val", FIG_DIR, f"{model_name}_cm_wp")
    plot_confusion(sp_zs, f"{model_name} SP Zero-Shot", FIG_DIR, f"{model_name}_cm_sp_zs")

    # Fine-tune
    print(f"\n  Fine-tuning on SP...")
    ft_model = finetune_model(model.state_dict(), cfg["class"], model_params, loaders,
                               use_time=use_time)
    sp_ft = evaluate(ft_model, loaders["sp_test"], use_time=use_time)
    print(f"  SP FT   — dir={sp_ft['dir_acc']:.3f} F1={sp_ft['dir_f1']:.3f} | int={sp_ft['int_acc']:.3f} F1={sp_ft['int_f1']:.3f}")
    torch.save(ft_model.state_dict(), CKPT_DIR / f"{model_name}_best_ft.pt")
    plot_confusion(sp_ft, f"{model_name} SP Fine-Tuned", FIG_DIR, f"{model_name}_cm_sp_ft")

    # Save full results
    results = {
        "model": model_name, "params": n_params, "config": {**model_params, "lr":lr, "wd":wd, "dw":dw, "bs":bs},
        "wp_val": {"dir_acc": wp_m["dir_acc"], "dir_f1": wp_m["dir_f1"],
                   "int_acc": wp_m["int_acc"], "int_f1": wp_m["int_f1"]},
        "sp_zeroshot": {"dir_acc": sp_zs["dir_acc"], "dir_f1": sp_zs["dir_f1"],
                        "int_acc": sp_zs["int_acc"], "int_f1": sp_zs["int_f1"]},
        "sp_finetuned": {"dir_acc": sp_ft["dir_acc"], "dir_f1": sp_ft["dir_f1"],
                         "int_acc": sp_ft["int_acc"], "int_f1": sp_ft["int_f1"]},
        "epochs_trained": len(history["train_loss"]),
    }
    with open(LOG_DIR / f"{model_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f" {model_name} — Final Results")
    print(f"{'='*60}")
    for name, m in [("WP Val", wp_m), ("SP Zero-Shot", sp_zs), ("SP Fine-Tuned", sp_ft)]:
        print(f"  {name:15s} | dir={m['dir_acc']:.1%} F1={m['dir_f1']:.1%} | int={m['int_acc']:.1%} F1={m['int_f1']:.1%}")

    del model, ft_model; torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--hpo-epochs", type=int, default=50)
    parser.add_argument("--hpo-patience", type=int, default=12)
    parser.add_argument("--final-epochs", type=int, default=0, help="0 = HPO only")
    parser.add_argument("--final-patience", type=int, default=50)
    parser.add_argument("--skip-hpo", action="store_true", help="Skip HPO, use defaults")
    args = parser.parse_args()

    start = time.time()

    if args.skip_hpo:
        cfg = MODEL_REGISTRY[args.model]
        best_params = {**cfg["defaults"], **TRAIN_HPO["defaults"], "batch_size": 64}
    else:
        best_params, best_val = run_hpo(args.model, args.trials, args.hpo_epochs, args.hpo_patience)

    if args.final_epochs > 0:
        run_full_training(args.model, best_params, args.final_epochs, args.final_patience)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
