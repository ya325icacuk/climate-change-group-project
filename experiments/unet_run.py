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

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("..").resolve()
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ═══════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS (Optuna-tuned, see Section 2b)
# ═══════════════════════════════════════════════════════════════════════
#
# Selected via 60-trial Optuna HPO (50 epochs/trial, median pruner).
# Best trial: 0.622 val direction accuracy (50-epoch proxy).
# Full training with EMA reaches ~0.615–0.619 on WP val.
#
# VRAM budget: 15 GB (shared GPU). Peak usage ~9.5 GB with bc=128.
# ═══════════════════════════════════════════════════════════════════════

# ── Architecture ───────────────────────────────────────────────────────
# 4-level encoder: 128→256→512→1024 channels, bottleneck=2048
# Spatial trace: 81→41→21→11→6 (ceil_mode pooling for odd dims)
N_LEVELS      = 4
BASE_CHANNELS = 128     # wider network to fully utilise 15 GB VRAM
HEAD_DIM      = 256     # 3-layer MLP fusion head (256→128→K)
IN_CHANNELS   = 15      # SST(1) + u,v,z × 4 pressure levels(12) + shear + vorticity

# ── Regularisation ─────────────────────────────────────────────────────
DROPOUT       = 0.2     # Dropout2d in ConvBlocks + head dropout
DROP_PATH     = 0.0     # stochastic depth off (Optuna chose 0.0)
LABEL_SMOOTH  = 0.0     # label smoothing off (Optuna chose 0.0)

# ── Training ──────────────────────────────────────────────────────────
BATCH_SIZE    = 64
LR            = 5e-4    # OneCycleLR base (peak = 3× = 1.5e-3)
WEIGHT_DECAY  = 1.3e-3
EPOCHS        = 300     # max — early stopping on val_dir_acc, patience 50
PATIENCE      = 50
DIR_WEIGHT    = 0.5     # loss = DIR_WEIGHT·L_dir + (1-DIR_WEIGHT)·L_int
SCHEDULER     = "onecycle"
OPTIMIZER     = "adamw"

# ── Augmentation ──────────────────────────────────────────────────────
# Spatial flips disabled: flipping changes direction label meaning.
# Mixup works because it interpolates without altering spatial orientation.
AUG_HFLIP     = False
AUG_VFLIP     = False
USE_MIXUP     = True    # alpha=0.2, applied 50% of batches

# ── Multimodal fusion ─────────────────────────────────────────────────
USE_ENV = True    # 40-dim environmental features
USE_1D  = True    # 4-dim 1D features (lat, lon, wind, pressure offsets)

# ── Fine-tuning (WP → SP) ────────────────────────────────────────────
FT_LR       = 1e-4
FT_EPOCHS   = 80
FT_PATIENCE = 15

# ── Task ──────────────────────────────────────────────────────────────
N_DIR_CLASSES = 8
N_INT_CLASSES = 4
DIR_LABELS  = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
INTE_LABELS = ["Weakening", "Steady", "Slow-intens.", "Rapid-intens."]

# ── Reproducibility ───────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})")
print(f"Data dir: {DATA_DIR}")


# ── Load all splits ──
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
        "env":    torch.load(DATA_DIR / "env"     / f"{split}_env.pt",    weights_only=False),
        "data1d": torch.load(DATA_DIR / "data1d"  / f"{split}_1d.pt",    weights_only=False),
        "labels": torch.load(DATA_DIR / "labels"  / f"{split}_labels.pt", weights_only=False),
    }

print(f"Loaded {len(SPLITS)} splits")
for split in SPLITS:
    n_storms = len(raw[split]["grids"])
    n_ts = sum(v.shape[0] for v in raw[split]["grids"].values())
    print(f"  {split:15s}: {n_storms:3d} storms, {n_ts:5d} timesteps")

class CycloneDataset(Dataset):
    """Flattens storm-level dicts into timestep-level samples.

    Filters out sentinel labels (-1).  Grids are already normalised
    (channel-wise z-score) by the preprocessing pipeline.
    1D features are z-scored using training-set statistics passed at init.
    For SP splits, uses hemisphere-reflected direction labels.
    """
    def __init__(self, grids, env, data1d, labels, use_reflected=False,
                 d1d_mean=None, d1d_std=None):
        self.samples = []
        dir_key = "direction_reflected" if use_reflected else "direction"

        for storm_id in grids:
            g = grids[storm_id]        # (N_t, 15, 81, 81)
            e = env[storm_id]          # (N_t, 40)
            d = data1d[storm_id]       # (N_t, 4)
            d_lbl = labels[storm_id][dir_key]     # (N_t,)
            i_lbl = labels[storm_id]["intensity"]  # (N_t,)

            for t in range(g.shape[0]):
                if d_lbl[t].item() == -1 or i_lbl[t].item() == -1:
                    continue
                self.samples.append((
                    g[t], e[t], d[t],
                    d_lbl[t].long(), i_lbl[t].long(),
                ))

        # Compute or store 1D normalisation stats
        if d1d_mean is None:
            all_1d = torch.stack([s[2] for s in self.samples])
            self.d1d_mean = all_1d.mean(dim=0)
            self.d1d_std  = all_1d.std(dim=0).clamp(min=1e-6)
        else:
            self.d1d_mean = d1d_mean
            self.d1d_std  = d1d_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        grid, env, d1d, dir_lbl, int_lbl = self.samples[idx]
        d1d = (d1d - self.d1d_mean) / self.d1d_std
        return grid, env, d1d, dir_lbl, int_lbl

# ── Build datasets and loaders ──
# Build WP train first to get 1D normalisation stats
datasets = {}
datasets["wp_train"] = CycloneDataset(
    raw["wp_train"]["grids"], raw["wp_train"]["env"],
    raw["wp_train"]["data1d"], raw["wp_train"]["labels"],
    use_reflected=False
)
# Reuse WP train 1D stats for all other splits (prevent leakage)
d1d_mean = datasets["wp_train"].d1d_mean
d1d_std  = datasets["wp_train"].d1d_std
print(f"1D norm stats — mean: {d1d_mean.tolist()}, std: {d1d_std.tolist()}")

for split, cfg in SPLITS.items():
    if split == "wp_train":
        continue
    datasets[split] = CycloneDataset(
        raw[split]["grids"], raw[split]["env"], raw[split]["data1d"],
        raw[split]["labels"], use_reflected=cfg["reflected"],
        d1d_mean=d1d_mean, d1d_std=d1d_std
    )

loaders = {}
for split in SPLITS:
    shuffle = "train" in split
    loaders[split] = DataLoader(
        datasets[split], batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=1, pin_memory=True, drop_last=False
    )

for split, ds in datasets.items():
    print(f"{split:15s}: {len(ds):5d} valid samples")

# Verify a sample
g, e, d, dl, il = datasets["wp_train"][0]
print(f"\nSample shapes — grid: {g.shape}, env: {e.shape}, 1d: {d.shape}, dir: {dl}, int: {il}")
print(f"Grid stats  — mean: {g.mean():.4f}, std: {g.std():.4f}")
print(f"1D (normed) — {d}")

# ── Class weights (inverse frequency from WP train) ──
dir_counts = Counter()
int_counts = Counter()
for _, _, _, dl, il in datasets["wp_train"].samples:
    dir_counts[dl.item()] += 1
    int_counts[il.item()] += 1

n_total = len(datasets["wp_train"])

dir_weights = torch.zeros(N_DIR_CLASSES)
for c in range(N_DIR_CLASSES):
    dir_weights[c] = n_total / (N_DIR_CLASSES * max(dir_counts[c], 1))

int_weights = torch.zeros(N_INT_CLASSES)
for c in range(N_INT_CLASSES):
    int_weights[c] = n_total / (N_INT_CLASSES * max(int_counts[c], 1))

print("Direction class distribution (WP train):")
for c in range(N_DIR_CLASSES):
    print(f"  {DIR_LABELS[c]:4s}: {dir_counts[c]:4d} ({dir_counts[c]/n_total*100:5.1f}%)  weight={dir_weights[c]:.2f}")

print(f"\nIntensity class distribution (WP train):")
for c in range(N_INT_CLASSES):
    print(f"  {INTE_LABELS[c]:14s}: {int_counts[c]:4d} ({int_counts[c]/n_total*100:5.1f}%)  weight={int_weights[c]:.2f}")

def mixup_data(x, y1, y2, alpha=0.2):
    """Mixup augmentation: interpolate between random pairs of samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y1, y2, y1[idx], y2[idx], lam

print("mixup_data helper defined")

class DropPath(nn.Module):
    """Stochastic depth — randomly drop entire residual branches during training."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = torch.rand(x.size(0), 1, 1, 1, device=x.device) > self.drop_prob
        return x * keep / (1 - self.drop_prob)


class ConvBlock(nn.Module):
    """Double convolution with residual + drop path: Conv → BN → GELU → Drop → Conv → BN → GELU + skip."""
    def __init__(self, in_ch, out_ch, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return self.drop_path(self.net(x)) + self.residual(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, max(ch // reduction, 4)),
            nn.GELU(),
            nn.Linear(max(ch // reduction, 4), ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class EncoderBlock(nn.Module):
    """ConvBlock + SE attention + MaxPool2d for downsampling."""
    def __init__(self, in_ch, out_ch, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout, drop_path)
        self.se = SEBlock(out_ch)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)

    def forward(self, x):
        skip = self.se(self.conv(x))
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """Upsample → crop/pad to match skip → concatenate → ConvBlock."""
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout, drop_path)

    def forward(self, x, skip):
        x = self.up(x)
        dh = x.size(2) - skip.size(2)
        dw = x.size(3) - skip.size(3)
        if dh > 0 or dw > 0:
            x = x[:, :, dh // 2 : dh // 2 + skip.size(2),
                        dw // 2 : dw // 2 + skip.size(3)]
        elif dh < 0 or dw < 0:
            x = F.pad(x, [0, -dw, 0, -dh])
        return self.conv(torch.cat([x, skip], dim=1))

class UNet2dClassifier(nn.Module):
    """U-Net encoder-decoder with SE attention, residual connections, and stochastic depth."""
    def __init__(self, in_channels=15, base_channels=32, n_levels=4,
                 n_dir_classes=8, n_int_classes=4,
                 env_dim=40, d1d_dim=4,
                 use_env=True, use_1d=True,
                 dropout=0.0, head_dim=128, drop_path=0.1):
        super().__init__()
        self.n_levels = n_levels
        self.use_env = use_env
        self.use_1d = use_1d
        self.base_channels = base_channels

        # Linearly increasing drop path rate per level
        dp_rates = [drop_path * i / max(n_levels, 1) for i in range(n_levels + 1)]

        # ── Encoder ──
        self.encoders = nn.ModuleList()
        ch_in = in_channels
        for i in range(n_levels):
            ch_out = base_channels * (2 ** i)
            self.encoders.append(EncoderBlock(ch_in, ch_out, dropout, dp_rates[i]))
            ch_in = ch_out

        # ── Bottleneck ──
        bottleneck_ch = base_channels * (2 ** n_levels)
        self.bottleneck = ConvBlock(ch_in, bottleneck_ch, dropout, dp_rates[n_levels])

        # ── Decoder ──
        self.decoders = nn.ModuleList()
        ch_in = bottleneck_ch
        for i in range(n_levels - 1, -1, -1):
            skip_ch = base_channels * (2 ** i)
            self.decoders.append(DecoderBlock(ch_in, skip_ch, skip_ch, dropout, dp_rates[i]))
            ch_in = skip_ch

        # ── Global average pooling → fusion → heads ──
        self.gap = nn.AdaptiveAvgPool2d(1)
        fusion_dim = base_channels
        if use_env:
            fusion_dim += env_dim
        if use_1d:
            fusion_dim += d1d_dim

        self.head_dir = nn.Sequential(
            nn.Linear(fusion_dim, head_dim), nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(head_dim, head_dim // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim // 2, n_dir_classes),
        )
        self.head_int = nn.Sequential(
            nn.Linear(fusion_dim, head_dim), nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(head_dim, head_dim // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim // 2, n_int_classes),
        )

    def forward(self, grid, env=None, d1d=None):
        skips = []
        x = grid
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        x = self.gap(x).flatten(1)

        parts = [x]
        if self.use_env and env is not None:
            parts.append(env)
        if self.use_1d and d1d is not None:
            parts.append(d1d)
        fused = torch.cat(parts, dim=1)

        return self.head_dir(fused), self.head_int(fused)


# ── Instantiate and inspect ──
model = UNet2dClassifier(
    in_channels=IN_CHANNELS, base_channels=BASE_CHANNELS,
    n_levels=N_LEVELS, n_dir_classes=N_DIR_CLASSES,
    n_int_classes=N_INT_CLASSES, env_dim=40, d1d_dim=4,
    use_env=USE_ENV, use_1d=USE_1D, dropout=0.1,
    head_dim=HEAD_DIM, drop_path=0.1,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"UNet2dClassifier — {n_params:,} parameters")
print(f"  Levels: {N_LEVELS}, Base channels: {BASE_CHANNELS}")
print(f"  Head dim: {HEAD_DIM}")

# Smoke test
with torch.no_grad():
    g = torch.randn(2, 15, 81, 81, device=DEVICE)
    e = torch.randn(2, 40, device=DEVICE)
    d = torch.randn(2, 4, device=DEVICE)
    d_out, i_out = model(g, e, d)
print(f"  Smoke test passed: dir={d_out.shape}, int={i_out.shape}")


import optuna, gc
from optuna.trial import TrialState
optuna.logging.set_verbosity(optuna.logging.WARNING)

_loader_train = loaders["wp_train"]
_loader_val   = loaders["wp_val"]
_dir_weights  = dir_weights.to(DEVICE)
_int_weights  = int_weights.to(DEVICE)

# ── HPO config ─────────────────────────────────────────────────────────
OPTUNA_EPOCHS   = 30    # longer proxy training for more reliable signal
OPTUNA_PATIENCE = 8    # early stopping within each trial
N_TRIALS        = 30


def _cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def objective(trial):
    # ── Architecture (fit within 15 GB VRAM) ──
    n_levels = trial.suggest_int("n_levels", 3, 4)
    base_ch  = trial.suggest_categorical("base_channels", [64, 96, 128])
    head     = trial.suggest_int("head_dim", 128, 256, step=64)

    # ── Regularisation ──
    dropout      = trial.suggest_float("dropout", 0.1, 0.3, step=0.05)
    drop_path    = trial.suggest_float("drop_path", 0.0, 0.2, step=0.05)
    label_smooth = trial.suggest_float("label_smoothing", 0.0, 0.1, step=0.05)
    dir_weight   = trial.suggest_float("dir_weight", 0.4, 0.7, step=0.1)

    # ── Augmentation ──
    use_mixup = trial.suggest_categorical("use_mixup", [True, False])

    # ── Optimiser ──
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 5e-4, 5e-3, log=True)

    # ── Scheduler ──
    sched_name = trial.suggest_categorical("scheduler", ["cosine", "onecycle"])

    # ── Build model ──
    torch.manual_seed(SEED)
    m = UNet2dClassifier(
        in_channels=IN_CHANNELS, base_channels=base_ch, n_levels=n_levels,
        n_dir_classes=N_DIR_CLASSES, n_int_classes=N_INT_CLASSES,
        env_dim=40, d1d_dim=4, use_env=True, use_1d=True,
        dropout=dropout, head_dim=head, drop_path=drop_path,
    )
    try:
        m = m.to(DEVICE)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
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

    loss_d = nn.CrossEntropyLoss(weight=_dir_weights, label_smoothing=label_smooth)
    loss_i = nn.CrossEntropyLoss(weight=_int_weights, label_smoothing=label_smooth)

    best_val_acc = 0.0
    patience_ctr = 0

    try:
        for epoch in range(1, OPTUNA_EPOCHS + 1):
            m.train()
            for grid, env, d1d, dl, il in _loader_train:
                grid, env, d1d = grid.to(DEVICE), env.to(DEVICE), d1d.to(DEVICE)
                dl, il = dl.to(DEVICE), il.to(DEVICE)

                if use_mixup and torch.rand(1).item() > 0.5:
                    grid, dl, il, dl_b, il_b, lam = mixup_data(grid, dl, il)
                    d_out, i_out = m(grid, env, d1d)
                    loss = dir_weight * (lam * loss_d(d_out, dl) + (1 - lam) * loss_d(d_out, dl_b)) + \
                           (1 - dir_weight) * (lam * loss_i(i_out, il) + (1 - lam) * loss_i(i_out, il_b))
                else:
                    d_out, i_out = m(grid, env, d1d)
                    loss = dir_weight * loss_d(d_out, dl) + (1 - dir_weight) * loss_i(i_out, il)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                if step_per_batch:
                    sched.step()

            if not step_per_batch:
                sched.step()

            m.eval()
            preds, trues = [], []
            with torch.no_grad():
                for grid, env, d1d, dl, il in _loader_val:
                    grid, env, d1d = grid.to(DEVICE), env.to(DEVICE), d1d.to(DEVICE)
                    preds.extend(m(grid, env, d1d)[0].argmax(1).cpu().tolist())
                    trues.extend(dl.tolist())
            val_acc = accuracy_score(trues, preds)

            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= OPTUNA_PATIENCE:
                    break
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            raise optuna.TrialPruned()
        raise
    finally:
        try: m.cpu()
        except Exception: pass
        del m, opt; _cleanup_gpu()

    return best_val_acc


_cleanup_gpu()
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8),
    study_name="unet_hpo",
)
study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,),
               show_progress_bar=True)

print(f"\n{'='*60}")
print(f" Optuna search complete — {len(study.trials)} trials")
print(f"{'='*60}")
completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
print(f" Completed: {len(completed)}, Pruned: {len(study.trials) - len(completed)}")
if completed:
    print(f" Best val direction accuracy: {study.best_value:.4f}")
    print(f" Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"   {k:25s}: {v}")
else:
    print(" WARNING: No trials completed. Using default hyperparameters.")


# ── Optuna visualisation ──
from optuna.trial import TrialState
completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
if len(completed) > 2:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Optimization history
    vals = [t.value for t in completed]
    axes[0].plot(vals, marker='o', ms=4, alpha=0.6)
    axes[0].set_xlabel("Completed trial")
    axes[0].set_ylabel("Val direction accuracy")
    axes[0].set_title("Optimisation history")

    # 2. Param importances (top 8)
    try:
        importances = optuna.importance.get_param_importances(study)
        names = list(importances.keys())[:8]
        vals_imp = [importances[n] for n in names]
        axes[1].barh(names[::-1], vals_imp[::-1])
        axes[1].set_xlabel("Importance")
        axes[1].set_title("Hyperparameter importance")
    except Exception as e:
        axes[1].text(0.5, 0.5, f"Importance unavailable:\n{e}",
                     ha="center", va="center", transform=axes[1].transAxes)

    # 3. Parallel coordinate (top 5 trials)
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    param_names = list(top5[0].params.keys())[:6]
    for i, t in enumerate(top5):
        vals_p = [t.params.get(p, 0) for p in param_names]
        normed = []
        for j, v in enumerate(vals_p):
            all_v = [tt.params.get(param_names[j], 0) for tt in completed]
            mn, mx = min(all_v), max(all_v)
            normed.append((v - mn) / (mx - mn + 1e-8))
        axes[2].plot(range(len(param_names)), normed, marker='o', label=f"#{i+1} ({t.value:.3f})")
    axes[2].set_xticks(range(len(param_names)))
    axes[2].set_xticklabels(param_names, rotation=45, ha="right")
    axes[2].set_title("Top-5 trials (normalised params)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures" / "unet_optuna_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: figures/unet_optuna_results.png")
else:
    print(f"Only {len(completed)} completed trials — skipping Optuna plots")


# ═══════════════════════════════════════════════════════════════════════
# Apply best hyperparameters from Optuna
# ═══════════════════════════════════════════════════════════════════════
completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
if completed:
    bp = study.best_params
    print(f"Using Optuna best (val_dir_acc={study.best_value:.4f}):")
else:
    # Fallback: proven best from prior runs
    bp = {"n_levels": 4, "base_channels": 128, "head_dim": 256,
          "dropout": 0.2, "drop_path": 0.0,
          "label_smoothing": 0.0, "dir_weight": 0.5,
          "use_mixup": True,
          "lr": 5e-4, "weight_decay": 1.3e-3,
          "scheduler": "onecycle"}
    print("No completed trials — using fallback defaults:")

N_LEVELS       = bp["n_levels"]
BASE_CHANNELS  = bp["base_channels"]
BEST_HEAD_DIM  = bp["head_dim"]
BEST_DROPOUT   = bp["dropout"]
BEST_DROP_PATH = bp.get("drop_path", 0.0)
BEST_LABEL_SMOOTH = bp["label_smoothing"]
DIR_WEIGHT     = bp["dir_weight"]
BEST_AUG_HFLIP = bp.get("aug_hflip", False)
BEST_AUG_VFLIP = bp.get("aug_vflip", False)
BEST_MIXUP     = bp["use_mixup"]
LR             = bp["lr"]
WEIGHT_DECAY   = bp["weight_decay"]
OPTIMIZER      = bp.get("optimizer", "adamw")
BEST_SCHEDULER = bp["scheduler"]

for k, v in bp.items():
    print(f"  {k:25s}: {v}")

# ── Rebuild model with best params ──
torch.manual_seed(SEED)
model = UNet2dClassifier(
    in_channels=IN_CHANNELS, base_channels=BASE_CHANNELS,
    n_levels=N_LEVELS, n_dir_classes=N_DIR_CLASSES,
    n_int_classes=N_INT_CLASSES, env_dim=40, d1d_dim=4,
    use_env=USE_ENV, use_1d=USE_1D, dropout=BEST_DROPOUT,
    head_dim=BEST_HEAD_DIM, drop_path=BEST_DROP_PATH,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {n_params:,} parameters")
print(f"Estimated VRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")


# ═══════════════════════════════════════════════════════════════════════
# Loss functions, optimiser, and learning rate scheduler
# ═══════════════════════════════════════════════════════════════════════

# ── Loss: weighted cross-entropy for class imbalance ──────────────────
# Class weights computed from WP train distribution (inverse frequency).
# Label smoothing softens targets to prevent overconfident predictions.
loss_dir_fn = nn.CrossEntropyLoss(weight=dir_weights.to(DEVICE),
                                   label_smoothing=BEST_LABEL_SMOOTH)
loss_int_fn = nn.CrossEntropyLoss(weight=int_weights.to(DEVICE),
                                   label_smoothing=BEST_LABEL_SMOOTH)

# ── Optimiser: AdamW ──────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                              weight_decay=WEIGHT_DECAY)
print(f"Optimizer: AdamW (lr={LR:.5f}, wd={WEIGHT_DECAY:.5f})")

# ── Scheduler ─────────────────────────────────────────────────────────
# OneCycleLR: warmup to 3x base LR, then cosine decay.
# This aggressive schedule trains more epochs before plateauing,
# and consistently outperformed plain cosine annealing in our HPO.
if BEST_SCHEDULER == "onecycle":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 3, epochs=EPOCHS,
        steps_per_epoch=len(loaders["wp_train"]))
    STEP_PER_BATCH = True
    print(f"Scheduler: OneCycleLR (max_lr={LR*3:.5f})")
elif BEST_SCHEDULER == "cosine_warm_restarts":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    STEP_PER_BATCH = False
    print(f"Scheduler: CosineWarmRestarts (T_0=30, T_mult=2)")
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)
    STEP_PER_BATCH = False
    print(f"Scheduler: CosineAnnealing (T_max={EPOCHS})")


class EMA:
    """Exponential Moving Average of model parameters for more stable evaluation."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.clone()

    def apply(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)


def mixup_data(x, y1, y2, alpha=0.2):
    """Mixup augmentation: interpolate between random pairs of samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y1, y2, y1[idx], y2[idx], lam


def train_one_epoch(model, loader, optimizer, device, scheduler=None,
                    step_per_batch=False, aug_hflip=False, aug_vflip=False,
                    use_mixup=False, ema=None):
    model.train()
    total_loss = 0.0
    correct_dir = correct_int = total = 0

    for grid, env, d1d, dir_lbl, int_lbl in loader:
        grid    = grid.to(device)
        env     = env.to(device)
        d1d     = d1d.to(device)
        dir_lbl = dir_lbl.to(device)
        int_lbl = int_lbl.to(device)

        if aug_hflip and torch.rand(1).item() > 0.5:
            grid = grid.flip(-1)
        if aug_vflip and torch.rand(1).item() > 0.5:
            grid = grid.flip(-2)

        if use_mixup and torch.rand(1).item() > 0.5:
            grid, dir_lbl, int_lbl, dir_lbl_b, int_lbl_b, lam = mixup_data(
                grid, dir_lbl, int_lbl, alpha=0.2)
            dir_logits, int_logits = model(grid, env, d1d)
            l_dir = lam * loss_dir_fn(dir_logits, dir_lbl) + (1 - lam) * loss_dir_fn(dir_logits, dir_lbl_b)
            l_int = lam * loss_int_fn(int_logits, int_lbl) + (1 - lam) * loss_int_fn(int_logits, int_lbl_b)
        else:
            dir_logits, int_logits = model(grid, env, d1d)
            l_dir = loss_dir_fn(dir_logits, dir_lbl)
            l_int = loss_int_fn(int_logits, int_lbl)

        loss = DIR_WEIGHT * l_dir + (1 - DIR_WEIGHT) * l_int

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        if step_per_batch and scheduler is not None:
            scheduler.step()

        bs = grid.size(0)
        total_loss  += loss.item() * bs
        correct_dir += (dir_logits.argmax(1) == dir_lbl).sum().item()
        correct_int += (int_logits.argmax(1) == int_lbl).sum().item()
        total       += bs

    return total_loss / total, correct_dir / total, correct_int / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_dir_pred, all_dir_true = [], []
    all_int_pred, all_int_true = [], []
    total = 0

    for grid, env, d1d, dir_lbl, int_lbl in loader:
        grid    = grid.to(device)
        env     = env.to(device)
        d1d     = d1d.to(device)
        dir_lbl = dir_lbl.to(device)
        int_lbl = int_lbl.to(device)

        dir_logits, int_logits = model(grid, env, d1d)
        l_dir = loss_dir_fn(dir_logits, dir_lbl)
        l_int = loss_int_fn(int_logits, int_lbl)
        loss  = DIR_WEIGHT * l_dir + (1 - DIR_WEIGHT) * l_int

        bs = grid.size(0)
        total_loss += loss.item() * bs
        total      += bs

        all_dir_pred.extend(dir_logits.argmax(1).cpu().tolist())
        all_dir_true.extend(dir_lbl.cpu().tolist())
        all_int_pred.extend(int_logits.argmax(1).cpu().tolist())
        all_int_true.extend(int_lbl.cpu().tolist())

    metrics = {
        "loss":    total_loss / total,
        "dir_acc": accuracy_score(all_dir_true, all_dir_pred),
        "int_acc": accuracy_score(all_int_true, all_int_pred),
        "dir_f1":  f1_score(all_dir_true, all_dir_pred, average="macro",
                            zero_division=0),
        "int_f1":  f1_score(all_int_true, all_int_pred, average="macro",
                            zero_division=0),
        "dir_pred": all_dir_pred, "dir_true": all_dir_true,
        "int_pred": all_int_pred, "int_true": all_int_true,
    }
    return metrics


# ── Training loop with early stopping on val_dir_acc + EMA ──
history = {"train_loss": [], "val_loss": [],
           "train_dir_acc": [], "val_dir_acc": [],
           "train_int_acc": [], "val_int_acc": []}

best_val_dir_acc = 0.0
best_model_state = None
patience_counter = 0
ema = EMA(model, decay=0.998)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_dir_acc, train_int_acc = train_one_epoch(
        model, loaders["wp_train"], optimizer, DEVICE,
        scheduler=scheduler if STEP_PER_BATCH else None,
        step_per_batch=STEP_PER_BATCH,
        aug_hflip=BEST_AUG_HFLIP, aug_vflip=BEST_AUG_VFLIP,
        use_mixup=True, ema=ema)

    # Evaluate with EMA weights
    ema.apply(model)
    val_metrics = evaluate(model, loaders["wp_val"], DEVICE)
    ema.restore(model)

    if not STEP_PER_BATCH:
        scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_metrics["loss"])
    history["train_dir_acc"].append(train_dir_acc)
    history["val_dir_acc"].append(val_metrics["dir_acc"])
    history["train_int_acc"].append(train_int_acc)
    history["val_int_acc"].append(val_metrics["int_acc"])

    # Early stopping on val direction accuracy (not loss)
    if val_metrics["dir_acc"] > best_val_dir_acc:
        best_val_dir_acc = val_metrics["dir_acc"]
        best_model_state = deepcopy(dict(ema.shadow))  # save EMA weights
        patience_counter = 0
        marker = " ★"
    else:
        patience_counter += 1
        marker = ""

    if epoch % 5 == 0 or epoch == 1 or marker:
        print(f"Epoch {epoch:3d}/{EPOCHS} │ "
              f"Train loss={train_loss:.4f} dir={train_dir_acc:.3f} int={train_int_acc:.3f} │ "
              f"Val loss={val_metrics['loss']:.4f} dir={val_metrics['dir_acc']:.3f} "
              f"int={val_metrics['int_acc']:.3f}{marker}")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
        break

# Restore best EMA model
model.load_state_dict(best_model_state)
print(f"\nBest val dir acc (EMA): {best_val_dir_acc:.4f}")
print(f"Best val loss: {min(history['val_loss']):.4f}")


# ── Learning curves ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"],   label="Val")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Combined Loss"); axes[0].legend()

axes[1].plot(history["train_dir_acc"], label="Train")
axes[1].plot(history["val_dir_acc"],   label="Val")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Direction Accuracy"); axes[1].legend()

axes[2].plot(history["train_int_acc"], label="Train")
axes[2].plot(history["val_int_acc"],   label="Val")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy")
axes[2].set_title("Intensity Accuracy"); axes[2].legend()

for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "figures" / "unet_learning_curves.png", dpi=150,
            bbox_inches="tight")
plt.show()

def plot_confusion_matrices(metrics, title_prefix, labels_dir=DIR_LABELS,
                            labels_int=INTE_LABELS):
    """Plot direction and intensity confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm_dir = confusion_matrix(metrics["dir_true"], metrics["dir_pred"],
                              labels=list(range(len(labels_dir))))
    cm_int = confusion_matrix(metrics["int_true"], metrics["int_pred"],
                              labels=list(range(len(labels_int))))
    
    sns.heatmap(cm_dir, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_dir, yticklabels=labels_dir, ax=axes[0])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].set_title(f"{title_prefix} — Direction")
    
    sns.heatmap(cm_int, annot=True, fmt="d", cmap="Oranges",
                xticklabels=labels_int, yticklabels=labels_int, ax=axes[1])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].set_title(f"{title_prefix} — Intensity")
    
    plt.tight_layout()
    return fig


def print_metrics(metrics, split_name):
    """Print accuracy and F1 summary."""
    print(f"\n{'='*50}")
    print(f" {split_name}")
    print(f"{'='*50}")
    print(f"  Direction — Acc: {metrics['dir_acc']:.3f}  "
          f"F1 (macro): {metrics['dir_f1']:.3f}")
    print(f"  Intensity — Acc: {metrics['int_acc']:.3f}  "
          f"F1 (macro): {metrics['int_f1']:.3f}")

# ── Evaluate on WP validation ──
wp_val_metrics = evaluate(model, loaders["wp_val"], DEVICE)
print(f"WP Validation:  dir_acc={wp_val_metrics['dir_acc']:.4f}  "
      f"dir_f1={wp_val_metrics['dir_f1']:.4f}  "
      f"int_acc={wp_val_metrics['int_acc']:.4f}  "
      f"int_f1={wp_val_metrics['int_f1']:.4f}")


fig = plot_confusion_matrices(wp_val_metrics, "WP Validation")
fig.savefig(PROJECT_ROOT / "figures" / "unet_wp_val_confusion.png", dpi=150,
            bbox_inches="tight")
plt.show()

# ── Zero-shot evaluation on SP test ──
sp_zs_metrics = evaluate(model, loaders["sp_test"], DEVICE)
print(f"SP Test (zero-shot):  dir_acc={sp_zs_metrics['dir_acc']:.4f}  "
      f"dir_f1={sp_zs_metrics['dir_f1']:.4f}  "
      f"int_acc={sp_zs_metrics['int_acc']:.4f}  "
      f"int_f1={sp_zs_metrics['int_f1']:.4f}")


fig = plot_confusion_matrices(sp_zs_metrics, "SP Test (zero-shot)")
fig.savefig(PROJECT_ROOT / "figures" / "unet_sp_zeroshot_confusion.png", dpi=150,
            bbox_inches="tight")
plt.show()

# ── Transfer gap analysis ──
print("\n" + "=" * 65)
print(" Transfer Gap: WP Val vs SP Zero-Shot")
print("=" * 65)
print(f"{'Metric':<20s} {'WP Val':>10s} {'SP Zero-Shot':>12s} {'Gap':>10s}")
print("-" * 55)
for metric, label in [("dir_acc", "Dir Accuracy"),
                       ("dir_f1",  "Dir F1 (macro)"),
                       ("int_acc", "Int Accuracy"),
                       ("int_f1",  "Int F1 (macro)")]:
    wp = wp_val_metrics[metric]
    sp = sp_zs_metrics[metric]
    gap = sp - wp
    print(f"{label:<20s} {wp:>10.3f} {sp:>12.3f} {gap:>+10.3f}")

def finetune(model_state, loaders_ft_train, loaders_ft_val,
             freeze_backbone=False, lr=FT_LR, epochs=FT_EPOCHS,
             patience=FT_PATIENCE):
    """Fine-tune a model from a saved state dict."""
    ft_model = UNet2dClassifier(
        in_channels=IN_CHANNELS, base_channels=BASE_CHANNELS,
        n_levels=N_LEVELS, n_dir_classes=N_DIR_CLASSES,
        n_int_classes=N_INT_CLASSES, env_dim=40, d1d_dim=4,
        use_env=USE_ENV, use_1d=USE_1D, dropout=BEST_DROPOUT,
        head_dim=BEST_HEAD_DIM, drop_path=BEST_DROP_PATH,
    ).to(DEVICE)
    ft_model.load_state_dict(model_state)

    if freeze_backbone:
        for name, p in ft_model.named_parameters():
            if "head" not in name:
                p.requires_grad = False
        trainable = [p for p in ft_model.parameters() if p.requires_grad]
        print(f"Head-only: {sum(p.numel() for p in trainable):,} trainable params")
    else:
        trainable = ft_model.parameters()
        print("Full fine-tune: all params trainable")

    ft_optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)
    ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        ft_optimizer, T_max=epochs, eta_min=1e-6)

    best_dir_acc = 0.0
    best_state = None
    wait = 0
    ft_history = {"train_loss": [], "val_loss": [],
                  "val_dir_acc": [], "val_int_acc": []}

    for epoch in range(1, epochs + 1):
        t_loss, _, _ = train_one_epoch(ft_model, loaders_ft_train,
                                        ft_optimizer, DEVICE)
        v = evaluate(ft_model, loaders_ft_val, DEVICE)
        ft_scheduler.step()

        ft_history["train_loss"].append(t_loss)
        ft_history["val_loss"].append(v["loss"])
        ft_history["val_dir_acc"].append(v["dir_acc"])
        ft_history["val_int_acc"].append(v["int_acc"])

        if v["dir_acc"] > best_dir_acc:
            best_dir_acc = v["dir_acc"]
            best_state = deepcopy(ft_model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  FT Epoch {epoch:3d} │ train_loss={t_loss:.4f} "
                  f"val_loss={v['loss']:.4f} dir={v['dir_acc']:.3f} "
                  f"int={v['int_acc']:.3f}")

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    ft_model.load_state_dict(best_state)
    return ft_model, ft_history


# ── Strategy 1: Full fine-tune ──
print("Strategy 1: Full Fine-Tuning")
print("-" * 40)
ft_full_model, ft_full_hist = finetune(
    best_model_state, loaders["sp_ft_train"], loaders["sp_ft_val"],
    freeze_backbone=False
)
ft_full_metrics = evaluate(ft_full_model, loaders["sp_ft_val"], DEVICE)
print_metrics(ft_full_metrics, "SP Fine-Tuned (full) — Val")

# ── Strategy 2: Head-only fine-tune ──
print("Strategy 2: Head-Only Fine-Tuning")
print("-" * 40)
ft_head_model, ft_head_hist = finetune(
    best_model_state, loaders["sp_ft_train"], loaders["sp_ft_val"],
    freeze_backbone=True
)
ft_head_metrics = evaluate(ft_head_model, loaders["sp_ft_val"], DEVICE)
print_metrics(ft_head_metrics, "SP Fine-Tuned (head-only) — Val")

# ── Fine-tuning learning curves ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for hist, label in [(ft_full_hist, "Full FT"), (ft_head_hist, "Head-only FT")]:
    axes[0].plot(hist["val_loss"], label=label)
    axes[1].plot(hist["val_dir_acc"], label=label)

axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val Loss")
axes[0].set_title("Fine-Tuning Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dir Accuracy")
axes[1].set_title("Fine-Tuning Dir Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "figures" / "unet_finetune_curves.png", dpi=150,
            bbox_inches="tight")
plt.show()

# ── Evaluate best fine-tuned model on sp_test ──
ft_full_best = max(ft_full_hist["val_dir_acc"])
ft_head_best = max(ft_head_hist["val_dir_acc"])

if ft_full_best >= ft_head_best:
    best_ft_model = ft_full_model
    best_ft_strategy = "full"
    best_ft_history = ft_full_hist
else:
    best_ft_model = ft_head_model
    best_ft_strategy = "head-only"
    best_ft_history = ft_head_hist

print(f"Best fine-tuning strategy: {best_ft_strategy}")
sp_ft_test_metrics = evaluate(best_ft_model, loaders["sp_test"], DEVICE)
print(f"SP Test (fine-tuned {best_ft_strategy}):  "
      f"dir_acc={sp_ft_test_metrics['dir_acc']:.4f}  "
      f"dir_f1={sp_ft_test_metrics['dir_f1']:.4f}  "
      f"int_acc={sp_ft_test_metrics['int_acc']:.4f}  "
      f"int_f1={sp_ft_test_metrics['int_f1']:.4f}")


fig = plot_confusion_matrices(sp_ft_test_metrics,
                              f"SP Test (fine-tuned, {best_ft_strategy})")
fig.savefig(PROJECT_ROOT / "figures" / "unet_sp_finetuned_confusion.png", dpi=150,
            bbox_inches="tight")
plt.show()

# ── Three-way comparison ──
print("\n" + "=" * 75)
print(" Cross-Basin Transfer Summary")
print("=" * 75)
print(f"{'Setting':<30s} {'Dir Acc':>8s} {'Dir F1':>8s} "
      f"{'Int Acc':>8s} {'Int F1':>8s}")
print("-" * 65)

for name, m in [
    ("WP Val (in-basin)",          wp_val_metrics),
    ("SP Test (zero-shot)",        sp_zs_metrics),
    (f"SP Test (FT {best_ft_strategy})", sp_ft_test_metrics),
]:
    print(f"{name:<30s} {m['dir_acc']:>8.3f} {m['dir_f1']:>8.3f} "
          f"{m['int_acc']:>8.3f} {m['int_f1']:>8.3f}")

# ── Encoder feature activation visualization ──
# Run a single sample through the encoder and visualise activations at each level
model.eval()
sample_grid, sample_env, sample_d1d, _, _ = datasets["wp_val"][0]
sample_grid = sample_grid.unsqueeze(0).to(DEVICE)

# Extract encoder activations
encoder_acts = []
x = sample_grid
with torch.no_grad():
    for enc in model.encoders:
        skip, x = enc(x)
        encoder_acts.append(skip.cpu())

n_levels_actual = len(encoder_acts)
fig, axes = plt.subplots(2, n_levels_actual, figsize=(4 * n_levels_actual, 7))
if n_levels_actual == 1:
    axes = axes.reshape(2, 1)

for i, act in enumerate(encoder_acts):
    # Mean activation across channels
    mean_act = act[0].mean(dim=0).numpy()
    im = axes[0, i].imshow(mean_act, cmap="viridis", aspect="auto")
    axes[0, i].set_title(f"Level {i} — Mean ({act.shape[1]}ch, {act.shape[2]}×{act.shape[3]})")
    plt.colorbar(im, ax=axes[0, i], shrink=0.8)

    # Max activation across channels
    max_act = act[0].max(dim=0)[0].numpy()
    im = axes[1, i].imshow(max_act, cmap="magma", aspect="auto")
    axes[1, i].set_title(f"Level {i} — Max")
    plt.colorbar(im, ax=axes[1, i], shrink=0.8)

axes[0, 0].set_ylabel("Mean activation")
axes[1, 0].set_ylabel("Max activation")
plt.suptitle("Encoder Feature Activations by Level", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "figures" / "unet_encoder_activations.png", dpi=150,
            bbox_inches="tight")
plt.show()

# ── Level importance analysis ──
# Measure the L2 norm of encoder conv weights at each level as a proxy for importance
level_norms = {"encoder": [], "decoder": []}

for i, enc in enumerate(model.encoders):
    norm = sum(p.data.norm().item() for p in enc.conv.net.parameters() if p.ndim >= 2)
    level_norms["encoder"].append(norm)

for i, dec in enumerate(model.decoders):
    norm = sum(p.data.norm().item() for p in dec.conv.net.parameters() if p.ndim >= 2)
    level_norms["decoder"].append(norm)

# Also measure activation magnitudes over a batch
model.eval()
batch_grid = next(iter(loaders["wp_val"]))[0][:16].to(DEVICE)
enc_act_norms = []
x = batch_grid
with torch.no_grad():
    for enc in model.encoders:
        skip, x = enc(x)
        enc_act_norms.append(skip.norm(dim=(2, 3)).mean().item())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Weight norms
levels = list(range(n_levels_actual))
axes[0].bar(levels, level_norms["encoder"], color="steelblue", label="Encoder")
axes[0].bar([l + 0.35 for l in levels], level_norms["decoder"][::-1],
            width=0.35, color="coral", label="Decoder")
axes[0].set_xlabel("Level"); axes[0].set_ylabel("Sum of Weight L2 Norms")
axes[0].set_title("Conv Weight Magnitude per Level")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Activation norms
axes[1].bar(levels, enc_act_norms, color="teal")
axes[1].set_xlabel("Level"); axes[1].set_ylabel("Mean Activation L2 Norm")
axes[1].set_title("Encoder Activation Magnitude")
axes[1].grid(True, alpha=0.3)

# Channel counts
enc_channels = [BASE_CHANNELS * (2 ** i) for i in range(n_levels_actual)]
axes[2].bar(levels, enc_channels, color="mediumpurple")
axes[2].set_xlabel("Level"); axes[2].set_ylabel("Channels")
axes[2].set_title("Channel Count per Level")
axes[2].grid(True, alpha=0.3)

plt.suptitle("U-Net Level Importance Analysis", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "figures" / "unet_level_importance.png", dpi=150,
            bbox_inches="tight")
plt.show()

print("Level importance summary:")
for i in range(n_levels_actual):
    print(f"  Level {i}: channels={enc_channels[i]:3d}, "
          f"weight_norm={level_norms['encoder'][i]:.2f}, "
          f"act_norm={enc_act_norms[i]:.2f}")

# ── Per-storm prediction timeline (sample SP storm) ──
# Pick one storm from sp_test and show predictions over time
sp_grids  = raw["sp_test"]["grids"]
sp_env    = raw["sp_test"]["env"]
sp_d1d    = raw["sp_test"]["data1d"]
sp_labels = raw["sp_test"]["labels"]

# Pick the longest storm
storm_id = max(sp_grids, key=lambda k: sp_grids[k].shape[0])
n_ts = sp_grids[storm_id].shape[0]
print(f"Storm: {storm_id} ({n_ts} timesteps)")

# Run predictions for each timestep
model.eval()
dir_preds, int_preds = [], []
dir_confs, int_confs = [], []
dir_true_list, int_true_list = [], []

with torch.no_grad():
    for t in range(n_ts):
        d_lbl = sp_labels[storm_id]["direction_reflected"][t].item()
        i_lbl = sp_labels[storm_id]["intensity"][t].item()
        if d_lbl == -1 or i_lbl == -1:
            continue

        g = sp_grids[storm_id][t].unsqueeze(0).to(DEVICE)
        e = sp_env[storm_id][t].unsqueeze(0).to(DEVICE)
        d = sp_d1d[storm_id][t].unsqueeze(0).to(DEVICE)

        d_logits, i_logits = model(g, e, d)
        d_probs = F.softmax(d_logits, dim=1)
        i_probs = F.softmax(i_logits, dim=1)

        dir_preds.append(d_logits.argmax(1).item())
        int_preds.append(i_logits.argmax(1).item())
        dir_confs.append(d_probs.max().item())
        int_confs.append(i_probs.max().item())
        dir_true_list.append(d_lbl)
        int_true_list.append(i_lbl)

timesteps = np.arange(len(dir_preds))

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

# Direction
axes[0].step(timesteps, dir_true_list, where="mid", label="True",
             color="#2c3e50", linewidth=2)
axes[0].step(timesteps, dir_preds, where="mid", label="Predicted",
             color="#e74c3c", linewidth=2, linestyle="--")
axes[0].fill_between(timesteps, 0, [c * 7 for c in dir_confs],
                     alpha=0.1, color="#e74c3c", label="Confidence")
axes[0].set_yticks(range(8))
axes[0].set_yticklabels(DIR_LABELS)
axes[0].set_ylabel("Direction")
axes[0].set_title(f"Storm {storm_id} — Direction Predictions")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

# Intensity
axes[1].step(timesteps, int_true_list, where="mid", label="True",
             color="#2c3e50", linewidth=2)
axes[1].step(timesteps, int_preds, where="mid", label="Predicted",
             color="#e67e22", linewidth=2, linestyle="--")
axes[1].set_yticks(range(4))
axes[1].set_yticklabels(INTE_LABELS)
axes[1].set_xlabel("Timestep (6-hourly)")
axes[1].set_ylabel("Intensity Change")
axes[1].set_title(f"Storm {storm_id} — Intensity Predictions")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "figures" / "unet_storm_timeline.png", dpi=150,
            bbox_inches="tight")
plt.show()

# ── Final summary ──
print("\n" + "=" * 80)
print(" U-Net Cross-Basin Transfer — Final Results")
print("=" * 80)
print(f"\nModel: UNet2dClassifier")
print(f"  Encoder levels: {N_LEVELS}, Base channels: {BASE_CHANNELS}")
print(f"  Parameters: {n_params:,}")
print(f"  Multimodal: grid (15ch) + env ({40 if USE_ENV else 0}d) "
      f"+ 1D ({4 if USE_1D else 0}d)")
print(f"  Optimizer: {OPTIMIZER.upper()}, LR: {LR}")

print(f"\n{'Setting':<35s} {'Dir Acc':>8s} {'Dir F1':>8s} "
      f"{'Int Acc':>8s} {'Int F1':>8s}")
print("-" * 70)
for name, m in [
    ("WP Validation (in-basin)",          wp_val_metrics),
    ("SP Test (zero-shot)",               sp_zs_metrics),
    (f"SP Test (fine-tuned, {best_ft_strategy})", sp_ft_test_metrics),
]:
    print(f"{name:<35s} {m['dir_acc']:>8.3f} {m['dir_f1']:>8.3f} "
          f"{m['int_acc']:>8.3f} {m['int_f1']:>8.3f}")

# Transfer efficiency
zs_gap_dir = sp_zs_metrics["dir_acc"] - wp_val_metrics["dir_acc"]
ft_recovery = (sp_ft_test_metrics["dir_acc"] - sp_zs_metrics["dir_acc"])
print(f"\nZero-shot transfer gap (direction):  {zs_gap_dir:+.3f}")
print(f"Fine-tuning recovery (direction):     {ft_recovery:+.3f}")

# Save model
save_path = PROJECT_ROOT / "experiments" / "unet_best_wp.pt"
torch.save(best_model_state, save_path)
print(f"\nBest WP model saved to: {save_path}")

ft_save_path = PROJECT_ROOT / "experiments" / "unet_best_ft.pt"
torch.save(best_ft_model.state_dict(), ft_save_path)
print(f"Best fine-tuned model saved to: {ft_save_path}")
