import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from collections import OrderedDict
import warnings, os, copy

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── Paths ──
PROJECT_ROOT = Path("..").resolve()
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EXP_DIR  = PROJECT_ROOT / "experiments"
FIG_DIR  = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

N_DIR_CLASSES = 8
N_INT_CLASSES = 4
DIR_LABELS  = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
INTE_LABELS = ["Weakening", "Steady", "Slow-intens.", "Rapid-intens."]

# ── Channel / feature names ──
CH_NAMES = ["SST", "u_200", "u_500", "u_850", "u_925",
            "v_200", "v_500", "v_850", "v_925",
            "z_200", "z_500", "z_850", "z_925",
            "wind_shear", "vorticity"]

GRID_GROUPS = OrderedDict({
    "SST":           [0],
    "u_wind":        [1, 2, 3, 4],
    "v_wind":        [5, 6, 7, 8],
    "geopotential":  [9, 10, 11, 12],
    "wind_shear":    [13],
    "vorticity":     [14],
})

ENV_GROUPS = OrderedDict({
    "wind":            slice(0, 1),
    "move_velocity":   slice(1, 2),
    "intensity_class": slice(2, 8),
    "month":           slice(8, 20),
    "history_dir_12h": slice(20, 28),
    "history_dir_24h": slice(28, 36),
    "history_int_24h": slice(36, 40),
})

ENV_FEATURE_NAMES = (
    ["wind"] +
    ["move_velocity"] +
    [f"intensity_class_{i}" for i in range(6)] +
    [f"month_{i}" for i in range(12)] +
    [f"hist_dir_12h_{i}" for i in range(8)] +
    [f"hist_dir_24h_{i}" for i in range(8)] +
    [f"hist_int_24h_{i}" for i in range(4)]
)

print(f"Device: {DEVICE}")
print(f"Grid groups: {list(GRID_GROUPS.keys())}")
print(f"Env groups:  {list(ENV_GROUPS.keys())}")

# ── Dataset (same as comparison.ipynb) ──
class CycloneDataset(Dataset):
    """Flattens storm-level dicts into timestep-level samples.
    Filters sentinel labels (-1). 1D features z-scored with WP train stats."""
    def __init__(self, grids, env, data1d, labels, use_reflected=False,
                 d1d_mean=None, d1d_std=None):
        self.samples = []
        dir_key = "direction_reflected" if use_reflected else "direction"
        for storm_id in grids:
            g = grids[storm_id]
            e = env[storm_id]
            d = data1d[storm_id]
            d_lbl = labels[storm_id][dir_key]
            i_lbl = labels[storm_id]["intensity"]
            for t in range(g.shape[0]):
                if d_lbl[t].item() == -1 or i_lbl[t].item() == -1:
                    continue
                self.samples.append((
                    g[t], e[t], d[t], d_lbl[t].long(), i_lbl[t].long()))
        if d1d_mean is None:
            all_1d = torch.stack([s[2] for s in self.samples])
            self.d1d_mean = all_1d.mean(dim=0)
            self.d1d_std  = all_1d.std(dim=0).clamp(min=1e-6)
        else:
            self.d1d_mean, self.d1d_std = d1d_mean, d1d_std

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        grid, env, d1d, dir_lbl, int_lbl = self.samples[idx]
        d1d = (d1d - self.d1d_mean) / self.d1d_std
        return grid, env, d1d, dir_lbl, int_lbl


# ── Load data splits ──
SPLITS = {
    "wp_train":    {"reflected": False},
    "wp_val":      {"reflected": False},
}

raw = {}
for split in SPLITS:
    raw[split] = {
        "grids":  torch.load(DATA_DIR / "grids"  / f"{split}_grids.pt",  weights_only=False),
        "env":    torch.load(DATA_DIR / "env"     / f"{split}_env.pt",    weights_only=False),
        "data1d": torch.load(DATA_DIR / "data1d"  / f"{split}_1d.pt",    weights_only=False),
        "labels": torch.load(DATA_DIR / "labels"  / f"{split}_labels.pt", weights_only=False),
    }

datasets = {}
datasets["wp_train"] = CycloneDataset(
    raw["wp_train"]["grids"], raw["wp_train"]["env"],
    raw["wp_train"]["data1d"], raw["wp_train"]["labels"])
d1d_mean = datasets["wp_train"].d1d_mean
d1d_std  = datasets["wp_train"].d1d_std

datasets["wp_val"] = CycloneDataset(
    raw["wp_val"]["grids"], raw["wp_val"]["env"], raw["wp_val"]["data1d"],
    raw["wp_val"]["labels"], use_reflected=False,
    d1d_mean=d1d_mean, d1d_std=d1d_std)

loaders = {s: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)
           for s, ds in datasets.items()}

for s, ds in datasets.items():
    print(f"  {s:15s}: {len(ds):5d} samples")

# ═══════════════════════════════════════════════════════════════
# U-Net with SE attention (copied from comparison.ipynb)
# ═══════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = torch.rand(x.size(0), 1, 1, 1, device=x.device) > self.drop_prob
        return x * keep / (1 - self.drop_prob)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU())
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.drop_path = DropPath(drop_path)
    def forward(self, x):
        return self.drop_path(self.net(x)) + self.residual(x)


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // reduction, 4)), nn.GELU(),
            nn.Linear(max(ch // reduction, 4), ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout, drop_path)
        self.se = SEBlock(out_ch)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
    def forward(self, x):
        skip = self.se(self.conv(x))
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout, drop_path)
    def forward(self, x, skip):
        x = self.up(x)
        dh = x.size(2) - skip.size(2)
        dw = x.size(3) - skip.size(3)
        if dh > 0 or dw > 0:
            x = x[:, :, dh//2:dh//2+skip.size(2), dw//2:dw//2+skip.size(3)]
        elif dh < 0 or dw < 0:
            x = F.pad(x, [0, -dw, 0, -dh])
        return self.conv(torch.cat([x, skip], dim=1))


class UNet2dClassifier(nn.Module):
    """U-Net encoder-decoder with SE attention for TC classification."""
    def __init__(self, in_channels=15, base_channels=32, n_levels=4,
                 n_dir_classes=8, n_int_classes=4,
                 env_dim=40, d1d_dim=4, use_env=True, use_1d=True,
                 dropout=0.0, head_dim=256, drop_path=0.1):
        super().__init__()
        self.use_env = use_env
        self.use_1d  = use_1d
        dp_rates = [drop_path * i / max(n_levels, 1) for i in range(n_levels + 1)]
        self.encoders = nn.ModuleList()
        ch_in = in_channels
        for i in range(n_levels):
            ch_out = base_channels * (2 ** i)
            self.encoders.append(EncoderBlock(ch_in, ch_out, dropout, dp_rates[i]))
            ch_in = ch_out
        bottleneck_ch = base_channels * (2 ** n_levels)
        self.bottleneck = ConvBlock(ch_in, bottleneck_ch, dropout, dp_rates[n_levels])
        self.decoders = nn.ModuleList()
        ch_in = bottleneck_ch
        for i in range(n_levels - 1, -1, -1):
            skip_ch = base_channels * (2 ** i)
            self.decoders.append(DecoderBlock(ch_in, skip_ch, skip_ch, dropout, dp_rates[i]))
            ch_in = skip_ch
        self.gap = nn.AdaptiveAvgPool2d(1)
        fusion_dim = base_channels + (env_dim if use_env else 0) + (d1d_dim if use_1d else 0)
        self.head_dir = nn.Sequential(
            nn.Linear(fusion_dim, head_dim), nn.GELU(), nn.Dropout(dropout * 2),
            nn.Linear(head_dim, head_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_dim // 2, n_dir_classes))
        self.head_int = nn.Sequential(
            nn.Linear(fusion_dim, head_dim), nn.GELU(), nn.Dropout(dropout * 2),
            nn.Linear(head_dim, head_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_dim // 2, n_int_classes))

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
        if self.use_env and env is not None: parts.append(env)
        if self.use_1d  and d1d is not None: parts.append(d1d)
        fused = torch.cat(parts, dim=1)
        return self.head_dir(fused), self.head_int(fused)

# ── Load checkpoint ──
model = UNet2dClassifier(in_channels=15, base_channels=32, n_levels=4,
                         head_dim=256, dropout=0.0, drop_path=0.0)
state = torch.load(EXP_DIR / "unet_best_wp.pt", map_location="cpu", weights_only=False)
model.load_state_dict(state)
model.to(DEVICE).eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"U-Net loaded: {n_params:,d} parameters")

@torch.no_grad()
def evaluate(model, loader, mask_fn=None):
    """Evaluate model on a DataLoader.
    
    mask_fn: optional callable(grid, env, d1d) -> (grid, env, d1d)
             that zeros out specific features before forward pass.
    Returns dict with dir_acc, dir_f1, int_acc, int_f1.
    """
    model.eval()
    dir_preds, dir_trues = [], []
    int_preds, int_trues = [], []
    for grid, env, d1d, dl, il in loader:
        grid = grid.to(DEVICE)
        env  = env.to(DEVICE)
        d1d  = d1d.to(DEVICE)
        if mask_fn is not None:
            grid, env, d1d = mask_fn(grid, env, d1d)
        d_out, i_out = model(grid, env, d1d)
        dir_preds.extend(d_out.argmax(1).cpu().tolist())
        int_preds.extend(i_out.argmax(1).cpu().tolist())
        dir_trues.extend(dl.tolist())
        int_trues.extend(il.tolist())
    return {
        "dir_acc": accuracy_score(dir_trues, dir_preds),
        "dir_f1":  f1_score(dir_trues, dir_preds, average="macro", zero_division=0),
        "int_acc": accuracy_score(int_trues, int_preds),
        "int_f1":  f1_score(int_trues, int_preds, average="macro", zero_division=0),
    }


# ── Baseline (all features intact) ──
baseline = evaluate(model, loaders["wp_val"])
print("Baseline (WP val, all features):")
for k, v in baseline.items():
    print(f"  {k}: {v:.4f}")

# ── Leave-one-group-out: zero out each grid channel group ──
grid_ablation = {}

for group_name, channels in GRID_GROUPS.items():
    def make_mask(chs):
        def mask_fn(grid, env, d1d):
            g = grid.clone()
            g[:, chs, :, :] = 0.0
            return g, env, d1d
        return mask_fn

    result = evaluate(model, loaders["wp_val"], mask_fn=make_mask(channels))
    grid_ablation[group_name] = result
    drop_dir = baseline["dir_acc"] - result["dir_acc"]
    drop_int = baseline["int_acc"] - result["int_acc"]
    print(f"  Drop {group_name:15s} | dir_acc: {result['dir_acc']:.4f} "
          f"(drop {drop_dir:+.4f}) | int_acc: {result['int_acc']:.4f} "
          f"(drop {drop_int:+.4f})")

# ── Plot: grid channel group ablation ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

groups = list(GRID_GROUPS.keys())
dir_drops = [baseline["dir_acc"] - grid_ablation[g]["dir_acc"] for g in groups]
int_drops = [baseline["int_acc"] - grid_ablation[g]["int_acc"] for g in groups]

for ax, drops, title, color in zip(
        axes,
        [dir_drops, int_drops],
        ["Direction Accuracy Drop", "Intensity Accuracy Drop"],
        ["#2196F3", "#FF5722"]):
    bars = ax.barh(groups, drops, color=color, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Accuracy Drop (higher = more important)")
    ax.set_title(f"Grid Group Ablation: {title}")
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, drops):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", fontsize=9, fontweight="bold")

fig.suptitle("Leave-One-Group-Out: Grid Channel Importance", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "ablation_grid_groups.png", dpi=200, bbox_inches="tight")
plt.close('all')

# ── Leave-one-group-out: zero out each env feature group ──
env_ablation = {}

for group_name, slc in ENV_GROUPS.items():
    def make_mask(s):
        def mask_fn(grid, env, d1d):
            e = env.clone()
            e[:, s] = 0.0
            return grid, e, d1d
        return mask_fn

    result = evaluate(model, loaders["wp_val"], mask_fn=make_mask(slc))
    env_ablation[group_name] = result
    drop_dir = baseline["dir_acc"] - result["dir_acc"]
    drop_int = baseline["int_acc"] - result["int_acc"]
    print(f"  Drop {group_name:18s} | dir_acc: {result['dir_acc']:.4f} "
          f"(drop {drop_dir:+.4f}) | int_acc: {result['int_acc']:.4f} "
          f"(drop {drop_int:+.4f})")

# ── Plot: env feature group ablation ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

env_groups_list = list(ENV_GROUPS.keys())
dir_drops = [baseline["dir_acc"] - env_ablation[g]["dir_acc"] for g in env_groups_list]
int_drops = [baseline["int_acc"] - env_ablation[g]["int_acc"] for g in env_groups_list]

for ax, drops, title, color in zip(
        axes,
        [dir_drops, int_drops],
        ["Direction Accuracy Drop", "Intensity Accuracy Drop"],
        ["#9C27B0", "#E91E63"]):
    bars = ax.barh(env_groups_list, drops, color=color, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Accuracy Drop (higher = more important)")
    ax.set_title(f"Env Group Ablation: {title}")
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, drops):
        offset = 0.002 if val >= 0 else -0.002
        ha = "left" if val >= 0 else "right"
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")

fig.suptitle("Leave-One-Group-Out: Env Feature Importance", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "ablation_env_groups.png", dpi=200, bbox_inches="tight")
plt.close('all')

# ── Modality-level ablation: zero entire modalities ──
modality_ablation = {}

# Drop grid (all channels zeroed)
def mask_no_grid(grid, env, d1d):
    return torch.zeros_like(grid), env, d1d

# Drop env (all env dims zeroed)
def mask_no_env(grid, env, d1d):
    return grid, torch.zeros_like(env), d1d

# Drop 1d (all 1d dims zeroed)
def mask_no_d1d(grid, env, d1d):
    return grid, env, torch.zeros_like(d1d)

# Drop grid + env (keep only 1d)
def mask_only_d1d(grid, env, d1d):
    return torch.zeros_like(grid), torch.zeros_like(env), d1d

# Drop grid + 1d (keep only env)
def mask_only_env(grid, env, d1d):
    return torch.zeros_like(grid), env, torch.zeros_like(d1d)

# Drop env + 1d (keep only grid)
def mask_only_grid(grid, env, d1d):
    return grid, torch.zeros_like(env), torch.zeros_like(d1d)

modality_configs = {
    "No Grid":      mask_no_grid,
    "No Env":       mask_no_env,
    "No 1D":        mask_no_d1d,
    "Only Grid":    mask_only_grid,
    "Only Env":     mask_only_env,
    "Only 1D":      mask_only_d1d,
}

print(f"{'Config':15s} | {'Dir Acc':>8s} | {'Drop':>7s} | {'Int Acc':>8s} | {'Drop':>7s}")
print("-" * 60)
print(f"{'Baseline':15s} | {baseline['dir_acc']:>8.4f} |    --   | {baseline['int_acc']:>8.4f} |    --  ")

for name, mask_fn in modality_configs.items():
    result = evaluate(model, loaders["wp_val"], mask_fn=mask_fn)
    modality_ablation[name] = result
    d_drop = baseline["dir_acc"] - result["dir_acc"]
    i_drop = baseline["int_acc"] - result["int_acc"]
    print(f"{name:15s} | {result['dir_acc']:>8.4f} | {d_drop:>+7.4f} | "
          f"{result['int_acc']:>8.4f} | {i_drop:>+7.4f}")

# ── Plot: modality ablation ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# "Drop" configs
drop_names = ["No Grid", "No Env", "No 1D"]
# "Only" configs
only_names = ["Only Grid", "Only Env", "Only 1D"]

for ax, metric, title in zip(axes, ["dir_acc", "int_acc"],
                              ["Direction Accuracy", "Intensity Accuracy"]):
    # Drop bars
    x = np.arange(3)
    w = 0.35
    drop_vals = [modality_ablation[n][metric] for n in drop_names]
    only_vals = [modality_ablation[n][metric] for n in only_names]

    bars1 = ax.bar(x - w/2, drop_vals, w, label="Drop one modality",
                   color="#EF5350", edgecolor="white", alpha=0.85)
    bars2 = ax.bar(x + w/2, only_vals, w, label="Keep only one modality",
                   color="#66BB6A", edgecolor="white", alpha=0.85)

    ax.axhline(baseline[metric], color="black", linestyle="--", linewidth=1.2,
               label=f"Baseline ({baseline[metric]:.3f})")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Grid", "Env", "1D"])
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, max(baseline[metric] + 0.15, max(drop_vals + only_vals) + 0.08))
    ax.legend(fontsize=8)

fig.suptitle("Modality Ablation: Drop vs Keep Single Modality",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "ablation_modality.png", dpi=200, bbox_inches="tight")
plt.close('all')

# ── Add-one-in: start from nothing, add one grid group at a time ──
# "Nothing" baseline: all inputs zeroed
nothing_result = evaluate(model, loaders["wp_val"],
                          mask_fn=lambda g, e, d: (torch.zeros_like(g),
                                                    torch.zeros_like(e),
                                                    torch.zeros_like(d)))
print(f"All zeroed:  dir_acc={nothing_result['dir_acc']:.4f}  "
      f"int_acc={nothing_result['int_acc']:.4f}\n")

# Add one grid group (all else zeroed)
grid_add_one = {}
for group_name, channels in GRID_GROUPS.items():
    def make_mask(chs):
        def mask_fn(grid, env, d1d):
            g = torch.zeros_like(grid)
            g[:, chs, :, :] = grid[:, chs, :, :]
            return g, torch.zeros_like(env), torch.zeros_like(d1d)
        return mask_fn

    result = evaluate(model, loaders["wp_val"], mask_fn=make_mask(channels))
    grid_add_one[group_name] = result
    gain_dir = result["dir_acc"] - nothing_result["dir_acc"]
    print(f"  Add {group_name:15s} | dir_acc: {result['dir_acc']:.4f} "
          f"(gain {gain_dir:+.4f})")

# Add one env group (all else zeroed)
print()
env_add_one = {}
for group_name, slc in ENV_GROUPS.items():
    def make_mask(s):
        def mask_fn(grid, env, d1d):
            e = torch.zeros_like(env)
            e[:, s] = env[:, s]
            return torch.zeros_like(grid), e, torch.zeros_like(d1d)
        return mask_fn

    result = evaluate(model, loaders["wp_val"], mask_fn=make_mask(slc))
    env_add_one[group_name] = result
    gain_dir = result["dir_acc"] - nothing_result["dir_acc"]
    print(f"  Add {group_name:18s} | dir_acc: {result['dir_acc']:.4f} "
          f"(gain {gain_dir:+.4f})")

# ── Plot: add-one-in combined (grid + env groups) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

all_groups = list(GRID_GROUPS.keys()) + list(ENV_GROUPS.keys())
all_add_one = {**grid_add_one, **env_add_one}
colors = (["#2196F3"] * len(GRID_GROUPS) + ["#9C27B0"] * len(ENV_GROUPS))

for ax, metric, title in zip(axes, ["dir_acc", "int_acc"],
                              ["Direction Accuracy", "Intensity Accuracy"]):
    gains = [all_add_one[g][metric] - nothing_result[metric] for g in all_groups]
    bars = ax.barh(all_groups, gains, color=colors, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Accuracy Gain over All-Zero (higher = more useful alone)")
    ax.set_title(f"Add-One-In: {title}")
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, gains):
        offset = 0.002 if val >= 0 else -0.002
        ha = "left" if val >= 0 else "right"
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=8, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#2196F3", label="Grid group"),
                       Patch(facecolor="#9C27B0", label="Env group")],
              fontsize=8)

fig.suptitle("Add-One-In Ablation: Standalone Feature Group Value",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "ablation_add_one_in.png", dpi=200, bbox_inches="tight")
plt.close('all')

import shap

# ── Collect all validation data into tensors ──
all_grids, all_envs, all_d1ds, all_dir_labels = [], [], [], []
for grid, env, d1d, dl, il in loaders["wp_val"]:
    all_grids.append(grid)
    all_envs.append(env)
    all_d1ds.append(d1d)
    all_dir_labels.append(dl)

all_grids = torch.cat(all_grids, dim=0)
all_envs  = torch.cat(all_envs, dim=0)
all_d1ds  = torch.cat(all_d1ds, dim=0)
all_dir_labels = torch.cat(all_dir_labels, dim=0)

# Compute mean grid and mean d1d for fixing
mean_grid = all_grids.mean(dim=0, keepdim=True).to(DEVICE)  # (1, 15, 81, 81)
mean_d1d  = all_d1ds.mean(dim=0, keepdim=True).to(DEVICE)   # (1, 4)

print(f"Val samples: {all_envs.shape[0]}")
print(f"Mean grid shape: {mean_grid.shape}, Mean d1d shape: {mean_d1d.shape}")

# ── Wrapper: model that takes only env as input ──
class EnvWrapper(nn.Module):
    """Wraps the U-Net so SHAP only sees the env vector.
    Grid and d1d are fixed to their dataset means."""
    def __init__(self, model, fixed_grid, fixed_d1d):
        super().__init__()
        self.model = model
        self.register_buffer("fixed_grid", fixed_grid)
        self.register_buffer("fixed_d1d", fixed_d1d)

    def forward(self, env):
        B = env.shape[0]
        grid = self.fixed_grid.expand(B, -1, -1, -1)
        d1d = self.fixed_d1d.expand(B, -1)
        dir_logits, _ = self.model(grid, env, d1d)
        return dir_logits  # (B, 8) direction logits


env_wrapper = EnvWrapper(model, mean_grid, mean_d1d).to(DEVICE).eval()

# Quick sanity check
with torch.no_grad():
    test_out = env_wrapper(all_envs[:2].to(DEVICE))
    print(f"EnvWrapper output shape: {test_out.shape}")  # Should be (2, 8)

# ── SHAP GradientExplainer on env features ──
background_env = all_envs[:100].to(DEVICE)
explain_env    = all_envs[:200].to(DEVICE)

explainer = shap.GradientExplainer(env_wrapper, background_env)
shap_values_env = explainer.shap_values(explain_env)

# shap_values_env may be list-of-arrays or a single array depending on shap version
shap_arr = np.array(shap_values_env)  # try to get (n_samples, n_features, n_classes) or similar
print(f"SHAP raw shape: {shap_arr.shape}")

# Normalize to (n_samples, n_features) by averaging absolute values across classes
if shap_arr.ndim == 3:
    # Could be (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
    if shap_arr.shape[0] == 200 and shap_arr.shape[2] == 8:
        # (n_samples, n_features, n_classes)
        shap_abs_mean = np.mean(np.abs(shap_arr), axis=2)  # (200, 40)
        shap_avg = np.mean(shap_arr, axis=2)  # (200, 40)
    elif shap_arr.shape[0] == 8 and shap_arr.shape[1] == 200:
        # (n_classes, n_samples, n_features)
        shap_abs_mean = np.mean(np.abs(shap_arr), axis=0)  # (200, 40)
        shap_avg = np.mean(shap_arr, axis=0)  # (200, 40)
    else:
        # Fallback: average over last dim
        shap_abs_mean = np.mean(np.abs(shap_arr), axis=-1)
        shap_avg = np.mean(shap_arr, axis=-1)
elif shap_arr.ndim == 2:
    shap_abs_mean = np.abs(shap_arr)
    shap_avg = shap_arr
else:
    raise ValueError(f"Unexpected SHAP shape: {shap_arr.shape}")

print(f"SHAP avg shape: {shap_avg.shape}")

# Create SHAP Explanation object for plotting
explanation = shap.Explanation(
    values=shap_avg,
    data=explain_env.cpu().numpy(),
    feature_names=ENV_FEATURE_NAMES,
)

fig = plt.figure(figsize=(10, 10))
shap.plots.beeswarm(explanation, max_display=20, show=False)
plt.title("SHAP Beeswarm: Env Features (Direction Prediction, avg across classes)",
          fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_env_beeswarm.png", dpi=200, bbox_inches="tight")
plt.close('all')

# ── SHAP bar plot: mean |SHAP| per env feature group ──
# Aggregate SHAP by env group
group_importance = {}
for group_name, slc in ENV_GROUPS.items():
    # Mean absolute SHAP across all samples and all classes for this group
    group_vals = shap_abs_mean[:, slc]  # (200, group_size)
    group_importance[group_name] = group_vals.mean()

# Sort by importance
sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
g_names = [g[0] for g in sorted_groups]
g_vals  = [g[1] for g in sorted_groups]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(g_names[::-1], g_vals[::-1], color="#FF9800", edgecolor="white", alpha=0.85)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("SHAP: Env Feature Group Importance (Direction Prediction)",
             fontsize=12, fontweight="bold")
for bar, val in zip(bars, g_vals[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_env_group_bar.png", dpi=200, bbox_inches="tight")
plt.close('all')

# ── Grid channel importance via gradient-based attribution ──
# Instead of full SHAP on (15,81,81), we compute per-channel gradient magnitude.
# This is faster and gives per-channel importance directly.

all_grids_dev = all_grids[:200].to(DEVICE).requires_grad_(True)
all_envs_dev  = all_envs[:200].to(DEVICE)
all_d1ds_dev  = all_d1ds[:200].to(DEVICE)

model.eval()
# Forward pass
dir_logits, _ = model(all_grids_dev, all_envs_dev, all_d1ds_dev)
# Use the predicted class logit as the target
pred_classes = dir_logits.argmax(dim=1)
target_logits = dir_logits[torch.arange(len(pred_classes)), pred_classes]
target_logits.sum().backward()

# Per-channel mean absolute gradient: (200, 15, 81, 81) -> mean over (samples, H, W)
grad = all_grids_dev.grad.detach().cpu()  # (200, 15, 81, 81)
channel_importance = grad.abs().mean(dim=(0, 2, 3))  # (15,)

print("Per-channel gradient importance:")
for i, (name, imp) in enumerate(zip(CH_NAMES, channel_importance)):
    print(f"  {name:12s}: {imp:.6f}")

# ── Plot: per-channel grid importance ──
fig, ax = plt.subplots(figsize=(10, 6))

sorted_idx = channel_importance.argsort(descending=True)
sorted_names = [CH_NAMES[i] for i in sorted_idx]
sorted_vals  = channel_importance[sorted_idx].numpy()

bars = ax.barh(sorted_names[::-1], sorted_vals[::-1],
               color="#4CAF50", edgecolor="white", alpha=0.85)
ax.set_xlabel("Mean |Gradient| (higher = more important)")
ax.set_title("Grid Channel Importance (Gradient-based Attribution)",
             fontsize=12, fontweight="bold")
for bar, val in zip(bars, sorted_vals[::-1]):
    ax.text(bar.get_width() + max(sorted_vals) * 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{val:.5f}", va="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_grid_channel_importance.png", dpi=200, bbox_inches="tight")
plt.close('all')

# ── Grid channel importance aggregated by group ──
grid_group_importance = {}
for group_name, channels in GRID_GROUPS.items():
    grid_group_importance[group_name] = channel_importance[channels].mean().item()

sorted_gg = sorted(grid_group_importance.items(), key=lambda x: x[1], reverse=True)
gg_names = [g[0] for g in sorted_gg]
gg_vals  = [g[1] for g in sorted_gg]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(gg_names[::-1], gg_vals[::-1],
               color="#2196F3", edgecolor="white", alpha=0.85)
ax.set_xlabel("Mean |Gradient| per group")
ax.set_title("Grid Group Importance (Gradient-based, grouped)",
             fontsize=12, fontweight="bold")
for bar, val in zip(bars, gg_vals[::-1]):
    ax.text(bar.get_width() + max(gg_vals) * 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{val:.5f}", va="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_grid_group_importance.png", dpi=200, bbox_inches="tight")
plt.close('all')

import pandas as pd

# ── Summary table: ablation results ──
print("=" * 80)
print(" ABLATION STUDY SUMMARY")
print("=" * 80)

# Grid groups
print("\n--- Grid Channel Group Ablation (Leave-One-Out) ---")
print(f"{'Group':15s} | {'Dir Acc':>8s} | {'Dir Drop':>9s} | {'Int Acc':>8s} | {'Int Drop':>9s}")
print("-" * 60)
for g in GRID_GROUPS:
    r = grid_ablation[g]
    dd = baseline["dir_acc"] - r["dir_acc"]
    di = baseline["int_acc"] - r["int_acc"]
    print(f"{g:15s} | {r['dir_acc']:>8.4f} | {dd:>+9.4f} | {r['int_acc']:>8.4f} | {di:>+9.4f}")

# Env groups
print("\n--- Env Feature Group Ablation (Leave-One-Out) ---")
print(f"{'Group':18s} | {'Dir Acc':>8s} | {'Dir Drop':>9s} | {'Int Acc':>8s} | {'Int Drop':>9s}")
print("-" * 65)
for g in ENV_GROUPS:
    r = env_ablation[g]
    dd = baseline["dir_acc"] - r["dir_acc"]
    di = baseline["int_acc"] - r["int_acc"]
    print(f"{g:18s} | {r['dir_acc']:>8.4f} | {dd:>+9.4f} | {r['int_acc']:>8.4f} | {di:>+9.4f}")

# Modality ablation
print("\n--- Modality Ablation ---")
print(f"{'Config':15s} | {'Dir Acc':>8s} | {'Dir Drop':>9s} | {'Int Acc':>8s} | {'Int Drop':>9s}")
print("-" * 60)
print(f"{'Baseline':15s} | {baseline['dir_acc']:>8.4f} |       --  | {baseline['int_acc']:>8.4f} |       --")
for name, r in modality_ablation.items():
    dd = baseline["dir_acc"] - r["dir_acc"]
    di = baseline["int_acc"] - r["int_acc"]
    print(f"{name:15s} | {r['dir_acc']:>8.4f} | {dd:>+9.4f} | {r['int_acc']:>8.4f} | {di:>+9.4f}")

# ── Combined importance comparison: ablation drop vs gradient importance ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Grid groups — ablation drop vs gradient importance
grid_names = list(GRID_GROUPS.keys())
abl_drops  = [baseline["dir_acc"] - grid_ablation[g]["dir_acc"] for g in grid_names]
grad_imps  = [grid_group_importance[g] for g in grid_names]

# Normalize both to [0, 1] for comparison
def normalize(vals):
    arr = np.array(vals)
    if arr.max() - arr.min() < 1e-10:
        return np.ones_like(arr) * 0.5
    return (arr - arr.min()) / (arr.max() - arr.min())

ax = axes[0]
x = np.arange(len(grid_names))
w = 0.35
ax.bar(x - w/2, normalize(abl_drops), w, label="Ablation Drop (norm.)",
       color="#EF5350", edgecolor="white", alpha=0.85)
ax.bar(x + w/2, normalize(grad_imps), w, label="Gradient Importance (norm.)",
       color="#4CAF50", edgecolor="white", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(grid_names, rotation=30, ha="right")
ax.set_ylabel("Normalized Importance")
ax.set_title("Grid Groups: Ablation vs Gradient")
ax.legend(fontsize=8)

# Right: Env groups — ablation drop vs SHAP importance
env_names = list(ENV_GROUPS.keys())
abl_drops_env = [baseline["dir_acc"] - env_ablation[g]["dir_acc"] for g in env_names]
shap_imps_env = [group_importance[g] for g in env_names]

ax = axes[1]
x = np.arange(len(env_names))
ax.bar(x - w/2, normalize(abl_drops_env), w, label="Ablation Drop (norm.)",
       color="#EF5350", edgecolor="white", alpha=0.85)
ax.bar(x + w/2, normalize(shap_imps_env), w, label="SHAP Importance (norm.)",
       color="#FF9800", edgecolor="white", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(env_names, rotation=30, ha="right")
ax.set_ylabel("Normalized Importance")
ax.set_title("Env Groups: Ablation vs SHAP")
ax.legend(fontsize=8)

fig.suptitle("Feature Importance: Ablation vs Attribution Methods",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "ablation_vs_attribution.png", dpi=200, bbox_inches="tight")
plt.close('all')

# ── List all saved figures ──
print("\nFigures saved:")
for f in sorted(FIG_DIR.glob("ablation_*.png")) + sorted(FIG_DIR.glob("shap_*.png")):
    print(f"  {f.name}")
