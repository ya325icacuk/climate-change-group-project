"""
TCND Data Analysis — Basin Generalization Study
================================================
Comprehensive cross-basin analysis of the TropiCycloneNet test dataset.
Produces statistics, distributions, and a basin similarity matrix to
inform transfer-learning experiment design (WP → other basins).

Usage:
    python src/data_analysis.py

Outputs figures to figures/ and prints summary statistics to console.
"""

import os
import sys
import glob
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "tropicyclonenet" / "TCND_test" / "TCND_test"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

BASINS = ["EP", "NA", "NI", "SI", "SP", "WP"]
BASIN_COLORS = {
    "EP": "#e41a1c", "NA": "#377eb8", "NI": "#4daf4a",
    "SI": "#984ea3", "SP": "#ff7f00", "WP": "#a65628",
}

DIRECTION_LABELS = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]  # 0-7
INTENSITY_CLASSES = ["TD", "TS", "STS", "TY", "STY", "SuperTY"]   # 0-5

DATA1D_COLS = ["index", "category", "lat_offset", "lon_offset",
               "wind_norm", "pressure_norm", "datetime", "name"]

# Max NetCDF files to sample per basin for SST/shear stats (speed vs coverage)
NC_SAMPLE_PER_BASIN = 80

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
})


# ---------------------------------------------------------------------------
# 1. Data loading helpers
# ---------------------------------------------------------------------------

def load_data1d(basin: str) -> pd.DataFrame:
    """Load all Data1D .txt files for a basin into a single DataFrame."""
    pattern = DATA_ROOT / "Data1D" / basin / "test" / "*.txt"
    files = sorted(glob.glob(str(pattern)))
    frames = []
    for f in files:
        storm_name = Path(f).stem  # e.g. EP2017BSTDORA
        df = pd.read_csv(f, sep="\t", header=None, names=DATA1D_COLS,
                         dtype={"datetime": str})
        df["storm_id"] = storm_name
        df["basin"] = basin
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=DATA1D_COLS + ["storm_id", "basin"])
    return pd.concat(frames, ignore_index=True)


def load_env_data(basin: str) -> list[dict]:
    """Load all Env-Data .npy files for a basin. Returns list of dicts."""
    basin_dir = DATA_ROOT / "Env-Data" / basin
    records = []
    if not basin_dir.exists():
        return records
    for year_dir in sorted(basin_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for storm_dir in sorted(year_dir.iterdir()):
            if not storm_dir.is_dir():
                continue
            for npy_file in sorted(storm_dir.glob("*.npy")):
                try:
                    d = np.load(str(npy_file), allow_pickle=True).item()
                    d["_basin"] = basin
                    d["_storm"] = storm_dir.name
                    d["_year"] = year_dir.name
                    d["_datetime"] = npy_file.stem
                    records.append(d)
                except Exception:
                    pass
    return records


def sample_nc_files(basin: str, n: int = NC_SAMPLE_PER_BASIN) -> list[str]:
    """Return up to n randomly sampled NetCDF file paths for a basin."""
    basin_dir = DATA_ROOT / "Data3D" / basin
    if not basin_dir.exists():
        return []
    all_nc = sorted(glob.glob(str(basin_dir / "**" / "*.nc"), recursive=True))
    if len(all_nc) <= n:
        return all_nc
    rng = np.random.default_rng(42)
    return list(rng.choice(all_nc, size=n, replace=False))


def compute_shear_sst_from_nc(nc_path: str) -> dict | None:
    """Extract area-averaged vertical wind shear and mean SST from a NetCDF file."""
    try:
        import netCDF4 as nc4
        ds = nc4.Dataset(nc_path, "r")
        # SST — 2D field
        sst_var = None
        for vname in ["sst", "SST"]:
            if vname in ds.variables:
                sst_var = ds.variables[vname]
                break
        mean_sst = float(np.nanmean(sst_var[:])) if sst_var is not None else np.nan

        # Wind shear: |V_200 - V_850|
        # Pressure levels order: 200, 500, 850, 925 (indices 0, 1, 2, 3)
        u = ds.variables.get("u") or ds.variables.get("u_component_of_wind")
        v = ds.variables.get("v") or ds.variables.get("v_component_of_wind")
        if u is not None and v is not None:
            u_data = u[:]
            v_data = v[:]
            # shape: (time, levels, lat, lon) or (levels, lat, lon)
            if u_data.ndim == 4:
                u_200 = u_data[0, 0, :, :]  # level index 0 = 200 hPa
                u_850 = u_data[0, 2, :, :]  # level index 2 = 850 hPa
                v_200 = v_data[0, 0, :, :]
                v_850 = v_data[0, 2, :, :]
            elif u_data.ndim == 3:
                u_200 = u_data[0, :, :]
                u_850 = u_data[2, :, :]
                v_200 = v_data[0, :, :]
                v_850 = v_data[2, :, :]
            else:
                ds.close()
                return {"sst": mean_sst, "shear": np.nan}
            # Inner 5° box = center 21x21 pixels (81 total, center at 40)
            c = 40
            r = 10
            du = u_200[c-r:c+r+1, c-r:c+r+1] - u_850[c-r:c+r+1, c-r:c+r+1]
            dv = v_200[c-r:c+r+1, c-r:c+r+1] - v_850[c-r:c+r+1, c-r:c+r+1]
            shear = float(np.nanmean(np.sqrt(du**2 + dv**2)))
        else:
            shear = np.nan
        ds.close()
        return {"sst": mean_sst, "shear": shear}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2. Load all data
# ---------------------------------------------------------------------------

def load_all():
    print("=" * 60)
    print("TCND Data Analysis — Basin Generalization Study")
    print("=" * 60)

    # --- Data1D ---
    print("\n[1/3] Loading Data1D...")
    d1d_frames = []
    for b in BASINS:
        df = load_data1d(b)
        print(f"  {b}: {df['storm_id'].nunique()} storms, {len(df)} timesteps")
        d1d_frames.append(df)
    df_1d = pd.concat(d1d_frames, ignore_index=True)

    # Parse datetime
    df_1d["year"] = df_1d["datetime"].str[:4].astype(int)
    df_1d["month"] = df_1d["datetime"].str[4:6].astype(int)

    # --- Env-Data ---
    print("\n[2/3] Loading Env-Data...")
    env_records = []
    for b in BASINS:
        recs = load_env_data(b)
        print(f"  {b}: {len(recs)} timesteps")
        env_records.extend(recs)

    # Build a flat DataFrame from env records
    env_rows = []
    for r in env_records:
        row = {
            "basin": r["_basin"],
            "storm": r["_storm"],
            "year": r["_year"],
            "datetime": r["_datetime"],
            "wind": r.get("wind", np.nan),
            "move_velocity": r.get("move_velocity", np.nan),
            "future_direction24": r.get("future_direction24", -1),
            "future_inte_change24": r.get("future_inte_change24", -1),
            "history_direction12": r.get("history_direction12", -1),
            "history_direction24": r.get("history_direction24", -1),
            "history_inte_change24": r.get("history_inte_change24", -1),
        }
        # Intensity class: argmax of one-hot
        ic = r.get("intensity_class", None)
        if ic is not None:
            row["intensity_class"] = int(np.argmax(ic))
        else:
            row["intensity_class"] = -1
        env_rows.append(row)
    df_env = pd.DataFrame(env_rows)

    # Convert sentinel -1 to proper handling
    for col in ["future_direction24", "future_inte_change24",
                "history_direction12", "history_direction24", "history_inte_change24"]:
        df_env[col] = pd.to_numeric(df_env[col], errors="coerce")

    # --- Data3D (sampled) ---
    print("\n[3/3] Sampling Data3D for SST/shear stats...")
    nc_stats = []
    for b in BASINS:
        nc_files = sample_nc_files(b)
        count = 0
        for f in nc_files:
            result = compute_shear_sst_from_nc(f)
            if result:
                result["basin"] = b
                nc_stats.append(result)
                count += 1
        print(f"  {b}: {count}/{len(nc_files)} files processed")
    df_nc = pd.DataFrame(nc_stats) if nc_stats else pd.DataFrame(columns=["basin", "sst", "shear"])

    return df_1d, df_env, df_nc


# ---------------------------------------------------------------------------
# 3. Statistics & Figures
# ---------------------------------------------------------------------------

def print_basin_summary(df_1d: pd.DataFrame, df_env: pd.DataFrame):
    """Print summary table of storms and timesteps per basin."""
    print("\n" + "=" * 60)
    print("BASIN SUMMARY")
    print("=" * 60)

    summary = df_1d.groupby("basin").agg(
        storms=("storm_id", "nunique"),
        timesteps=("index", "count"),
        years=("year", lambda x: f"{x.min()}-{x.max()}"),
        mean_wind=("wind_norm", "mean"),
    ).reindex(BASINS)
    print(summary.to_string())

    # Missing data (sentinels) from env
    print("\n--- Sentinel -1 rates in Env-Data ---")
    for col in ["future_direction24", "future_inte_change24",
                "history_direction12", "history_direction24"]:
        rates = df_env.groupby("basin")[col].apply(
            lambda s: (s == -1).sum() / len(s) * 100 if len(s) > 0 else 0
        ).reindex(BASINS)
        print(f"  {col}:")
        for b in BASINS:
            print(f"    {b}: {rates.get(b, 0):.1f}% missing")


def fig_storms_per_basin(df_1d: pd.DataFrame):
    """Bar chart: storms and timesteps per basin."""
    summary = df_1d.groupby("basin").agg(
        storms=("storm_id", "nunique"),
        timesteps=("index", "count"),
    ).reindex(BASINS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    bars1 = ax1.bar(BASINS, summary["storms"],
                    color=[BASIN_COLORS[b] for b in BASINS])
    ax1.set_title("Storms per Basin")
    ax1.set_ylabel("Count")
    for bar, val in zip(bars1, summary["storms"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha="center", fontsize=9)

    bars2 = ax2.bar(BASINS, summary["timesteps"],
                    color=[BASIN_COLORS[b] for b in BASINS])
    ax2.set_title("Timesteps per Basin (6-hourly)")
    ax2.set_ylabel("Count")
    for bar, val in zip(bars2, summary["timesteps"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 str(val), ha="center", fontsize=9)

    fig.suptitle("Data Imbalance Across Basins", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "storms_per_basin.png")
    plt.close(fig)
    print("  Saved: figures/storms_per_basin.png")


def fig_temporal_distribution(df_1d: pd.DataFrame):
    """Storms per year and per month, by basin."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Per year
    for b in BASINS:
        sub = df_1d[df_1d["basin"] == b]
        yearly = sub.groupby("year")["storm_id"].nunique()
        ax1.plot(yearly.index, yearly.values, "o-", label=b,
                 color=BASIN_COLORS[b], markersize=4)
    ax1.set_title("Storms per Year")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of storms")
    ax1.legend(ncol=2, fontsize=8)

    # Per month (all basins stacked)
    month_counts = df_1d.groupby(["basin", "month"])["storm_id"].nunique().unstack(level=0, fill_value=0)
    month_counts = month_counts.reindex(range(1, 13), fill_value=0)[BASINS]
    month_counts.plot(kind="bar", stacked=True, ax=ax2,
                      color=[BASIN_COLORS[b] for b in BASINS])
    ax2.set_title("Storms per Month (all years)")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Count")
    ax2.set_xticklabels([str(m) for m in range(1, 13)], rotation=0)
    ax2.legend(fontsize=8)

    fig.suptitle("Temporal Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "temporal_distribution.png")
    plt.close(fig)
    print("  Saved: figures/temporal_distribution.png")


def fig_intensity_by_basin(df_env: pd.DataFrame):
    """Distribution of intensity classes by basin."""
    valid = df_env[df_env["intensity_class"] >= 0].copy()
    fig, ax = plt.subplots(figsize=(10, 5))

    counts = valid.groupby(["basin", "intensity_class"]).size().unstack(fill_value=0)
    # Normalize to percentages
    pcts = counts.div(counts.sum(axis=1), axis=0) * 100
    pcts = pcts.reindex(BASINS).reindex(columns=range(6), fill_value=0)
    pcts.columns = INTENSITY_CLASSES

    pcts.plot(kind="bar", ax=ax, colormap="RdYlGn_r", edgecolor="white")
    ax.set_title("Intensity Class Distribution by Basin", fontsize=13, fontweight="bold")
    ax.set_xlabel("Basin")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(BASINS, rotation=0)
    ax.legend(title="Intensity", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intensity_by_basin.png")
    plt.close(fig)
    print("  Saved: figures/intensity_by_basin.png")


def fig_direction_by_basin(df_env: pd.DataFrame):
    """Polar plots of 24h direction distributions per basin."""
    valid = df_env[(df_env["future_direction24"] >= 0) &
                   (df_env["future_direction24"] <= 7)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8),
                             subplot_kw={"projection": "polar"})
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    # Compass: E=0°, SE=45°, S=90°, ... map to polar (90° - compass)
    # polar angle: E=0, NE=π/4, N=π/2, NW=3π/4, W=π, SW=5π/4, S=3π/2, SE=7π/4
    # Direction labels order: E(0), SE(1), S(2), SW(3), W(4), NW(5), N(6), NE(7)
    polar_angles = np.array([0, 7, 6, 5, 4, 3, 2, 1]) * (2 * np.pi / 8)

    for idx, b in enumerate(BASINS):
        ax = axes[idx // 3, idx % 3]
        sub = valid[valid["basin"] == b]
        if len(sub) == 0:
            ax.set_title(f"{b} (no data)")
            continue
        counts = sub["future_direction24"].value_counts().reindex(range(8), fill_value=0)
        pcts = counts / counts.sum() * 100

        bars = ax.bar(polar_angles, pcts.values, width=0.6,
                      color=BASIN_COLORS[b], alpha=0.8, edgecolor="white")
        ax.set_thetagrids(np.degrees(polar_angles), DIRECTION_LABELS)
        ax.set_title(f"{b} (n={len(sub)})", fontsize=11, pad=15)
        ax.set_rlabel_position(135)

    fig.suptitle("24-Hour Future Direction Distribution by Basin",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "direction_by_basin.png")
    plt.close(fig)
    print("  Saved: figures/direction_by_basin.png")


def fig_intensity_change_by_basin(df_env: pd.DataFrame):
    """Distribution of 24h intensity change classes by basin."""
    valid = df_env[df_env["future_inte_change24"] >= 0].copy()
    change_labels = {0: "Weaken", 1: "Stable", 2: "Intensify", 3: "Rapid\nIntensify"}

    fig, ax = plt.subplots(figsize=(10, 5))
    counts = valid.groupby(["basin", "future_inte_change24"]).size().unstack(fill_value=0)
    pcts = counts.div(counts.sum(axis=1), axis=0) * 100
    pcts = pcts.reindex(BASINS)
    pcts.columns = [change_labels.get(c, str(c)) for c in pcts.columns]

    pcts.plot(kind="bar", ax=ax, color=["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"],
              edgecolor="white")
    ax.set_title("24h Intensity Change Distribution by Basin",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Basin")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(BASINS, rotation=0)
    ax.legend(title="Change", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "intensity_change_by_basin.png")
    plt.close(fig)
    print("  Saved: figures/intensity_change_by_basin.png")


def fig_wind_distribution(df_env: pd.DataFrame):
    """KDE of normalized wind speed by basin."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for b in BASINS:
        sub = df_env[(df_env["basin"] == b) & (df_env["wind"].notna())]
        if len(sub) > 0:
            sub["wind"].plot(kind="kde", ax=ax, label=f"{b} (n={len(sub)})",
                             color=BASIN_COLORS[b], linewidth=2)
    ax.set_title("Wind Speed Distribution by Basin (normalized)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Normalized wind speed")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "wind_by_basin.png")
    plt.close(fig)
    print("  Saved: figures/wind_by_basin.png")


def fig_move_velocity(df_env: pd.DataFrame):
    """Box plot of move_velocity by basin."""
    valid = df_env[df_env["move_velocity"] != -1].copy()
    # move_velocity can be 0 for first timesteps — keep those
    fig, ax = plt.subplots(figsize=(8, 5))
    data_by_basin = [valid[valid["basin"] == b]["move_velocity"].dropna().values
                     for b in BASINS]
    bp = ax.boxplot(data_by_basin, labels=BASINS, patch_artist=True, showfliers=False)
    for patch, b in zip(bp["boxes"], BASINS):
        patch.set_facecolor(BASIN_COLORS[b])
        patch.set_alpha(0.7)
    ax.set_title("Movement Velocity by Basin", fontsize=13, fontweight="bold")
    ax.set_ylabel("Move velocity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "move_velocity_by_basin.png")
    plt.close(fig)
    print("  Saved: figures/move_velocity_by_basin.png")


def fig_sst_shear(df_nc: pd.DataFrame):
    """Distributions of SST and vertical wind shear by basin."""
    if df_nc.empty or len(df_nc) < 10:
        print("  [SKIP] Not enough Data3D samples for SST/shear figures.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # SST
    for b in BASINS:
        sub = df_nc[(df_nc["basin"] == b) & (df_nc["sst"].notna())]
        if len(sub) > 2:
            sub["sst"].plot(kind="kde", ax=ax1, label=f"{b} (n={len(sub)})",
                            color=BASIN_COLORS[b], linewidth=2)
    ax1.set_title("SST Distribution by Basin")
    ax1.set_xlabel("SST (K or °C)")
    ax1.legend(fontsize=8)

    # Shear
    for b in BASINS:
        sub = df_nc[(df_nc["basin"] == b) & (df_nc["shear"].notna())]
        if len(sub) > 2:
            sub["shear"].plot(kind="kde", ax=ax2, label=f"{b} (n={len(sub)})",
                              color=BASIN_COLORS[b], linewidth=2)
    ax2.set_title("Vertical Wind Shear (200-850 hPa)")
    ax2.set_xlabel("Shear (m/s)")
    ax2.legend(fontsize=8)

    fig.suptitle("Physical Features — Cross-Basin Comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sst_shear_by_basin.png")
    plt.close(fig)
    print("  Saved: figures/sst_shear_by_basin.png")


def fig_missing_data(df_env: pd.DataFrame):
    """Heatmap of missing data (sentinel -1) rates by basin and feature."""
    sentinel_cols = ["future_direction24", "future_inte_change24",
                     "history_direction12", "history_direction24",
                     "history_inte_change24", "move_velocity"]
    rates = {}
    for b in BASINS:
        sub = df_env[df_env["basin"] == b]
        if len(sub) == 0:
            rates[b] = {c: 0 for c in sentinel_cols}
            continue
        rates[b] = {}
        for c in sentinel_cols:
            rates[b][c] = (sub[c] == -1).sum() / len(sub) * 100

    df_rates = pd.DataFrame(rates).T.reindex(BASINS)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df_rates, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "% missing (-1 sentinel)"})
    ax.set_title("Missing Data Rates by Basin & Feature",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "missing_data_by_basin.png")
    plt.close(fig)
    print("  Saved: figures/missing_data_by_basin.png")


# ---------------------------------------------------------------------------
# 4. Basin similarity matrix (Wasserstein distance)
# ---------------------------------------------------------------------------

def compute_basin_similarity(df_env: pd.DataFrame, df_nc: pd.DataFrame):
    """Compute Wasserstein distance between basins for key features.
    Returns a 6x6 similarity matrix (lower = more similar)."""

    features_env = ["wind"]  # continuous features from Env-Data
    features_nc = ["sst", "shear"]  # from Data3D

    # Collect distributions per basin
    basin_dists = {b: {} for b in BASINS}
    for b in BASINS:
        sub_env = df_env[df_env["basin"] == b]
        basin_dists[b]["wind"] = sub_env["wind"].dropna().values

        # Direction distribution (as histogram)
        valid_dir = sub_env[(sub_env["future_direction24"] >= 0) &
                            (sub_env["future_direction24"] <= 7)]
        if len(valid_dir) > 0:
            dir_counts = valid_dir["future_direction24"].value_counts().reindex(
                range(8), fill_value=0)
            basin_dists[b]["direction"] = (dir_counts / dir_counts.sum()).values
        else:
            basin_dists[b]["direction"] = np.ones(8) / 8

        # Intensity class distribution
        valid_ic = sub_env[sub_env["intensity_class"] >= 0]
        if len(valid_ic) > 0:
            ic_counts = valid_ic["intensity_class"].value_counts().reindex(
                range(6), fill_value=0)
            basin_dists[b]["intensity_class"] = (ic_counts / ic_counts.sum()).values
        else:
            basin_dists[b]["intensity_class"] = np.ones(6) / 6

        if not df_nc.empty:
            sub_nc = df_nc[df_nc["basin"] == b]
            basin_dists[b]["sst"] = sub_nc["sst"].dropna().values
            basin_dists[b]["shear"] = sub_nc["shear"].dropna().values

    # Compute pairwise Wasserstein distances
    feature_keys = ["wind", "direction", "intensity_class"]
    if not df_nc.empty:
        feature_keys += ["sst", "shear"]

    dist_matrix = np.zeros((6, 6))
    for i, b1 in enumerate(BASINS):
        for j, b2 in enumerate(BASINS):
            if i == j:
                continue
            total_dist = 0
            n_features = 0
            for fk in feature_keys:
                d1 = basin_dists[b1].get(fk, np.array([]))
                d2 = basin_dists[b2].get(fk, np.array([]))
                if len(d1) > 1 and len(d2) > 1:
                    w = wasserstein_distance(d1, d2)
                    total_dist += w
                    n_features += 1
            dist_matrix[i, j] = total_dist / max(n_features, 1)

    return dist_matrix


def fig_basin_similarity(dist_matrix: np.ndarray):
    """Plot basin similarity heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(dist_matrix, xticklabels=BASINS, yticklabels=BASINS,
                annot=True, fmt=".3f", cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "Avg Wasserstein distance (lower = more similar)"})
    ax.set_title("Basin Similarity Matrix\n(avg Wasserstein across wind, direction, intensity, SST, shear)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "basin_similarity_matrix.png")
    plt.close(fig)
    print("  Saved: figures/basin_similarity_matrix.png")

    # Print transfer recommendations
    print("\n--- Transfer Recommendations (from WP) ---")
    wp_idx = BASINS.index("WP")
    dists = [(BASINS[j], dist_matrix[wp_idx, j]) for j in range(6) if j != wp_idx]
    dists.sort(key=lambda x: x[1])
    for basin, d in dists:
        print(f"  WP → {basin}: distance = {d:.4f}")
    print(f"  Best target: {dists[0][0]} | Hardest target: {dists[-1][0]}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    df_1d, df_env, df_nc = load_all()

    print_basin_summary(df_1d, df_env)

    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    fig_storms_per_basin(df_1d)
    fig_temporal_distribution(df_1d)
    fig_intensity_by_basin(df_env)
    fig_direction_by_basin(df_env)
    fig_intensity_change_by_basin(df_env)
    fig_wind_distribution(df_env)
    fig_move_velocity(df_env)
    fig_sst_shear(df_nc)
    fig_missing_data(df_env)

    print("\n" + "=" * 60)
    print("BASIN SIMILARITY ANALYSIS")
    print("=" * 60)
    dist_matrix = compute_basin_similarity(df_env, df_nc)
    fig_basin_similarity(dist_matrix)

    print("\n" + "=" * 60)
    print(f"Done! All figures saved to {FIG_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
