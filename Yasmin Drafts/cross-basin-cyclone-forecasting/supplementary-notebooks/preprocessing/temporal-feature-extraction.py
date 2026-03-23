"""
Extract temporal features from split_index.csv and save per-split .pt files.

For each split and each storm, computes a tensor of shape (N_t, 6):
    [storm_progress, hour_sin, hour_cos, month_sin, month_cos, storm_duration_norm]

Timesteps missing from Data1D are filtered out to stay aligned with the
grid/env/data1d/label .pt files produced by data-preprocessing-pipeline.ipynb.

Output: data/processed-v2/time/{split}_time.pt  (dict of {storm_name: tensor})
"""

import math
import pandas as pd
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed-v2"
INDEX_PATH = DATA_DIR / "split_index.csv"
OUT_DIR = DATA_DIR / "time"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TCND_ROOT = PROJECT_ROOT / "data" / "tropicyclonenet" / "TCND_test" / "TCND_test"
BASINS = ["WP", "SP"]
SPLIT_NAMES = ["wp_train", "wp_val", "sp_test", "sp_ft_train", "sp_ft_val"]
TWO_PI = 2.0 * math.pi

DATA1D_COLS = ["index", "category", "lat_offset", "lon_offset",
               "wind_norm", "pressure_norm", "datetime", "name"]


def parse_datetime(dt_str: str):
    """Parse YYYYMMDDhh string into (month, hour)."""
    s = str(dt_str)
    month = int(s[4:6])
    hour = int(s[8:10])
    return month, hour


def build_time_features(group_df: pd.DataFrame, max_storm_len: int) -> torch.Tensor:
    """Build (N_t, 6) tensor for one storm from its rows (already sorted)."""
    n = len(group_df)
    feats = torch.zeros(n, 6)
    duration_norm = n / max(max_storm_len, 1)

    for i, (_, row) in enumerate(group_df.iterrows()):
        month, hour = parse_datetime(row["datetime"])
        progress = i / max(n - 1, 1)

        feats[i] = torch.tensor([
            progress,
            math.sin(TWO_PI * hour / 24.0),
            math.cos(TWO_PI * hour / 24.0),
            math.sin(TWO_PI * (month - 1) / 12.0),
            math.cos(TWO_PI * (month - 1) / 12.0),
            duration_norm,
        ])

    return feats


def load_data1d_datetimes():
    """Load all Data1D text files and return a set of (storm_file_stem, datetime) tuples."""
    datetimes = set()
    all_names = []
    for basin in BASINS:
        d1d_dir = TCND_ROOT / "Data1D" / basin / "test"
        for f in sorted(d1d_dir.glob("*.txt")):
            storm_name = f.stem
            all_names.append(storm_name)
            d1d_df = pd.read_csv(f, sep="\t", header=None, names=DATA1D_COLS,
                                 dtype={"datetime": str})
            for dt in d1d_df["datetime"].str.strip():
                datetimes.add((storm_name, dt))
    return datetimes, all_names


def find_data1d_storm(storm_name, basin, all_names):
    """Find the Data1D key matching a storm from the split index."""
    candidates = [k for k in all_names if storm_name in k and k.startswith(basin)]
    return candidates[0] if candidates else None


def main():
    print(f"Reading {INDEX_PATH}")
    df_raw = pd.read_csv(INDEX_PATH)
    df_raw = df_raw.sort_values(["split", "storm", "datetime"]).reset_index(drop=True)
    df_raw["datetime"] = df_raw["datetime"].astype(str)
    print(f"Raw: {len(df_raw)} rows, {df_raw['storm'].nunique()} storms")

    # Filter to timesteps with matching Data1D entries
    data1d_datetimes, data1d_all_names = load_data1d_datetimes()
    print(f"Data1D lookup: {len(data1d_datetimes)} entries from {len(data1d_all_names)} storms")

    storm_name_map = {}
    for _, row in df_raw[["basin", "storm"]].drop_duplicates().iterrows():
        storm_name_map[(row["basin"], row["storm"])] = find_data1d_storm(
            row["storm"], row["basin"], data1d_all_names)

    keep_mask = []
    for _, row in df_raw.iterrows():
        d1d_name = storm_name_map.get((row["basin"], row["storm"]))
        dt = str(row["datetime"]).strip()
        keep_mask.append(d1d_name is not None and (d1d_name, dt) in data1d_datetimes)

    df = df_raw[keep_mask].reset_index(drop=True)
    print(f"After filtering: {len(df)} rows, {df['storm'].nunique()} storms")

    max_storm_len = df.groupby(["split", "storm"]).size().max()
    print(f"Max storm length: {max_storm_len}")

    for split in SPLIT_NAMES:
        split_df = df[df["split"] == split]
        if split_df.empty:
            print(f"  {split}: no rows found, skipping")
            continue

        time_dict = {}
        for storm_name, storm_df in split_df.groupby("storm", sort=False):
            storm_df_sorted = storm_df.sort_values("datetime")
            time_dict[storm_name] = build_time_features(storm_df_sorted, max_storm_len)

        out_path = OUT_DIR / f"{split}_time.pt"
        torch.save(time_dict, out_path)

        total_ts = sum(v.shape[0] for v in time_dict.values())
        print(f"  {split:15s}: {len(time_dict):3d} storms, "
              f"{total_ts:5d} timesteps -> {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
