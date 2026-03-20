"""
Extract temporal features from split_index.csv and save per-split .pt files.

For each split and each storm, computes a tensor of shape (N_t, 6):
    [storm_progress, hour_sin, hour_cos, month_sin, month_cos, timestep_idx_normalized]

Output: data/processed/time/{split}_time.pt  (dict of {storm_name: tensor})
"""

import math
import pandas as pd
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_PATH = DATA_DIR / "split_index.csv"
OUT_DIR = DATA_DIR / "time"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_NAMES = ["wp_train", "wp_val", "sp_test", "sp_ft_train", "sp_ft_val"]
TWO_PI = 2.0 * math.pi


def parse_datetime(dt_str: str):
    """Parse YYYYMMDDhh string into (month, hour)."""
    s = str(dt_str)
    month = int(s[4:6])
    hour = int(s[8:10])
    return month, hour


def build_time_features(group_df: pd.DataFrame) -> torch.Tensor:
    """Build (N_t, 6) tensor for one storm from its rows (already sorted)."""
    n = len(group_df)
    feats = torch.zeros(n, 6)

    for i, (_, row) in enumerate(group_df.iterrows()):
        month, hour = parse_datetime(row["datetime"])

        # storm_progress: 0.0 at start, 1.0 at end
        progress = i / max(n - 1, 1)

        # hour cyclical (period = 24)
        hour_sin = math.sin(TWO_PI * hour / 24.0)
        hour_cos = math.cos(TWO_PI * hour / 24.0)

        # month cyclical (period = 12)
        month_sin = math.sin(TWO_PI * (month - 1) / 12.0)
        month_cos = math.cos(TWO_PI * (month - 1) / 12.0)

        # timestep_idx normalized by storm length
        timestep_norm = i / max(n - 1, 1)

        feats[i] = torch.tensor([
            progress, hour_sin, hour_cos, month_sin, month_cos, timestep_norm
        ])

    return feats


def main():
    print(f"Reading {INDEX_PATH}")
    df = pd.read_csv(INDEX_PATH)

    # Sort by storm and datetime to ensure correct ordering
    df = df.sort_values(["split", "storm", "datetime"]).reset_index(drop=True)

    for split in SPLIT_NAMES:
        split_df = df[df["split"] == split]
        if split_df.empty:
            print(f"  {split}: no rows found, skipping")
            continue

        time_dict = {}
        for storm_name, storm_df in split_df.groupby("storm", sort=False):
            storm_df_sorted = storm_df.sort_values("datetime")
            time_dict[storm_name] = build_time_features(storm_df_sorted)

        out_path = OUT_DIR / f"{split}_time.pt"
        torch.save(time_dict, out_path)

        # Summary
        total_ts = sum(v.shape[0] for v in time_dict.values())
        print(f"  {split:15s}: {len(time_dict):3d} storms, "
              f"{total_ts:5d} timesteps -> {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
