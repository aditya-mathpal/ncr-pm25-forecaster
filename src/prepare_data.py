from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "src" / "delhi_pm25_hourly.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_parquet(DATA_PATH)

df["period_start"] = pd.to_datetime(df["period_start"])

df = (
    df
    .sort_values("period_start")
    .reset_index(drop=True)
)


# --------------------------------------------------
# Verify dataset
# --------------------------------------------------

expected_index = pd.date_range(
    start="2021-01-01 00:00:00",
    end="2024-12-31 23:00:00",
    freq="h"
)

actual_index = pd.DatetimeIndex(df["period_start"])

missing = expected_index.difference(actual_index)

if len(missing) != 0:
    raise ValueError(
        f"Dataset contains {len(missing)} missing timestamps."
    )

if df["city_pm25"].isna().any():
    raise ValueError(
        "Dataset contains missing PM2.5 values."
    )

if df["period_start"].duplicated().any():
    raise ValueError(
        "Dataset contains duplicate timestamps."
    )


# --------------------------------------------------
# Chronological split
# --------------------------------------------------

train = df[
    df["period_start"] < "2023-01-01"
].copy()

validation = df[
    (df["period_start"] >= "2023-01-01") &
    (df["period_start"] < "2024-01-01")
].copy()

test = df[
    df["period_start"] >= "2024-01-01"
].copy()


# --------------------------------------------------
# Save splits
# --------------------------------------------------

train.to_parquet(
    OUTPUT_DIR / "train.parquet",
    engine="pyarrow",
    compression="snappy",
    index=False
)

validation.to_parquet(
    OUTPUT_DIR / "validation.parquet",
    engine="pyarrow",
    compression="snappy",
    index=False
)

test.to_parquet(
    OUTPUT_DIR / "test.parquet",
    engine="pyarrow",
    compression="snappy",
    index=False
)


# --------------------------------------------------
# Report
# --------------------------------------------------

print("=" * 60)
print("DATASET SPLIT")
print("=" * 60)

for name, split in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:

    print(f"\n{name}")
    print(f"Rows: {len(split):,}")
    print(
        f"Range: "
        f"{split['period_start'].min()} → "
        f"{split['period_start'].max()}"
    )
    print(
        f"Mean PM2.5: "
        f"{split['city_pm25'].mean():.2f}"
    )

print("\nFiles written to:")
print(OUTPUT_DIR)