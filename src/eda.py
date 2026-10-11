from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "src" / "delhi_pm25_hourly.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "eda"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_parquet(DATA_PATH)

df["period_start"] = pd.to_datetime(df["period_start"])
df = df.sort_values("period_start").reset_index(drop=True)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Rows: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")

print(
    f"Date range: "
    f"{df['period_start'].min()} → "
    f"{df['period_start'].max()}"
)

print(f"Missing PM2.5: {df['city_pm25'].isna().sum():,}")
print(
    f"Duplicate timestamps: "
    f"{df['period_start'].duplicated().sum():,}"
)


# --------------------------------------------------
# Verify hourly continuity
# --------------------------------------------------

expected_index = pd.date_range(
    start="2021-01-01 00:00:00",
    end="2024-12-31 23:00:00",
    freq="h"
)

actual_index = pd.DatetimeIndex(df["period_start"])

missing_timestamps = expected_index.difference(actual_index)

print("\n" + "=" * 60)
print("TIMESTAMP CONTINUITY")
print("=" * 60)

print(f"Expected hours: {len(expected_index):,}")
print(f"Actual hours:   {len(actual_index):,}")
print(f"Missing hours:  {len(missing_timestamps):,}")


# --------------------------------------------------
# PM2.5 statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("PM2.5 STATISTICS")
print("=" * 60)

print(df["city_pm25"].describe())

print(f"\nZero values: {(df['city_pm25'] == 0).sum():,}")
print(f"Negative values: {(df['city_pm25'] < 0).sum():,}")
print(
    f"Values >= 500: "
    f"{(df['city_pm25'] >= 500).sum():,}"
)
print(
    f"Values >= 1000: "
    f"{(df['city_pm25'] >= 1000).sum():,}"
)


# --------------------------------------------------
# Station coverage
# --------------------------------------------------

print("\n" + "=" * 60)
print("STATION COVERAGE")
print("=" * 60)

print(df["station_count"].describe())

for threshold in [5, 10, 15, 20, 25, 30, 35]:
    count = (df["station_count"] < threshold).sum()
    pct = count / len(df) * 100

    print(
        f"Hours with < {threshold:2d} stations: "
        f"{count:5d} ({pct:.2f}%)"
    )


# --------------------------------------------------
# Add temporal features
# --------------------------------------------------

df["year"] = df["period_start"].dt.year
df["month"] = df["period_start"].dt.month
df["hour"] = df["period_start"].dt.hour
df["day_of_week"] = df["period_start"].dt.dayofweek


month_names = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

df["month_name"] = df["month"].map(month_names)


# --------------------------------------------------
# Monthly summary
# --------------------------------------------------

monthly = (
    df.groupby("month")
      .agg(
          mean_pm25=("city_pm25", "mean"),
          median_pm25=("city_pm25", "median"),
          std_pm25=("city_pm25", "std"),
          min_pm25=("city_pm25", "min"),
          max_pm25=("city_pm25", "max"),
          hours=("city_pm25", "size"),
          mean_station_count=("station_count", "mean"),
      )
      .reset_index()
)

monthly["month_name"] = monthly["month"].map(month_names)

monthly.to_csv(
    RESULTS_DIR / "monthly_summary.csv",
    index=False
)

print("\n" + "=" * 60)
print("MONTHLY SUMMARY")
print("=" * 60)

print(
    monthly[
        [
            "month_name",
            "mean_pm25",
            "median_pm25",
            "std_pm25",
            "mean_station_count",
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# Yearly summary
# --------------------------------------------------

yearly = (
    df.groupby("year")
      .agg(
          mean_pm25=("city_pm25", "mean"),
          median_pm25=("city_pm25", "median"),
          std_pm25=("city_pm25", "std"),
          min_pm25=("city_pm25", "min"),
          max_pm25=("city_pm25", "max"),
          mean_station_count=("station_count", "mean"),
      )
      .reset_index()
)

yearly.to_csv(
    RESULTS_DIR / "yearly_summary.csv",
    index=False
)

print("\n" + "=" * 60)
print("YEARLY SUMMARY")
print("=" * 60)

print(yearly.to_string(index=False))


# --------------------------------------------------
# Hour-of-day profile
# --------------------------------------------------

hourly_profile = (
    df.groupby("hour")
      .agg(
          mean_pm25=("city_pm25", "mean"),
          median_pm25=("city_pm25", "median"),
      )
      .reset_index()
)

hourly_profile.to_csv(
    RESULTS_DIR / "hourly_profile.csv",
    index=False
)

print("\n" + "=" * 60)
print("HOURLY PROFILE")
print("=" * 60)

print(hourly_profile.to_string(index=False))


# --------------------------------------------------
# Plot 1: Full time series
# --------------------------------------------------

plt.figure(figsize=(16, 6))

plt.plot(
    df["period_start"],
    df["city_pm25"],
    linewidth=0.6
)

plt.xlabel("Date")
plt.ylabel("PM2.5 (µg/m³)")
plt.title("Delhi Hourly PM2.5 — 2021 to 2024")

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "full_time_series.png",
    dpi=200
)
plt.close()


# --------------------------------------------------
# Plot 2: Monthly mean PM2.5
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    monthly["month_name"],
    monthly["mean_pm25"]
)

plt.xlabel("Month")
plt.ylabel("Mean PM2.5 (µg/m³)")
plt.title("Average Delhi PM2.5 by Month")

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "monthly_mean.png",
    dpi=200
)
plt.close()


# --------------------------------------------------
# Plot 3: Hour-of-day profile
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    hourly_profile["hour"],
    hourly_profile["mean_pm25"],
    marker="o"
)

plt.xlabel("Hour of Day")
plt.ylabel("Mean PM2.5 (µg/m³)")
plt.title("Average Delhi PM2.5 by Hour of Day")

plt.xticks(np.arange(0, 24, 2))

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "hourly_profile.png",
    dpi=200
)
plt.close()


# --------------------------------------------------
# Plot 4: PM2.5 distribution
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    df["city_pm25"],
    bins=100
)

plt.xlabel("PM2.5 (µg/m³)")
plt.ylabel("Frequency")
plt.title("Distribution of Delhi PM2.5")

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "pm25_distribution.png",
    dpi=200
)
plt.close()


# --------------------------------------------------
# Plot 5: Station count distribution
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.hist(
    df["station_count"],
    bins=np.arange(
        df["station_count"].min() - 0.5,
        df["station_count"].max() + 1.5,
        1
    )
)

plt.xlabel("Number of Reporting Stations")
plt.ylabel("Number of Hours")
plt.title("Station Coverage per Hour")

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "station_count_distribution.png",
    dpi=200
)
plt.close()


print("\nEDA complete.")
print(f"Results saved to: {RESULTS_DIR}")