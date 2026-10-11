import io
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AQI_API_KEY")

if not API_KEY:
    raise ValueError("AQI_API_KEY not found in .env")

URL = "https://airquality.xkdr.org/v1/measurements"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

# --------------------------------------------------
# Download monthly station-level data
# --------------------------------------------------

all_data = []

months = pd.date_range(
    start="2021-01-01",
    end="2024-12-01",
    freq="MS"
)

for month_start in months:

    month_end = month_start + pd.offsets.MonthEnd(1)

    start = month_start.strftime("%Y-%m-%d")
    end = month_end.strftime("%Y-%m-%d")

    print(f"Downloading {start} → {end}")

    params = {
        "city": "Delhi",
        "source": "cpcb_caaqm",
        "parameter": "PM2.5",
        "start": start,
        "end": end,
        "agg": "hourly",
        "format": "csv"
    }

    response = requests.get(
        URL,
        headers=HEADERS,
        params=params,
        timeout=90
    )

    response.raise_for_status()

    month_df = pd.read_csv(
        io.StringIO(response.text),
        parse_dates=["period_start"]
    )

    # Convert numerical columns explicitly
    numeric_cols = ["mean", "min", "max", "n"]

    for col in numeric_cols:
        month_df[col] = pd.to_numeric(
            month_df[col],
            errors="coerce"
        )

    print(
        f"  {len(month_df):,} station-hour records "
        f"from {month_df['station_id'].nunique()} stations"
    )

    all_data.append(month_df)


# --------------------------------------------------
# Combine station-level data
# --------------------------------------------------

df = pd.concat(
    all_data,
    ignore_index=True
)

df = df.sort_values(
    ["period_start", "station_id"]
)

# Save station-level dataset
df.to_parquet(
    "delhi_pm25_station_hourly.parquet",
    engine="pyarrow",
    compression="snappy",
    index=False
)

print("\nStation-level dataset:")
print(df.info())


# --------------------------------------------------
# Construct Delhi-wide hourly PM2.5
# --------------------------------------------------

city_df = (
    df.groupby("period_start")
      .agg(
          city_pm25=("mean", "mean"),
          station_count=("station_id", "nunique")
      )
      .reset_index()
)

city_df["city_pm25"] = pd.to_numeric(
    city_df["city_pm25"],
    errors="coerce"
)

city_df = city_df.sort_values("period_start")


# --------------------------------------------------
# Verify complete modelling period
# --------------------------------------------------

expected_index = pd.date_range(
    start="2021-01-01 00:00:00",
    end="2024-12-31 23:00:00",
    freq="h"
)

actual_index = pd.DatetimeIndex(
    city_df["period_start"]
)

missing_hours = expected_index.difference(actual_index)

print("\nExpected hours:", len(expected_index))
print("Actual hours:", len(actual_index))
print("Missing hours:", len(missing_hours))

if len(missing_hours) > 0:
    print("\nFirst missing hours:")
    print(missing_hours[:20])


# --------------------------------------------------
# Save final modelling dataset
# --------------------------------------------------

city_df.to_parquet(
    "delhi_pm25_hourly.parquet",
    engine="pyarrow",
    compression="snappy",
    index=False
)

print("\nCity-level dataset:")
print(city_df.info())

print("\nDate range:")
print(
    city_df["period_start"].min(),
    "→",
    city_df["period_start"].max()
)

print("\nRows:")
print(len(city_df))

print("\nMissing PM2.5:")
print(city_df["city_pm25"].isna().sum())

print("\nStation count:")
print(city_df["station_count"].describe())

print("\nFirst rows:")
print(city_df.head())

print("\nLast rows:")
print(city_df.tail())