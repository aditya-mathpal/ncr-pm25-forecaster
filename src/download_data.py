import io
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("AQI_API_KEY")
if not API_KEY:
    raise ValueError("AQI_API_KEY was not found. Add it to the project's .env file.")

URL = "https://airquality.xkdr.org/v1/measurements"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
RAW_OUTPUT_DIR = PROJECT_ROOT / "src"
RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_monthly_station_data() -> pd.DataFrame:
    """Download the available hourly PM2.5 station records one month at a time."""
    frames = []
    months = pd.date_range(start="2021-01-01", end="2024-12-01", freq="MS")

    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(1)
        start = month_start.strftime("%Y-%m-%d")
        end = month_end.strftime("%Y-%m-%d")
        print(f"Downloading {start} to {end}...")

        params = {
            "city": "Delhi",
            "source": "cpcb_caaqm",
            "parameter": "PM2.5",
            "start": start,
            "end": end,
            "agg": "hourly",
            "format": "csv",
        }

        response = requests.get(URL, headers=HEADERS, params=params, timeout=90)
        response.raise_for_status()
        if not response.text.strip():
            print("  No rows returned for this month.")
            continue

        month_df = pd.read_csv(io.StringIO(response.text))
        if month_df.empty:
            print("  No rows returned for this month.")
            continue

        required = {"period_start", "station_id", "mean"}
        missing_columns = required.difference(month_df.columns)
        if missing_columns:
            raise ValueError(
                f"The API response for {start} is missing columns: {sorted(missing_columns)}"
            )

        month_df["period_start"] = pd.to_datetime(month_df["period_start"])
        for column in ["mean", "min", "max", "n"]:
            if column in month_df.columns:
                month_df[column] = pd.to_numeric(month_df[column], errors="coerce")

        print(
            f"  {len(month_df):,} station-hour records "
            f"from {month_df['station_id'].nunique()} stations"
        )
        frames.append(month_df)

    if not frames:
        raise RuntimeError("The API returned no station records for the requested period.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["period_start", "station_id"]).reset_index(drop=True)
    return df


def build_city_hourly(station_df: pd.DataFrame) -> pd.DataFrame:
    """Take the mean of available station-hour means for each hour in Delhi."""
    city_df = (
        station_df.groupby("period_start")
        .agg(
            city_pm25=("mean", "mean"),
            station_count=("station_id", "nunique"),
        )
        .reset_index()
        .sort_values("period_start")
        .reset_index(drop=True)
    )
    city_df["city_pm25"] = pd.to_numeric(city_df["city_pm25"], errors="coerce")
    return city_df


def main() -> None:
    station_df = download_monthly_station_data()
    station_path = RAW_OUTPUT_DIR / "delhi_pm25_station_hourly.parquet"
    station_df.to_parquet(station_path, engine="pyarrow", compression="snappy", index=False)
    print(f"\nStation-level data saved to: {station_path}")
    print(f"Station-hour rows: {len(station_df):,}")

    city_df = build_city_hourly(station_df)
    expected_index = pd.date_range(
        start="2021-01-01 00:00:00",
        end="2024-12-31 23:00:00",
        freq="h",
    )
    actual_index = pd.DatetimeIndex(city_df["period_start"])
    missing_hours = expected_index.difference(actual_index)
    extra_hours = actual_index.difference(expected_index)

    print(f"\nExpected hourly timestamps: {len(expected_index):,}")
    print(f"Observed city-level hours: {len(actual_index):,}")
    print(f"Missing hours: {len(missing_hours):,}")
    print(f"Timestamps outside requested period: {len(extra_hours):,}")
    if len(missing_hours):
        print("First missing timestamps:")
        print(missing_hours[:20].to_list())
    if len(extra_hours):
        print("First out-of-range timestamps:")
        print(extra_hours[:20].to_list())

    city_path = RAW_OUTPUT_DIR / "delhi_pm25_hourly.parquet"
    city_df.to_parquet(city_path, engine="pyarrow", compression="snappy", index=False)
    print(f"\nCity-level data saved to: {city_path}")
    print(f"Date range: {city_df['period_start'].min()} to {city_df['period_start'].max()}")
    print(f"Missing PM2.5 values: {city_df['city_pm25'].isna().sum():,}")
    print(f"Mean PM2.5: {city_df['city_pm25'].mean():.2f} µg/m³")

    if len(missing_hours) or len(extra_hours) or city_df["city_pm25"].isna().any():
        raise ValueError(
            "The city-level dataset does not cover the expected hourly period cleanly. "
            "Check the download output before running prepare_data.py."
        )


if __name__ == "__main__":
    main()
