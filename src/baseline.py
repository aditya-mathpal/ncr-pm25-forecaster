from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 6, 24]


def load_and_check_data(path: Path, name: str) -> pd.DataFrame:
    """Load one split and make sure the timestamps and measurements look usable."""
    df = pd.read_parquet(path)
    required = {"period_start", "city_pm25"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{name} is missing columns: {sorted(missing_columns)}")

    df["period_start"] = pd.to_datetime(df["period_start"])
    df = df.sort_values("period_start").reset_index(drop=True)

    if df["period_start"].duplicated().any():
        raise ValueError(f"{name} has duplicate timestamps.")
    if df["city_pm25"].isna().any():
        raise ValueError(f"{name} contains missing PM2.5 values.")
    if len(df) > 1 and not (df["period_start"].diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{name} is not continuous hourly data.")

    return df


train = load_and_check_data(DATA_DIR / "train.parquet", "Training data")
validation = load_and_check_data(DATA_DIR / "validation.parquet", "Validation data")

if train["period_start"].max() >= validation["period_start"].min():
    raise ValueError("Training and validation periods overlap or are out of order.")

# We keep the fitted model fixed during validation, but measurements from earlier
# in 2023 are available as time moves forward. That is a rolling-origin evaluation.
observations = pd.concat([train, validation], ignore_index=True)
observations = observations.sort_values("period_start").reset_index(drop=True)
observed_series = observations.set_index("period_start")["city_pm25"]
validation_actual = validation.set_index("period_start")["city_pm25"]
validation_target_times = pd.DatetimeIndex(validation["period_start"])

results = []

for horizon in HORIZONS:
    # For a target at T, a forecast h hours ahead was issued at T-h.
    target_times = validation_target_times
    forecast_origins = target_times - pd.to_timedelta(horizon, unit="h")

    predictions = observed_series.reindex(forecast_origins).to_numpy(dtype=float)
    actual = validation_actual.reindex(target_times).to_numpy(dtype=float)

    if np.isnan(predictions).any():
        raise ValueError(f"Could not find all persistence inputs for horizon {horizon}.")
    if np.isnan(actual).any():
        raise ValueError("Some validation targets have no matching actual measurement.")

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    prediction_df = pd.DataFrame({
        "target_time": target_times,
        "forecast_origin": forecast_origins,
        "horizon_hours": horizon,
        "actual": actual,
        "prediction": predictions,
        "method": "persistence",
    })
    prediction_path = RESULTS_DIR / f"validation_predictions_{horizon}h.csv"
    prediction_df.to_csv(prediction_path, index=False)

    results.append({
        "method": "persistence",
        "horizon_hours": horizon,
        "n_validation": len(actual),
        "MAE": mae,
        "RMSE": rmse,
    })

    print(f"\nPersistence baseline — {horizon}-hour horizon")
    print(f"Validation targets: {len(actual):,}")
    print(f"Target period: {target_times.min()} to {target_times.max()}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Predictions saved to: {prediction_path}")

results_df = pd.DataFrame(results)
results_path = RESULTS_DIR / "validation_results.csv"
results_df.to_csv(results_path, index=False)

print("\nPersistence validation summary")
print(results_df.to_string(index=False))
print(f"\nSummary saved to: {results_path}")
