from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "xgboost"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
HORIZONS = [1, 6, 24]

FEATURE_COLUMNS = [
    "current_pm25",
    *[f"lag_{lag}" for lag in LAGS],
    "rolling_mean_6",
    "rolling_mean_24",
    "rolling_std_24",
    "hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


def load_and_check_data(path: Path, name: str) -> pd.DataFrame:
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


def add_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build features available at time t and label the value at t+h."""
    data = df.copy()
    values = data["city_pm25"]

    # The current measurement is known when the forecast is issued.
    data["current_pm25"] = values
    for lag in LAGS:
        data[f"lag_{lag}"] = values.shift(lag)

    # These summaries include the current observed value, never a future one.
    data["rolling_mean_6"] = values.rolling(6).mean()
    data["rolling_mean_24"] = values.rolling(24).mean()
    data["rolling_std_24"] = values.rolling(24).std()

    data["hour"] = data["period_start"].dt.hour
    data["day_of_week"] = data["period_start"].dt.dayofweek
    data["month"] = data["period_start"].dt.month
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["month_sin"] = np.sin(2 * np.pi * (data["month"] - 1) / 12)
    data["month_cos"] = np.cos(2 * np.pi * (data["month"] - 1) / 12)

    data["target"] = values.shift(-horizon)
    data["target_time"] = data["period_start"] + pd.to_timedelta(horizon, unit="h")
    return data


def build_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )


validation_start = validation["period_start"].min()
validation_end = validation["period_start"].max()
validation_actual = validation.set_index("period_start")["city_pm25"]
results = []

for horizon in HORIZONS:
    print(f"\nXGBoost baseline — {horizon}-hour horizon")

    # Training labels come only from the training period. The final h rows
    # naturally fall out because their future targets are not in this split.
    train_features = add_features(train, horizon)
    train_features = train_features.dropna(subset=FEATURE_COLUMNS + ["target"])
    train_features = train_features.reset_index(drop=True)
    if train_features.empty:
        raise ValueError(f"No usable training rows for horizon {horizon}.")

    model = build_model()
    model.fit(train_features[FEATURE_COLUMNS], train_features["target"])

    # To forecast the earliest validation hours, we need the tail of training
    # history as well as the measurements that would have arrived during 2023.
    history_length = max(LAGS) + horizon
    history = train.tail(history_length)
    combined = pd.concat([history, validation], ignore_index=True)
    combined = combined.sort_values("period_start").reset_index(drop=True)
    combined_features = add_features(combined, horizon)

    # Validation membership is determined by the time being predicted, not
    # by the row where the forecast was issued.
    target_mask = combined_features["target_time"].between(validation_start, validation_end)
    validation_features = combined_features.loc[target_mask].copy()
    validation_features = validation_features.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    if len(validation_features) != len(validation):
        raise ValueError(
            f"Horizon {horizon}: expected {len(validation)} validation forecasts, "
            f"but created {len(validation_features)}. Check the hourly data and lags."
        )

    target_times = pd.DatetimeIndex(validation_features["target_time"])
    forecast_origins = target_times - pd.to_timedelta(horizon, unit="h")
    actual = validation_actual.reindex(target_times).to_numpy(dtype=float)
    predictions = model.predict(validation_features[FEATURE_COLUMNS])

    if np.isnan(actual).any():
        raise ValueError(f"Some actual values are missing for horizon {horizon}.")

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    prediction_df = pd.DataFrame({
        "target_time": target_times,
        "forecast_origin": forecast_origins,
        "horizon_hours": horizon,
        "actual": actual,
        "prediction": predictions,
        "method": "xgboost_baseline",
    })
    prediction_path = RESULTS_DIR / f"validation_predictions_{horizon}h.csv"
    prediction_df.to_csv(prediction_path, index=False)

    results.append({
        "method": "xgboost_baseline",
        "horizon_hours": horizon,
        "n_train": len(train_features),
        "n_validation": len(validation_features),
        "MAE": mae,
        "RMSE": rmse,
    })

    print(f"Training rows: {len(train_features):,}")
    print(f"Validation targets: {len(validation_features):,}")
    print(f"Target period: {target_times.min()} to {target_times.max()}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Predictions saved to: {prediction_path}")

results_df = pd.DataFrame(results)
results_path = RESULTS_DIR / "validation_results.csv"
results_df.to_csv(results_path, index=False)

print("\nXGBoost validation summary")
print(results_df.to_string(index=False))
print(f"\nSummary saved to: {results_path}")
