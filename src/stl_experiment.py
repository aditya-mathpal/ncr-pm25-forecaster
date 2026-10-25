from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "stl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
HORIZONS = [1, 6, 24]

FEATURE_COLUMNS = [
    "current_deseasonalized",
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


def get_seasonal_profile(series_df: pd.DataFrame) -> dict[int, float]:
    """Estimate a reusable 24-hour seasonal profile from the supplied period."""
    series = (
        series_df.set_index("period_start")["city_pm25"]
        .sort_index()
        .asfreq("h")
    )
    if series.isna().any():
        raise ValueError("STL input has missing observations after hourly reindexing.")

    decomposition = STL(series, period=24, robust=True).fit()
    profile = (
        pd.DataFrame({
            "hour": decomposition.seasonal.index.hour,
            "seasonal": decomposition.seasonal.to_numpy(),
        })
        .groupby("hour")["seasonal"]
        .mean()
        .to_dict()
    )
    if set(profile) != set(range(24)):
        raise ValueError("The STL profile must contain one value for each hour of day.")
    return {int(hour): float(value) for hour, value in profile.items()}


def create_features(
    df: pd.DataFrame,
    seasonal_profile: dict[int, float],
    horizon: int,
) -> pd.DataFrame:
    """Remove the fixed hour-of-day profile, then make features at time t for y(t+h)."""
    data = df.copy()
    data["seasonal"] = data["period_start"].dt.hour.map(seasonal_profile)
    if data["seasonal"].isna().any():
        raise ValueError("Could not assign an STL seasonal value to every observation.")

    data["deseasonalized"] = data["city_pm25"] - data["seasonal"]
    values = data["deseasonalized"]
    data["current_deseasonalized"] = values

    for lag in LAGS:
        data[f"lag_{lag}"] = values.shift(lag)

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
    # Keep the forecaster settings the same as the raw-series baseline so
    # the comparison focuses on the STL preprocessing, not a different model.
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


print("Estimating the primary STL profile from training data...")
train_only_profile = get_seasonal_profile(train)

# This second profile uses the validation period too. It is only a diagnostic
# for sensitivity to the decomposition window; it is not a deployable forecast.
print("Estimating the validation-informed profile for the diagnostic comparison...")
train_and_validation = pd.concat([train, validation], ignore_index=True)
train_and_validation = train_and_validation.sort_values("period_start").reset_index(drop=True)
validation_informed_profile = get_seasonal_profile(train_and_validation)

validation_start = validation["period_start"].min()
validation_end = validation["period_start"].max()
validation_actual = validation.set_index("period_start")["city_pm25"]
experiments = [
    ("train_only_stl", train_only_profile),
    ("validation_leaky_stl_diagnostic", validation_informed_profile),
]
results = []

for method, seasonal_profile in experiments:
    for horizon in HORIZONS:
        print(f"\n{method} — {horizon}-hour horizon")

        train_features = create_features(train, seasonal_profile, horizon)
        train_features = train_features.dropna(subset=FEATURE_COLUMNS + ["target"])
        train_features = train_features.reset_index(drop=True)
        if train_features.empty:
            raise ValueError(f"No usable STL training rows for horizon {horizon}.")

        # The forecast at the start of validation needs some observations from
        # the end of training to supply its lags.
        history_length = max(LAGS) + horizon
        history = train.tail(history_length)
        combined = pd.concat([history, validation], ignore_index=True)
        combined = combined.sort_values("period_start").reset_index(drop=True)
        combined_features = create_features(combined, seasonal_profile, horizon)

        target_mask = combined_features["target_time"].between(validation_start, validation_end)
        validation_features = combined_features.loc[target_mask].copy()
        validation_features = validation_features.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        if len(validation_features) != len(validation):
            raise ValueError(
                f"Horizon {horizon}: expected {len(validation)} validation forecasts, "
                f"but created {len(validation_features)}. Check the hourly data and lags."
            )

        model = build_model()
        model.fit(train_features[FEATURE_COLUMNS], train_features["target"])
        predicted_deseasonalized = model.predict(validation_features[FEATURE_COLUMNS])

        target_times = pd.DatetimeIndex(validation_features["target_time"])
        forecast_origins = target_times - pd.to_timedelta(horizon, unit="h")
        actual = validation_actual.reindex(target_times).to_numpy(dtype=float)
        seasonal_at_target = target_times.hour.map(seasonal_profile).to_numpy(dtype=float)
        predictions = predicted_deseasonalized + seasonal_at_target

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
            "method": method,
        })
        prediction_path = RESULTS_DIR / f"validation_predictions_{method}_{horizon}h.csv"
        prediction_df.to_csv(prediction_path, index=False)

        results.append({
            "method": method,
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

print("\nSTL validation summary")
print(results_df.to_string(index=False))
print(f"\nSummary saved to: {results_path}")
