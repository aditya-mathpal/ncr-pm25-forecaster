"""Evaluate the frozen forecasting setup on the untouched 2024 test period."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "test"
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

STL_FEATURE_COLUMNS = [
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
test = load_and_check_data(DATA_DIR / "test.parquet", "Test data")

if train["period_start"].max() >= validation["period_start"].min():
    raise ValueError("Training and validation periods overlap or are out of order.")
if validation["period_start"].max() >= test["period_start"].min():
    raise ValueError("Validation and test periods overlap or are out of order.")

# The model and seasonal profile are fitted on training data only. Earlier
# validation/test observations may be used as lagged inputs once their times
# have passed, just as they would be available in a real rolling forecast.
all_observations = pd.concat([train, validation, test], ignore_index=True)
all_observations = all_observations.sort_values("period_start").reset_index(drop=True)
observed_series = all_observations.set_index("period_start")["city_pm25"]
test_actual = test.set_index("period_start")["city_pm25"]
test_start = test["period_start"].min()
test_end = test["period_start"].max()


def build_model() -> XGBRegressor:
    # These are the same settings used in the validation experiments.
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


def add_raw_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    data = df.copy()
    values = data["city_pm25"]
    data["current_pm25"] = values
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


def get_seasonal_profile(series_df: pd.DataFrame) -> dict[int, float]:
    series = (
        series_df.set_index("period_start")["city_pm25"]
        .sort_index()
        .asfreq("h")
    )
    if series.isna().any():
        raise ValueError("STL training data has missing observations after hourly reindexing.")

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


def add_stl_features(
    df: pd.DataFrame,
    seasonal_profile: dict[int, float],
    horizon: int,
) -> pd.DataFrame:
    data = df.copy()
    data["seasonal"] = data["period_start"].dt.hour.map(seasonal_profile)
    if data["seasonal"].isna().any():
        raise ValueError("Could not assign an STL value to every observation.")

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


def save_scores(
    method: str,
    horizon: int,
    target_times: pd.DatetimeIndex,
    actual: np.ndarray,
    predictions: np.ndarray,
) -> list[dict]:
    prediction_df = pd.DataFrame({
        "target_time": target_times,
        "forecast_origin": target_times - pd.to_timedelta(horizon, unit="h"),
        "horizon_hours": horizon,
        "actual": actual,
        "prediction": predictions,
        "method": method,
    })
    prediction_path = RESULTS_DIR / f"test_predictions_{method}_{horizon}h.csv"
    prediction_df.to_csv(prediction_path, index=False)

    months = target_times.month
    windows = [
        ("all_test", np.ones(len(target_times), dtype=bool)),
        ("post_monsoon_oct_nov", months.isin([10, 11])),
        ("winter_dec_feb", months.isin([12, 1, 2])),
    ]
    rows = []
    for window_name, mask in windows:
        if not mask.any():
            continue
        rows.append({
            "method": method,
            "horizon_hours": horizon,
            "period": window_name,
            "n_test": int(mask.sum()),
            "MAE": mean_absolute_error(actual[mask], predictions[mask]),
            "RMSE": np.sqrt(mean_squared_error(actual[mask], predictions[mask])),
        })

    print(f"{method}, {horizon}h: test MAE={rows[0]['MAE']:.4f}, RMSE={rows[0]['RMSE']:.4f}")
    print(f"Predictions saved to: {prediction_path}")
    return rows


all_results = []

test_target_times = pd.DatetimeIndex(test["period_start"])
test_origins_by_horizon = {
    horizon: test_target_times - pd.to_timedelta(horizon, unit="h")
    for horizon in HORIZONS
}

# Persistence is deliberately simple: repeat the most recent measurement
# available at the forecast origin.
for horizon in HORIZONS:
    target_times = test_target_times
    forecast_origins = test_origins_by_horizon[horizon]
    actual = test_actual.reindex(target_times).to_numpy(dtype=float)
    predictions = observed_series.reindex(forecast_origins).to_numpy(dtype=float)
    if np.isnan(actual).any() or np.isnan(predictions).any():
        raise ValueError(f"Missing persistence values for test horizon {horizon}.")
    all_results.extend(save_scores("persistence", horizon, target_times, actual, predictions))

# Fit and evaluate the raw-series XGBoost model. Its settings are not retuned
# against the test set; each model sees training labels from 2021–2022 only.
for horizon in HORIZONS:
    training_features = add_raw_features(train, horizon)
    training_features = training_features.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    model = build_model()
    model.fit(training_features[FEATURE_COLUMNS], training_features["target"])

    all_features = add_raw_features(all_observations, horizon)
    target_mask = all_features["target_time"].between(test_start, test_end)
    test_features = all_features.loc[target_mask].copy()
    test_features = test_features.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if len(test_features) != len(test):
        raise ValueError(
            f"XGBoost horizon {horizon}: expected {len(test)} test forecasts, "
            f"but created {len(test_features)}."
        )

    target_times = pd.DatetimeIndex(test_features["target_time"])
    actual = test_actual.reindex(target_times).to_numpy(dtype=float)
    predictions = model.predict(test_features[FEATURE_COLUMNS])
    if np.isnan(actual).any():
        raise ValueError(f"Missing XGBoost actuals for test horizon {horizon}.")
    all_results.extend(save_scores("xgboost_baseline", horizon, target_times, actual, predictions))

# Fit the STL profile on training data only, matching the primary experiment.
print("Estimating the final STL profile from training data only...")
train_profile = get_seasonal_profile(train)
for horizon in HORIZONS:
    training_features = add_stl_features(train, train_profile, horizon)
    training_features = training_features.dropna(subset=STL_FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    model = build_model()
    model.fit(training_features[STL_FEATURE_COLUMNS], training_features["target"])

    all_features = add_stl_features(all_observations, train_profile, horizon)
    target_mask = all_features["target_time"].between(test_start, test_end)
    test_features = all_features.loc[target_mask].copy()
    test_features = test_features.dropna(subset=STL_FEATURE_COLUMNS).reset_index(drop=True)
    if len(test_features) != len(test):
        raise ValueError(
            f"STL horizon {horizon}: expected {len(test)} test forecasts, "
            f"but created {len(test_features)}."
        )

    target_times = pd.DatetimeIndex(test_features["target_time"])
    actual = test_actual.reindex(target_times).to_numpy(dtype=float)
    predicted_deseasonalized = model.predict(test_features[STL_FEATURE_COLUMNS])
    seasonal_at_target = target_times.hour.map(train_profile).to_numpy(dtype=float)
    predictions = predicted_deseasonalized + seasonal_at_target
    if np.isnan(actual).any():
        raise ValueError(f"Missing STL actuals for test horizon {horizon}.")
    all_results.extend(save_scores("train_only_stl", horizon, target_times, actual, predictions))

results_df = pd.DataFrame(all_results)
results_path = RESULTS_DIR / "test_results.csv"
results_df.to_csv(results_path, index=False)

print("\nFinal 2024 test results")
print(results_df.to_string(index=False))
print(f"\nResults saved to: {results_path}")
print("\nReminder: test results are for final reporting, not further model selection.")
