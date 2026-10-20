from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL
from xgboost import XGBRegressor


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "stl"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

train = pd.read_parquet(
    DATA_DIR / "train.parquet"
)

validation = pd.read_parquet(
    DATA_DIR / "validation.parquet"
)

for df in (train, validation):
    df["period_start"] = pd.to_datetime(
        df["period_start"]
    )

train = (
    train
    .sort_values("period_start")
    .reset_index(drop=True)
)

validation = (
    validation
    .sort_values("period_start")
    .reset_index(drop=True)
)


# --------------------------------------------------
# Basic data checks
# --------------------------------------------------

REQUIRED_COLUMNS = {
    "period_start",
    "city_pm25",
}


for name, df in [
    ("train", train),
    ("validation", validation),
]:
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df["period_start"].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate timestamps."
        )

    if df["city_pm25"].isna().any():
        raise ValueError(
            f"{name} contains missing city_pm25 values."
        )

    if not df["period_start"].is_monotonic_increasing:
        raise ValueError(
            f"{name} timestamps are not sorted."
        )


for name, df in [
    ("train", train),
    ("validation", validation),
]:
    timestamp_differences = (
        df["period_start"]
        .diff()
        .dropna()
    )

    if not (timestamp_differences == pd.Timedelta(hours=1)).all():
        raise ValueError(
            f"{name} does not have a continuous hourly index."
        )


if train["period_start"].max() >= validation["period_start"].min():
    raise ValueError(
        "Training and validation periods overlap or are out of order."
    )


# --------------------------------------------------
# XGBoost configuration
# --------------------------------------------------

LAGS = [
    1,
    2,
    3,
    6,
    12,
    24,
    48,
    72,
    168,
]

FEATURE_COLUMNS = [
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


HORIZONS = [1, 6, 24]


def build_model() -> XGBRegressor:
    """
    Build the XGBoost model used for the STL experiment.
    """

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


# --------------------------------------------------
# STL seasonal profile
# --------------------------------------------------

def get_seasonal_profile(
    series_df: pd.DataFrame,
) -> dict[int, float]:
    """
    Estimate an hour-of-day seasonal profile using STL.

    The hourly STL seasonal component is averaged by
    hour of day to obtain a reusable 24-hour profile.

    For the train-only condition, this function receives
    only the training data.

    For the deliberately leaky condition, it receives
    both training and validation data.
    """

    series = (
        series_df
        .set_index("period_start")["city_pm25"]
        .sort_index()
        .asfreq("h")
    )

    if series.isna().any():
        raise ValueError(
            "STL input contains missing values after "
            "conversion to an hourly frequency."
        )

    stl = STL(
        series,
        period=24,
        robust=True,
    )

    result = stl.fit()

    seasonal = result.seasonal

    profile = (
        pd.DataFrame({
            "hour": seasonal.index.hour,
            "seasonal": seasonal.to_numpy(),
        })
        .groupby("hour")["seasonal"]
        .mean()
        .to_dict()
    )

    expected_hours = set(range(24))

    if set(profile.keys()) != expected_hours:
        raise ValueError(
            "The STL profile does not contain all 24 hours."
        )

    return profile


# --------------------------------------------------
# Create forecasting features
# --------------------------------------------------

def create_features(
    df: pd.DataFrame,
    seasonal_profile: dict[int, float],
    horizon: int,
) -> pd.DataFrame:
    """
    Construct features for an H-hour-ahead forecast.

    Each row represents the timestamp of the target
    observation. For a forecast horizon H, all PM2.5
    features use observations no later than H hours
    before that target timestamp.

    For example, when H = 1, a prediction for time t
    can use observations through t-1.

    The target is the deseasonalized value at the row's
    own timestamp. Predictions are evaluated against
    raw PM2.5 at that same timestamp after adding back
    the seasonal profile for the target hour.
    """

    if horizon < 1:
        raise ValueError(
            "Forecast horizon must be at least one hour."
        )

    data = df.copy()

    # ----------------------------------------------
    # Remove the estimated hour-of-day seasonal effect
    # ----------------------------------------------

    data["seasonal"] = (
        data["period_start"]
        .dt.hour
        .map(seasonal_profile)
    )

    if data["seasonal"].isna().any():
        raise ValueError(
            "Could not assign a seasonal value to every row."
        )

    data["deseasonalized"] = (
        data["city_pm25"] - data["seasonal"]
    )

    # ----------------------------------------------
    # Lag features
    #
    # The target is at timestamp t.
    # The most recent usable observation is t - horizon.
    #
    # lag_1: observation at t - horizon
    # lag_2: observation at t - horizon - 1
    # and so on.
    # ----------------------------------------------

    for lag in LAGS:
        data[f"lag_{lag}"] = (
            data["deseasonalized"]
            .shift(horizon + lag - 1)
        )

    # ----------------------------------------------
    # Rolling features
    #
    # Shift by the forecast horizon first so that
    # the rolling windows contain only observations
    # available by the forecast cutoff.
    # ----------------------------------------------

    available_history = (
        data["deseasonalized"]
        .shift(horizon)
    )

    data["rolling_mean_6"] = (
        available_history
        .rolling(window=6)
        .mean()
    )

    data["rolling_mean_24"] = (
        available_history
        .rolling(window=24)
        .mean()
    )

    data["rolling_std_24"] = (
        available_history
        .rolling(window=24)
        .std()
    )

    # ----------------------------------------------
    # Calendar features for the TARGET timestamp
    # ----------------------------------------------

    data["hour"] = data["period_start"].dt.hour
    data["day_of_week"] = data["period_start"].dt.dayofweek
    data["month"] = data["period_start"].dt.month

    # Hour-of-day cyclical encoding
    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    # Month cyclical encoding
    data["month_sin"] = np.sin(
        2 * np.pi * (data["month"] - 1) / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * (data["month"] - 1) / 12
    )

    # ----------------------------------------------
    # Target
    #
    # The target is the deseasonalized PM2.5 value
    # at the row's timestamp. Its features only use
    # observations available H hours beforehand.
    # ----------------------------------------------

    data["target"] = data["deseasonalized"]

    return data


# --------------------------------------------------
# Create STL profiles
# --------------------------------------------------

print("Computing train-only STL profile...")

train_only_profile = get_seasonal_profile(
    train
)

print("Computing validation-leaky STL profile...")

full_data = (
    pd.concat(
        [train, validation],
        ignore_index=True,
    )
    .sort_values("period_start")
    .reset_index(drop=True)
)

validation_leaky_profile = get_seasonal_profile(
    full_data
)


# --------------------------------------------------
# Run experiments
# --------------------------------------------------

results = []

experiments = [
    ("train_only_stl", train_only_profile),
    ("validation_leaky_stl", validation_leaky_profile),
]


for method, seasonal_profile in experiments:

    print(
        f"\n{'=' * 60}\n"
        f"{method.upper()}\n"
        f"{'=' * 60}"
    )

    for horizon in HORIZONS:

        print(
            f"\nHorizon: {horizon} hours"
        )

        # ------------------------------------------
        # Training features
        # ------------------------------------------

        train_features = create_features(
            train,
            seasonal_profile,
            horizon,
        )

        train_features = (
            train_features
            .dropna(
                subset=FEATURE_COLUMNS + ["target"]
            )
            .reset_index(drop=True)
        )

        if train_features.empty:
            raise ValueError(
                "No usable training rows remain for "
                f"horizon {horizon}."
            )

        X_train = train_features[FEATURE_COLUMNS]
        y_train = train_features["target"]

        # ------------------------------------------
        # Validation history
        #
        # Retain enough training observations to
        # construct the longest lag for this horizon.
        # ------------------------------------------

        history_length = max(LAGS) + horizon - 1

        history = train.tail(history_length)

        combined = (
            pd.concat(
                [history, validation],
                ignore_index=True,
            )
            .sort_values("period_start")
            .reset_index(drop=True)
        )

        # ------------------------------------------
        # Validation features
        # ------------------------------------------

        validation_features = create_features(
            combined,
            seasonal_profile,
            horizon,
        )

        validation_start = validation["period_start"].min()
        validation_end = validation["period_start"].max()

        # Keep rows whose TARGET timestamps are in
        # the validation period.
        validation_features = (
            validation_features[
                (
                    validation_features["period_start"]
                    >= validation_start
                )
                &
                (
                    validation_features["period_start"]
                    <= validation_end
                )
            ]
            .dropna(
                subset=FEATURE_COLUMNS + ["target"]
            )
            .reset_index(drop=True)
        )

        if validation_features.empty:
            raise ValueError(
                "No usable validation rows remain for "
                f"horizon {horizon}."
            )

        X_val = validation_features[FEATURE_COLUMNS]

        # ------------------------------------------
        # Train model
        # ------------------------------------------

        model = build_model()

        model.fit(
            X_train,
            y_train,
        )

        # ------------------------------------------
        # Predict deseasonalized PM2.5
        # ------------------------------------------

        predicted_deseasonalized = model.predict(X_val)

        # ------------------------------------------
        # Restore seasonal component at the TARGET
        # timestamp.
        #
        # Validation rows are indexed by their target
        # timestamps, so no additional time shift is
        # applied here.
        # ------------------------------------------

        seasonal_future = (
            validation_features["period_start"]
            .dt.hour
            .map(seasonal_profile)
            .to_numpy()
        )

        predictions = (
            predicted_deseasonalized
            + seasonal_future
        )

        # ------------------------------------------
        # Actual raw PM2.5 at the SAME target times
        # ------------------------------------------

        actual = (
            validation_features["city_pm25"]
            .to_numpy()
        )

        target_timestamps = validation_features[
            "period_start"
        ]

        assert len(actual) == len(predictions), (
            "Prediction and actual lengths do not match."
        )

        assert not np.isnan(predictions).any(), (
            "Predictions contain NaN values."
        )

        assert not np.isnan(actual).any(), (
            "Actual values contain NaN values."
        )

        # ------------------------------------------
        # Metrics
        # ------------------------------------------

        mae = mean_absolute_error(
            actual,
            predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predictions,
            )
        )

        results.append({
            "method": method,
            "horizon_hours": horizon,
            "n_train": len(train_features),
            "n_validation": len(validation_features),
            "validation_start": target_timestamps.min(),
            "validation_end": target_timestamps.max(),
            "MAE": mae,
            "RMSE": rmse,
        })

        print(
            f"Training rows:   {len(train_features)}"
        )

        print(
            f"Validation rows: {len(validation_features)}"
        )

        print(
            f"MAE:  {mae:.4f}"
        )

        print(
            f"RMSE: {rmse:.4f}"
        )


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_path = RESULTS_DIR / "validation_results.csv"

results_df.to_csv(
    output_path,
    index=False,
)

print(
    "\n" + "=" * 60
)

print(
    "STL VALIDATION RESULTS"
)

print(
    "=" * 60
)

print(
    results_df[
        [
            "method",
            "horizon_hours",
            "n_train",
            "n_validation",
            "MAE",
            "RMSE",
        ]
    ].to_string(index=False)
)

print(
    f"\nResults saved to: {output_path}"
)