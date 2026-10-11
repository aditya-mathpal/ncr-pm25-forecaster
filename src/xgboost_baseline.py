from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "xgboost"

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

train["period_start"] = pd.to_datetime(train["period_start"])
validation["period_start"] = pd.to_datetime(
    validation["period_start"]
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
# Feature configuration
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
    168
]


# --------------------------------------------------
# Feature creation
# --------------------------------------------------

def add_features(
    df: pd.DataFrame,
    horizon: int
) -> pd.DataFrame:

    data = df.copy()

    # Lag features
    for lag in LAGS:
        data[f"lag_{lag}"] = (
            data["city_pm25"].shift(lag)
        )

    # Rolling statistics.
    # shift(1) guarantees that the current
    # observation is not included.
    data["rolling_mean_6"] = (
        data["city_pm25"]
        .shift(1)
        .rolling(6)
        .mean()
    )

    data["rolling_mean_24"] = (
        data["city_pm25"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    data["rolling_std_24"] = (
        data["city_pm25"]
        .shift(1)
        .rolling(24)
        .std()
    )

    # Calendar features
    data["hour"] = data["period_start"].dt.hour
    data["day_of_week"] = data["period_start"].dt.dayofweek
    data["month"] = data["period_start"].dt.month

    # Cyclical encodings
    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    # Future target
    data["target"] = (
        data["city_pm25"].shift(-horizon)
    )

    return data


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


# --------------------------------------------------
# Model
# --------------------------------------------------

def build_model() -> XGBRegressor:

    return XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )


# --------------------------------------------------
# Train + validation
# --------------------------------------------------

results = []

for horizon in [1, 6, 24]:

    print(
        f"\n{'=' * 60}\n"
        f"XGBOOST — {horizon}-HOUR HORIZON\n"
        f"{'=' * 60}"
    )

    # --------------------------------------------------
    # Training features
    # --------------------------------------------------

    train_features = add_features(
        train,
        horizon
    )

    train_features = train_features.dropna(
        subset=FEATURE_COLUMNS + ["target"]
    )

    X_train = train_features[
        FEATURE_COLUMNS
    ]

    y_train = train_features[
        "target"
    ]

    # --------------------------------------------------
    # Validation features
    #
    # Include the end of training history so that
    # the first validation observations have valid lags.
    # --------------------------------------------------

    history = train.tail(max(LAGS)).copy()

    combined = pd.concat(
        [history, validation],
        ignore_index=True
    )

    combined_features = add_features(
        combined,
        horizon
    )

    # Keep only rows whose prediction target
    # belongs to the validation period.
    validation_start = validation[
        "period_start"
    ].min()

    validation_end = validation[
        "period_start"
    ].max()

    validation_features = combined_features[
        (
            combined_features["period_start"]
            >= validation_start
        )
        &
        (
            combined_features["period_start"]
            <= validation_end
        )
    ].copy()

    validation_features = validation_features.dropna(
        subset=FEATURE_COLUMNS + ["target"]
    )

    X_val = validation_features[
        FEATURE_COLUMNS
    ]

    y_val = validation_features[
        "target"
    ]

    print(
        f"Training rows:   {len(X_train):,}"
    )

    print(
        f"Validation rows: {len(X_val):,}"
    )

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------

    model = build_model()

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------

    predictions = model.predict(
        X_val
    )

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_val,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            predictions
        )
    )

    results.append({
        "horizon_hours": horizon,
        "MAE": mae,
        "RMSE": rmse
    })

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

results_df.to_csv(
    RESULTS_DIR / "validation_results.csv",
    index=False
)


# --------------------------------------------------
# Final output
# --------------------------------------------------

print("\n" + "=" * 60)
print("XGBOOST VALIDATION RESULTS")
print("=" * 60)

print(
    results_df.to_string(index=False)
)