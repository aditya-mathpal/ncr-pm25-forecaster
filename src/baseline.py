from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load validation data
# --------------------------------------------------

val = pd.read_parquet(
    DATA_DIR / "validation.parquet"
)

val = (
    val
    .sort_values("period_start")
    .reset_index(drop=True)
)


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_persistence(
    df: pd.DataFrame,
    horizon: int
) -> tuple[float, float]:

    actual = df["city_pm25"].iloc[horizon:].to_numpy()
    predicted = df["city_pm25"].iloc[:-horizon].to_numpy()

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    return mae, rmse


# --------------------------------------------------
# Evaluate horizons
# --------------------------------------------------

results = []

for horizon in [1, 6, 24]:

    mae, rmse = evaluate_persistence(
        val,
        horizon
    )

    results.append({
        "horizon_hours": horizon,
        "MAE": mae,
        "RMSE": rmse
    })


results_df = pd.DataFrame(results)


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df.to_csv(
    RESULTS_DIR / "validation_results.csv",
    index=False
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("=" * 60)
print("PERSISTENCE BASELINE — VALIDATION")
print("=" * 60)

print(
    results_df.to_string(index=False)
)