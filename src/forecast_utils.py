"""Shared helpers for the extension experiments.

Used by leakage_experiment.py, delta_target_experiment.py and diagnostics.py.
The feature set, XGBoost settings and chronological split are identical to the
ones in xgboost_baseline.py / stl_experiment.py / final_evaluation.py, so the
new variants are directly comparable with the results you already have.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
HORIZONS = [1, 6, 24]

TRAIN_END = pd.Timestamp("2022-12-31 23:00")
SPLITS = {
    "val": (pd.Timestamp("2023-01-01 00:00"), pd.Timestamp("2023-12-31 23:00")),
    "test": (pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-12-31 23:00")),
}
EXPECTED_N = {"val": 8760, "test": 8784}


def load_series() -> pd.Series:
    """Hourly city-level PM2.5 for 2021-2024 as one continuous series."""
    parts = []
    for name in ("train", "validation", "test"):
        df = pd.read_parquet(DATA_DIR / f"{name}.parquet")
        df["period_start"] = pd.to_datetime(df["period_start"])
        parts.append(df)
    full = pd.concat(parts).sort_values("period_start")
    if full["period_start"].duplicated().any():
        raise ValueError("Duplicate timestamps in the data splits.")
    y = full.set_index("period_start")["city_pm25"].asfreq("h")
    if y.isna().any() or len(y) != 35064:
        raise ValueError("Expected a complete hourly series of 35,064 values.")
    return y


def build_model() -> XGBRegressor:
    # Same settings as xgboost_baseline.py and final_evaluation.py.
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


def build_features(v: pd.Series, extra: dict | None = None) -> pd.DataFrame:
    """Features known at the forecast origin t, built from series v.

    v is either the raw PM2.5 series or a de-seasonalised version of it.
    `extra` adds further columns (for example a trend estimate) aligned to v.
    """
    d = pd.DataFrame(index=v.index)
    d["current"] = v
    for lag in LAGS:
        d[f"lag_{lag}"] = v.shift(lag)
    d["rolling_mean_6"] = v.rolling(6).mean()
    d["rolling_mean_24"] = v.rolling(24).mean()
    d["rolling_std_24"] = v.rolling(24).std()
    hour = v.index.hour
    month = v.index.month
    d["hour"] = hour
    d["day_of_week"] = v.index.dayofweek
    d["month"] = month
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    d["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    d["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    for name, series in (extra or {}).items():
        d[name] = series.reindex(v.index).to_numpy()
    return d


def run_pipeline(
    name: str,
    y: pd.Series,
    out_dir: Path,
    seasonal_at_origin: pd.Series | None = None,
    seasonal_at_target=None,
    extra=None,
    log_target: bool = False,
) -> list[dict]:
    """Train on 2021-2022, predict 2023 (val) and 2024 (test), save predictions.

    seasonal_at_origin : series subtracted from y before building lag features
                         (value for time t, as known at time t).
    seasonal_at_target : function h -> series indexed by forecast origin t, giving
                         the amount subtracted from y[t+h] to form the training
                         target; the same amount is added back to the prediction.
                         For a plain model this is None (zero). For a
                         "predict the change" model it is y itself.
    extra              : extra feature columns, as a dict {name: series aligned to
                         the origin t}, or a function h -> such a dict when the
                         columns depend on the horizon.
    log_target         : train on log1p(y[t+h]) instead of y[t+h].
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    v = y if seasonal_at_origin is None else y - seasonal_at_origin
    rows, importances = [], []

    for h in HORIZONS:
        h_extra = extra(h) if callable(extra) else extra
        d = build_features(v, h_extra)
        feature_cols = [c for c in d.columns]
        offset = (
            pd.Series(0.0, index=y.index)
            if seasonal_at_target is None
            else seasonal_at_target(h)
        )
        future = y.shift(-h)
        d["target"] = np.log1p(future) if log_target else future - offset
        d["target_time"] = d.index + pd.Timedelta(hours=h)

        train_rows = d[d["target_time"] <= TRAIN_END].dropna(subset=feature_cols + ["target"])
        model = build_model()
        model.fit(train_rows[feature_cols], train_rows["target"])
        importances.append(
            pd.DataFrame(
                {
                    "variant": name,
                    "horizon_hours": h,
                    "feature": feature_cols,
                    "importance": model.feature_importances_,
                }
            )
        )

        for split, (start, end) in SPLITS.items():
            ev = d[(d["target_time"] >= start) & (d["target_time"] <= end)]
            ev = ev.dropna(subset=feature_cols)
            if len(ev) != EXPECTED_N[split]:
                raise ValueError(
                    f"{name} h={h} {split}: expected {EXPECTED_N[split]} forecasts, got {len(ev)}"
                )
            raw_pred = model.predict(ev[feature_cols])
            if log_target:
                pred = np.expm1(raw_pred)
            else:
                pred = raw_pred + offset.reindex(ev.index).to_numpy()
            target_times = pd.DatetimeIndex(ev["target_time"])
            actual = y.reindex(target_times).to_numpy()

            pd.DataFrame(
                {
                    "target_time": target_times,
                    "forecast_origin": ev.index,
                    "horizon_hours": h,
                    "actual": actual,
                    "prediction": pred,
                    "method": name,
                }
            ).to_csv(out_dir / f"{split}_{name}_{h}h.csv", index=False)

            rows.append(
                {
                    "variant": name,
                    "horizon_hours": h,
                    "split": split,
                    "MAE": float(np.mean(np.abs(actual - pred))),
                    "RMSE": float(np.sqrt(np.mean((actual - pred) ** 2))),
                }
            )
            print(
                f"  {name:<26} h={h:>2} {split:<4} "
                f"MAE={rows[-1]['MAE']:.3f}  RMSE={rows[-1]['RMSE']:.3f}",
                flush=True,
            )

    pd.concat(importances).to_csv(out_dir / f"feature_importance_{name}.csv", index=False)
    return rows
