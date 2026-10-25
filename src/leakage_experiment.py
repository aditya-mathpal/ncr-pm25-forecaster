"""Does look-ahead leakage in STL inflate forecast accuracy?

Every variant uses the same data, the same XGBoost settings and the same
targets. The only thing that changes is how the STL information is produced.

Existing design (replace the series by "series minus fixed hour-of-day profile"):
  raw_xgb                  no STL at all (your existing XGBoost baseline)
  stl_train_profile        profile fitted on 2021-2022 only (your existing STL variant, leak-free)
  stl_all_profile          the same profile fitted on all of 2021-2024 (mild leak)

New design (keep the raw features and ADD STL components as extra features:
the seasonal value now, the trend now, the remainder now, and the seasonal
value for the hour being forecast):
  stl_causal_14d           STL re-fitted at every hour on the previous 14 days only (no look-ahead)
  stl_causal_7d            same with a 7-day window (sensitivity check)
  stl_leaky_seasonal       STL fitted once on all 4 years, seasonal components only (look-ahead)
  stl_leaky_full           STL fitted once on all 4 years, seasonal + trend + remainder (look-ahead)
  stl_leaky_full_trend361  as above with a longer trend window (sensitivity check)

stl_causal_14d and stl_leaky_full are the matched pair: same kind of
decomposition, same features. The gap between them is the effect of looking ahead.

Run from the project root:   python src/leakage_experiment.py
Takes roughly 6-10 minutes (most of it is the rolling STL, which is cached).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.seasonal import STL

from forecast_utils import HORIZONS, RESULTS_DIR, TRAIN_END, load_series, run_pipeline

OUT_DIR = RESULTS_DIR / "leakage"
TIME_VARYING_ROBUST = False     # same setting for the rolling and the whole-series STL


def fixed_hour_profile(series: pd.Series) -> pd.Series:
    """Average STL seasonal value for each hour of the day (the existing method)."""
    result = STL(series, period=24, robust=True).fit()
    seasonal = pd.Series(result.seasonal.to_numpy(), index=series.index)
    return seasonal.groupby(series.index.hour).mean()


def full_series_stl(y: pd.Series, **kwargs) -> tuple[pd.Series, pd.Series]:
    """STL on the whole 2021-2024 series at once (this is the look-ahead step)."""
    result = STL(y, period=24, robust=TIME_VARYING_ROBUST, **kwargs).fit()
    return (
        pd.Series(result.seasonal.to_numpy(), index=y.index),
        pd.Series(result.trend.to_numpy(), index=y.index),
    )


def _causal_chunk(values: np.ndarray, start: int, end: int, window: int) -> np.ndarray:
    out = np.full((end - start, 5), np.nan)
    last = window - 1
    for i, t in enumerate(range(start, end)):
        if t < last:
            continue
        segment = values[t - window + 1 : t + 1]
        fit = STL(segment, period=24, robust=TIME_VARYING_ROBUST).fit()
        s = fit.seasonal
        # seasonal value at the forecast origin, the trend at the origin, and the
        # seasonal value for the hour being forecast, taken from the most recent
        # full cycle inside the window (so nothing after time t is used)
        out[i] = [s[-1], fit.trend[-1], s[last - 23], s[last - 18], s[last]]
    return out


def causal_stl(y: pd.Series, window: int) -> pd.DataFrame:
    """At every hour t, fit STL on the previous `window` hours only."""
    cache = OUT_DIR / f"causal_stl_cache_{window}h.parquet"
    if cache.exists():
        print(f"Using cached rolling STL: {cache}")
        return pd.read_parquet(cache)

    print(f"Fitting rolling STL on a {window}-hour trailing window (cached afterwards)...")
    values = y.to_numpy()
    edges = np.linspace(0, len(values), 41, dtype=int)
    chunks = Parallel(n_jobs=-1)(
        delayed(_causal_chunk)(values, a, b, window) for a, b in zip(edges[:-1], edges[1:])
    )
    arr = np.vstack(chunks)
    frame = pd.DataFrame(
        arr,
        index=y.index,
        columns=["seasonal_now", "trend_now", "seasonal_h1", "seasonal_h6", "seasonal_h24"],
    )
    frame.to_parquet(cache)
    return frame




def stl_feature_sets(seasonal: pd.Series, trend: pd.Series, y: pd.Series, full: bool):
    """Extra-feature function h -> columns, built from a whole-series STL fit."""
    def extra(h: int) -> dict:
        cols = {"stl_seasonal_now": seasonal, "stl_seasonal_target": seasonal.shift(-h)}
        if full:
            cols["stl_trend_now"] = trend
            cols["stl_remainder_now"] = y - seasonal - trend
        return cols
    return extra


def causal_feature_set(causal: pd.DataFrame, y: pd.Series):
    def extra(h: int) -> dict:
        return {
            "stl_seasonal_now": causal["seasonal_now"],
            "stl_seasonal_target": causal[f"seasonal_h{h}"],
            "stl_trend_now": causal["trend_now"],
            "stl_remainder_now": y - causal["seasonal_now"] - causal["trend_now"],
        }
    return extra


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y = load_series()
    hours = y.index.hour
    results = []

    print("\n[1/8] raw_xgb")
    results += run_pipeline("raw_xgb", y, OUT_DIR)

    print("\n[2/8] stl_train_profile (fixed profile, fitted on 2021-2022 only)")
    profile_train = fixed_hour_profile(y[:TRAIN_END])
    s_train = pd.Series(hours.map(profile_train).to_numpy(), index=y.index)
    results += run_pipeline(
        "stl_train_profile", y, OUT_DIR,
        seasonal_at_origin=s_train, seasonal_at_target=lambda h: s_train.shift(-h),
    )

    print("\n[3/8] stl_all_profile (fixed profile, fitted on all four years)")
    profile_all = fixed_hour_profile(y)
    s_all = pd.Series(hours.map(profile_all).to_numpy(), index=y.index)
    print(f"  largest difference between the two 24-hour profiles: "
          f"{(profile_train - profile_all).abs().max():.2f} ug/m3")
    results += run_pipeline(
        "stl_all_profile", y, OUT_DIR,
        seasonal_at_origin=s_all, seasonal_at_target=lambda h: s_all.shift(-h),
    )

    print("\n[4/8] stl_causal_14d (rolling 14-day window, no look-ahead)")
    results += run_pipeline(
        "stl_causal_14d", y, OUT_DIR, extra=causal_feature_set(causal_stl(y, 14 * 24), y)
    )

    print("\n[5/8] stl_causal_7d (rolling 7-day window, no look-ahead)")
    results += run_pipeline(
        "stl_causal_7d", y, OUT_DIR, extra=causal_feature_set(causal_stl(y, 7 * 24), y)
    )

    print("\n[6/8] stl_leaky_seasonal (STL on the whole series, seasonal part only)")
    seasonal, trend = full_series_stl(y)
    results += run_pipeline(
        "stl_leaky_seasonal", y, OUT_DIR, extra=stl_feature_sets(seasonal, trend, y, full=False)
    )

    print("\n[7/8] stl_leaky_full (STL on the whole series, seasonal + trend + remainder)")
    results += run_pipeline(
        "stl_leaky_full", y, OUT_DIR, extra=stl_feature_sets(seasonal, trend, y, full=True)
    )

    print("\n[8/8] stl_leaky_full_trend361 (sensitivity: 15-day trend window)")
    seasonal361, trend361 = full_series_stl(y, trend=361)
    results += run_pipeline(
        "stl_leaky_full_trend361", y, OUT_DIR,
        extra=stl_feature_sets(seasonal361, trend361, y, full=True),
    )

    table = pd.DataFrame(results)
    table.to_csv(OUT_DIR / "leakage_results.csv", index=False)
    pd.set_option("display.width", 200)
    for split in ("test", "val"):
        print(f"\nMAE, {split} set")
        print(table[table.split == split].pivot(
            index="variant", columns="horizon_hours", values="MAE").round(2).to_string())
    print(f"\nPredictions and tables saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
