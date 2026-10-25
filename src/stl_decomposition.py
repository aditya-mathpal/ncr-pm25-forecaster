"""Descriptive STL decomposition of the whole 2021-2024 series (for the EDA section).

This is a description of the data, not a forecasting step, so it is fine to use
all four years here. It produces the numbers behind the report's statements
about the diurnal cycle, the trend and the remainder, which were not computed
before.

Run from the project root:   python src/stl_decomposition.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL

from forecast_utils import RESULTS_DIR, load_series

OUT_DIR = RESULTS_DIR / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def main() -> None:
    y = load_series()
    fit = STL(y, period=24, robust=True).fit()
    parts = pd.DataFrame(
        {"observed": y, "trend": fit.trend, "seasonal": fit.seasonal, "remainder": fit.resid},
        index=y.index,
    )
    parts.to_parquet(OUT_DIR / "stl_components.parquet")

    # Size of the daily cycle: highest minus lowest seasonal value within each day
    daily = parts["seasonal"].groupby(parts.index.date)
    amplitude = (daily.max() - daily.min())
    amplitude.index = pd.to_datetime(amplitude.index)
    by_month = pd.DataFrame({
        "daily_cycle_amplitude": amplitude.groupby(amplitude.index.month).mean(),
        "trend_mean": parts["trend"].groupby(parts.index.month).mean(),
        "remainder_std": parts["remainder"].groupby(parts.index.month).std(),
    })
    by_month.index.name = "month"
    by_month.to_csv(OUT_DIR / "stl_by_month.csv")

    by_year = y.groupby(y.index.year).mean().rename("mean_pm25").to_frame()
    by_year.to_csv(OUT_DIR / "stl_yearly_means.csv")

    print("\nDaily-cycle amplitude by month (ug/m3):")
    print(by_month.round(1).to_string())
    print("\nYearly mean PM2.5:")
    print(by_year.round(1).to_string())

    window = parts.loc["2023-11-01":"2023-12-15"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, facecolor=SURFACE)
    for ax, col, title in zip(
        axes, ["observed", "trend", "seasonal", "remainder"],
        ["Observed PM2.5", "Trend", "Daily (24-hour) seasonal part", "Remainder"],
    ):
        ax.plot(window.index, window[col], color=BLUE, linewidth=1.1)
        ax.set_ylabel("µg/m³", color=INK2, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stl_decomposition.png", dpi=200, facecolor=SURFACE)
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
