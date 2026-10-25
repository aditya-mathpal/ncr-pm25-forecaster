"""Can we fix XGBoost's failure on extreme spikes?

Tree models cannot predict values higher than the ones they saw in training.
The 2024 test year contains a spike (820 ug/m3) that is above anything in
2021-2022 (maximum 744), and XGBoost predicted roughly half of it.

Two simple fixes, both using the same features and settings as raw_xgb:

  xgb_delta   predict the CHANGE from the current value (y[t+h] - y[t]),
              then add the current value back.
  xgb_log     predict log(1 + PM2.5) and transform back.

Run from the project root:   python src/delta_target_experiment.py
Takes about a minute.
"""

import pandas as pd

from forecast_utils import RESULTS_DIR, load_series, run_pipeline

OUT_DIR = RESULTS_DIR / "delta"


def main() -> None:
    y = load_series()
    results = []

    print("\n[1/2] xgb_delta (predict the change from the current value)")
    results += run_pipeline("xgb_delta", y, OUT_DIR, seasonal_at_target=lambda h: y)

    print("\n[2/2] xgb_log (predict log(1 + PM2.5))")
    results += run_pipeline("xgb_log", y, OUT_DIR, log_target=True)

    table = pd.DataFrame(results)
    table.to_csv(OUT_DIR / "delta_results.csv", index=False)
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
