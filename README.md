# Evaluating Look-Ahead Bias in STL-Based Hourly PM2.5 Forecasting in Delhi

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Decision Theory and Applied Statistics Minor Project*  
**School of Basic Sciences, Humanities & Management**  
**Manipal Institute of Technology, Manipal Academy of Higher Education (MAHE)**

**Authors:** Siddharth Sivakumar, Aditya Pankaj Mathpal, Raghavendra Rishi Rau  
**Faculty Guide:** Dr. Sunita Sharma

---

## Executive Summary

Fine particulate matter ($\text{PM}_{2.5}$) in Delhi regularly breaches hazardous air quality thresholds, demanding accurate short-term warnings for public health advisories and the Graded Response Action Plan (GRAP). While statistical and machine-learning hybrids—particularly **Seasonal-Trend decomposition using Loess (STL)** combined with gradient-boosted decision trees (**XGBoost**)—are widely adopted in the time-series literature, many published implementations decompose the entire dataset prior to splitting into train and test subsets. This introduces **look-ahead bias (data leakage)** by allowing centered smoothing filters to peek into future values.

This repository provides a reproducible, leak-free benchmarking framework for hourly $\text{PM}_{2.5}$ forecasting across 1-hour, 6-hour, and 24-hour prediction horizons using 4 years of continuous CAAQM ground monitoring data (2021–2024, 35,064 observations).

### Key Takeaways
1. **Look-Ahead Bias Heavily Inflates Accuracy:** Applying STL over the entire series before splitting falsely slashes test Mean Absolute Error (MAE) by **18% at 1 hour**, **55% at 6 hours**, and **27% at 24 hours** compared to a strictly causal rolling decomposition.
2. **Causal STL Yields No Tangible Advantage:** When restricted strictly to past information (via a rolling causal decomposition or training-only diurnal profiles), STL adds virtually **no meaningful improvement** over XGBoost trained directly on raw lagged features ($\pm 2\%$ MAE variance, statistically indistinguishable under Diebold–Mariano tests).
3. **Tree Models Extrapolate Poorly During Rare Extremes:** Tree ensembles cannot predict target values outside the range seen during training (training max: $744\,\mu\text{g/m}^3$; test peak: $820\,\mu\text{g/m}^3$). During the extreme November 2024 episode, standard XGBoost underpredicted the peak by half ($409\,\mu\text{g/m}^3$).
4. **The $\Delta$-Target Formulation Rescues Extreme Episodes:** Reformulating XGBoost to predict the *change* ($\Delta y_{t+h} = y_{t+h} - y_t$) and adding back $y_t$ reduces 1-hour test RMSE from **$15.83$ to $10.27\,\mu\text{g/m}^3$** (outperforming persistence at $14.45\,\mu\text{g/m}^3$) and forecasts the peak up to $756\,\mu\text{g/m}^3$.
5. **Alert Reliability at Longer Horizons:** When evaluated as an alert system for severe pollution ($>250\,\mu\text{g/m}^3$), persistence is a formidable benchmark. At a 24-hour horizon, no model achieves recall higher than $\sim 50\%$, missing roughly half of severe episode hours.

---

## Experimental Results

All models were trained strictly on **2021–2022** ($17,520$ hours), validated on **2023** ($8,760$ hours), and tested on the untouched **2024** holdout set ($8,784$ hours).

### 1. Main Benchmark (2024 Test Year)

| Model / Method | MAE 1h ($\mu\text{g/m}^3$) | MAE 6h ($\mu\text{g/m}^3$) | MAE 24h ($\mu\text{g/m}^3$) | RMSE 1h ($\mu\text{g/m}^3$) | RMSE 6h ($\mu\text{g/m}^3$) | RMSE 24h ($\mu\text{g/m}^3$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence (Naive Baseline)** | 8.85 | 35.43 | 32.68 | 14.45 | 55.82 | 55.66 |
| **XGBoost (Raw Lags)** | 6.82 | 24.25 | 30.80 | 15.83 | 42.22 | 51.72 |
| **STL Fixed Profile (Train-Only, Causal)** | 6.97 | 24.52 | 31.50 | 16.91 | 42.83 | 52.19 |
| **STL Rolling 14-Day (Causal)** | 6.96 | 23.71 | 30.72 | 14.88 | 41.24 | **50.89** |
| **XGBoost $\Delta$-Target (Change Prediction)** | **6.15** | **23.36** | 31.10 | **10.27** | **39.61** | 52.14 |

*Note: Persistence at 24 hours functions as a seasonal-naive forecast (predicting the same hour yesterday), making it substantially stronger than 6-hour persistence.*

---

### 2. STL Look-Ahead Bias Comparison (2024 Test Year)

To quantify the distortion caused by future data leakage, we compared causal decompositions against whole-series leaky decompositions under identical hyperparameters and feature structures:

| Variant | Protocol Status | MAE 1h | MAE 6h | MAE 24h |
| :--- | :--- | :---: | :---: | :---: |
| **XGBoost (Raw Lags)** | Valid / Leak-Free | 6.82 | 24.25 | 30.80 |
| **STL Fixed Profile (Train-Only)** | Valid / Leak-Free | 6.97 | 24.52 | 31.50 |
| **STL Fixed Profile (All Data)** | Mild Leak (Profile only) | 6.89 | 24.39 | 31.30 |
| **STL Rolling 14-Day** | Valid / Causal | 6.96 | 23.71 | 30.72 |
| **STL Rolling 7-Day** | Valid / Causal | 6.84 | 23.95 | 30.62 |
| **STL Whole-Series (Seasonal Only)** | **Look-Ahead Leakage** | 6.20 | 18.69 | 28.13 |
| **STL Whole-Series (Full Decomposition)** | **Look-Ahead Leakage** | **5.70** | **10.64** | **22.51** |
| **STL Whole-Series (Longer Trend)** | **Look-Ahead Leakage** | 6.17 | 18.47 | 23.12 |
| **Artificial Error Reduction (Leaky vs Rolling 14d)** | — | **-18.0%** | **-55.2%** | **-26.7%** |

*The leakage is most severe at $h = 6$, where a centered Loess trend smoother centered at $t$ incorporates future values around $t+6$, effectively handing the target level directly to the model.*

---

### 3. Severe Episode Alert Classification ($>250\,\mu\text{g/m}^3$)

Performance evaluated across the 605 hourly events exceeding the CPCB "Severe" threshold ($>250\,\mu\text{g/m}^3$) in the 2024 test year:

| Method | Recall 1h | Precision 1h | Recall 6h | Precision 6h | Recall 24h | Precision 24h |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence** | 0.89 | 0.89 | 0.54 | 0.54 | **0.55** | 0.55 |
| **XGBoost (Raw Lags)** | 0.90 | 0.94 | 0.55 | 0.69 | 0.50 | 0.58 |
| **STL Rolling 14-Day** | 0.90 | **0.94** | 0.53 | 0.70 | 0.49 | **0.60** |
| **XGBoost $\Delta$-Target** | **0.90** | 0.93 | **0.59** | **0.71** | 0.50 | 0.58 |

---

## Setup

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### API Configuration
Generate a free API key at [https://airquality.xkdr.org/signup](https://airquality.xkdr.org/signup).

Create a `.env` file in the project root directory:
```env
AQI_API_KEY=<your API key from XKDR>
```

### Download and Prepare Data
```bash
# 1. Download monthly station records for Delhi (2021-2024)
python src/download_data.py

# 2. Aggregate station-hour records to citywide mean and split datasets
python src/prepare_data.py
```

---

## Reproducing the Experiments

The pipeline is split into modular, sequential scripts. Each script can be executed independently from the project root:

```bash
# 1. Exploratory Data Analysis & full-series decomposition
python src/eda.py
python src/stl_decomposition.py

# 2. Run baselines
python src/baseline.py
python src/xgboost_baseline.py
python src/stl_experiment.py

# 3. Controlled leakage & look-ahead evaluation
python src/leakage_experiment.py

# 4. Out-of-range spike diagnostics (Delta target)
python src/delta_target_experiment.py

# 5. Final frozen test set evaluation
python src/final_evaluation.py

# 6. Generate final publication tables and figures
python src/diagnostics.py
```

All summary tables (CSV and LaTeX format) are saved to `results/tables/`, and generated figures are written to `results/figures/` and `results/eda/`.

---

## Project Structure

```text
ncr-pm25-forecaster/
├── data/                                 # Chronologically split Parquet datasets
│   ├── train.parquet                     # 2021-01-01 to 2022-12-31 (17,520 hrs)
│   ├── validation.parquet                # 2023-01-01 to 2023-12-31 (8,760 hrs)
│   └── test.parquet                      # 2024-01-01 to 2024-12-31 (8,784 hrs)
├── results/                              # Output artifacts, tables, and figures
│   ├── baseline/                         # Persistence baseline validation outputs
│   ├── delta/                            # Change-prediction experiment outputs
│   ├── eda/                              # Descriptive plots and distribution statistics
│   ├── figures/                          # Publication figures (leakage & Nov 2024 spike)
│   ├── leakage/                          # Look-ahead vs causal STL results
│   ├── tables/                           # CSV and LaTeX tables for the report
│   ├── test/                             # 2024 frozen test set predictions
│   └── xgboost/                          # Standard XGBoost validation predictions
├── src/                                  # Source code
│   ├── baseline.py                       # Persistence baseline implementation
│   ├── delta_target_experiment.py        # XGBoost change prediction (Delta target)
│   ├── diagnostics.py                    # Diebold-Mariano tests, alerts, and figure generation
│   ├── download_data.py                  # XKDR API client for CPCB station data
│   ├── eda.py                            # Seasonality, diurnal profiles, and distributions
│   ├── final_evaluation.py               # Frozen evaluation on 2024 test data
│   ├── forecast_utils.py                 # Shared features, pipeline runners, and configs
│   ├── leakage_experiment.py             # Matched causal vs leaky STL experiments
│   ├── prepare_data.py                   # Data cleaning, aggregation, and chronological split
│   ├── stl_decomposition.py              # Descriptive 4-year Loess decomposition
│   ├── stl_experiment.py                 # Train-only fixed profile STL pipeline
│   └── xgboost_baseline.py               # Direct multi-step XGBoost forecaster
├── requirements.txt                      # Python dependencies
├── LICENSE                               # MIT License
└── README.md                             # Project documentation
```

---

## Methodology Overview

### 1. Data Processing and Aggregation
- **Raw Inputs:** Hourly ground-monitoring $\text{PM}_{2.5}$ measurements gathered across 39 CAAQM stations in Delhi.
- **Aggregation:** Unweighted hourly citywide average across all reporting stations for that timestamp (minimum 6, median 36 reporting stations).
- **Temporal Integrity:** Strictly partitioned chronologically:
  - **Train:** 2021–2022 ($17,520$ hours)
  - **Validation:** 2023 ($8,760$ hours)
  - **Test:** 2024 ($8,784$ hours)

### 2. Feature Engineering
- **Autoregressive Lags:** $y_t, y_{t-1}, y_{t-2}, y_{t-3}, y_{t-6}, y_{t-12}, y_{t-24}, y_{t-48}, y_{t-72}, y_{t-168}$.
- **Rolling Statistics:** 6-hour and 24-hour rolling means ($\mu_6, \mu_{24}$), and 24-hour rolling standard deviation ($\sigma_{24}$).
- **Calendar & Harmonic Encodings:** Hour of day, day of week, month, and circular sine/cosine representations ($\sin(2\pi \cdot \text{hour}/24)$, $\cos(2\pi \cdot \text{hour}/24)$, etc.).

### 3. Decomposition Schemes
- **Fixed Profile (Train Only):** 24-hour diurnal profile extracted solely from 2021–2022 data.
- **Rolling STL (Causal):** Refitted at each origin $t$ over the trailing 336 hours (14 days), extracting trend $T_t$, seasonal $S_t$, remainder $R_t$, and target seasonal index $S_{t+h}$ without accessing any observation $> t$.
- **Whole-Series STL (Leaky):** STL fitted globally across all four years before train/test splitting, exposing future information through bidirectional Loess smoothing.

### 4. Hyperparameter Settings (XGBoost)
All gradient boosted models use standardized hyperparameters across all horizons and experiments:
- `n_estimators`: 500
- `max_depth`: 7
- `learning_rate`: 0.05
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `objective`: `reg:squarederror`
- `random_state`: 42

---

## Data Attribution

This project uses air quality data provided by the **Central Pollution Control Board (CPCB)** and the **United States Department of State**, accessed through the **XKDR Forum**.

Any publication resulting from this project will credit **CPCB, the US Department of State, and XKDR Forum** as the data sources/providers.

---

## Acknowledgements

We express our gratitude to **Dr. Sunita Sharma** (School of Basic Sciences, Humanities & Management, Manipal Institute of Technology, MAHE) for her supervision, valuable feedback, and guidance throughout this project.
