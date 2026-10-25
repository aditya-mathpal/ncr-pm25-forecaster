"""Turn the saved predictions into the tables and figures used in the report.

Reads the prediction files written by baseline.py, final_evaluation.py,
leakage_experiment.py and delta_target_experiment.py, so run those first.

Writes to results/tables (CSV and LaTeX) and results/figures (PNG).

Run from the project root:   python src/diagnostics.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_utils import HORIZONS, RESULTS_DIR

TABLES = RESULTS_DIR / "tables"
FIGURES = RESULTS_DIR / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Where each method's predictions live
# ----------------------------------------------------------------------------

LABELS = {
    "persistence": "Persistence",
    "raw_xgb": "XGBoost (raw lags)",
    "stl_train_profile": "STL fixed profile, train only",
    "stl_all_profile": "STL fixed profile, all data",
    "stl_causal_14d": "STL rolling 14-day (no look-ahead)",
    "stl_causal_7d": "STL rolling 7-day (no look-ahead)",
    "stl_leaky_seasonal": "STL whole series, seasonal only (look-ahead)",
    "stl_leaky_full": "STL whole series, full (look-ahead)",
    "stl_leaky_full_trend361": "STL whole series, longer trend (look-ahead)",
    "xgb_delta": "XGBoost predicting the change",
    "xgb_log": "XGBoost on log scale",
}


def prediction_path(method: str, split: str, h: int) -> Path:
    if method == "persistence":
        if split == "val":
            return RESULTS_DIR / "baseline" / f"validation_predictions_{h}h.csv"
        return RESULTS_DIR / "test" / f"test_predictions_persistence_{h}h.csv"
    folder = "delta" if method.startswith("xgb_") else "leakage"
    return RESULTS_DIR / folder / f"{split}_{method}_{h}h.csv"


SHORT = {
    "persistence": "Persistence", "raw_xgb": "XGB raw", "stl_train_profile": "STL fixed",
    "stl_causal_14d": "STL rolling", "stl_leaky_full": "STL look-ahead", "xgb_delta": "XGB change",
}

PRED = {}


def load_all() -> None:
    for method in LABELS:
        for split in ("val", "test"):
            for h in HORIZONS:
                path = prediction_path(method, split, h)
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing {path}. Run baseline.py, final_evaluation.py, "
                        "leakage_experiment.py and delta_target_experiment.py first."
                    )
                PRED[(method, split, h)] = pd.read_csv(path, parse_dates=["target_time"]).set_index("target_time")
    for split in ("val", "test"):
        for h in HORIZONS:
            ref = PRED[("persistence", split, h)]
            for method in LABELS:
                df = PRED[(method, split, h)]
                if not df.index.equals(ref.index) or not np.allclose(df["actual"], ref["actual"]):
                    raise ValueError(f"{method} {split} {h}h is not on the same targets as persistence.")


def errors(method: str, split: str, h: int) -> np.ndarray:
    df = PRED[(method, split, h)]
    return (df["actual"] - df["prediction"]).to_numpy()


def mae(method, split, h):
    return float(np.mean(np.abs(errors(method, split, h))))


def rmse(method, split, h):
    return float(np.sqrt(np.mean(errors(method, split, h) ** 2)))


# ----------------------------------------------------------------------------
# Small helpers: LaTeX output and the Diebold-Mariano test
# ----------------------------------------------------------------------------

def tex_table(df: pd.DataFrame, path: Path, bold_min_cols=None, fmt="{:.2f}") -> None:
    """Write a plain tabular (no caption) so the report can wrap it as it likes."""
    cols = list(df.columns)
    lines = ["\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", "\\hline",
             " & ".join(cols) + " \\\\", "\\hline"]
    bold_min_cols = bold_min_cols or []
    minima = {c: df[c].min() for c in bold_min_cols}
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            value = row[c]
            if isinstance(value, (float, np.floating)):
                text = fmt.format(value)
                if c in minima and value == minima[c]:
                    text = f"\\textbf{{{text}}}"
            else:
                text = str(value)
            cells.append(text.replace("%", "\\%").replace("_", "\\_"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def p_text(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = 48):
    """Two-sided DM test with a Newey-West variance. Negative mean diff: A is better."""
    d = loss_a - loss_b
    n = len(d)
    dc = d - d.mean()
    var = np.mean(dc * dc)
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * np.mean(dc[k:] * dc[:-k])
    z = d.mean() / math.sqrt(var / n)
    p = math.erfc(abs(z) / math.sqrt(2))
    return float(d.mean()), float(z), float(p)


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

MAIN_METHODS = ["persistence", "raw_xgb", "stl_train_profile", "stl_causal_14d", "xgb_delta"]
LEAK_METHODS = ["raw_xgb", "stl_train_profile", "stl_all_profile", "stl_causal_14d", "stl_causal_7d",
                "stl_leaky_seasonal", "stl_leaky_full", "stl_leaky_full_trend361"]


def main_tables() -> None:
    for split in ("test", "val"):
        rows = []
        for m in MAIN_METHODS:
            row = {"Method": LABELS[m]}
            for h in HORIZONS:
                row[f"MAE {h}h"] = mae(m, split, h)
            for h in HORIZONS:
                row[f"RMSE {h}h"] = rmse(m, split, h)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(TABLES / f"main_results_{split}.csv", index=False)
        tex_table(df, TABLES / f"main_results_{split}.tex",
                  bold_min_cols=[c for c in df.columns if c != "Method"])


def leakage_tables() -> None:
    for split in ("test", "val"):
        rows = [{"Variant": LABELS[m], **{f"MAE {h}h": mae(m, split, h) for h in HORIZONS}}
                for m in LEAK_METHODS]
        df = pd.DataFrame(rows)
        # how much lower is the leaky error than the matched leak-free one?
        gap = {"Variant": "Look-ahead effect: leaky vs rolling 14-day (%)"}
        for h in HORIZONS:
            gap[f"MAE {h}h"] = 100 * (mae("stl_leaky_full", split, h) / mae("stl_causal_14d", split, h) - 1)
        df_csv = pd.concat([df, pd.DataFrame([gap])], ignore_index=True)
        df_csv.to_csv(TABLES / f"leakage_{split}.csv", index=False)
        tex_table(df, TABLES / f"leakage_{split}.tex")
        # the effect row goes in its own tiny table so bold/format stay simple
        pd.DataFrame([gap]).to_csv(TABLES / f"leakage_effect_{split}.csv", index=False)


def significance_table() -> None:
    pairs = [
        ("raw_xgb", "persistence"),
        ("stl_train_profile", "raw_xgb"),
        ("stl_causal_14d", "raw_xgb"),
        ("stl_leaky_full", "stl_causal_14d"),
        ("xgb_delta", "raw_xgb"),
        ("xgb_delta", "persistence"),
    ]
    rows = []
    for split in ("test", "val"):
        for a, b in pairs:
            for h in HORIZONS:
                ea, eb = errors(a, split, h), errors(b, split, h)
                diff_abs, z_abs, p_abs = diebold_mariano(np.abs(ea), np.abs(eb))
                diff_sq, z_sq, p_sq = diebold_mariano(ea ** 2, eb ** 2)
                rows.append({
                    "split": split, "A": a, "B": b, "horizon": h,
                    "mean_abs_err_A_minus_B": diff_abs, "p_abs": p_abs,
                    "mean_sq_err_A_minus_B": diff_sq, "p_sq": p_sq,
                })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "significance_all.csv", index=False)

    for split in ("test", "val"):
        sub = df[df.split == split]
        out = pd.DataFrame({
            "Comparison (A vs B)": [f"{SHORT[a]} vs {SHORT[b]}" for a, b in zip(sub.A, sub.B)],
            "h": sub.horizon.astype(str),
            "MAE gap": sub.mean_abs_err_A_minus_B.to_numpy(),
            "p (abs)": [p_text(p) for p in sub.p_abs],
            "p (sq)": [p_text(p) for p in sub.p_sq],
        })
        tex_table(out, TABLES / f"significance_{split}.tex")


ALERT_COSTS = {"missed": 5.0, "false": 1.0}   # illustrative: a miss is 5x worse than a false alarm
ALERT_METHODS = ["persistence", "raw_xgb", "stl_causal_14d", "xgb_delta"]


def alert_tables() -> None:
    rows = []
    for threshold in (120, 250):
        for h in HORIZONS:
            for m in ALERT_METHODS:
                df = PRED[(m, "test", h)]
                event = df["actual"] > threshold
                alarm = df["prediction"] > threshold
                tp = int((event & alarm).sum()); fn = int((event & ~alarm).sum())
                fp = int((~event & alarm).sum())
                rows.append({
                    "threshold": threshold, "h": h, "method": m,
                    "events": int(event.sum()),
                    "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
                    "missed": fn, "false_alarms": fp,
                    "illustrative_cost": ALERT_COSTS["missed"] * fn + ALERT_COSTS["false"] * fp,
                })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "alerts_all.csv", index=False)
    for threshold in (120, 250):
        sub = df[df.threshold == threshold]
        wide = pd.DataFrame({"Method": [LABELS[m] for m in ALERT_METHODS]})
        for h in HORIZONS:
            s = sub[sub.h == h].set_index("method").loc[ALERT_METHODS]
            wide[f"Recall {h}h"] = s["recall"].to_numpy()
            wide[f"Precision {h}h"] = s["precision"].to_numpy()
        tex_table(wide, TABLES / f"alerts_{threshold}.tex")


def period_table() -> None:
    windows = {
        "Whole year": lambda t: np.ones(len(t), dtype=bool),
        "Oct-Nov": lambda t: t.month.isin([10, 11]),
        "Dec-Feb": lambda t: t.month.isin([12, 1, 2]),
        "Mar-Sep": lambda t: t.month.isin(range(3, 10)),
    }
    rows = []
    for m in ["persistence", "raw_xgb", "stl_causal_14d", "xgb_delta"]:
        for h in HORIZONS:
            df = PRED[(m, "test", h)]
            row = {"Method": LABELS[m], "h": str(h)}
            for name, fn in windows.items():
                mask = np.asarray(fn(df.index))
                row[name] = float(np.mean(np.abs(df["actual"][mask] - df["prediction"][mask])))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "mae_by_period_test.csv", index=False)
    tex_table(df, TABLES / "mae_by_period_test.tex")


def importance_table() -> None:
    imp = pd.read_csv(RESULTS_DIR / "leakage" / "feature_importance_raw_xgb.csv")
    out = []
    for h in HORIZONS:
        sub = imp[imp.horizon_hours == h].sort_values("importance", ascending=False).head(5)
        out.append({"h": h, "top5": ", ".join(f"{f} ({v:.2f})" for f, v in zip(sub.feature, sub.importance))})
    pd.DataFrame(out).to_csv(TABLES / "feature_importance_top5.csv", index=False)


def sanity_check_against_existing() -> None:
    """The new raw XGBoost run should reproduce final_evaluation.py (up to tiny numerical noise)."""
    path = RESULTS_DIR / "test" / "test_results.csv"
    if not path.exists():
        return
    old = pd.read_csv(path)
    old = old[(old.method == "xgboost_baseline") & (old.period == "all_test")].set_index("horizon_hours")
    for h in HORIZONS:
        diff = abs(old.loc[h, "MAE"] - mae("raw_xgb", "test", h))
        flag = "ok" if diff < 0.05 else "CHECK"
        print(f"  sanity: raw XGBoost {h}h test MAE {mae('raw_xgb', 'test', h):.3f} vs final_evaluation.py "
              f"{old.loc[h, 'MAE']:.3f}  [{flag}]")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, AQUA, ORANGE, VIOLET, GREY = "#2a78d6", "#1baf7a", "#eb6834", "#6250d6", "#8a8984"


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def leakage_figure() -> None:
    methods = [("raw_xgb", "No STL (raw XGBoost)", BLUE),
               ("stl_causal_14d", "STL, no look-ahead", AQUA),
               ("stl_leaky_full", "STL, with look-ahead", ORANGE)]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    width = 0.26
    x = np.arange(len(HORIZONS))
    for i, (m, label, color) in enumerate(methods):
        vals = [mae(m, "test", h) for h in HORIZONS]
        bars = ax.bar(x + (i - 1) * (width + 0.02), vals, width, color=color, label=label,
                      edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=10, color=INK)
    ax.set_xticks(x, [f"{h}-hour forecast" for h in HORIZONS])
    ax.set_ylabel("Mean absolute error, 2024 test year (µg/m³)", color=INK2)
    ax.set_title("Looking ahead inside the STL step makes forecasts look far better", color=INK,
                 fontsize=13, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "leakage_comparison.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def spike_figure() -> None:
    start, end = pd.Timestamp("2024-11-16"), pd.Timestamp("2024-11-20")
    series = [("actual", PRED[("persistence", "test", 1)]["actual"], INK, 2.4, "Actual"),
              ("persistence", PRED[("persistence", "test", 1)]["prediction"], GREY, 1.4, "Persistence"),
              ("raw_xgb", PRED[("raw_xgb", "test", 1)]["prediction"], BLUE, 1.8, "XGBoost (raw)"),
              ("xgb_delta", PRED[("xgb_delta", "test", 1)]["prediction"], ORANGE, 1.8, "XGBoost predicting the change")]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    for _, s, color, lw, label in series:
        s = s[(s.index >= start) & (s.index < end)]
        ax.plot(s.index, s.values, color=color, linewidth=lw, label=label)
    ax.axhline(744, color=GREY, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(end, 752, "Highest value seen in training (744)", color=INK2, fontsize=9, va="bottom", ha="right")
    ax.set_ylim(0, 900)
    ax.set_ylabel("PM2.5 (µg/m³)", color=INK2)
    ax.set_title("1-hour forecasts during the November 2024 spike", color=INK, fontsize=13, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK2, loc="lower center", ncol=2)
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    fig.tight_layout()
    fig.savefig(FIGURES / "spike_nov2024.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def spike_numbers() -> None:
    a = PRED[("raw_xgb", "test", 1)]
    e = (a["actual"] - a["prediction"]) ** 2
    top = int(0.01 * len(e))
    share = np.sort(e.to_numpy())[::-1][:top].sum() / e.sum()
    biggest = a.loc[(a["actual"] - a["prediction"]).abs().idxmax()]
    out = {
        "share_of_squared_error_from_worst_1pct_hours": share,
        "max_actual_test": a["actual"].max(), "max_xgb_prediction_test": a["prediction"].max(),
        "worst_hour_actual": biggest["actual"], "worst_hour_prediction": biggest["prediction"],
    }
    pd.Series(out).to_csv(TABLES / "spike_numbers.csv", header=["value"])
    print("  spike numbers:", {k: round(float(v), 2) for k, v in out.items()})



# ----------------------------------------------------------------------------
# Numbers for the report text (so the prose cannot drift away from the results)
# ----------------------------------------------------------------------------

MACRO_NAME = {
    "persistence": "Pers", "raw_xgb": "Raw", "stl_train_profile": "Fixed", "stl_all_profile": "FixedAll",
    "stl_causal_14d": "Causal", "stl_causal_7d": "CausalSeven", "stl_leaky_seasonal": "LeakySeas",
    "stl_leaky_full": "Leaky", "stl_leaky_full_trend361": "LeakyTrend", "xgb_delta": "Delta", "xgb_log": "Log",
}
H_NAME = {1: "One", 6: "Six", 24: "TwentyFour"}


def write_numbers() -> None:
    """results/tables/numbers.tex defines a macro for every number quoted in the report text."""
    lines = ["% Generated by diagnostics.py - do not edit by hand."]

    def add(name, value, decimals=2):
        lines.append(f"\\newcommand{{\\{name}}}{{{value:.{decimals}f}}}")

    for split, prefix in (("test", ""), ("val", "v")):
        for m, mname in MACRO_NAME.items():
            for h in HORIZONS:
                add(f"{prefix}mae{mname}{H_NAME[h]}", mae(m, split, h))
                add(f"{prefix}rmse{mname}{H_NAME[h]}", rmse(m, split, h))
        for h in HORIZONS:
            ref = mae("stl_causal_14d", split, h)
            # how much lower the look-ahead error is than the matched rolling (no look-ahead) error
            add(f"{prefix}leakEff{H_NAME[h]}", 100 * (1 - mae("stl_leaky_full", split, h) / ref), 0)
            add(f"{prefix}leakEffTrend{H_NAME[h]}", 100 * (1 - mae("stl_leaky_full_trend361", split, h) / ref), 0)
            add(f"{prefix}leakEffSeas{H_NAME[h]}", 100 * (1 - mae("stl_leaky_seasonal", split, h) / ref), 0)
            # skill of the models relative to persistence (positive = better than persistence)
            for m in ("raw_xgb", "xgb_delta", "stl_causal_14d"):
                add(f"{prefix}skill{MACRO_NAME[m]}{H_NAME[h]}",
                    100 * (1 - mae(m, split, h) / mae("persistence", split, h)), 0)
            add(f"{prefix}causalVsRaw{H_NAME[h]}",
                100 * (1 - mae("stl_causal_14d", split, h) / mae("raw_xgb", split, h)), 1)
            add(f"{prefix}deltaVsRaw{H_NAME[h]}",
                100 * (1 - mae("xgb_delta", split, h) / mae("raw_xgb", split, h)), 1)

    # alert metrics on the test year, threshold 250 (CPCB "severe")
    for m in ALERT_METHODS:
        for h in HORIZONS:
            df = PRED[(m, "test", h)]
            event, alarm = df["actual"] > 250, df["prediction"] > 250
            tp, fn, fp = int((event & alarm).sum()), int((event & ~alarm).sum()), int((~event & alarm).sum())
            add(f"recall{MACRO_NAME[m]}{H_NAME[h]}", tp / max(tp + fn, 1))
            add(f"prec{MACRO_NAME[m]}{H_NAME[h]}", tp / max(tp + fp, 1))
    n_events = int((PRED[("raw_xgb", "test", 1)]["actual"] > 250).sum())
    lines.append(f"\\newcommand{{\\severeHours}}{{{n_events}}}")

    # the spike
    a = PRED[("raw_xgb", "test", 1)]
    sq = ((a["actual"] - a["prediction"]) ** 2).to_numpy()
    top = int(0.01 * len(sq))
    add("spikeShare", 100 * np.sort(sq)[::-1][:top].sum() / sq.sum(), 0)
    worst = (a["actual"] - a["prediction"]).abs().idxmax()
    add("spikeActual", a.loc[worst, "actual"], 0)
    add("spikePredicted", a.loc[worst, "prediction"], 0)
    add("spikeMaxPredicted", a["prediction"].max(), 0)
    add("spikeDeltaPredicted", PRED[("xgb_delta", "test", 1)].loc[worst, "prediction"], 0)
    (TABLES / "numbers.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    print("Loading predictions...")
    load_all()
    sanity_check_against_existing()
    main_tables(); leakage_tables(); significance_table(); alert_tables()
    period_table(); importance_table(); spike_numbers(); write_numbers()
    leakage_figure(); spike_figure()
    print(f"Tables saved to:  {TABLES}")
    print(f"Figures saved to: {FIGURES}")


if __name__ == "__main__":
    main()
