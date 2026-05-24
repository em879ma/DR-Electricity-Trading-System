from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_all_figures(
    rf_forecast_df: pd.DataFrame,
    calibrated_forecast_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    optimal_schedule_df: pd.DataFrame,
    optimized_hourly_df: pd.DataFrame,
    evaluation_dashboard: pd.DataFrame,
    energy_mix_regime: pd.DataFrame | None = None,
    output_root: str = "outputs",
) -> None:
    """Generate the 6 required final figures."""
    fig_dir = Path(output_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_forecast_calibration(rf_forecast_df, calibrated_forecast_df, fig_dir / "01_forecast_calibration.png")
    plot_bootstrap_scenario_fan(scenario_df, calibrated_forecast_df, fig_dir / "02_bootstrap_scenario_fan_chart.png")
    plot_day_ahead_dr_schedule(optimal_schedule_df, fig_dir / "03_day_ahead_dr_schedule.png")
    plot_observed_vs_counterfactual(optimized_hourly_df, fig_dir / "04_observed_vs_counterfactual_price.png")
    plot_evaluation_dashboard(evaluation_dashboard, fig_dir / "05_evaluation_dashboard.png")
    if energy_mix_regime is not None and not energy_mix_regime.empty:
        plot_energy_mix_regime(energy_mix_regime, fig_dir / "06_energy_mix_regime_analysis.png")


# ---------------------------------------------------------------------------
# Figure 1: Forecast calibration
# ---------------------------------------------------------------------------

def plot_forecast_calibration(
    raw_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    path: Path,
    horizon: int = 24,
    tail_points: int = 200,
) -> None:
    p_raw = raw_df[(raw_df["target"] == "price") & (raw_df["horizon"] == horizon)].copy()
    p_cal = cal_df[(cal_df["target"] == "price") & (cal_df["horizon"] == horizon)].copy()
    merged = (
        p_raw[["target_datetime", "actual", "forecast"]]
        .merge(p_cal[["target_datetime", "forecast_calibrated"]], on="target_datetime", how="inner")
        .sort_values("target_datetime")
        .tail(tail_points)
    )
    if merged.empty:
        return

    err_raw = merged["forecast"] - merged["actual"]
    err_cal = merged["forecast_calibrated"] - merged["actual"]
    mae_raw = float(np.mean(np.abs(err_raw)))
    mae_cal = float(np.mean(np.abs(err_cal)))
    rmse_raw = float(np.sqrt(np.mean(err_raw ** 2)))
    rmse_cal = float(np.sqrt(np.mean(err_cal ** 2)))
    bias_raw = float(np.mean(err_raw))
    bias_cal = float(np.mean(err_cal))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(merged["target_datetime"], merged["actual"], label="Actual price", linewidth=1.5, color="black")
    ax.plot(merged["target_datetime"], merged["forecast"], label="RF raw forecast", linewidth=1.0, linestyle="--", color="steelblue")
    ax.plot(merged["target_datetime"], merged["forecast_calibrated"], label="RF calibrated forecast", linewidth=1.0, color="tomato")
    ax.set_title(f"Figure 1: Forecast Calibration (H={horizon}, last {tail_points} hours)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (€/MWh)")
    ax.legend(fontsize=8)
    annotation = (
        f"Raw:  MAE={mae_raw:.2f}  RMSE={rmse_raw:.2f}  Bias={bias_raw:.2f}\n"
        f"Cal:   MAE={mae_cal:.2f}  RMSE={rmse_cal:.2f}  Bias={bias_cal:.2f}"
    )
    ax.text(0.01, 0.97, annotation, transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Bootstrap scenario fan chart
# ---------------------------------------------------------------------------

def plot_bootstrap_scenario_fan(
    scenario_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    path: Path,
    horizon: int = 24,
) -> None:
    scen = scenario_df.copy()
    scen["target_datetime"] = pd.to_datetime(scen["target_datetime"])
    pivot = scen.groupby("target_datetime")["price_scenario"].quantile([0.05, 0.10, 0.50, 0.90, 0.95]).unstack()
    pivot.columns = ["p05", "p10", "p50", "p90", "p95"]

    # Get calibrated point forecast and actual
    p_cal = cal_df[(cal_df["target"] == "price") & (cal_df["horizon"] == horizon)].copy()
    p_cal["target_datetime"] = pd.to_datetime(p_cal["target_datetime"])
    merged = pivot.join(p_cal.set_index("target_datetime")[["forecast_calibrated", "actual"]], how="left")

    fig, ax = plt.subplots(figsize=(11, 4))
    x = merged.index
    ax.fill_between(x, merged["p05"], merged["p95"], alpha=0.12, color="steelblue", label="P05-P95 band")
    ax.fill_between(x, merged["p10"], merged["p90"], alpha=0.25, color="steelblue", label="P10-P90 band")
    ax.plot(x, merged["p50"], linewidth=1.0, color="steelblue", linestyle="--", label="Scenario median")
    if "forecast_calibrated" in merged.columns:
        ax.plot(x, merged["forecast_calibrated"], linewidth=1.2, color="tomato", label="Calibrated forecast")
    if "actual" in merged.columns:
        ax.plot(x, merged["actual"], linewidth=1.5, color="black", label="Actual price")
    ax.set_title("Figure 2: Bootstrap Scenario Fan Chart")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (€/MWh)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Day-ahead DR schedule
# ---------------------------------------------------------------------------

def plot_day_ahead_dr_schedule(schedule_df: pd.DataFrame, path: Path) -> None:
    s = schedule_df.copy()
    dt_col = "target_datetime" if "target_datetime" in s.columns else "datetime"
    s[dt_col] = pd.to_datetime(s[dt_col])
    s = s.sort_values(dt_col).tail(48)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    price_col = "expected_price_calibrated" if "expected_price_calibrated" in s.columns else "forecasted_price"
    if price_col in s.columns:
        ax1.plot(s[dt_col], s[price_col], color="black", linewidth=1.5, label="Expected price")
        if "price_p90" in s.columns:
            p90_thr = float(s[price_col].quantile(0.9))
            high_mask = s[price_col] >= p90_thr
            ax1.scatter(s.loc[high_mask, dt_col], s.loc[high_mask, price_col],
                        color="red", s=20, zorder=5, label="High-price hours")
        ax1.set_ylabel("Expected price (€/MWh)")
        ax1.legend(fontsize=8)

    q_col = "q_DA_optimal" if "q_DA_optimal" in s.columns else "q_DA"
    if q_col in s.columns:
        colors = []
        for _, row in s.iterrows():
            if row.get("low_price_constraint_triggered", False):
                colors.append("gray")
            elif row.get(price_col, 0) >= (s[price_col].quantile(0.9) if price_col in s.columns else 0):
                colors.append("tomato")
            else:
                colors.append("steelblue")
        ax2.bar(s[dt_col], s[q_col], color=colors, width=0.03)
        ax2.set_ylabel("q_DA (MW)")
        ax2.set_xlabel("Time")
        from matplotlib.patches import Patch
        legend_el = [
            Patch(facecolor="tomato", label="High-price hours"),
            Patch(facecolor="steelblue", label="Normal hours"),
            Patch(facecolor="gray", label="Guardrail blocked"),
        ]
        ax2.legend(handles=legend_el, fontsize=8)

    fig.suptitle("Figure 3: Optimized Day-Ahead DR Schedule", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Observed vs counterfactual price
# ---------------------------------------------------------------------------

def plot_observed_vs_counterfactual(hourly: pd.DataFrame, path: Path) -> None:
    h = hourly.copy()
    h["datetime"] = pd.to_datetime(h["datetime"])
    h = h.sort_values("datetime")

    price_col = "observed_price" if "observed_price" in h.columns else "price"
    cf_col = "counterfactual_price"

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(h["datetime"], h[price_col], color="black", linewidth=1.3, label="Observed actual price")
    if cf_col in h.columns:
        ax.plot(h["datetime"], h[cf_col], color="tomato", linewidth=1.0, linestyle="--",
                label="Counterfactual price (after DR)")
        diff = float((h[price_col] - h[cf_col]).mean())
        ax.text(0.01, 0.97, f"Avg price reduction: {diff:.2f} €/MWh",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Figure 4: Observed vs Counterfactual Price")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (€/MWh)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Four-level evaluation dashboard
# ---------------------------------------------------------------------------

def plot_evaluation_dashboard(eval_df: pd.DataFrame, path: Path) -> None:
    if eval_df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Figure 5: Four-Level Evaluation Dashboard", fontsize=13)

    level_cols = {
        "Level 1: Forecast Accuracy": [c for c in eval_df.columns if c in ["MAE", "RMSE", "Bias", "R2", "high_price_recall", "negative_price_recall"]],
        "Level 2: Scenario Quality": [c for c in eval_df.columns if c in ["coverage_10_90", "coverage_05_95", "scenario_bias", "mean_interval_width"]],
        "Level 3: Decision Performance": [c for c in eval_df.columns if c in ["total_market_cost_reduction", "avg_price_reduction", "high_price_hours_reduced", "constraint_violation_rate"]],
        "Level 4: Mechanism Validity": [c for c in eval_df.columns if c in ["mechanism_checks_passed", "mechanism_checks_total", "passed_rate"]],
    }

    for ax, (title, cols) in zip(axes.flat, level_cols.items()):
        ax.set_title(title, fontsize=10)
        available = [c for c in cols if c in eval_df.columns]
        if not available:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        row = eval_df[available].iloc[0]
        x = np.arange(len(available))
        bars = ax.barh(x, row.to_numpy(dtype=float), color="steelblue")
        ax.set_yticks(x)
        ax.set_yticklabels(available, fontsize=8)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Energy-mix regime analysis
# ---------------------------------------------------------------------------

def plot_energy_mix_regime(regime_df: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 6: Energy-Mix / Regime Analysis", fontsize=12)

    df = regime_df.copy()
    x = np.arange(len(df))
    w = 0.35

    ax1.bar(x - w / 2, df["avg_DR_quantity"].fillna(0), width=w, label="Avg DR quantity", color="steelblue")
    ax1.bar(x + w / 2, df["avg_price_reduction"].fillna(0), width=w, label="Avg price reduction", color="tomato")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["regime"], rotation=20, fontsize=8)
    ax1.set_title("DR effect by regime")
    ax1.set_ylabel("Value")
    ax1.legend(fontsize=8)

    ax2.bar(x, df["negative_price_frequency"].fillna(0), color="orange")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["regime"], rotation=20, fontsize=8)
    ax2.set_title("Negative-price frequency by regime")
    ax2.set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
