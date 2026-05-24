"""
Run 05: Generate figures 1-4 and energy-mix regime analysis (Figure 6).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.visualization import (
    plot_forecast_calibration,
    plot_bootstrap_scenario_fan,
    plot_day_ahead_dr_schedule,
    plot_observed_vs_counterfactual,
    plot_energy_mix_regime,
)
from src.energy_mix import build_energy_mix_regime_summary


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out_root = cfg["output"]["root"]
    tables_dir = Path(out_root) / "tables"
    fig_dir = Path(out_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=== Figure 1: Forecast Calibration ===")
    raw_path = tables_dir / "rf_raw_forecasts.csv"
    cal_path = tables_dir / "rf_calibrated_forecasts.csv"
    if raw_path.exists() and cal_path.exists():
        rf_raw = pd.read_csv(raw_path, parse_dates=["target_datetime", "forecast_origin_datetime"])
        rf_cal = pd.read_csv(cal_path, parse_dates=["target_datetime", "forecast_origin_datetime"])
        plot_forecast_calibration(rf_raw, rf_cal, fig_dir / "01_forecast_calibration.png")
        print(f"  Saved 01_forecast_calibration.png")
    else:
        print(f"  Skipped (missing raw/calibrated forecast tables).")

    print("=== Figure 2: Bootstrap Scenario Fan Chart ===")
    scen_path = Path(out_root) / "scenarios" / "bootstrap_scenarios_calibrated.csv"
    if scen_path.exists() and cal_path.exists():
        scen_df = pd.read_csv(scen_path, parse_dates=["target_datetime"])
        rf_cal = pd.read_csv(cal_path, parse_dates=["target_datetime"])
        plot_bootstrap_scenario_fan(scen_df, rf_cal, fig_dir / "02_bootstrap_scenario_fan_chart.png")
        print(f"  Saved 02_bootstrap_scenario_fan_chart.png")
    else:
        print(f"  Skipped (missing scenario/calibrated files).")

    print("=== Figure 3: Day-Ahead DR Schedule ===")
    sched_path = tables_dir / "day_ahead_optimal_dr_schedule.csv"
    if sched_path.exists():
        sched = pd.read_csv(sched_path, parse_dates=["target_datetime"])
        plot_day_ahead_dr_schedule(sched, fig_dir / "03_day_ahead_dr_schedule.png")
        print(f"  Saved 03_day_ahead_dr_schedule.png")
    else:
        print(f"  Skipped (missing schedule).")

    print("=== Figure 4: Observed vs Counterfactual Price ===")
    hourly_path = tables_dir / "optimized_price_vs_observed_hourly.csv"
    if hourly_path.exists():
        hourly = pd.read_csv(hourly_path, parse_dates=["datetime"])
        plot_observed_vs_counterfactual(hourly, fig_dir / "04_observed_vs_counterfactual_price.png")
        print(f"  Saved 04_observed_vs_counterfactual_price.png")
    else:
        print(f"  Skipped (missing hourly counterfactual table).")

    print("=== Energy-Mix Regime Analysis ===")
    panel_path = Path("data/processed/market_panel_with_dr.csv")
    if hourly_path.exists() and panel_path.exists():
        hourly = pd.read_csv(hourly_path, parse_dates=["datetime"])
        panel = pd.read_csv(panel_path, parse_dates=["datetime"])
        # Compute energy-structure features if not already present
        if "renewable_share" not in panel.columns:
            panel["renewable_share"] = panel["renewable_total"] / (panel["load"] + 1e-6)
        if "scarcity_index" not in panel.columns:
            panel["scarcity_index"] = panel["load"] / (
                panel["available_capacity"] + panel["imports"] + 1e-6
            )
        if "oversupply_index" not in panel.columns:
            panel["oversupply_index"] = (
                panel["renewable_total"] + panel["imports"] - panel["load"]
            )
        merged = hourly.merge(
            panel[["datetime", "renewable_share", "scarcity_index", "oversupply_index", "rolling_elasticity"]],
            on="datetime", how="left",
        )
        regime_df = build_energy_mix_regime_summary(
            merged,
            output_path=tables_dir / "energy_mix_regime_summary.csv",
        )
        print(regime_df.to_string(index=False))
        plot_energy_mix_regime(regime_df, fig_dir / "06_energy_mix_regime_analysis.png")
        print(f"  Saved 06_energy_mix_regime_analysis.png")
    else:
        print(f"  Skipped (missing hourly or panel data).")

    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
