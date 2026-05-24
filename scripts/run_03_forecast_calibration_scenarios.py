"""
Run 03: RF forecast → calibration → bootstrap scenarios.
RF is an operational baseline. The forecast is NOT the optimized price.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.forecasting import forecast_rf_day_ahead, fit_negative_price_binary_forecast, reshape_day_ahead_forecasts, write_rf_outputs
from src.calibration import (
    calibrate_forecasts,
    reshape_calibrated_day_ahead_forecasts,
    write_calibration_outputs,
)
from src.scenarios import build_residual_history, generate_bootstrap_scenarios, write_scenarios


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out_root = cfg["output"]["root"]

    dr_path = Path("data/processed/market_panel_with_dr.csv")
    if not dr_path.exists():
        raise FileNotFoundError(f"Run 02 first. Missing: {dr_path}")

    print("Loading market panel with DR features...")
    dr_df = pd.read_csv(dr_path, parse_dates=["datetime"])

    print("Running RF day-ahead forecast (baseline model, not optimized price)...")
    rf_raw_long, rf_acc = forecast_rf_day_ahead(
        dr_df,
        horizons=cfg["forecast"]["horizons"],
        rf_params=cfg["forecast"]["rf"],
        split=cfg["forecast"]["split"],
    )

    tables_dir = Path(out_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rf_raw_long.to_csv(tables_dir / "rf_raw_forecasts.csv", index=False)
    rf_acc.to_csv(tables_dir / "forecast_accuracy_level1_raw.csv", index=False)

    day_ahead_rf_raw = reshape_day_ahead_forecasts(rf_raw_long, horizon=24)
    write_rf_outputs(day_ahead_rf_raw, rf_acc, output_root=out_root)
    print(f"  RF forecast rows: {len(rf_raw_long)}")

    print("Calibrating forecasts...")
    rf_cal_long, calibration_summary = calibrate_forecasts(rf_raw_long)
    write_calibration_outputs(rf_cal_long, calibration_summary, output_root=out_root)
    day_ahead_rf = reshape_calibrated_day_ahead_forecasts(rf_cal_long, horizon=24)
    day_ahead_rf.to_csv(tables_dir / "day_ahead_rf_forecasts.csv", index=False)
    rf_cal_long.to_csv(tables_dir / "rf_calibrated_forecasts.csv", index=False)
    print(calibration_summary.to_string(index=False))

    print("Fitting negative-price binary classifier...")
    neg_price_df, neg_metrics = fit_negative_price_binary_forecast(
        dr_df,
        rf_params=cfg["forecast"]["rf"],
        split=cfg["forecast"]["split"],
        horizon=24,
    )
    if not neg_price_df.empty:
        rf_cal_long = rf_cal_long.merge(neg_price_df, on="target_datetime", how="left")
        rf_cal_long.to_csv(tables_dir / "rf_calibrated_forecasts.csv", index=False)
        print(f"  negative_price_recall_classifier: {neg_metrics.get('negative_price_recall_classifier', float('nan')):.3f}")
        print(f"  negative_price_precision_classifier: {neg_metrics.get('negative_price_precision_classifier', float('nan')):.3f}")
        print(f"  classifier_threshold: {neg_metrics.get('classifier_threshold', 0.5):.3f}")

    print("Generating bootstrap scenarios...")
    resid_input = rf_cal_long.copy()
    resid_input["forecast"] = resid_input["forecast_calibrated"]
    resid = build_residual_history(resid_input)
    boot = generate_bootstrap_scenarios(
        day_ahead_rf=day_ahead_rf,
        residual_history=resid,
        n_scenarios=cfg["scenario"]["n_scenarios"],
        block_bootstrap=cfg["scenario"]["block_bootstrap"],
        block_length=cfg["scenario"]["block_length"],
        random_seed=cfg["scenario"]["random_seed"],
    )
    write_scenarios(boot, output_root=out_root)
    print(f"  Scenarios: {boot['scenario_id'].nunique()} scenarios × {boot['target_datetime'].nunique()} hours")
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
