"""
Run 04: Fit counterfactual price model and optimize day-ahead DR schedule.

The optimized price is the COUNTERFACTUAL price after DR actions — NOT the RF forecast.
Negative-price guardrails are mandatory.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.counterfactual_price_model import (
    fit_counterfactual_price_models,
    predict_counterfactual_price,
    demand_sensitivity_analysis,
    write_counterfactual_model_outputs,
)
from src.day_ahead_decision import optimize_dr_scenario_grid, evaluate_constraints
from src.scenarios import build_residual_history, generate_bootstrap_scenarios
from src.calibration import slice_calibrated_day_ahead_block
from src.evaluation_decision import build_hourly_counterfactual_panel


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out_root = cfg["output"]["root"]
    tables_dir = Path(out_root) / "tables"
    models_dir = Path(out_root) / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    dr_path = Path("data/processed/market_panel_with_dr.csv")
    cal_path = tables_dir / "rf_calibrated_forecasts.csv"
    scen_path = Path(out_root) / "scenarios" / "bootstrap_scenarios_calibrated.csv"

    for p in [dr_path, cal_path, scen_path]:
        if not p.exists():
            raise FileNotFoundError(f"Run prior steps first. Missing: {p}")

    print("Loading data...")
    dr_df = pd.read_csv(dr_path, parse_dates=["datetime"])
    rf_cal_long = pd.read_csv(cal_path, parse_dates=["target_datetime", "forecast_origin_datetime"])
    boot = pd.read_csv(scen_path, parse_dates=["target_datetime"])

    print("Fitting counterfactual price model (structural Ridge + RF)...")
    model_pack, eval_df, coef_df = fit_counterfactual_price_models(dr_df)

    print("Demand sensitivity analysis (mechanism check)...")
    sens_df = demand_sensitivity_analysis(dr_df, model_pack)
    write_counterfactual_model_outputs(eval_df, coef_df, sens_df, output_root=out_root)

    with (models_dir / "counterfactual_price_model.pkl").open("wb") as f:
        pickle.dump(model_pack, f)
    print(f"  Counterfactual model saved.")
    print(eval_df.to_string(index=False))

    flex_cfg = float(cfg["decision"]["flexibility_ratios"][1]) if "flexibility_ratios" in cfg["decision"] else 0.10
    c_dr = float(cfg["decision"].get("c_dr", 3.0))
    pen = float(cfg["decision"].get("penalty_rate", 20.0))
    p_hist_25 = float(dr_df["price"].quantile(0.25))
    p_hist_90 = float(dr_df["price"].quantile(0.90))
    low_price_thr = float(cfg["decision"].get("low_price_threshold", p_hist_25))
    low_price_prob_thr = float(cfg["decision"].get("low_price_probability_threshold", 0.30))

    print(f"\nOptimizing DR schedule (objective=average_cost, flex={flex_cfg:.0%})...")
    print(f"  Guardrails: low_price_threshold={low_price_thr:.1f}, p_negative≥{low_price_prob_thr}")

    optimal_schedule = optimize_dr_scenario_grid(
        dr_df,
        boot,
        flexibility_ratio=flex_cfg,
        objective_mode="average_cost",
        high_price_threshold=p_hist_90,
        cvar_alpha=0.90,
        use_refit_price_model=True,
        refit_model_pack=model_pack,
        demand_col_for_price_model="demand_base",
        c_dr=c_dr,
        penalty_rate=pen,
        low_price_threshold=low_price_thr,
        low_price_probability_threshold=low_price_prob_thr,
    )

    if optimal_schedule.empty:
        print("  WARNING: optimal schedule is empty. Check data alignment.")
        return

    optimal_schedule.to_csv(tables_dir / "day_ahead_optimal_dr_schedule.csv", index=False)

    constraint_df = evaluate_constraints(optimal_schedule)
    constraint_df.to_csv(tables_dir / "constraint_check_summary.csv", index=False)
    print(constraint_df.to_string(index=False))

    # Build hourly counterfactual panel
    hourly = build_hourly_counterfactual_panel(optimal_schedule, dr_df)
    hourly.to_csv(tables_dir / "optimized_price_vs_observed_hourly.csv", index=False)

    # Summary table
    high_thr = float(hourly["observed_price"].quantile(0.9))
    summary = pd.DataFrame([{
        "avg_observed_price": float(hourly["observed_price"].mean()),
        "avg_counterfactual_price": float(hourly["counterfactual_price"].mean()),
        "avg_price_reduction": float(hourly["price_reduction"].mean()),
        "percent_price_reduction": float(hourly["price_reduction"].mean() / (hourly["observed_price"].mean() + 1e-9) * 100),
        "high_price_hours_observed": int((hourly["observed_price"] >= high_thr).sum()),
        "high_price_hours_counterfactual": int((hourly["counterfactual_price"] >= high_thr).sum()),
        "high_price_hours_reduced": int((hourly["observed_price"] >= high_thr).sum() - (hourly["counterfactual_price"] >= high_thr).sum()),
        "observed_total_market_cost": float((hourly["observed_price"] * hourly["demand_base"]).sum()),
        "counterfactual_total_market_cost": float((hourly["counterfactual_price"] * hourly["counterfactual_demand"]).sum()),
        "total_cost_reduction": float((hourly["observed_price"] * hourly["demand_base"]).sum() - (hourly["counterfactual_price"] * hourly["counterfactual_demand"]).sum()),
        "constraints_satisfied": bool(constraint_df["passed"].all()),
        "flexibility_ratio": flex_cfg,
        "objective_mode": "average_cost",
    }])
    summary.to_csv(tables_dir / "optimized_price_vs_observed_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
