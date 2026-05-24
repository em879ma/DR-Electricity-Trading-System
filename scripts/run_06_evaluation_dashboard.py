"""
Run 06: Four-Level Evaluation Dashboard.
Loads outputs from prior steps, computes all four evaluation levels,
saves tables, and generates Figure 5 (evaluation dashboard).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation_forecast import evaluate_forecast_accuracy
from src.evaluation_scenarios import evaluate_scenario_quality
from src.evaluation_decision import evaluate_decision_performance, evaluate_constraints
from src.evaluation_mechanism import evaluate_economic_mechanisms
from src.reporting import build_evaluation_dashboard_table, write_final_research_summary
from src.visualization import plot_evaluation_dashboard


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out_root = cfg["output"]["root"]
    tables_dir = Path(out_root) / "tables"
    fig_dir = Path(out_root) / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=== Level 1: Forecast Accuracy ===")
    forecast_level1 = pd.DataFrame()
    cal_path = tables_dir / "rf_calibrated_forecasts.csv"
    if cal_path.exists():
        cal_long = pd.read_csv(cal_path)
        forecast_level1 = evaluate_forecast_accuracy(cal_long, horizon=24, target="price")
        forecast_level1.to_csv(tables_dir / "forecast_accuracy_level1.csv", index=False)
        print(forecast_level1.to_string(index=False))
    else:
        print(f"  Calibrated forecast not found at {cal_path}. Skipping Level 1.")

    print("\n=== Level 2: Scenario Quality ===")
    scenario_level2 = pd.DataFrame()
    scen_path = Path(out_root) / "scenarios" / "bootstrap_scenarios_calibrated.csv"
    panel_path = Path("data/processed/market_panel_with_dr.csv")
    if not panel_path.exists():
        panel_path = Path("data/processed/market_panel.csv")
    if scen_path.exists() and panel_path.exists():
        scen_df = pd.read_csv(scen_path)
        panel = pd.read_csv(panel_path)
        scenario_level2 = evaluate_scenario_quality(scen_df, panel)
        scenario_level2.to_csv(tables_dir / "scenario_quality_level2.csv", index=False)
        print(scenario_level2.to_string(index=False))
    else:
        print(f"  Scenario or panel file not found. Skipping Level 2.")

    print("\n=== Level 3: Decision Performance ===")
    decision_level3 = pd.DataFrame()
    sched_path = tables_dir / "day_ahead_optimal_dr_schedule.csv"
    panel_path2 = Path("data/processed/market_panel_with_dr.csv")
    if not panel_path2.exists():
        panel_path2 = panel_path
    if sched_path.exists() and panel_path2.exists():
        sched = pd.read_csv(sched_path)
        panel2 = pd.read_csv(panel_path2)
        flex = float(cfg["decision"]["flexibility_ratios"][1]) if "flexibility_ratios" in cfg["decision"] else 0.10
        c_dr = float(cfg["decision"].get("c_dr", 3.0))
        pen = float(cfg["decision"].get("penalty_rate", 20.0))
        decision_level3 = evaluate_decision_performance(sched, panel2, flexibility_ratio=flex, c_dr=c_dr, penalty_rate=pen)
        decision_level3.to_csv(tables_dir / "decision_performance_level3.csv", index=False)
        constraint_df = evaluate_constraints(sched)
        constraint_df.to_csv(tables_dir / "constraint_check_summary.csv", index=False)
        print(decision_level3.to_string(index=False))
    else:
        print(f"  Schedule or panel not found. Skipping Level 3.")

    print("\n=== Level 4: Economic Mechanism Validity ===")
    mechanism_level4 = pd.DataFrame()
    hourly_path = tables_dir / "optimized_price_vs_observed_hourly.csv"
    sens_path = tables_dir / "counterfactual_demand_sensitivity.csv"
    coef_path = tables_dir / "counterfactual_price_model_coefficients.csv"
    if hourly_path.exists():
        hourly = pd.read_csv(hourly_path)
        sens_df = pd.read_csv(sens_path) if sens_path.exists() else None
        coef_df = pd.read_csv(coef_path) if coef_path.exists() else None
        mechanism_level4 = evaluate_economic_mechanisms(hourly, sensitivity_df=sens_df, model_coef_df=coef_df)
        mechanism_level4.to_csv(tables_dir / "mechanism_validity_level4.csv", index=False)
        print(mechanism_level4.to_string(index=False))
    else:
        print(f"  Hourly panel not found at {hourly_path}. Skipping Level 4.")

    print("\n=== Building Evaluation Dashboard Table ===")
    dashboard = build_evaluation_dashboard_table(forecast_level1, scenario_level2, decision_level3, mechanism_level4)
    dashboard.to_csv(tables_dir / "evaluation_dashboard.csv", index=False)

    print("\n=== Generating Figure 5: Evaluation Dashboard ===")
    # Pivot dashboard to wide for plotting
    if not dashboard.empty:
        wide = dashboard.pivot_table(index="level", columns="metric", values="value", aggfunc="first")
        wide = wide.reset_index()
        # Include L4 pass_rate in summary row
        summary_row = {}
        for lvl in ["L1_forecast", "L2_scenario", "L3_decision", "L4_mechanism"]:
            sub = dashboard[dashboard["level"] == lvl]
            if not sub.empty:
                summary_row.update({f"{lvl}_{r['metric']}": r["value"] for _, r in sub.iterrows()})
        summary_df = pd.DataFrame([summary_row])
        plot_evaluation_dashboard(summary_df, fig_dir / "05_evaluation_dashboard.png")
        print(f"  Saved to {fig_dir / '05_evaluation_dashboard.png'}")

    print("\n=== Writing Final Research Summary ===")
    cal_summary_path = tables_dir / "forecast_calibration_summary.csv"
    cal_summary = pd.read_csv(cal_summary_path) if cal_summary_path.exists() else None
    write_final_research_summary(
        forecast_level1=forecast_level1,
        scenario_level2=scenario_level2,
        decision_level3=decision_level3,
        mechanism_level4=mechanism_level4,
        calibration_summary=cal_summary,
        output_root=out_root,
    )
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
