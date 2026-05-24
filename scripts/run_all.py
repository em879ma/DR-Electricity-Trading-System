"""
run_all.py — Execute all pipeline steps sequentially.

Pipeline:
  01: Prepare data (OPSD + Ember → market_panel.csv)
  02: Features, elasticity, baseline demand
  03: RF forecast + calibration + bootstrap scenarios
  04: Counterfactual price model + day-ahead DR decision
  05: Figures 1-4 + energy-mix regime analysis
  06: Four-level evaluation dashboard + final research summary

Note:
  - RF forecast is the operational baseline, NOT the optimized price.
  - The optimized price is the counterfactual price after feasible DR actions.
  - rolling_elasticity is a feature (rolling OLS slope), not causal elasticity.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def main(config_path: str = "config.yaml") -> None:
    from scripts.run_01_prepare_data import main as run_01
    from scripts.run_02_features_elasticity_dr import main as run_02
    from scripts.run_03_forecast_calibration_scenarios import main as run_03
    from scripts.run_04_day_ahead_decision import main as run_04
    from scripts.run_05_counterfactuals import main as run_05
    from scripts.run_06_evaluation_dashboard import main as run_06

    print("=" * 60)
    print("DR Day-Ahead Decision System — Full Pipeline")
    print("=" * 60)

    print("\n[01/06] Preparing data...")
    run_01(config_path)

    print("\n[02/06] Features, elasticity, baseline demand...")
    run_02(config_path)

    print("\n[03/06] RF forecast, calibration, scenarios...")
    run_03(config_path)

    print("\n[04/06] Counterfactual price model + DR optimization...")
    run_04(config_path)

    print("\n[05/06] Figures + energy-mix regime analysis...")
    run_05(config_path)

    print("\n[06/06] Four-level evaluation dashboard...")
    run_06(config_path)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("Key outputs:")
    print("  outputs/tables/forecast_accuracy_level1.csv")
    print("  outputs/tables/scenario_quality_level2.csv")
    print("  outputs/tables/decision_performance_level3.csv")
    print("  outputs/tables/mechanism_validity_level4.csv")
    print("  outputs/tables/evaluation_dashboard.csv")
    print("  outputs/tables/day_ahead_optimal_dr_schedule.csv")
    print("  outputs/tables/optimized_price_vs_observed_summary.csv")
    print("  outputs/tables/energy_mix_regime_summary.csv")
    print("  outputs/figures/01_forecast_calibration.png")
    print("  outputs/figures/02_bootstrap_scenario_fan_chart.png")
    print("  outputs/figures/03_day_ahead_dr_schedule.png")
    print("  outputs/figures/04_observed_vs_counterfactual_price.png")
    print("  outputs/figures/05_evaluation_dashboard.png")
    print("  outputs/figures/06_energy_mix_regime_analysis.png")
    print("  outputs/reports/final_research_summary.md")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full DR pipeline.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
