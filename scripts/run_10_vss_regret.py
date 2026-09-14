"""
Run 10: Value of Stochastic Solution (VSS) and Decision Regret.

Computes two paper-aligned L3 metrics:

  VSS = Cost(deterministic-point-forecast DR)  −  Cost(scenario-based DR)
      A positive VSS justifies the cost of generating scenarios.

  Regret = Cost(model-based DR)  −  Cost(perfect-information DR)
      Lower regret means the model captures more of the achievable saving.

Inputs (must exist):
  outputs/tables/day_ahead_optimal_dr_schedule.csv   (from run_04)
  data/processed/market_panel_with_dr.csv            (from run_02)

Outputs:
  outputs/tables/vss_regret_summary.csv
  outputs/tables/vss_regret_breakdown.csv  (per-month / per-regime breakdown)
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

from src.evaluation_decision import (
    build_hourly_counterfactual_panel,
    compute_regret,
    compute_vss,
)
from src.day_ahead_decision import optimize_dr_scenario_grid


def _hourly_cost(prices: np.ndarray, demands: np.ndarray) -> float:
    return float(np.sum(prices * demands))


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out_root = Path(cfg["output"]["root"])
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    sched_path = tables_dir / "day_ahead_optimal_dr_schedule.csv"
    panel_path = Path("data/processed/market_panel_with_dr.csv")
    model_path = out_root / "models" / "counterfactual_price_model.pkl"
    scen_path = out_root / "scenarios" / "bootstrap_scenarios_calibrated.csv"

    for p in [sched_path, panel_path, model_path, scen_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing dependency: {p}")

    print("Loading data...")
    sched = pd.read_csv(sched_path, parse_dates=["target_datetime"])
    panel = pd.read_csv(panel_path, parse_dates=["datetime"])
    with model_path.open("rb") as f:
        model_pack = pickle.load(f)

    flex_cfg = float(cfg["decision"]["flexibility_ratios"][1])
    c_dr = float(cfg["decision"].get("c_dr", 3.0))
    pen = float(cfg["decision"].get("penalty_rate", 20.0))
    p_hist_90 = float(panel["price"].quantile(0.90))
    p_hist_25 = float(panel["price"].quantile(0.25))
    low_thr = float(cfg["decision"].get("low_price_threshold", p_hist_25))
    low_prob_thr = float(cfg["decision"].get("low_price_probability_threshold", 0.30))

    # ── (1) Stochastic baseline cost (already in run_04 output) ─────────
    print("Building stochastic-DR hourly panel (from run_04 schedule)...")
    stoch_hourly = build_hourly_counterfactual_panel(sched, panel)
    stoch_cost = _hourly_cost(
        stoch_hourly["counterfactual_price"].to_numpy(),
        stoch_hourly["counterfactual_demand"].to_numpy(),
    )
    obs_cost = _hourly_cost(
        stoch_hourly["observed_price"].to_numpy(),
        stoch_hourly["demand_base"].to_numpy(),
    )
    print(f"  Observed market cost:   €{obs_cost:,.0f}")
    print(f"  Stochastic-DR cost:     €{stoch_cost:,.0f}")
    print(f"  Stochastic saving:      €{obs_cost - stoch_cost:,.0f}")

    # ── (2) Deterministic baseline: 1-scenario point forecast ───────────
    # Derive the point forecast by averaging the bootstrap scenarios per hour
    # (this matches what `ablate_no_scenarios` does in src/ablation.py).
    print("\nComputing deterministic-DR schedule (single scenario = point forecast)...")
    boot = pd.read_csv(scen_path, parse_dates=["target_datetime", "forecast_origin_datetime"])
    degen = (
        boot.groupby(["forecast_origin_datetime", "target_datetime", "horizon"], as_index=False)
        .agg({
            "price_scenario":     "mean",
            "load_scenario":      "mean",
            "renewable_scenario": "mean",
        })
        .assign(scenario_id=1, scenario_probability=1.0)
        [["scenario_id", "scenario_probability", "forecast_origin_datetime",
          "target_datetime", "horizon",
          "price_scenario", "load_scenario", "renewable_scenario"]]
    )
    det_sched = optimize_dr_scenario_grid(
        panel, degen,
        flexibility_ratio=flex_cfg, objective_mode="average_cost",
        high_price_threshold=p_hist_90,
        use_refit_price_model=True, refit_model_pack=model_pack,
        c_dr=c_dr, penalty_rate=pen,
        low_price_threshold=low_thr,
        low_price_probability_threshold=low_prob_thr,
    )
    det_hourly = build_hourly_counterfactual_panel(det_sched, panel)
    det_cost = _hourly_cost(
        det_hourly["counterfactual_price"].to_numpy(),
        det_hourly["counterfactual_demand"].to_numpy(),
    )
    print(f"  Deterministic-DR cost:  €{det_cost:,.0f}")

    # ── (3) Perfect-information oracle ──────────────────────────────────
    # The oracle re-runs the SAME optimizer but with the realized values
    # supplied as the (degenerate) scenario — i.e. price/load/renewable
    # are known exactly. This is the strongest possible lower bound that is
    # still apples-to-apples with the model-based and deterministic schedules.
    print("\nComputing perfect-information oracle schedule (same optimizer, actuals as scenarios)...")
    # Build a one-scenario "forecast" frame using the realized values.
    oracle_scen = (
        panel[["datetime", "price", "load", "renewable_total"]]
        .rename(columns={
            "datetime": "target_datetime",
            "price": "price_scenario",
            "load": "load_scenario",
            "renewable_total": "renewable_scenario",
        })
        .merge(sched[["target_datetime"]], on="target_datetime", how="inner")
        .assign(
            scenario_id=1, scenario_probability=1.0, horizon=24,
            forecast_origin_datetime=lambda d: d["target_datetime"] - pd.Timedelta(hours=24),
        )
        [["scenario_id", "scenario_probability", "forecast_origin_datetime",
          "target_datetime", "horizon",
          "price_scenario", "load_scenario", "renewable_scenario"]]
    )
    oracle_sched = optimize_dr_scenario_grid(
        panel, oracle_scen,
        flexibility_ratio=flex_cfg, objective_mode="average_cost",
        high_price_threshold=p_hist_90,
        use_refit_price_model=True, refit_model_pack=model_pack,
        c_dr=c_dr, penalty_rate=pen,
        low_price_threshold=low_thr,
        low_price_probability_threshold=low_prob_thr,
    )
    oracle_hourly = build_hourly_counterfactual_panel(oracle_sched, panel)
    oracle_cost = _hourly_cost(
        oracle_hourly["counterfactual_price"].to_numpy(),
        oracle_hourly["counterfactual_demand"].to_numpy(),
    )
    print(f"  Perfect-information cost: €{oracle_cost:,.0f}")

    # ── (4) Compose summary ─────────────────────────────────────────────
    vss = compute_vss(stoch_cost, det_cost)
    regret = compute_regret(stoch_cost, oracle_cost)
    det_regret = compute_regret(det_cost, oracle_cost)

    summary = pd.DataFrame([{
        "observed_total_cost":            obs_cost,
        "stochastic_DR_cost":             stoch_cost,
        "deterministic_DR_cost":          det_cost,
        "perfect_information_cost":       oracle_cost,
        "stochastic_saving_vs_observed":  obs_cost - stoch_cost,
        "deterministic_saving_vs_observed": obs_cost - det_cost,
        "VSS_absolute":                   vss["VSS_absolute"],
        "VSS_relative":                   vss["VSS_relative"],
        "regret_stochastic_absolute":     regret["regret_absolute"],
        "regret_stochastic_relative":     regret["regret_relative"],
        "regret_deterministic_absolute":  det_regret["regret_absolute"],
        "regret_deterministic_relative":  det_regret["regret_relative"],
    }])
    out_path = tables_dir / "vss_regret_summary.csv"
    summary.to_csv(out_path, index=False)
    print("\n=== Summary ===")
    print(summary.T.to_string())
    print(f"\nSaved to {out_path}")
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
