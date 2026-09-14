"""
Run 13: Scenario-count sweep.

Hypothesis (from v2.2): with 48 block-bootstrap scenarios the DR optimizer
is "overfitting" to the scenario fan — the noise tail averages out into a
risk-averse plan that under-uses DR. Reducing n_scenarios should move the
average-cost objective closer to the deterministic baseline (which already
beats the stochastic one by €191M / VSS = −1.66%).

For each n ∈ {1, 6, 12, 24, 48, 96}:
  1. Regenerate scenarios from saved residuals using src/scenarios.py
  2. Re-run optimize_dr_scenario_grid (same model_pack as run_04)
  3. Compute hourly counterfactual cost + VSS + Regret

Outputs:
  outputs/tables/scenario_sweep_summary.csv  (one row per n_scenarios)
  outputs/figures/13_scenario_sweep.png      (cost / VSS / regret vs n)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.scenarios import generate_bootstrap_scenarios
from src.day_ahead_decision import optimize_dr_scenario_grid
from src.evaluation_decision import (
    build_hourly_counterfactual_panel,
    compute_regret,
)


N_GRID = [6, 12, 24]  # n=1 and n=48 are read from existing outputs (vss_regret_summary, run_04)


def _wide_forecasts(rf_cal: pd.DataFrame) -> pd.DataFrame:
    """Pivot rf_calibrated_forecasts.csv (long) → wide with columns expected by
    generate_bootstrap_scenarios. Filters to horizon=24 (day-ahead)."""
    sub = rf_cal[
        rf_cal["model"].isin(["random_forest", "regime_ensemble"])
        & (rf_cal["horizon"] == 24)
    ].copy()
    wide = sub.pivot_table(
        index=["forecast_origin_datetime", "target_datetime", "horizon"],
        columns="target", values="forecast_calibrated", aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    rename = {"price": "price_forecast", "load": "load_forecast",
              "renewable_total": "renewable_forecast"}
    wide = wide.rename(columns=rename)
    return wide.dropna(subset=["price_forecast", "load_forecast", "renewable_forecast"])


def _build_residuals(rf_cal: pd.DataFrame) -> pd.DataFrame:
    sub = rf_cal[rf_cal["horizon"] == 24].copy()
    sub["residual"] = sub["actual"] - sub["forecast_calibrated"]
    return sub[["target", "residual"]].dropna()


def _hourly_cost(price: np.ndarray, demand: np.ndarray) -> float:
    return float(np.sum(price * demand))


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    tables = Path(cfg["output"]["root"]) / "tables"
    figs = Path(cfg["output"]["root"]) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    models = Path(cfg["output"]["root"]) / "models"

    print("Loading dependencies (panel, calibrated forecasts, oracle prices, model pack)...")
    panel = pd.read_csv("data/processed/market_panel_with_dr.csv", parse_dates=["datetime"])
    rf_cal = pd.read_csv(tables / "rf_calibrated_forecasts.csv",
                         parse_dates=["forecast_origin_datetime", "target_datetime"])
    with (models / "counterfactual_price_model.pkl").open("rb") as f:
        model_pack = pickle.load(f)

    day_ahead = _wide_forecasts(rf_cal)
    resid = _build_residuals(rf_cal)
    print(f"  day_ahead rows: {len(day_ahead):,}")
    print(f"  residual samples: price={len(resid[resid['target']=='price']):,}")

    flex = float(cfg["decision"]["flexibility_ratios"][1])
    c_dr = float(cfg["decision"].get("c_dr", 3.0))
    pen = float(cfg["decision"].get("penalty_rate", 20.0))
    p_hist_25 = float(panel["price"].quantile(0.25))
    p_hist_90 = float(panel["price"].quantile(0.90))
    low_thr = float(cfg["decision"].get("low_price_threshold", p_hist_25))
    low_prob = float(cfg["decision"].get("low_price_probability_threshold", 0.30))
    block_length = int(cfg["scenario"].get("block_length", 24))
    seed = int(cfg["scenario"].get("random_seed", 42))

    # ── Reference: oracle (perfect information) ──────────────────────
    print("\nRunning perfect-information oracle (one pass, reused for all n)...")
    oracle_scen = (
        panel[["datetime", "price", "load", "renewable_total"]]
        .rename(columns={
            "datetime": "target_datetime",
            "price": "price_scenario",
            "load": "load_scenario",
            "renewable_total": "renewable_scenario",
        })
        .merge(day_ahead[["target_datetime"]].drop_duplicates(),
               on="target_datetime", how="inner")
        .assign(scenario_id=1, scenario_probability=1.0, horizon=24,
                forecast_origin_datetime=lambda d: d["target_datetime"] - pd.Timedelta(hours=24))
        [["scenario_id", "scenario_probability", "forecast_origin_datetime",
          "target_datetime", "horizon",
          "price_scenario", "load_scenario", "renewable_scenario"]]
    )
    oracle_sched = optimize_dr_scenario_grid(
        panel, oracle_scen,
        flexibility_ratio=flex, objective_mode="average_cost",
        high_price_threshold=p_hist_90,
        use_refit_price_model=True, refit_model_pack=model_pack,
        c_dr=c_dr, penalty_rate=pen,
        low_price_threshold=low_thr,
        low_price_probability_threshold=low_prob,
    )
    oracle_hourly = build_hourly_counterfactual_panel(oracle_sched, panel)
    oracle_cost = _hourly_cost(
        oracle_hourly["counterfactual_price"].to_numpy(),
        oracle_hourly["counterfactual_demand"].to_numpy(),
    )
    obs_cost = _hourly_cost(
        oracle_hourly["observed_price"].to_numpy(),
        oracle_hourly["demand_base"].to_numpy(),
    )
    print(f"  Observed total:       €{obs_cost:,.0f}")
    print(f"  Perfect-info cost:    €{oracle_cost:,.0f}")

    # ── Sweep ────────────────────────────────────────────────────────
    rows = []
    for n in N_GRID:
        print(f"\n── n_scenarios = {n} ──", flush=True)
        scen = generate_bootstrap_scenarios(
            day_ahead, resid, n_scenarios=n,
            block_bootstrap=True, block_length=block_length,
            random_seed=seed,
        )
        sched = optimize_dr_scenario_grid(
            panel, scen,
            flexibility_ratio=flex, objective_mode="average_cost",
            high_price_threshold=p_hist_90,
            use_refit_price_model=True, refit_model_pack=model_pack,
            c_dr=c_dr, penalty_rate=pen,
            low_price_threshold=low_thr,
            low_price_probability_threshold=low_prob,
        )
        hourly = build_hourly_counterfactual_panel(sched, panel)
        cost = _hourly_cost(
            hourly["counterfactual_price"].to_numpy(),
            hourly["counterfactual_demand"].to_numpy(),
        )

        avg_q = float(sched["q_DA_optimal"].mean()) if "q_DA_optimal" in sched.columns else float("nan")
        regret = compute_regret(cost, oracle_cost)
        rows.append({
            "n_scenarios": n,
            "stochastic_DR_cost":  cost,
            "saving_vs_observed":  obs_cost - cost,
            "avg_q_DR":            avg_q,
            "regret_absolute":     regret["regret_absolute"],
            "regret_relative":     regret["regret_relative"],
        })
        print(f"  saving = €{obs_cost - cost:,.0f}  "
              f"regret = €{regret['regret_absolute']:,.0f} ({regret['regret_relative']:.2%})  "
              f"avg q = {avg_q:.0f} MW")

    # ── Append n=1 and n=48 from existing outputs (reuse rather than recompute) ──
    vss_path = tables / "vss_regret_summary.csv"
    if vss_path.exists():
        existing = pd.read_csv(vss_path).iloc[0]
        det_cost = float(existing["deterministic_DR_cost"])
        stoch_cost = float(existing["stochastic_DR_cost"])
        obs_cost_ref = float(existing["observed_total_cost"])
        oracle_cost_ref = float(existing["perfect_information_cost"])
        for n_val, cost_val in [(1, det_cost), (48, stoch_cost)]:
            r_abs = cost_val - oracle_cost_ref
            rows.append({
                "n_scenarios": n_val,
                "stochastic_DR_cost":  cost_val,
                "saving_vs_observed":  obs_cost_ref - cost_val,
                "avg_q_DR":            float("nan"),
                "regret_absolute":     r_abs,
                "regret_relative":     r_abs / (abs(oracle_cost_ref) + 1e-9),
            })

    sweep = pd.DataFrame(rows).sort_values("n_scenarios").reset_index(drop=True)
    # VSS for each n vs the n=1 deterministic case
    if 1 in sweep["n_scenarios"].values:
        det_cost = float(sweep.loc[sweep["n_scenarios"] == 1, "stochastic_DR_cost"].iloc[0])
        sweep["VSS_absolute"] = det_cost - sweep["stochastic_DR_cost"]
        sweep["VSS_relative"] = sweep["VSS_absolute"] / (abs(det_cost) + 1e-9)
    else:
        sweep["VSS_absolute"] = float("nan")
        sweep["VSS_relative"] = float("nan")

    out_csv = tables / "scenario_sweep_summary.csv"
    sweep.to_csv(out_csv, index=False)
    print("\n=== Sweep summary ===")
    cols_show = ["n_scenarios", "saving_vs_observed", "VSS_relative",
                 "regret_relative", "avg_q_DR"]
    print(sweep[cols_show].to_string(index=False))
    print(f"\nSaved to {out_csv}")

    # ── Figure: cost / VSS / regret vs n_scenarios ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(sweep["n_scenarios"], sweep["saving_vs_observed"] / 1e9,
                 "-o", color="#1F3A68")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("n_scenarios (log scale)")
    axes[0].set_ylabel("DR saving (€B)")
    axes[0].set_title("Saving vs scenario count")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(sweep["n_scenarios"], sweep["VSS_relative"] * 100,
                 "-o", color="#E85D04")
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("n_scenarios (log scale)")
    axes[1].set_ylabel("VSS (%)")
    axes[1].set_title("VSS (>0 = scenarios help)")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(sweep["n_scenarios"], sweep["regret_relative"] * 100,
                 "-o", color="#27AE60")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("n_scenarios (log scale)")
    axes[2].set_ylabel("Regret (%)")
    axes[2].set_title("Regret vs perfect info")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "13_scenario_sweep.png", dpi=130)
    plt.close(fig)
    print(f"Figure saved to {figs / '13_scenario_sweep.png'}")
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
