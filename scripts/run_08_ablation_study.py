"""
Run 08: Ablation study for the DR decision pipeline.

Six ablations against the XGBoost + block-bootstrap baseline:
  1. no_scenarios      — degenerate 1-scenario (point forecast)
  2. iid_bootstrap     — IID vs 24h block bootstrap
  3. no_holiday        — remove holiday / Christmas features
  4. no_sincos         — replace cyclic hour encoding with raw integer
  5. no_elasticity     — remove rolling_elasticity feature
  6. no_guardrail      — disable negative-price guardrail

Outputs:
  outputs/ablation/tables/ablation_summary.csv
  outputs/ablation/figures/08a_forecast_ablations.png
  outputs/ablation/figures/08b_scenario_ablation.png
  outputs/ablation/figures/08c_decision_ablations.png
  outputs/ablation/figures/08d_ablation_heatmap.png
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ablation import (
    ablate_no_holiday,
    ablate_no_sincos,
    ablate_no_elasticity,
    ablate_no_scenarios,
    ablate_iid_bootstrap,
    ablate_no_guardrail,
)
from src.scenarios import build_residual_history
from src.evaluation_scenarios import evaluate_scenario_quality
from src.evaluation_decision import evaluate_decision_performance, build_hourly_counterfactual_panel

STYLE = {
    "baseline": "#2c7bb6",
    "ablated":  "#d7191c",
    "neutral":  "#fdae61",
}


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bar_compare(ax, labels, baseline_vals, ablated_vals, title, ylabel, fmt=".2f"):
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, baseline_vals, w, label="Baseline", color=STYLE["baseline"], alpha=0.85)
    b2 = ax.bar(x + w / 2, ablated_vals,  w, label="Ablated",  color=STYLE["ablated"],  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                f"{h:{fmt}}", ha="center", va="bottom", fontsize=7)


# ──────────────────────────────────────────────────────────────────────────────
# Figure functions
# ──────────────────────────────────────────────────────────────────────────────

def plot_forecast_ablations(results: dict, baseline: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Forecast Ablations — L1 Metrics (H=24, price)", fontsize=12, fontweight="bold")

    ablation_keys = ["no_holiday", "no_sincos", "no_elasticity"]
    labels = ["No Holiday\nFeatures", "No Sin/Cos\nEncoding", "No Elasticity\nFeature"]

    for i, (metric, ylabel) in enumerate([
        ("mae",  "MAE (€/MWh)"),
        ("rmse", "RMSE (€/MWh)"),
        ("r2",   "R²"),
    ]):
        bl_vals = [baseline[metric]] * len(ablation_keys)
        ab_vals = [results[k][metric] for k in ablation_keys]
        _bar_compare(axes[i], labels, bl_vals, ab_vals, ylabel, ylabel, fmt=".3f")

    fig.tight_layout()
    fig.savefig(out_dir / "08a_forecast_ablations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 08a_forecast_ablations.png")


def plot_scenario_ablation(
    baseline_l2: dict, iid_l2: dict, out_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Ablation 2: Block Bootstrap vs IID Bootstrap — L2 Scenario Quality",
                 fontsize=12, fontweight="bold")

    coverage_metrics = ["coverage_10_90", "coverage_05_95",
                        "high_price_tail_coverage", "neg_price_tail_coverage"]
    cov_labels = ["P10–P90\nCoverage", "P5–P95\nCoverage",
                  "High-Price\nTail Coverage", "Neg-Price\nTail Coverage"]
    bl_cov = [baseline_l2[m] for m in coverage_metrics]
    iid_cov = [iid_l2[m] for m in coverage_metrics]
    _bar_compare(axes[0], cov_labels, bl_cov, iid_cov,
                 "Coverage Metrics", "Fraction of Hours", fmt=".3f")
    axes[0].axhline(0.80, color="gray", linestyle="--", linewidth=0.8, label="80 % target")

    width_metrics = ["mean_interval_width", "scenario_bias"]
    wl = ["Interval Width\n(€/MWh)", "Scenario Bias\n(€/MWh)"]
    bl_w = [baseline_l2[m] for m in width_metrics]
    iid_w = [iid_l2[m] for m in width_metrics]
    _bar_compare(axes[1], wl, bl_w, iid_w,
                 "Width & Bias", "€/MWh", fmt=".2f")

    fig.tight_layout()
    fig.savefig(out_dir / "08b_scenario_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 08b_scenario_ablation.png")


def plot_decision_ablations(results: dict, baseline_l3: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Decision Ablations — L3 Metrics (flex=10 %)", fontsize=12, fontweight="bold")

    ablation_keys = ["no_scenarios", "iid_bootstrap", "no_guardrail"]
    labels = ["No Scenarios\n(point forecast)", "IID Bootstrap\n(no blocks)", "No Guardrail\n(neg-price)"]

    metrics = [
        ("avg_price_reduction",  "Avg Price Reduction (€/MWh)", ".2f"),
        ("total_cost_reduction", "Total Cost Reduction (€)",    ".0f"),
        ("realized_profit",      "Realized Profit (€)",         ".0f"),
    ]
    for i, (metric, ylabel, fmt) in enumerate(metrics):
        bl_v = [baseline_l3[metric]] * len(ablation_keys)
        ab_v = [results[k][metric] for k in ablation_keys]
        _bar_compare(axes[i], labels, bl_v, ab_v, ylabel, ylabel, fmt=fmt)

    fig.tight_layout()
    fig.savefig(out_dir / "08c_decision_ablations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 08c_decision_ablations.png")


def plot_ablation_heatmap(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Normalised % change from baseline for all ablations and metrics."""
    hm = summary_df.set_index("ablation").drop(columns=["description"], errors="ignore")
    # Keep only delta_pct columns
    pct_cols = [c for c in hm.columns if c.endswith("_delta_pct")]
    if not pct_cols:
        return
    hm = hm[pct_cols].rename(columns=lambda c: c.replace("_delta_pct", ""))
    hm = hm.astype(float)

    fig, ax = plt.subplots(figsize=(max(8, len(pct_cols) * 1.1), max(4, len(hm) * 0.8)))
    vmax = max(abs(hm.values[np.isfinite(hm.values)]).max(), 1)
    im = ax.imshow(hm.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(hm.columns)))
    ax.set_xticklabels(hm.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(hm.index)))
    ax.set_yticklabels(hm.index, fontsize=9)
    for r in range(len(hm.index)):
        for c in range(len(hm.columns)):
            v = hm.values[r, c]
            txt = f"{v:+.1f}%" if np.isfinite(v) else "n/a"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=7.5, color="black")
    plt.colorbar(im, ax=ax, label="% change from baseline (positive = worse for lower-is-better metrics)")
    ax.set_title("Ablation Study — % Change from Baseline", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "08d_ablation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 08d_ablation_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["output"]["root"]) / "ablation"
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    tables_dir = Path(cfg["output"]["root"]) / "tables"
    models_dir = Path(cfg["output"]["root"]) / "models"

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    dr_df  = pd.read_csv("data/processed/market_panel_with_dr.csv",
                         parse_dates=["datetime"])
    rf_cal = pd.read_csv(tables_dir / "rf_calibrated_forecasts.csv",
                         parse_dates=["target_datetime", "forecast_origin_datetime"])
    boot   = pd.read_csv(
        Path(cfg["output"]["root"]) / "scenarios" / "bootstrap_scenarios.csv",
        parse_dates=["target_datetime"],
    )
    da_rf  = pd.read_csv(tables_dir / "day_ahead_rf_forecasts.csv",
                         parse_dates=["target_datetime", "forecast_origin_datetime"])
    sched_base = pd.read_csv(tables_dir / "day_ahead_optimal_dr_schedule.csv",
                             parse_dates=["target_datetime"])
    with (models_dir / "counterfactual_price_model.pkl").open("rb") as f:
        model_pack = pickle.load(f)

    split     = cfg["forecast"]["split"]
    rf_params = cfg["forecast"]["rf"]
    flex      = float(cfg["decision"]["flexibility_ratios"][1])
    c_dr      = float(cfg["decision"].get("c_dr", 3.0))
    pen       = float(cfg["decision"].get("penalty_rate", 20.0))

    # ── Build residual history for scenario re-generation ─────────────────────
    price_cal = rf_cal[rf_cal["target"] == "price"].copy()
    price_cal["forecast"] = price_cal["forecast_calibrated"]
    resid_df = build_residual_history(rf_cal.assign(
        forecast=rf_cal["forecast_calibrated"]))

    # ── Baseline metrics ───────────────────────────────────────────────────────
    print("Computing baseline metrics...")
    # L1: read from existing accuracy CSV (calibrated XGBoost)
    acc_path = tables_dir / "forecast_accuracy_level1_calibrated.csv"
    if acc_path.exists():
        acc_df = pd.read_csv(acc_path)
        xgb_row = acc_df[
            (acc_df["model"].str.contains("xgboost", case=False)) &
            (acc_df["target"] == "price") &
            (acc_df["horizon"] == 24)
        ]
        baseline_l1 = xgb_row.iloc[0].to_dict() if not xgb_row.empty else {}
    else:
        # Re-compute from scratch using the full pipeline baseline
        baseline_l1 = {}

    # Fall back: re-run XGBoost with full features to get fresh L1 numbers
    if not baseline_l1 or "mae" not in baseline_l1:
        from src.ablation import _xgb_l1
        print("  (re-computing baseline L1 from XGBoost)")
        baseline_l1 = _xgb_l1(dr_df, split, rf_params)

    # L2: baseline scenario quality
    bl2 = evaluate_scenario_quality(boot, dr_df)
    baseline_l2 = {
        "coverage_10_90":           float(bl2["coverage_10_90"].iloc[0]),
        "coverage_05_95":           float(bl2["coverage_05_95"].iloc[0]),
        "mean_interval_width":      float(bl2["mean_interval_width"].iloc[0]),
        "high_price_tail_coverage": float(bl2["high_price_tail_coverage"].iloc[0]),
        "neg_price_tail_coverage":  float(bl2["negative_price_tail_coverage"].iloc[0]),
        "scenario_bias":            float(bl2["scenario_bias"].iloc[0]),
    }

    # L3: baseline decision performance
    hourly_base = build_hourly_counterfactual_panel(sched_base, dr_df)
    l3_base_df  = evaluate_decision_performance(sched_base, dr_df,
                                                flexibility_ratio=flex,
                                                c_dr=c_dr, penalty_rate=pen)
    obs_cost = float((hourly_base["observed_price"] * hourly_base["demand_base"]).sum())
    cf_cost  = float((hourly_base["counterfactual_price"] * hourly_base["counterfactual_demand"]).sum())
    baseline_l3 = {
        "avg_price_reduction":      float(hourly_base["price_reduction"].mean()),
        "total_cost_reduction":     obs_cost - cf_cost,
        "high_price_hours_reduced": int(l3_base_df["high_price_hours_reduced"].iloc[0]),
        "DR_utilization_high":      float(l3_base_df["DR_utilization_high_price"].iloc[0]),
        "DR_utilization_low":       float(l3_base_df["DR_utilization_low_price"].iloc[0]),
        "realized_profit":          float(l3_base_df["realized_profit"].iloc[0]),
        "n_neg_price_dr":           0,
    }

    # ── Run ablations ──────────────────────────────────────────────────────────
    results: dict[str, dict] = {}

    print("\n[1/6] Ablation: no_scenarios (degenerate point-forecast scenarios)...")
    results["no_scenarios"] = ablate_no_scenarios(da_rf, dr_df, model_pack, cfg)
    print(f"      avg_price_reduction: {results['no_scenarios']['avg_price_reduction']:.2f} "
          f"(baseline {baseline_l3['avg_price_reduction']:.2f})")

    print("[2/6] Ablation: iid_bootstrap ...")
    results["iid_bootstrap"] = ablate_iid_bootstrap(da_rf, resid_df, dr_df, model_pack, cfg)
    print(f"      coverage_10_90: {results['iid_bootstrap']['coverage_10_90']:.3f} "
          f"(baseline {baseline_l2['coverage_10_90']:.3f})")

    print("[3/6] Ablation: no_holiday ...")
    results["no_holiday"] = ablate_no_holiday(dr_df, split, rf_params)
    print(f"      MAE: {results['no_holiday']['mae']:.2f} (baseline {baseline_l1.get('mae', float('nan')):.2f})")

    print("[4/6] Ablation: no_sincos ...")
    results["no_sincos"] = ablate_no_sincos(dr_df, split, rf_params)
    print(f"      MAE: {results['no_sincos']['mae']:.2f} (baseline {baseline_l1.get('mae', float('nan')):.2f})")

    print("[5/6] Ablation: no_elasticity ...")
    results["no_elasticity"] = ablate_no_elasticity(dr_df, split, rf_params)
    print(f"      MAE: {results['no_elasticity']['mae']:.2f} (baseline {baseline_l1.get('mae', float('nan')):.2f})")

    print("[6/6] Ablation: no_guardrail ...")
    results["no_guardrail"] = ablate_no_guardrail(boot, dr_df, model_pack, cfg)
    print(f"      avg_price_reduction: {results['no_guardrail']['avg_price_reduction']:.2f}, "
          f"neg-price DR hours: {results['no_guardrail']['n_neg_price_dr']}")

    # ── Build summary table ────────────────────────────────────────────────────
    print("\nBuilding summary table...")
    descriptions = {
        "no_scenarios":   "Replace 48-scenario fan with point forecast",
        "iid_bootstrap":  "IID residual bootstrap (no 24h blocks)",
        "no_holiday":     "Remove holiday / Christmas / neg-price-dummy features",
        "no_sincos":      "Replace hour_sin/cos with raw integer hour",
        "no_elasticity":  "Remove rolling_elasticity feature",
        "no_guardrail":   "Disable negative-price guardrail in DR optimizer",
    }

    # Combine baselines by metric type
    combined_baseline = {**baseline_l1, **baseline_l2, **baseline_l3}

    rows = []
    for key, desc in descriptions.items():
        row = {"ablation": key, "description": desc}
        m = results[key]
        for k, v in m.items():
            row[k] = v
            bl = combined_baseline.get(k)
            if bl is not None and isinstance(v, float) and np.isfinite(v) and isinstance(bl, (int, float)) and bl != 0:
                row[f"{k}_delta_pct"] = (v - bl) / abs(bl) * 100
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "tables" / "ablation_summary.csv", index=False)

    # ── Generate figures ───────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig_dir = out_dir / "figures"

    # Ensure baseline_l1 has the keys we need
    _bl1 = {
        "mae":  float(baseline_l1.get("mae",  float("nan"))),
        "rmse": float(baseline_l1.get("rmse", float("nan"))),
        "r2":   float(baseline_l1.get("r2",   float("nan"))),
    }
    plot_forecast_ablations(results, _bl1, fig_dir)

    iid_l2 = {k: results["iid_bootstrap"][k] for k in baseline_l2 if k in results["iid_bootstrap"]}
    plot_scenario_ablation(baseline_l2, iid_l2, fig_dir)

    combined_bl3 = {**baseline_l3}
    plot_decision_ablations(results, combined_bl3, fig_dir)

    plot_ablation_heatmap(summary_df, fig_dir)

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ABLATION STUDY RESULTS")
    print("=" * 72)

    print("\n--- Forecast Ablations (L1: MAE / RMSE / R²) ---")
    for k in ["no_holiday", "no_sincos", "no_elasticity"]:
        m = results[k]
        bl_mae = _bl1["mae"]
        print(f"  {k:20s}  MAE={m['mae']:.2f} ({m['mae']-bl_mae:+.2f})  "
              f"RMSE={m['rmse']:.2f}  R²={m['r2']:.3f}  "
              f"n_features={m['n_features']}")

    print("\n--- Scenario Ablations (L2) ---")
    print(f"  {'baseline (block)':20s}  coverage_10_90={baseline_l2['coverage_10_90']:.3f}  "
          f"width={baseline_l2['mean_interval_width']:.1f}  "
          f"bias={baseline_l2['scenario_bias']:.2f}")
    r = results["iid_bootstrap"]
    print(f"  {'iid_bootstrap':20s}  coverage_10_90={r['coverage_10_90']:.3f}  "
          f"width={r['mean_interval_width']:.1f}  bias={r['scenario_bias']:.2f}")

    print("\n--- Decision Ablations (L3) ---")
    bl3 = baseline_l3
    print(f"  {'baseline':20s}  price_reduction={bl3['avg_price_reduction']:.2f}  "
          f"cost_reduction={bl3['total_cost_reduction']/1e6:.1f}M  "
          f"profit={bl3['realized_profit']/1e6:.1f}M  "
          f"neg_DR={bl3['n_neg_price_dr']}")
    for k in ["no_scenarios", "iid_bootstrap", "no_guardrail"]:
        r = results[k]
        print(f"  {k:20s}  price_reduction={r['avg_price_reduction']:.2f}  "
              f"cost_reduction={r['total_cost_reduction']/1e6:.1f}M  "
              f"profit={r['realized_profit']/1e6:.1f}M  "
              f"neg_DR={r['n_neg_price_dr']}")

    print(f"\nOutputs saved to {out_dir}/")
    print("  tables/ablation_summary.csv")
    print("  figures/08a_forecast_ablations.png")
    print("  figures/08b_scenario_ablation.png")
    print("  figures/08c_decision_ablations.png")
    print("  figures/08d_ablation_heatmap.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
