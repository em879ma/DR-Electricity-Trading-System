from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_final_research_summary(
    forecast_level1: pd.DataFrame,
    scenario_level2: pd.DataFrame,
    decision_level3: pd.DataFrame,
    mechanism_level4: pd.DataFrame,
    calibration_summary: pd.DataFrame | None = None,
    output_root: str = "outputs",
) -> None:
    rep_dir = Path(output_root) / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    lines.append("# Final Research Summary: Day-Ahead DR Decision System")
    lines.append("")
    lines.append("## 1. Research Objective")
    lines.append("")
    lines.append(
        "This project is **not** a pure electricity price forecasting model. "
        "It is a **day-ahead demand-response (DR) decision system under forecast uncertainty** "
        "for the German electricity market."
    )
    lines.append("")
    lines.append("The decision pipeline is:")
    lines.append("")
    lines.append("```")
    lines.append("forecast → scenario generation → day-ahead DR decision → counterfactual market outcome")
    lines.append("```")
    lines.append("")

    lines.append("## 2. Data Used")
    lines.append("")
    lines.append("| Dataset | Source | Coverage |")
    lines.append("|---------|--------|----------|")
    lines.append("| OPSD hourly time series | data.open-power-system-data.org | Hourly: load, price, wind, solar |")
    lines.append("| Ember yearly electricity data | ember-energy.org | Annual: generation mix, fuel shares |")
    lines.append("")

    lines.append("## 3. Forecasting Method")
    lines.append("")
    lines.append(
        "**Random Forest** is used as the operational baseline. "
        "It is a reasonable nonlinear baseline for day-ahead price forecasting, "
        "but is **not** claimed to be the global-best model. "
        "Simple baselines (persistence, rolling same-hour mean) are also computed for comparison."
    )
    lines.append("")
    lines.append("Features include: lags, calendar dummies, rolling statistics, rolling elasticity, "
                 "net load, renewable share, scarcity index, oversupply index, and negative-price lag.")
    lines.append("")
    if calibration_summary is not None and not calibration_summary.empty:
        lines.append("### Calibration Summary")
        lines.append("")
        lines.append(calibration_summary.to_markdown(index=False) if hasattr(calibration_summary, "to_markdown") else calibration_summary.to_string(index=False))
        lines.append("")

    lines.append("## 4. Scenario Generation")
    lines.append("")
    lines.append(
        "Bootstrap scenarios are generated around the calibrated RF point forecast. "
        "Residuals from validation data are resampled using **block bootstrap** (block_length=24h, n_scenarios=48). "
        "Each scenario provides a plausible joint path for price, load, and renewable total."
    )
    lines.append("")

    lines.append("## 5. Day-Ahead DR Decision Formulation")
    lines.append("")
    lines.append(
        "For each hour t, the decision variable is `q_DA_t` = committed load reduction. "
        "Constraints: `0 ≤ q_DA_t ≤ flexibility_ratio × demand_base_t` and "
        "`load_after_DR ≥ 0.70 × demand_base_t`."
    )
    lines.append("")
    lines.append("**Low-price/negative-price guardrail** (mandatory):")
    lines.append("- If E[price_t] ≤ P25_historical → q_DA_t = 0")
    lines.append("- If Pr(price_t < 0) ≥ 0.30 → q_DA_t = 0")
    lines.append("")
    lines.append("Three objective modes: `average_cost`, `high_price_only`, `cvar_tail`.")
    lines.append("")
    lines.append(
        "The **optimized price is NOT the RF forecast**. "
        "It is the counterfactual price predicted by the structural price model "
        "after applying feasible DR actions."
    )
    lines.append("")

    lines.append("## 6. Energy-Structure Variables")
    lines.append("")
    lines.append("| Variable | Formula |")
    lines.append("|----------|---------|")
    lines.append("| net_load | load − renewable_total |")
    lines.append("| renewable_share | renewable_total / load |")
    lines.append("| scarcity_index | load / (capacity + imports + ε) |")
    lines.append("| oversupply_index | renewable_total + imports − load |")
    lines.append("| negative_price_dummy | 1(price < 0) |")
    lines.append("")

    lines.append("## 7. Four-Level Evaluation Results")
    lines.append("")

    lines.append("### Level 1: Forecast Accuracy")
    if not forecast_level1.empty:
        lines.append(forecast_level1.to_string(index=False))
    else:
        lines.append("(not yet computed)")
    lines.append("")

    lines.append("### Level 2: Scenario Quality")
    if not scenario_level2.empty:
        lines.append(scenario_level2.to_string(index=False))
    else:
        lines.append("(not yet computed)")
    lines.append("")

    lines.append("### Level 3: Decision Performance")
    if not decision_level3.empty:
        lines.append(decision_level3.to_string(index=False))
    else:
        lines.append("(not yet computed)")
    lines.append("")

    lines.append("### Level 4: Economic Mechanism Validity")
    if not mechanism_level4.empty:
        lines.append(mechanism_level4.to_string(index=False))
    else:
        lines.append("(not yet computed)")
    lines.append("")

    lines.append("## 8. Key Findings")
    lines.append("")
    _append_key_findings(lines, forecast_level1, decision_level3, mechanism_level4)

    lines.append("## 9. Limitations and Next Steps")
    lines.append("")
    lines.append("- **rolling_elasticity** is a rolling OLS slope (correlation feature), not a structural causal elasticity.")
    lines.append("- RF forecast accuracy is not the sole success criterion; decision and mechanism quality matter more.")
    lines.append("- Counterfactual price model uses in-sample fit; out-of-sample cross-validation is recommended.")
    lines.append("- ENTSO-E real-time data (with API token) would improve forecast quality.")
    lines.append("- Extension: multi-period DR with energy balance constraints (rebound effect).")
    lines.append("")

    (rep_dir / "final_research_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {rep_dir / 'final_research_summary.md'}")


def _append_key_findings(
    lines: list[str],
    forecast_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    mechanism_df: pd.DataFrame,
) -> None:
    if not forecast_df.empty and "MAE" in forecast_df.columns:
        cal_rows = forecast_df[forecast_df["model"].str.contains("calibrat", case=False, na=False)]
        if not cal_rows.empty:
            best = cal_rows.sort_values("MAE").iloc[0]
            lines.append(f"- Best calibrated forecast MAE: **{best['MAE']:.2f} €/MWh** (model: {best['model']})")

    if not decision_df.empty:
        if "total_market_cost_reduction" in decision_df.columns:
            cr = float(decision_df["total_market_cost_reduction"].iloc[0])
            lines.append(f"- Total market cost reduction: **{cr:,.0f} €**")
        if "avg_price_reduction" in decision_df.columns:
            pr = float(decision_df["avg_price_reduction"].iloc[0])
            lines.append(f"- Average price reduction: **{pr:.2f} €/MWh**")
        if "high_price_hours_reduced" in decision_df.columns:
            hr = int(decision_df["high_price_hours_reduced"].iloc[0])
            lines.append(f"- High-price hours reduced: **{hr}**")

    if not mechanism_df.empty and "passed" in mechanism_df.columns:
        n_passed = int(mechanism_df["passed"].sum())
        n_total = int(len(mechanism_df))
        lines.append(f"- Economic mechanism checks passed: **{n_passed}/{n_total}**")

    lines.append("")


def build_evaluation_dashboard_table(
    forecast_level1: pd.DataFrame,
    scenario_level2: pd.DataFrame,
    decision_level3: pd.DataFrame,
    mechanism_level4: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if not forecast_level1.empty:
        for _, r in forecast_level1.iterrows():
            for col in ["MAE", "RMSE", "Bias", "R2", "high_price_recall", "negative_price_recall"]:
                if col in r.index and pd.notna(r[col]):
                    rows.append({"level": "L1_forecast", "metric": f"{r.get('model', '')}_{col}", "value": float(r[col])})

    if not scenario_level2.empty:
        for col in ["coverage_10_90", "coverage_05_95", "scenario_bias", "mean_interval_width"]:
            if col in scenario_level2.columns:
                rows.append({"level": "L2_scenario", "metric": col, "value": float(scenario_level2[col].iloc[0])})

    if not decision_level3.empty:
        for col in ["total_market_cost_reduction", "avg_price_reduction", "high_price_hours_reduced", "constraint_violation_rate"]:
            if col in decision_level3.columns:
                rows.append({"level": "L3_decision", "metric": col, "value": float(decision_level3[col].iloc[0])})

    if not mechanism_level4.empty and "passed" in mechanism_level4.columns:
        n_passed = int(mechanism_level4["passed"].sum())
        n_total = int(len(mechanism_level4))
        rows.append({"level": "L4_mechanism", "metric": "checks_passed", "value": float(n_passed)})
        rows.append({"level": "L4_mechanism", "metric": "checks_total", "value": float(n_total)})
        rows.append({"level": "L4_mechanism", "metric": "pass_rate", "value": float(n_passed / max(n_total, 1))})

    return pd.DataFrame(rows)
