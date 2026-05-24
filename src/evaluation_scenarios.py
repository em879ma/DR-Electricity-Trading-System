from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_scenario_quality(
    scenarios_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    target_col: str = "price_scenario",
    actual_col: str = "price",
    datetime_col: str = "target_datetime",
) -> pd.DataFrame:
    """
    Level 2: Scenario (uncertainty) quality evaluation.

    Metrics:
    - coverage_10_90: fraction of actuals within P10-P90 scenario band
    - coverage_05_95: fraction of actuals within P05-P95 scenario band
    - mean_interval_width: average P10-P90 width
    - high_price_tail_coverage: coverage of actual high-price (>=P90) hours
    - negative_price_tail_coverage: coverage of actual negative-price hours
    - scenario_bias: mean(scenario_median - actual)
    """
    scen = scenarios_df.copy()
    scen[datetime_col] = pd.to_datetime(scen[datetime_col])
    actual = actual_df.copy()
    actual_dt_col = "datetime" if "datetime" in actual.columns else "target_datetime"
    actual[actual_dt_col] = pd.to_datetime(actual[actual_dt_col])

    quantiles = scen.groupby(datetime_col)[target_col].quantile([0.05, 0.10, 0.50, 0.90, 0.95]).unstack()
    quantiles.columns = ["p05", "p10", "p50", "p90", "p95"]
    quantiles = quantiles.reset_index()

    merged = quantiles.merge(
        actual[[actual_dt_col, actual_col]].rename(columns={actual_dt_col: datetime_col, actual_col: "actual"}),
        on=datetime_col,
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    act = merged["actual"]
    p90_actual = float(act.quantile(0.9))

    within_1090 = ((act >= merged["p10"]) & (act <= merged["p90"])).mean()
    within_0595 = ((act >= merged["p05"]) & (act <= merged["p95"])).mean()
    interval_width = float((merged["p90"] - merged["p10"]).mean())
    high_mask = act >= p90_actual
    high_cov = float(((act[high_mask] >= merged.loc[high_mask, "p10"]) & (act[high_mask] <= merged.loc[high_mask, "p90"])).mean()) if high_mask.any() else float("nan")
    neg_mask = act < 0
    neg_cov = float(((act[neg_mask] >= merged.loc[neg_mask, "p05"]) & (act[neg_mask] <= merged.loc[neg_mask, "p95"])).mean()) if neg_mask.any() else float("nan")
    scenario_bias = float((merged["p50"] - act).mean())

    return pd.DataFrame([{
        "coverage_10_90": float(within_1090),
        "coverage_05_95": float(within_0595),
        "mean_interval_width": interval_width,
        "high_price_tail_coverage": high_cov,
        "negative_price_tail_coverage": neg_cov,
        "scenario_bias": scenario_bias,
        "n_matched_hours": int(len(merged)),
    }])
