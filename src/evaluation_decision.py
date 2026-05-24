from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def evaluate_decision_performance(
    optimal_schedule: pd.DataFrame,
    actual_panel: pd.DataFrame,
    flexibility_ratio: float = 0.10,
    c_dr: float = 3.0,
    penalty_rate: float = 20.0,
) -> pd.DataFrame:
    """
    Level 3: Decision performance evaluation.

    Metrics: realized_profit, expected_profit, total_market_cost_reduction,
    avg_price_reduction, high_price_hours_reduced, penalty,
    constraint_violation_rate, DR_utilization_extreme_price, DR_utilization_low_price.
    """
    sched = optimal_schedule.copy()
    dt_col = "target_datetime" if "target_datetime" in sched.columns else "datetime"
    sched[dt_col] = pd.to_datetime(sched[dt_col])

    act = actual_panel.copy()
    act_dt_col = "datetime" if "datetime" in act.columns else "timestamp"
    act[act_dt_col] = pd.to_datetime(act[act_dt_col])

    merged = sched.merge(
        act[[act_dt_col, "price", "demand_base"]].rename(columns={act_dt_col: dt_col}),
        on=dt_col,
        how="left",
        suffixes=("", "_actual"),
    )

    q = merged["q_DA_optimal"] if "q_DA_optimal" in merged.columns else merged.get("q_DA", pd.Series(0.0, index=merged.index))
    price_actual = merged.get("price_actual", merged.get("price"))
    db = merged.get("demand_base_actual", merged.get("demand_base"))
    cf_price = merged.get("expected_counterfactual_price", price_actual)

    max_flex = flexibility_ratio * db
    realized_revenue = float((price_actual * q).sum())
    realized_dr_cost = float(c_dr * q.sum())
    realized_penalty = float((penalty_rate * np.clip(q - max_flex, 0.0, None)).sum())
    realized_profit = realized_revenue - realized_dr_cost - realized_penalty

    obs_cost = float((price_actual * db).sum())
    cf_cost = float((cf_price * (db - q)).sum())
    cost_reduction = obs_cost - cf_cost

    high_thr = float(price_actual.quantile(0.9))
    low_thr = float(price_actual.quantile(0.25))
    high_mask = price_actual >= high_thr
    low_mask = price_actual <= low_thr

    high_hours_obs = int(high_mask.sum())
    high_hours_cf = int((cf_price >= high_thr).sum())

    constraint_ok = (
        (q >= 0) &
        (q <= max_flex + 1e-9) &
        ((db - q) >= 0.70 * db - 1e-9)
    )
    violation_rate = float((~constraint_ok).mean())

    util_high = float(q[high_mask].sum() / (max_flex[high_mask].sum() + 1e-9)) if high_mask.any() else float("nan")
    util_low = float(q[low_mask].sum() / (max_flex[low_mask].sum() + 1e-9)) if low_mask.any() else float("nan")

    expected_profit = float(merged.get("expected_profit", q * 0.0).sum()) if "expected_profit" in merged.columns else float("nan")

    return pd.DataFrame([{
        "realized_profit": realized_profit,
        "expected_profit": expected_profit,
        "total_market_cost_reduction": cost_reduction,
        "avg_price_reduction": float((price_actual - cf_price).mean()),
        "high_price_hours_observed": high_hours_obs,
        "high_price_hours_counterfactual": high_hours_cf,
        "high_price_hours_reduced": high_hours_obs - high_hours_cf,
        "total_penalty": realized_penalty,
        "constraint_violation_rate": violation_rate,
        "DR_utilization_high_price": util_high,
        "DR_utilization_low_price": util_low,
        "flexibility_ratio": flexibility_ratio,
    }])


def evaluate_constraints(
    schedule_df: pd.DataFrame,
    min_load_ratio: float = 0.70,
) -> pd.DataFrame:
    s = schedule_df.copy()
    q_col = "q_DA_optimal" if "q_DA_optimal" in s.columns else "q_DA"
    min_allowed = min_load_ratio * s["demand_base"]
    checks = [
        ("q_non_negative", s[q_col] >= 0.0),
        ("q_within_max_flexibility", s[q_col] <= s["max_flexibility"] + 1e-9),
        ("counterfactual_load_minimum", s["counterfactual_demand"] >= min_allowed - 1e-9),
    ]
    rows = []
    for name, passed_mask in checks:
        total = int(len(passed_mask))
        violations = int((~passed_mask).sum())
        rows.append({
            "constraint_name": name,
            "violations": violations,
            "total_checks": total,
            "violation_rate": (violations / total) if total else float("nan"),
            "passed": violations == 0,
        })
    return pd.DataFrame(rows)


def build_hourly_counterfactual_panel(
    optimal_schedule: pd.DataFrame,
    actual_panel: pd.DataFrame,
) -> pd.DataFrame:
    sched = optimal_schedule.copy()
    dt_col = "target_datetime" if "target_datetime" in sched.columns else "datetime"
    act = actual_panel.copy()
    act_dt = "datetime" if "datetime" in act.columns else "timestamp"

    merged = sched.merge(
        act[[act_dt, "price"]].rename(columns={act_dt: dt_col}),
        on=dt_col, how="left",
    )
    return pd.DataFrame({
        "datetime": merged[dt_col],
        "observed_price": merged["price"],
        "counterfactual_price": merged["expected_counterfactual_price"],
        "price_reduction": merged["price"] - merged["expected_counterfactual_price"],
        "demand_base": merged["demand_base"],
        "q_DA_optimal": merged["q_DA_optimal"],
        "counterfactual_demand": merged["counterfactual_demand"],
        "max_flexibility": merged["max_flexibility"],
        "constraint_feasible": merged.get("constraint_feasible", pd.Series(True, index=merged.index)),
        "low_price_constraint_triggered": merged.get("low_price_constraint_triggered", pd.Series(False, index=merged.index)),
    })


def dr_allocation_by_regime(hourly: pd.DataFrame, price_col: str = "observed_price") -> pd.DataFrame:
    p = hourly[price_col]
    p50, p75, p90 = float(p.quantile(0.5)), float(p.quantile(0.75)), float(p.quantile(0.9))

    def _regime(v: float) -> str:
        if v < p50:
            return "low"
        if v < p75:
            return "medium"
        if v < p90:
            return "high"
        return "extreme"

    h = hourly.copy()
    h["regime"] = h[price_col].map(_regime)
    rows = []
    for reg in ["low", "medium", "high", "extreme"]:
        sub = h[h["regime"] == reg]
        if sub.empty:
            rows.append({"regime": reg, "number_of_hours": 0, "avg_q_DA": float("nan"),
                         "utilization_rate": float("nan"), "avg_price_reduction": float("nan")})
            continue
        tq = float(sub["q_DA_optimal"].sum())
        tmf = float(sub["max_flexibility"].sum())
        rows.append({
            "regime": reg,
            "number_of_hours": int(len(sub)),
            "avg_q_DA": float(sub["q_DA_optimal"].mean()),
            "total_q_DA": tq,
            "avg_max_flexibility": float(sub["max_flexibility"].mean()),
            "utilization_rate": float(tq / (tmf + 1e-9)),
            "avg_price_reduction": float(sub["price_reduction"].mean()),
        })
    return pd.DataFrame(rows)
