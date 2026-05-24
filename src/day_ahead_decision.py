from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from .counterfactual_price_model import (
    predict_counterfactual_price,
    predict_negative_price_probability,
    update_counterfactual_row,
)


def build_rule_based_day_ahead_schedule(
    base_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    flexibility_ratio: float = 0.10,
    alpha: float = 0.5,
    high_price_quantile: float = 0.75,
    c_dr: float = 3.0,
    penalty_rate: float = 20.0,
    forecast_type: str = "point",
    method: str = "rolling_mean",
) -> pd.DataFrame:
    fdf = forecast_df.copy()
    panel = base_df[["timestamp", "demand_base", "load", "renewable_total"]].rename(columns={"timestamp": "datetime"})
    out = (
        fdf.merge(panel, left_on="target_datetime", right_on="datetime", how="left")
        .drop(columns=["datetime"])
        .copy()
    )
    out = out.sort_values("target_datetime").tail(24).reset_index(drop=True)
    out["max_flexibility"] = flexibility_ratio * out["demand_base"]
    threshold = out["price_forecast"].quantile(high_price_quantile)
    out["q_DA"] = np.where(out["price_forecast"] >= threshold, alpha * out["max_flexibility"], 0.0)
    out["q_DA"] = out["q_DA"].clip(lower=0.0, upper=out["max_flexibility"])

    out["actual_available_response"] = np.clip(out["demand_base"] - out["load"], 0.0, out["max_flexibility"])
    out["expected_revenue"] = out["price_forecast"] * out["q_DA"]
    out["expected_cost"] = c_dr * out["q_DA"]
    shortfall = np.clip(out["q_DA"] - out["actual_available_response"], 0.0, None)
    out["expected_penalty"] = penalty_rate * shortfall
    out["expected_profit"] = out["expected_revenue"] - out["expected_cost"] - out["expected_penalty"]
    out["forecast_type"] = forecast_type
    out["method"] = method

    return out[
        [
            "target_datetime",
            "price_forecast",
            "load_forecast",
            "renewable_forecast",
            "max_flexibility",
            "q_DA",
            "expected_revenue",
            "expected_cost",
            "expected_penalty",
            "expected_profit",
            "forecast_type",
            "method",
        ]
    ].rename(columns={"target_datetime": "datetime", "price_forecast": "forecasted_price", "load_forecast": "forecasted_load", "renewable_forecast": "forecasted_renewable"})


def build_scenario_based_day_ahead_schedule(
    base_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    flexibility_ratio: float = 0.10,
    alpha: float = 0.5,
    high_price_quantile: float = 0.75,
    c_dr: float = 3.0,
    penalty_rate: float = 20.0,
    method: str = "bootstrap",
) -> pd.DataFrame:
    sdf = scenarios_df.copy()
    sdf["target_datetime"] = pd.to_datetime(sdf["target_datetime"])
    expected = (
        sdf.assign(
            wp=lambda d: d["price_scenario"] * d["scenario_probability"],
            wl=lambda d: d["load_scenario"] * d["scenario_probability"],
            wr=lambda d: d["renewable_scenario"] * d["scenario_probability"],
        )
        .groupby("target_datetime", as_index=False)[["wp", "wl", "wr"]]
        .sum()
        .rename(columns={"wp": "price_forecast", "wl": "load_forecast", "wr": "renewable_forecast"})
    )

    schedule = build_rule_based_day_ahead_schedule(
        base_df=base_df,
        forecast_df=expected,
        flexibility_ratio=flexibility_ratio,
        alpha=alpha,
        high_price_quantile=high_price_quantile,
        c_dr=c_dr,
        penalty_rate=penalty_rate,
        forecast_type="scenario",
        method=method,
    )

    # Replace point expected values with probability-weighted expected values from scenarios.
    sched = schedule.rename(columns={"datetime": "target_datetime"}).copy()
    merged = sdf.merge(sched[["target_datetime", "q_DA"]], on="target_datetime", how="inner")
    merged["max_flexibility_scenario"] = flexibility_ratio * merged["load_scenario"]
    merged["actual_available_response"] = np.clip(0.8 * merged["max_flexibility_scenario"], 0.0, None)
    merged["revenue"] = merged["price_scenario"] * merged["q_DA"]
    merged["cost"] = c_dr * merged["q_DA"]
    merged["penalty"] = penalty_rate * np.clip(merged["q_DA"] - merged["actual_available_response"], 0.0, None)
    merged["profit"] = merged["revenue"] - merged["cost"] - merged["penalty"]

    agg = merged.groupby("target_datetime", as_index=False).agg(
        expected_revenue=("revenue", lambda s: float(np.sum(s * merged.loc[s.index, "scenario_probability"]))),
        expected_cost=("cost", lambda s: float(np.sum(s * merged.loc[s.index, "scenario_probability"]))),
        expected_penalty=("penalty", lambda s: float(np.sum(s * merged.loc[s.index, "scenario_probability"]))),
        expected_profit=("profit", lambda s: float(np.sum(s * merged.loc[s.index, "scenario_probability"]))),
    )
    out = sched.merge(agg, on="target_datetime", how="left", suffixes=("", "_scenario"))
    for col in ["expected_revenue", "expected_cost", "expected_penalty", "expected_profit"]:
        if f"{col}_scenario" in out.columns:
            out[col] = out[f"{col}_scenario"]
            out = out.drop(columns=[f"{col}_scenario"])
    out["forecast_type"] = "scenario"
    out["method"] = method
    return out.rename(columns={"target_datetime": "datetime"})


def summarize_day_ahead_schedule(schedule_df: pd.DataFrame) -> pd.DataFrame:
    peak_idx = schedule_df["forecasted_price"].idxmax()
    return pd.DataFrame(
        [
            {
                "forecast_type": schedule_df["forecast_type"].iloc[0],
                "method": schedule_df["method"].iloc[0],
                "total_committed_DR": float(schedule_df["q_DA"].sum()),
                "avg_committed_DR": float(schedule_df["q_DA"].mean()),
                "peak_hour_committed_DR": float(schedule_df.loc[peak_idx, "q_DA"]),
                "expected_total_revenue": float(schedule_df["expected_revenue"].sum()),
                "expected_total_cost": float(schedule_df["expected_cost"].sum()),
                "expected_total_penalty": float(schedule_df["expected_penalty"].sum()),
                "expected_total_profit": float(schedule_df["expected_profit"].sum()),
            }
        ]
    )


def build_forecast_decision_comparison(
    point_schedule: pd.DataFrame,
    scenario_schedule: pd.DataFrame,
    actual_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
) -> pd.DataFrame:
    point = _compare_row(point_schedule, actual_df, accuracy_df, "point_forecast")
    scenario = _compare_row(scenario_schedule, actual_df, accuracy_df, "scenario_forecast")
    return pd.DataFrame([point, scenario])


def write_day_ahead_outputs(
    bid_schedule_df: pd.DataFrame,
    decision_summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    output_root: str = "outputs",
) -> None:
    tables_dir = Path(output_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    bid_schedule_df.to_csv(tables_dir / "day_ahead_bid_schedule.csv", index=False)
    decision_summary_df.to_csv(tables_dir / "day_ahead_decision_summary.csv", index=False)
    comparison_df.to_csv(tables_dir / "day_ahead_forecast_decision_comparison.csv", index=False)


def _compare_row(
    schedule: pd.DataFrame,
    actual_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    forecast_type_label: str,
) -> dict[str, float | str]:
    actual = actual_df[["timestamp", "price"]].rename(columns={"timestamp": "datetime"})
    merged = schedule.merge(actual, on="datetime", how="left")
    realized_revenue = float(np.sum(merged["price"] * merged["q_DA"]))
    realized_profit = float(realized_revenue - merged["expected_cost"].sum() - merged["expected_penalty"].sum())
    high_thr = float(actual_df["price"].quantile(0.9))
    high_covered = int((merged.loc[merged["price"] >= high_thr, "q_DA"] > 0).sum())
    acc = accuracy_df[accuracy_df["target"] == "price"]
    return {
        "method": str(schedule["method"].iloc[0]),
        "forecast_type": forecast_type_label,
        "total_committed_DR": float(schedule["q_DA"].sum()),
        "expected_profit": float(schedule["expected_profit"].sum()),
        "realized_profit": realized_profit,
        "total_penalty": float(schedule["expected_penalty"].sum()),
        "high_price_hours_covered": float(high_covered),
        "avg_forecast_error": float(acc["mae"].mean()) if len(acc) else float("nan"),
        "rmse_forecast_error": float(acc["rmse"].mean()) if len(acc) else float("nan"),
    }


def _scenario_price_objective(
    p_cf: np.ndarray,
    w: np.ndarray,
    price_scenario: np.ndarray,
    objective_mode: str,
    high_price_threshold: float,
    cvar_alpha: float,
) -> float:
    # Supported modes: average_cost, high_price_only, cvar_tail
    if objective_mode in ("average_cost", "average_price"):
        return float(np.sum(w * p_cf))
    if objective_mode == "high_price_only":
        m = price_scenario >= high_price_threshold
        if not np.any(m):
            return 0.0
        return float(np.sum(w[m] * p_cf[m]))
    if objective_mode in ("cvar_tail", "cvar_price"):
        q = float(np.quantile(p_cf, cvar_alpha))
        tail = p_cf[p_cf >= q]
        return float(np.mean(tail)) if len(tail) else float(np.mean(p_cf))
    raise ValueError(f"unknown objective_mode={objective_mode}")


def optimize_dr_scenario_grid(
    base_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    *,
    flexibility_ratio: float,
    objective_mode: str,
    high_price_threshold: float,
    cvar_alpha: float = 0.90,
    use_refit_price_model: bool = False,
    price_model_rf=None,
    refit_model_pack: dict | None = None,
    demand_col_for_price_model: str = "demand_base",
    c_dr: float = 3.0,
    penalty_rate: float = 20.0,
    min_load_ratio: float = 0.70,
    low_price_threshold: float = 0.0,
    low_price_probability_threshold: float = 0.30,
    q_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    decision_method: str = "grid_search_scenarios",
) -> pd.DataFrame:
    """
    Per-hour grid search over q using full scenario fan; counterfactual price from price model
    under each scenario's load/renewable draw.
    """
    base = base_df.copy()
    base["datetime"] = pd.to_datetime(base["datetime"])
    by_t = base.set_index("datetime", drop=False)

    scen = scenarios_df.copy()
    scen["target_datetime"] = pd.to_datetime(scen["target_datetime"])
    rows: list[dict[str, float | str | bool]] = []
    for t, grp in scen.groupby("target_datetime"):
        if t not in by_t.index:
            continue
        base_row = by_t.loc[t]
        if isinstance(base_row, pd.DataFrame):
            base_row = base_row.iloc[0]
        demand_base = float(base_row["demand_base"])
        max_flex = float(flexibility_ratio * demand_base)
        p_s = grp["price_scenario"].to_numpy(dtype=float)
        l_s = grp["load_scenario"].to_numpy(dtype=float)
        r_s = grp["renewable_scenario"].to_numpy(dtype=float)
        w = grp["scenario_probability"].to_numpy(dtype=float)
        w = w / (w.sum() + 1e-12)

        exp_ep = float(np.sum(w * p_s))
        exp_p10 = float(np.quantile(p_s, 0.10))
        exp_p50 = float(np.quantile(p_s, 0.50))
        exp_p90 = float(np.quantile(p_s, 0.90))
        p_low_scenario = float(np.sum(w[p_s <= low_price_threshold]))

        p_low_classifier = 0.0
        if use_refit_price_model and refit_model_pack is not None:
            p_low_classifier = float(
                predict_negative_price_probability(refit_model_pack, pd.DataFrame([base_row])).iloc[0]
            )
        force_no_dr = bool((exp_ep <= low_price_threshold) or (p_low_scenario >= low_price_probability_threshold) or (p_low_classifier >= low_price_probability_threshold))

        best = None
        for frac in q_grid:
            q = float(frac * max_flex)
            if force_no_dr and q > 0.0:
                continue
            # Build batch of counterfactual rows for all scenarios at once
            cf_rows = [
                update_counterfactual_row(
                    base_row, q, float(l_s[i]), float(r_s[i]),
                    demand_col=demand_col_for_price_model,
                    min_load_ratio=min_load_ratio,
                )
                for i in range(len(grp))
            ]
            cf_batch = pd.DataFrame(cf_rows)
            if use_refit_price_model and refit_model_pack is not None:
                p_cf_arr = predict_counterfactual_price(refit_model_pack, cf_batch).to_numpy(dtype=float)
            else:
                p_cf_arr = predict_counterfactual_price(price_model_rf, cf_batch).to_numpy(dtype=float)
            pen_list = [max(q - flexibility_ratio * float(l_s[i]), 0.0) for i in range(len(grp))]
            row_cf = cf_rows[-1]  # use last row for feasibility check
            price_term = _scenario_price_objective(p_cf_arr, w, p_s, objective_mode, high_price_threshold, cvar_alpha)
            penalty = float(np.sum(w * np.array(pen_list)) * penalty_rate)
            demand_after = max(demand_base - q, min_load_ratio * demand_base)
            econ_cost = price_term * demand_after + c_dr * q + penalty
            p_mean = float(np.sum(w * p_cf_arr))
            feasible = (0.0 <= q <= max_flex + 1e-9) and (float(row_cf[demand_col_for_price_model]) >= min_load_ratio * demand_base - 1e-9)
            if feasible and (best is None or econ_cost < best["econ_cost"]):
                best = {
                    "q": q,
                    "frac": frac,
                    "d_cf": float(row_cf[demand_col_for_price_model]),
                    "p_mean": p_mean,
                    "econ_cost": econ_cost,
                    "feasible": feasible,
                }
        if best is None:
            br = base_row
            db = demand_base
            best = {"q": 0.0, "frac": 0.0, "d_cf": db, "p_mean": exp_ep, "econ_cost": 0.0, "feasible": False}
        expected_cost_reduction = exp_ep * demand_base - best["p_mean"] * best["d_cf"]

        rows.append(
            {
                "target_datetime": t,
                "expected_price_calibrated": exp_ep,
                "price_p10": exp_p10,
                "price_p50": exp_p50,
                "price_p90": exp_p90,
                "demand_base": demand_base,
                "max_flexibility": max_flex,
                "q_DA_optimal": float(best["q"]),
                "q_DA_fraction": float(best["frac"]),
                "counterfactual_demand": float(best["d_cf"]),
                "expected_counterfactual_price": float(best["p_mean"]),
                "expected_price_reduction": float(exp_ep - best["p_mean"]),
                "expected_cost_reduction": float(expected_cost_reduction),
                "constraint_feasible": bool(best["feasible"]),
                "prob_low_price_scenario": p_low_scenario,
                "prob_low_price_classifier": float(p_low_classifier),
                "low_price_constraint_triggered": force_no_dr,
                "decision_method": decision_method,
                "objective_mode": objective_mode,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "target_datetime",
                "expected_price_calibrated",
                "price_p10",
                "price_p50",
                "price_p90",
                "demand_base",
                "max_flexibility",
                "q_DA_optimal",
                "q_DA_fraction",
                "counterfactual_demand",
                "expected_counterfactual_price",
                "expected_price_reduction",
                "expected_cost_reduction",
                "constraint_feasible",
                "prob_low_price_scenario",
                "prob_low_price_classifier",
                "low_price_constraint_triggered",
                "decision_method",
                "objective_mode",
            ]
        )
    return out.sort_values("target_datetime").reset_index(drop=True)


def optimize_day_ahead_dr_schedule(
    base_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    price_model,
    demand_col_for_price_model: str = "demand_base",
    flexibility_ratio: float = 0.10,
    c_dr: float = 3.0,
    penalty_rate: float = 20.0,
    min_load_ratio: float = 0.70,
    low_price_threshold: float = 0.0,
    low_price_probability_threshold: float = 0.30,
    q_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    *,
    objective_mode: str = "average_price",
    high_price_threshold: float = 50.0,
    cvar_alpha: float = 0.90,
    use_refit_price_model: bool = False,
    refit_model_pack: dict | None = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper: scenario-based grid optimization."""
    return optimize_dr_scenario_grid(
        base_df,
        scenario_df,
        flexibility_ratio=flexibility_ratio,
        objective_mode=objective_mode,
        high_price_threshold=high_price_threshold,
        cvar_alpha=cvar_alpha,
        use_refit_price_model=use_refit_price_model,
        price_model_rf=price_model if not use_refit_price_model else None,
        refit_model_pack=refit_model_pack,
        demand_col_for_price_model=demand_col_for_price_model,
        c_dr=c_dr,
        penalty_rate=penalty_rate,
        min_load_ratio=min_load_ratio,
        low_price_threshold=low_price_threshold,
        low_price_probability_threshold=low_price_probability_threshold,
        q_grid=q_grid,
        decision_method="grid_search_scenarios",
    )


def build_heuristic_dr_schedule(
    base_df: pd.DataFrame,
    day_ahead_forecast_block: pd.DataFrame,
    *,
    p75_hist: float,
    p90_hist: float,
    flexibility_ratio: float,
    use_refit_price_model: bool,
    price_model_rf,
    refit_model_pack: dict | None,
    demand_col: str = "demand_base",
    min_load_ratio: float = 0.70,
    low_price_threshold: float = 0.0,
    low_price_probability_threshold: float = 0.30,
    c_dr: float = 3.0,
    penalty_rate: float = 20.0,
) -> pd.DataFrame:
    """Heuristic baseline: full flex at >=P90 forecast, half at >=P75, else zero."""
    def _pred_one(sr: pd.Series) -> float:
        one = pd.DataFrame([sr])
        if use_refit_price_model:
            assert refit_model_pack is not None
            return float(predict_counterfactual_price(refit_model_pack, one).iloc[0])
        assert price_model_rf is not None
        return float(predict_counterfactual_price(price_model_rf, one).iloc[0])

    b = base_df.copy()
    b["datetime"] = pd.to_datetime(b["datetime"])
    f = day_ahead_forecast_block.copy()
    f["target_datetime"] = pd.to_datetime(f["target_datetime"])
    m = b.merge(f, left_on="datetime", right_on="target_datetime", how="inner")

    rows = []
    for _, r in m.iterrows():
        ep = float(r["price_forecast"])
        db = float(r["demand_base"])
        max_flex = flexibility_ratio * db
        p_low_classifier = 0.0
        if use_refit_price_model and refit_model_pack is not None:
            p_low_classifier = float(predict_negative_price_probability(refit_model_pack, pd.DataFrame([r])).iloc[0])

        if (ep <= low_price_threshold) or (p_low_classifier >= low_price_probability_threshold):
            q = 0.0
        elif ep >= p90_hist:
            q = max_flex
        elif ep >= p75_hist:
            q = 0.5 * max_flex
        else:
            q = 0.0
        row_cf = update_counterfactual_row(
            r,
            q,
            float(r["load_forecast"]),
            float(r["renewable_forecast"]),
            demand_col=demand_col,
            min_load_ratio=min_load_ratio,
        )
        p_cf = _pred_one(row_cf)
        rows.append(
            {
                "target_datetime": pd.Timestamp(r["target_datetime"]),
                "expected_price_calibrated": ep,
                "price_p10": float("nan"),
                "price_p50": float("nan"),
                "price_p90": float("nan"),
                "demand_base": db,
                "max_flexibility": max_flex,
                "q_DA_optimal": float(q),
                "q_DA_fraction": float(q / (max_flex + 1e-9)),
                "counterfactual_demand": float(row_cf[demand_col]),
                "expected_counterfactual_price": float(p_cf),
                "expected_price_reduction": float(ep - p_cf),
                "expected_cost_reduction": float(ep * db - p_cf * float(row_cf[demand_col])),
                "constraint_feasible": bool((0 <= q <= max_flex + 1e-9) and (float(row_cf[demand_col]) >= min_load_ratio * db - 1e-9)),
                "prob_low_price_classifier": p_low_classifier,
                "low_price_constraint_triggered": bool((ep <= low_price_threshold) or (p_low_classifier >= low_price_probability_threshold)),
                "decision_method": "high_price_heuristic",
                "objective_mode": "heuristic",
            }
        )
    return pd.DataFrame(rows).sort_values("target_datetime").reset_index(drop=True)


def evaluate_constraints(
    schedule_df: pd.DataFrame,
    min_load_ratio: float = 0.70,
) -> pd.DataFrame:
    s = schedule_df.copy()
    min_allowed = min_load_ratio * s["demand_base"]
    checks = [
        ("q_non_negative", (s["q_DA_optimal"] >= 0.0)),
        ("q_within_max_flex", (s["q_DA_optimal"] <= s["max_flexibility"] + 1e-9)),
        ("counterfactual_load_minimum", (s["counterfactual_demand"] >= min_allowed - 1e-9)),
    ]
    rows = []
    for name, passed_mask in checks:
        total = int(len(passed_mask))
        violations = int((~passed_mask).sum())
        rows.append(
            {
                "constraint_name": name,
                "violations": violations,
                "total_checks": total,
                "violation_rate": (violations / total) if total else float("nan"),
                "passed": violations == 0,
            }
        )
    return pd.DataFrame(rows)
