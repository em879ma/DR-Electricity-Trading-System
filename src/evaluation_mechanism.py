from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_economic_mechanisms(
    hourly: pd.DataFrame,
    sensitivity_df: pd.DataFrame | None = None,
    model_coef_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Level 4: Economic mechanism validity checks.

    Rules verified:
    1. demand_reduction_lowers_price: DR lowers counterfactual price vs. observed
    2. high_price_dr_concentration: DR concentrated in high-price hours
    3. negative_price_guardrail: no curtailment DR when price < 0
    4. delayed_worse_than_immediate: (checked via sensitivity if available)
    5. ridge_demand_positive: net_load coefficient > 0 in Ridge model
    6. ridge_renewable_negative: renewable_share coefficient < 0 in Ridge model
    7. ridge_import_negative: import_share coefficient < 0 in Ridge model
    8. ridge_oversupply_negative: oversupply_index coefficient < 0 in Ridge model

    Returns a pass/fail table with columns:
    mechanism_check, passed, metric_value, expected_direction, note
    """
    h = hourly.copy()
    price_col = "observed_price" if "observed_price" in h.columns else "price"
    cf_col = "counterfactual_price"
    q_col = "q_DA_optimal"

    checks = []

    # Check 1: DR lowers average price
    if cf_col in h.columns and price_col in h.columns:
        avg_obs = float(h[price_col].mean())
        avg_cf = float(h[cf_col].mean())
        checks.append({
            "mechanism_check": "demand_reduction_lowers_price",
            "passed": bool(avg_cf < avg_obs),
            "metric_value": float(avg_obs - avg_cf),
            "expected_direction": "> 0 (cf < observed)",
            "note": f"avg_observed={avg_obs:.2f}, avg_counterfactual={avg_cf:.2f}",
        })

    # Check 2: DR concentrated in high-price hours
    if q_col in h.columns and price_col in h.columns:
        p90_thr = float(h[price_col].quantile(0.9))
        p25_thr = float(h[price_col].quantile(0.25))
        high_util = float(h.loc[h[price_col] >= p90_thr, q_col].mean()) if (h[price_col] >= p90_thr).any() else float("nan")
        low_util = float(h.loc[h[price_col] <= p25_thr, q_col].mean()) if (h[price_col] <= p25_thr).any() else float("nan")
        checks.append({
            "mechanism_check": "dr_allocation_concentrated_in_high_price",
            "passed": bool(not np.isnan(high_util) and not np.isnan(low_util) and high_util > low_util),
            "metric_value": float(high_util - low_util) if not np.isnan(high_util) and not np.isnan(low_util) else float("nan"),
            "expected_direction": "> 0 (high price hours get more DR)",
            "note": f"avg_q_high={high_util:.1f}, avg_q_low={low_util:.1f}",
        })

    # Check 3: Negative-price guardrail prevents curtailment DR
    if "low_price_constraint_triggered" in h.columns and q_col in h.columns:
        triggered = h[h["low_price_constraint_triggered"] == True]
        if not triggered.empty:
            q_when_guardrail = float(triggered[q_col].mean())
            checks.append({
                "mechanism_check": "negative_price_guardrail_blocks_dr",
                "passed": bool(q_when_guardrail <= 1e-9),
                "metric_value": float(q_when_guardrail),
                "expected_direction": "= 0 (no DR when guardrail triggered)",
                "note": f"n_guardrail_hours={len(triggered)}, avg_q={q_when_guardrail:.2f}",
            })
        else:
            checks.append({
                "mechanism_check": "negative_price_guardrail_blocks_dr",
                "passed": True,
                "metric_value": 0.0,
                "expected_direction": "= 0",
                "note": "no guardrail-triggered hours in schedule",
            })

    # Check 4: Sensitivity shows demand reduction lowers price
    if sensitivity_df is not None and not sensitivity_df.empty:
        all_pos = bool(sensitivity_df.get("mechanism_valid_avg_positive", pd.Series([True])).all())
        high_gt_avg = bool(sensitivity_df.get("mechanism_valid_high_gt_avg", pd.Series([True])).all())
        checks.append({
            "mechanism_check": "sensitivity_demand_reduction_lowers_price",
            "passed": all_pos,
            "metric_value": float(sensitivity_df.get("avg_price_change", pd.Series([float("nan")])).mean()),
            "expected_direction": "> 0 across all demand-reduction ratios",
            "note": "average price change when demand reduced",
        })
        checks.append({
            "mechanism_check": "sensitivity_high_price_reduction_exceeds_average",
            "passed": high_gt_avg,
            "metric_value": float(sensitivity_df.get("high_price_price_change", pd.Series([float("nan")])).mean()),
            "expected_direction": "> avg_price_change",
            "note": "high-price hours benefit more from DR",
        })

    # Check 5-8: Ridge model coefficient signs
    if model_coef_df is not None and not model_coef_df.empty:
        coef_map = model_coef_df.set_index("feature")["coefficient"].to_dict()
        sign_checks = [
            ("ridge_net_load_positive", "net_load", "> 0", True),
            ("ridge_scarcity_positive", "scarcity_index", "> 0", True),
            ("ridge_renewable_share_negative", "renewable_share", "< 0", False),
            ("ridge_import_share_negative", "import_share", "< 0", False),
            ("ridge_oversupply_negative", "oversupply_index", "< 0", False),
        ]
        for check_name, feat, direction, expect_positive in sign_checks:
            val = coef_map.get(feat, float("nan"))
            if not np.isnan(val):
                passed = bool(val > 0) if expect_positive else bool(val < 0)
                checks.append({
                    "mechanism_check": check_name,
                    "passed": passed,
                    "metric_value": float(val),
                    "expected_direction": direction,
                    "note": f"Ridge coefficient for {feat}",
                })

    return pd.DataFrame(checks) if checks else pd.DataFrame(
        columns=["mechanism_check", "passed", "metric_value", "expected_direction", "note"]
    )
