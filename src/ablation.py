"""
Ablation study helpers for the DR decision pipeline.

Each function runs one ablation and returns a flat dict of key metrics.
All ablations compare against the baseline XGBoost + block-bootstrap pipeline.

Ablation map
------------
1. no_scenarios      — degenerate single-scenario (point forecast) → DR decision
2. iid_bootstrap     — IID residual sampling instead of 24h block bootstrap
3. no_holiday        — remove holiday/Christmas/neg-price-dummy features
4. no_sincos         — replace hour_sin/hour_cos with raw integer hour
5. no_elasticity     — remove rolling_elasticity from feature set
6. no_guardrail      — disable negative-price guardrail in DR optimizer
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.forecasting import RF_FEATURES
from src.scenarios import generate_bootstrap_scenarios
from src.day_ahead_decision import optimize_dr_scenario_grid
from src.evaluation_scenarios import evaluate_scenario_quality
from src.evaluation_decision import evaluate_decision_performance, build_hourly_counterfactual_panel

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False

# Feature groups
HOLIDAY_FEATURES = [
    "is_public_holiday", "holiday_load_window", "is_christmas_week",
    "is_public_holiday_lead_24", "holiday_load_window_lead_24",
    "is_christmas_week_lead_24", "negative_price_dummy_lag_24",
]
CYCLIC_FEATURES = ["hour_sin", "hour_cos"]
ELASTICITY_FEATURES = ["rolling_elasticity"]


# ──────────────────────────────────────────────────────────────────────────────
# Internal XGBoost helper
# ──────────────────────────────────────────────────────────────────────────────

def _xgb_l1(
    df: pd.DataFrame,
    split: dict,
    rf_params: dict,
    horizon: int = 24,
    exclude: list[str] | None = None,
    extra: list[str] | None = None,
) -> dict:
    """Train XGBoost with modified features; return L1 metrics dict."""
    if not _XGB:
        raise ImportError("xgboost not installed")

    work = df.copy()
    work["target_price"] = work["price"].shift(-horizon)

    feats = list(RF_FEATURES)
    if exclude:
        feats = [f for f in feats if f not in exclude]
    if extra:
        feats = feats + [f for f in extra if f not in feats]
    feat_cols = [c for c in feats if c in work.columns]

    fit = (work[feat_cols + ["target_price", "datetime"]]
           .dropna()
           .reset_index(drop=True))
    n = len(fit)
    n_tr = int(n * split["train_ratio"])
    n_va = int(n * (split["train_ratio"] + split.get("valid_ratio", 0.15)))
    train, valid = fit.iloc[:n_tr], fit.iloc[n_tr:n_va]
    eval_data = fit.iloc[n_tr:].reset_index(drop=True)  # valid + test

    params = {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": int(rf_params.get("random_seed", 42)), "n_jobs": 1,
    }
    mdl = xgb.XGBRegressor(**params, early_stopping_rounds=20, eval_metric="rmse")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl.fit(
            train[feat_cols].to_numpy(), train["target_price"].to_numpy(),
            eval_set=[(valid[feat_cols].to_numpy(), valid["target_price"].to_numpy())],
            verbose=False,
        )

    y = eval_data["target_price"].to_numpy()
    yhat = mdl.predict(eval_data[feat_cols].to_numpy())
    err = yhat - y
    pos = y > 0
    return {
        "mae":  float(mean_absolute_error(y, yhat)),
        "rmse": float(mean_squared_error(y, yhat) ** 0.5),
        "mape": float(np.mean(np.abs(err[pos] / y[pos])) * 100) if pos.any() else float("nan"),
        "bias": float(np.mean(err)),
        "r2":   float(r2_score(y, yhat)),
        "n_features": len(feat_cols),
    }


# ──────────────────────────────────────────────────────────────────────────────
# DR decision helper (scenarios → schedule → L3 metrics)
# ──────────────────────────────────────────────────────────────────────────────

def _run_dr(
    scenarios_df: pd.DataFrame,
    dr_df: pd.DataFrame,
    model_pack: dict,
    cfg: dict,
    low_price_threshold: float | None = None,
    low_price_probability_threshold: float | None = None,
) -> dict:
    flex  = float(cfg["decision"]["flexibility_ratios"][1])
    c_dr  = float(cfg["decision"].get("c_dr", 3.0))
    pen   = float(cfg["decision"].get("penalty_rate", 20.0))
    p90   = float(dr_df["price"].quantile(0.90))
    lp_thr  = float(cfg["decision"].get("low_price_threshold", 0.0)) if low_price_threshold is None else low_price_threshold
    lp_prob = float(cfg["decision"].get("low_price_probability_threshold", 0.30)) if low_price_probability_threshold is None else low_price_probability_threshold

    sched = optimize_dr_scenario_grid(
        dr_df, scenarios_df,
        flexibility_ratio=flex, objective_mode="average_cost",
        high_price_threshold=p90,
        use_refit_price_model=True, refit_model_pack=model_pack,
        c_dr=c_dr, penalty_rate=pen,
        low_price_threshold=lp_thr,
        low_price_probability_threshold=lp_prob,
    )
    hourly = build_hourly_counterfactual_panel(sched, dr_df)
    l3 = evaluate_decision_performance(sched, dr_df, flexibility_ratio=flex, c_dr=c_dr, penalty_rate=pen)

    n_neg_with_dr = 0
    if "q_DA_optimal" in sched.columns:
        tmp = sched.merge(
            dr_df[["datetime", "price"]].rename(columns={"datetime": "target_datetime"}),
            on="target_datetime", how="inner"
        )
        n_neg_with_dr = int(((tmp["price"] < 0) & (tmp["q_DA_optimal"] > 1e-6)).sum())

    obs_cost = float((hourly["observed_price"] * hourly["demand_base"]).sum())
    cf_cost  = float((hourly["counterfactual_price"] * hourly["counterfactual_demand"]).sum())

    return {
        "avg_price_reduction":    float(hourly["price_reduction"].mean()),
        "total_cost_reduction":   float(obs_cost - cf_cost),
        "high_price_hours_reduced": int(l3["high_price_hours_reduced"].iloc[0]),
        "DR_utilization_high":    float(l3["DR_utilization_high_price"].iloc[0]),
        "DR_utilization_low":     float(l3["DR_utilization_low_price"].iloc[0]),
        "realized_profit":        float(l3["realized_profit"].iloc[0]),
        "n_neg_price_dr":         n_neg_with_dr,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public ablation functions
# ──────────────────────────────────────────────────────────────────────────────

def ablate_no_holiday(df: pd.DataFrame, split: dict, rf_params: dict) -> dict:
    """Remove all holiday / Christmas / neg-price dummy features."""
    return _xgb_l1(df, split, rf_params, exclude=HOLIDAY_FEATURES)


def ablate_no_sincos(df: pd.DataFrame, split: dict, rf_params: dict) -> dict:
    """Replace hour_sin / hour_cos with raw integer hour."""
    return _xgb_l1(df, split, rf_params, exclude=CYCLIC_FEATURES, extra=["hour"])


def ablate_no_elasticity(df: pd.DataFrame, split: dict, rf_params: dict) -> dict:
    """Remove rolling_elasticity from the feature set."""
    return _xgb_l1(df, split, rf_params, exclude=ELASTICITY_FEATURES)


def ablate_no_scenarios(
    day_ahead_rf: pd.DataFrame,
    dr_df: pd.DataFrame,
    model_pack: dict,
    cfg: dict,
) -> dict:
    """Replace 48-scenario fan with a degenerate single-scenario (point forecast)."""
    degen = (
        day_ahead_rf
        .rename(columns={
            "price_forecast":      "price_scenario",
            "load_forecast":       "load_scenario",
            "renewable_forecast":  "renewable_scenario",
        })
        .assign(scenario_id=1, scenario_probability=1.0)
        [["scenario_id", "scenario_probability", "forecast_origin_datetime",
          "target_datetime", "horizon",
          "price_scenario", "load_scenario", "renewable_scenario"]]
    )
    return _run_dr(degen, dr_df, model_pack, cfg)


def ablate_iid_bootstrap(
    day_ahead_rf: pd.DataFrame,
    resid_df: pd.DataFrame,
    dr_df: pd.DataFrame,
    model_pack: dict,
    cfg: dict,
) -> tuple[dict, dict]:
    """
    IID bootstrap vs block bootstrap.
    Returns (l2_metrics, l3_metrics) as a merged dict plus coverage metrics.
    """
    iid_boot = generate_bootstrap_scenarios(
        day_ahead_rf, resid_df,
        n_scenarios=cfg["scenario"]["n_scenarios"],
        block_bootstrap=False,  # key change
        block_length=cfg["scenario"]["block_length"],
        random_seed=cfg["scenario"]["random_seed"],
    )
    l2 = evaluate_scenario_quality(iid_boot, dr_df)
    l3 = _run_dr(iid_boot, dr_df, model_pack, cfg)

    metrics = {
        "coverage_10_90":           float(l2["coverage_10_90"].iloc[0]),
        "coverage_05_95":           float(l2["coverage_05_95"].iloc[0]),
        "mean_interval_width":      float(l2["mean_interval_width"].iloc[0]),
        "high_price_tail_coverage": float(l2["high_price_tail_coverage"].iloc[0]),
        "neg_price_tail_coverage":  float(l2["negative_price_tail_coverage"].iloc[0]),
        "scenario_bias":            float(l2["scenario_bias"].iloc[0]),
        **l3,
    }
    return metrics


def ablate_no_guardrail(
    boot: pd.DataFrame,
    dr_df: pd.DataFrame,
    model_pack: dict,
    cfg: dict,
) -> dict:
    """Remove negative-price guardrail (both classifier and scenario thresholds)."""
    return _run_dr(
        boot, dr_df, model_pack, cfg,
        low_price_threshold=-999.0,        # expected-price condition never fires
        low_price_probability_threshold=1.1,  # probability conditions never fire
    )
