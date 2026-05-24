from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def estimate_baseline_demand(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_load"] = np.log(out["load"].clip(lower=1e-6))
    out["log_load_lag_1"] = out["log_load"].shift(1)
    out["log_load_lag_24"] = out["log_load"].shift(24)
    fit = out.dropna(subset=["log_load", "log_load_lag_1", "log_load_lag_24"]).copy()
    x_num = fit[["log_load_lag_1", "log_load_lag_24"]]
    x_cat = pd.get_dummies(fit[["hour", "dayofweek", "month"]].astype("category"), drop_first=True)
    x = pd.concat([pd.Series(1.0, index=fit.index, name="intercept"), x_num, x_cat], axis=1)
    x_fit = np.nan_to_num(x.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y_fit = np.nan_to_num(fit["log_load"].to_numpy(dtype=float), nan=float(np.nanmedian(fit["log_load"])))
    beta, *_ = np.linalg.lstsq(x_fit, y_fit, rcond=1e-8)
    beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
    x_all = pd.concat(
        [
            pd.Series(1.0, index=out.index, name="intercept"),
            out[["log_load_lag_1", "log_load_lag_24"]].bfill(),
            pd.get_dummies(out[["hour", "dayofweek", "month"]].astype("category"), drop_first=True),
        ],
        axis=1,
    ).reindex(columns=x.columns, fill_value=0.0)
    x_all_mat = np.nan_to_num(x_all.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    out["log_demand_base"] = np.sum(x_all_mat * beta.reshape(1, -1), axis=1)
    out["demand_base"] = np.exp(out["log_demand_base"])
    out["price_ref"] = out["price"].rolling(168, min_periods=24).mean().bfill()
    p = out["price"].clip(lower=1e-3)
    pr = out["price_ref"].clip(lower=1e-3)
    out["log_price_gap"] = np.log(p / pr)
    out["log_price_gap_lag_1"] = out["log_price_gap"].shift(1).bfill()
    out["log_price_gap_lag_2"] = out["log_price_gap"].shift(2).bfill()
    out["log_price_gap_lag_24"] = out["log_price_gap"].shift(24).bfill()
    return out


def construct_responsive_demand(df: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    e0 = float(params.get("epsilon_0", -0.05))
    e1 = float(params.get("epsilon_1", -0.01))
    e2 = float(params.get("epsilon_2", -0.005))
    e24 = float(params.get("epsilon_24", -0.005))
    # e0 is negative (demand falls when price rises); high-price-only uses e0 directly
    high_elasticity = e0

    out["demand_no_response"] = out["demand_base"]
    out["demand_responsive_immediate"] = out["demand_base"] * np.exp(e0 * out["log_price_gap"])
    # Delayed response uses only lagged price gaps — no current-period response (no shifting)
    out["demand_responsive_delayed"] = out["demand_base"] * np.exp(
        e1 * out["log_price_gap_lag_1"] + e2 * out["log_price_gap_lag_2"] + e24 * out["log_price_gap_lag_24"]
    )
    out["demand_responsive_dynamic"] = out["demand_base"] * np.exp(
        e0 * out["log_price_gap"]
        + e1 * out["log_price_gap_lag_1"]
        + e2 * out["log_price_gap_lag_2"]
        + e24 * out["log_price_gap_lag_24"]
    )
    out["demand_responsive_high_price_only"] = out["demand_base"] * np.exp(
        out["high_price_dummy"] * high_elasticity * out["log_price_gap"]
    )
    for c in [
        "demand_no_response",
        "demand_responsive_immediate",
        "demand_responsive_delayed",
        "demand_responsive_dynamic",
        "demand_responsive_high_price_only",
    ]:
        out[c] = out[c].clip(lower=0.5 * out["demand_base"], upper=1.3 * out["demand_base"])
    return out


def write_market_panel_with_dr(df: pd.DataFrame) -> None:
    out_path = Path("data/processed/market_panel_with_dr.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
