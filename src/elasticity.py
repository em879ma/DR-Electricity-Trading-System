from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def estimate_rolling_elasticity(df: pd.DataFrame, window: int = 24 * 7) -> pd.Series:
    log_p = np.log(df["price"].clip(lower=1e-3))
    log_q = np.log(df["load"].clip(lower=1e-3))
    cov = log_q.rolling(window=window, min_periods=max(24, window // 3)).cov(log_p)
    var = log_p.rolling(window=window, min_periods=max(24, window // 3)).var()
    elasticity = cov / var.replace(0.0, np.nan)
    elasticity = elasticity.clip(lower=-1.5, upper=0.2)
    elasticity = elasticity.fillna(elasticity.median()).fillna(-0.12)
    return elasticity.rename("rolling_elasticity")


def estimate_dynamic_elasticity(df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    work = df.copy()
    work["log_load"] = np.log(work["load"].clip(lower=1e-6))
    work["log_price"] = np.log(work["price"].clip(lower=1e-6))
    for l in [1, 2, 24]:
        work[f"log_price_lag_{l}"] = work["log_price"].shift(l)
    work["log_load_lag_1"] = work["log_load"].shift(1)
    work["log_load_lag_24"] = work["log_load"].shift(24)
    fit = work.dropna(
        subset=[
            "log_load",
            "log_price",
            "log_price_lag_1",
            "log_price_lag_2",
            "log_price_lag_24",
            "log_load_lag_1",
            "log_load_lag_24",
            "hour",
            "dayofweek",
            "month",
        ]
    ).copy()

    x_num = fit[
        [
            "log_price",
            "log_price_lag_1",
            "log_price_lag_2",
            "log_price_lag_24",
            "log_load_lag_1",
            "log_load_lag_24",
        ]
    ]
    x_cat = pd.get_dummies(fit[["hour", "dayofweek", "month"]].astype("category"), drop_first=True)
    x = pd.concat([pd.Series(1.0, index=fit.index, name="intercept"), x_num, x_cat], axis=1)
    y = fit["log_load"]
    x_mat = np.nan_to_num(x.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y_vec = np.nan_to_num(y.to_numpy(dtype=float), nan=float(np.nanmedian(y)), posinf=float(np.nanmedian(y)), neginf=float(np.nanmedian(y)))
    beta, *_ = np.linalg.lstsq(x_mat, y_vec, rcond=1e-8)
    beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.sum(x_mat * beta.reshape(1, -1), axis=1)
    resid = y_vec - pred
    n, k = len(y), x.shape[1]
    sigma2 = float((resid @ resid) / max(n - k, 1))
    xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 1e-12, None))

    coef_map = dict(zip(x.columns.tolist(), beta))
    se_map = dict(zip(x.columns.tolist(), se))
    params = {
        "epsilon_0": float(coef_map.get("log_price", -0.05)),
        "epsilon_1": float(coef_map.get("log_price_lag_1", -0.01)),
        "epsilon_2": float(coef_map.get("log_price_lag_2", -0.005)),
        "epsilon_24": float(coef_map.get("log_price_lag_24", -0.005)),
        "r_squared": float(1.0 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)),
        "n_obs": int(n),
    }
    table = pd.DataFrame(
        {
            "parameter": ["epsilon_0", "epsilon_1", "epsilon_2", "epsilon_24"],
            "estimate": [params["epsilon_0"], params["epsilon_1"], params["epsilon_2"], params["epsilon_24"]],
            "standard_error": [
                float(se_map.get("log_price", np.nan)),
                float(se_map.get("log_price_lag_1", np.nan)),
                float(se_map.get("log_price_lag_2", np.nan)),
                float(se_map.get("log_price_lag_24", np.nan)),
            ],
            "p_value": [np.nan, np.nan, np.nan, np.nan],
            "r_squared": params["r_squared"],
            "n_obs": params["n_obs"],
        }
    )
    return params, table


def write_elasticity_outputs(params: dict[str, float], table: pd.DataFrame, output_root: str = "outputs") -> None:
    tables = Path(output_root) / "tables"
    models = Path(output_root) / "models"
    tables.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / "elasticity_estimates.csv", index=False)
    with (models / "elasticity_params.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
