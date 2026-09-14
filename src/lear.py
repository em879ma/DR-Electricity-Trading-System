"""
LEAR (Lasso-Estimated AutoRegressive) baseline for day-ahead electricity prices.

Reference: Lago et al. (2021), "Forecasting day-ahead electricity prices:
A review of state-of-the-art algorithms, best practices and an open-access
benchmark." Applied Energy 293, 116983.

Design (paper §3.1):
  - 24 separate LASSO regressions, one per target hour h ∈ {0..23}.
  - Features per target hour:
      • price lags  : p(d-1, h), p(d-2, h), p(d-3, h), p(d-7, h)
      • load fcst   : forecast of load on day d, hour h (next-day available)
      • renewable   : forecast of renewable on day d, hour h
      • day-of-week dummies (6 binary)
  - α (LASSO strength) is selected with LassoCV (time-series fold).

The output matches the standard forecaster.py contract:
    pred_df, acc_df, neg_df, metrics

Note: Unlike Lago et al., we have hourly OPSD data with explicit forecast
columns for load/renewable. We use the actual columns named in market_panel
('load_forecast', 'renewable_forecast'), falling back to lagged actuals if
those columns are absent.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────────────────────────────────
# Feature construction (one model per target hour)
# ──────────────────────────────────────────────────────────────────────────────

_PRICE_LAGS_DAYS = (1, 2, 3, 7)  # in days; will be converted to hours


def _build_lear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the LEAR feature matrix on the full hourly panel.

    For each hour t we compute:
      - price_lag_<k>d : p(t - 24k) for k ∈ {1,2,3,7}
      - load_forecast  : OPSD day-ahead load forecast at t (if available)
      - renewable_forecast : OPSD day-ahead renewable forecast at t (if available)
      - dow_0..dow_5 : weekday dummies (Monday baseline omitted)

    The target column 'target_price' is p(t + 24).
    """
    work = df[["datetime", "price"]].copy()
    work["datetime"] = pd.to_datetime(work["datetime"])

    # price lags
    for k in _PRICE_LAGS_DAYS:
        work[f"price_lag_{k}d"] = work["price"].shift(24 * k)

    # Day-ahead exogenous variables — use lag-168h (same hour, 7d ago) as proxy
    # for next-day forecast, ensuring the value is genuinely available at decision
    # time (no leakage). 7-day lag captures weekly seasonality of load/renewable.
    if "load" in df.columns:
        work["lear_load"] = df["load"].shift(168).to_numpy()
    else:
        work["lear_load"] = 0.0

    if "renewable_total" in df.columns:
        work["lear_renew"] = df["renewable_total"].shift(168).to_numpy()
    elif {"wind_generation", "solar_generation"}.issubset(df.columns):
        work["lear_renew"] = (
            df["wind_generation"].fillna(0) + df["solar_generation"].fillna(0)
        ).shift(168).to_numpy()
    else:
        work["lear_renew"] = 0.0

    # weekday dummies (Monday=0 is baseline → omit dow_0)
    dow = work["datetime"].dt.dayofweek
    for d in range(1, 7):
        work[f"dow_{d}"] = (dow == d).astype(float)

    work["target_price"] = work["price"].shift(-24)
    work["hour"] = work["datetime"].dt.hour
    return work


def _feature_cols() -> list[str]:
    cols = [f"price_lag_{k}d" for k in _PRICE_LAGS_DAYS]
    cols += ["lear_load", "lear_renew"]
    cols += [f"dow_{d}" for d in range(1, 7)]
    return cols


# ──────────────────────────────────────────────────────────────────────────────
# LEAR model — 24 LASSO regressions
# ──────────────────────────────────────────────────────────────────────────────

class LEARModel:
    """24 separate LassoCV models, one per target hour."""

    def __init__(self, cv: int = 5, max_iter: int = 5000,
                 random_state: int = 42) -> None:
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state
        self.models_: dict[int, LassoCV] = {}
        self.scalers_: dict[int, StandardScaler] = {}
        self.feat_cols_: list[str] = _feature_cols()

    def fit(self, train_df: pd.DataFrame) -> "LEARModel":
        """Train one LassoCV per target hour h ∈ {0..23}."""
        for h in range(24):
            sub = train_df[train_df["hour"] == h]
            X = sub[self.feat_cols_].to_numpy()
            y = sub["target_price"].to_numpy()
            if len(y) < 20:
                continue
            sc = StandardScaler()
            X_sc = sc.fit_transform(X)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdl = LassoCV(
                    cv=self.cv, max_iter=self.max_iter,
                    random_state=self.random_state, n_jobs=1,
                ).fit(X_sc, y)
            self.models_[h] = mdl
            self.scalers_[h] = sc
        return self

    def predict(self, eval_df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(eval_df), np.nan, dtype=float)
        for h, mdl in self.models_.items():
            mask = (eval_df["hour"].to_numpy() == h)
            if not mask.any():
                continue
            X = eval_df.loc[mask, self.feat_cols_].to_numpy()
            X_sc = self.scalers_[h].transform(X)
            out[mask] = mdl.predict(X_sc)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline entry point — matches src/forecaster.py contract
# ──────────────────────────────────────────────────────────────────────────────

def fit_lear_pipeline(
    df: pd.DataFrame,
    split: dict,
    rf_params: dict,
    horizon: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Train LEAR and return (pred_df, acc_df, neg_df, metrics)."""
    work = _build_lear_features(df)
    fit = work.dropna(subset=self_required_cols()).reset_index(drop=True)

    n = len(fit)
    n_train = int(n * split["train_ratio"])
    train = fit.iloc[:n_train]
    eval_data = fit.iloc[n_train:].reset_index(drop=True)

    model = LEARModel().fit(train)
    pred = model.predict(eval_data)

    pred_df = pd.DataFrame({
        "forecast_origin_datetime": pd.to_datetime(eval_data["datetime"]),
        "target_datetime": (pd.to_datetime(eval_data["datetime"])
                            + pd.to_timedelta(horizon, "h")),
        "horizon": horizon,
        "model": "lear",
        "target": "price",
        "forecast": pred,
        "actual": eval_data["target_price"].to_numpy(),
    })

    y = pred_df["actual"].to_numpy()
    yhat = pred_df["forecast"].to_numpy()
    mask = ~np.isnan(yhat)
    err = yhat[mask] - y[mask]
    pos = y[mask] > 0
    acc_df = pd.DataFrame([{
        "method": "lear", "target": "price", "horizon": horizon,
        "mae":  float(mean_absolute_error(y[mask], yhat[mask])),
        "rmse": float(mean_squared_error(y[mask], yhat[mask]) ** 0.5),
        "mape": float(np.mean(np.abs(err[pos] / y[mask][pos])) * 100) if pos.any() else float("nan"),
        "bias": float(np.mean(err)),
        "r2":   float(r2_score(y[mask], yhat[mask])),
    }])

    target_dt = pred_df["target_datetime"]
    p_neg = (yhat < 0).astype(float)
    neg_df = pd.DataFrame({"target_datetime": target_dt, "p_negative_price": p_neg})

    metrics = {
        "mae": float(acc_df["mae"].iloc[0]),
        "rmse": float(acc_df["rmse"].iloc[0]),
        "n_train_per_hour": int(len(train) / 24),
        "n_active_hour_models": int(len(model.models_)),
    }
    return pred_df, acc_df, neg_df, metrics


def self_required_cols() -> list[str]:
    return _feature_cols() + ["target_price", "hour"]
