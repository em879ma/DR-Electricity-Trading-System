"""
Unified price forecasting interface for the DR decision pipeline.

Usage — select model in config.yaml:
    forecast:
      model: "xgboost"   # or: two_stage_rf | ridge | lstm | tcn

Entry point:
    from src.forecaster import fit_price_forecast
    pred_df, acc_df, neg_price_df, metrics = fit_price_forecast(
        df, model_name, split, rf_params
    )

Return contract (same for every model):
    pred_df      — forecast_origin_datetime, target_datetime, horizon, model,
                   target, forecast, actual
    acc_df       — method, target, horizon, mae, rmse, mape, bias, r2
    neg_price_df — target_datetime, p_negative_price  (RF binary classifier,
                   shared across all models)
    metrics      — dict with recall/precision/threshold + model-specific stats
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.forecasting import RF_FEATURES, fit_two_stage_price_forecast, get_feature_set

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

AVAILABLE_MODELS = ["two_stage_rf", "xgboost", "ridge", "lstm", "tcn",
                    "naive_seasonal", "regime_ensemble"]
_HORIZON = 24


# ══════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════

def fit_price_forecast(
    df: pd.DataFrame,
    model_name: str,
    split: dict,
    rf_params: dict,
    horizon: int = _HORIZON,
    feature_set: str = "full",
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Train a price forecasting model and return standardised outputs.

    Parameters
    ----------
    df         : market panel DataFrame (output of run_01/02)
    model_name : one of AVAILABLE_MODELS
    split      : {"train_ratio": 0.7, "valid_ratio": 0.15}
    rf_params  : RF/XGB hyperparameters from config.yaml
    horizon    : forecast horizon in hours (default 24)
    **kwargs   : model-specific overrides (e.g. xgb_params={"n_estimators": 300})

    Returns
    -------
    pred_df      : predictions on the eval set (valid + test)
    acc_df       : accuracy metrics
    neg_price_df : neg-price probability from a shared RF classifier
    metrics      : summary dict (recall, precision, threshold, …)
    """
    _check_model(model_name)
    kwargs["feature_set"] = feature_set

    if model_name == "two_stage_rf":
        return fit_two_stage_price_forecast(df, rf_params, split, horizon=horizon)

    if model_name == "xgboost":
        return _xgboost_pipeline(df, split, rf_params, horizon, **kwargs)

    if model_name == "ridge":
        return _ridge_pipeline(df, split, rf_params, horizon, **kwargs)

    if model_name == "lstm":
        return _lstm_pipeline(df, split, rf_params, horizon, **kwargs)

    if model_name == "tcn":
        return _tcn_pipeline(df, split, rf_params, horizon, **kwargs)

    if model_name == "naive_seasonal":
        return _naive_seasonal_pipeline(df, split, rf_params, horizon, **kwargs)

    if model_name == "regime_ensemble":
        from src.ensemble import fit_regime_ensemble_pipeline
        return fit_regime_ensemble_pipeline(df, split, rf_params, horizon, **kwargs)


def available_models() -> list[str]:
    """Return list of supported model names."""
    return list(AVAILABLE_MODELS)


def _check_model(name: str) -> None:
    if name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {AVAILABLE_MODELS}"
        )


# ══════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════

def _prep_tabular(df: pd.DataFrame, split: dict, horizon: int,
                  feature_set: str = "full"):
    """Returns (feat_cols, train_df, valid_df, test_df).

    `feature_set='pruned'` uses the v2.2 diagnostic-driven reduced feature list.
    """
    work = df.copy()
    work["target_price"] = work["price"].shift(-horizon)
    feat_cols = [c for c in get_feature_set(feature_set) if c in work.columns]
    fit = (work[feat_cols + ["target_price", "datetime"]]
           .dropna()
           .reset_index(drop=True))
    n = len(fit)
    n_train = int(n * split["train_ratio"])
    n_valid = int(n * (split["train_ratio"] + split.get("valid_ratio", 0.15)))
    return feat_cols, fit.iloc[:n_train], fit.iloc[n_train:n_valid], fit.iloc[n_valid:]


def _make_pred_df(
    eval_data: pd.DataFrame,
    pred: np.ndarray,
    model_name: str,
    horizon: int,
) -> pd.DataFrame:
    return pd.DataFrame({
        "forecast_origin_datetime": pd.to_datetime(eval_data["datetime"]),
        "target_datetime": (pd.to_datetime(eval_data["datetime"])
                            + pd.to_timedelta(horizon, "h")),
        "horizon": horizon,
        "model": model_name,
        "target": "price",
        "forecast": pred,
        "actual": eval_data["target_price"].to_numpy(),
    })


def _neg_clf(
    train: pd.DataFrame,
    eval_data: pd.DataFrame,
    feat_cols: list[str],
    rf_params: dict,
    horizon: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Shared RF binary classifier for negative-price probability.
    Identical to the Stage-1 classifier in Two-Stage RF.
    Used by XGBoost, Ridge, LSTM, TCN so the neg-price signal is consistent.
    """
    target_dt = (pd.to_datetime(eval_data["datetime"])
                 + pd.to_timedelta(horizon, "h"))
    train_y_bin = (train["target_price"] < 0).astype(int)

    if train_y_bin.nunique() < 2:
        return (
            pd.DataFrame({"target_datetime": target_dt,
                          "p_negative_price": 0.0}),
            {},
        )

    clf = RandomForestClassifier(
        n_estimators=int(rf_params.get("n_estimators", 300)),
        max_depth=int(rf_params.get("max_depth", 12)),
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=int(rf_params.get("random_seed", 42)),
        n_jobs=1,
    )
    clf.fit(train[feat_cols], train_y_bin)
    p_neg = clf.predict_proba(eval_data[feat_cols])[:, 1]

    eval_y_bin = (eval_data["target_price"].to_numpy() < 0).astype(int)
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.02, 0.51, 0.01):
        pb = (p_neg >= thr).astype(int)
        tp = int(((eval_y_bin == 1) & (pb == 1)).sum())
        fp = int(((eval_y_bin == 0) & (pb == 1)).sum())
        fn = int(((eval_y_bin == 1) & (pb == 0)).sum())
        if 2 * tp + fp + fn == 0:
            continue
        f1 = 2 * tp / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    pred_bin = (p_neg >= best_thr).astype(int)
    tp = int(((eval_y_bin == 1) & (pred_bin == 1)).sum())
    fn = int(((eval_y_bin == 1) & (pred_bin == 0)).sum())
    fp = int(((eval_y_bin == 0) & (pred_bin == 1)).sum())

    neg_df = pd.DataFrame({"target_datetime": target_dt, "p_negative_price": p_neg})
    metrics = {
        "negative_price_recall_classifier":
            tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "negative_price_precision_classifier":
            tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "n_neg_train_strict": int((train["target_price"] < 0).sum()),
        "classifier_threshold": float(best_thr),
    }
    return neg_df, metrics


def _build_acc(pred_df: pd.DataFrame, model_name: str, horizon: int) -> pd.DataFrame:
    y = pred_df["actual"].to_numpy()
    yhat = pred_df["forecast"].to_numpy()
    err = yhat - y
    pos = y > 0
    return pd.DataFrame([{
        "method": model_name, "target": "price", "horizon": horizon,
        "mae":  float(mean_absolute_error(y, yhat)),
        "rmse": float(mean_squared_error(y, yhat) ** 0.5),
        "mape": float(np.mean(np.abs(err[pos] / y[pos])) * 100) if pos.any() else float("nan"),
        "bias": float(np.mean(err)),
        "r2":   float(r2_score(y, yhat)),
    }])


# ══════════════════════════════════════════════════════════════════════
# XGBoost pipeline
# ══════════════════════════════════════════════════════════════════════

def _xgboost_pipeline(
    df: pd.DataFrame,
    split: dict,
    rf_params: dict,
    horizon: int,
    xgb_params: dict | None = None,
    feature_set: str = "full",
    **_,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not _XGB:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    feat_cols, train, valid, test = _prep_tabular(df, split, horizon, feature_set)
    eval_data = pd.concat([valid, test]).reset_index(drop=True)

    params = xgb_params or {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": int(rf_params.get("random_seed", 42)),
        "n_jobs": 1,
    }
    model = xgb.XGBRegressor(**params, early_stopping_rounds=20, eval_metric="rmse")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            train[feat_cols].to_numpy(),
            train["target_price"].to_numpy(),
            eval_set=[(valid[feat_cols].to_numpy(),
                       valid["target_price"].to_numpy())],
            verbose=False,
        )

    pred = model.predict(eval_data[feat_cols].to_numpy())
    pred_df = _make_pred_df(eval_data, pred, "xgboost", horizon)
    acc_df  = _build_acc(pred_df, "xgboost", horizon)
    neg_df, clf_m = _neg_clf(train, eval_data, feat_cols, rf_params, horizon)

    metrics = {
        **clf_m,
        "mae":  float(acc_df["mae"].iloc[0]),
        "rmse": float(acc_df["rmse"].iloc[0]),
        "n_neg_train_boundary": int((train["target_price"] < 10.0).sum()),
        "xgb_best_iteration": int(getattr(model, "best_iteration", -1)),
    }
    return pred_df, acc_df, neg_df, metrics


# ══════════════════════════════════════════════════════════════════════
# Ridge pipeline
# ══════════════════════════════════════════════════════════════════════

def _ridge_pipeline(
    df: pd.DataFrame,
    split: dict,
    rf_params: dict,
    horizon: int,
    alpha: float = 10.0,
    feature_set: str = "full",
    **_,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    feat_cols, train, valid, test = _prep_tabular(df, split, horizon, feature_set)
    eval_data = pd.concat([valid, test]).reset_index(drop=True)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[feat_cols])
    X_ev = scaler.transform(eval_data[feat_cols])

    pred = Ridge(alpha=alpha).fit(X_tr, train["target_price"]).predict(X_ev)
    pred_df = _make_pred_df(eval_data, pred, "ridge", horizon)
    acc_df  = _build_acc(pred_df, "ridge", horizon)
    neg_df, clf_m = _neg_clf(train, eval_data, feat_cols, rf_params, horizon)

    return pred_df, acc_df, neg_df, {**clf_m, "mae": float(acc_df["mae"].iloc[0])}


# ══════════════════════════════════════════════════════════════════════
# LSTM pipeline
# ══════════════════════════════════════════════════════════════════════

def _lstm_pipeline(
    df: pd.DataFrame,
    split: dict,
    rf_params: dict,
    horizon: int,
    **_,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not _TORCH:
        raise ImportError("torch not installed. Run: pip install torch")
    from src.forecast_benchmark import (
        _prep_sequences, _LSTMNet, _train_torch, SEQ_LEN,
    )

    seqs = _prep_sequences(df, split, SEQ_LEN)
    mdl = _train_torch(
        _LSTMNet(seqs["n_features"]),
        seqs["X_train"], seqs["y_train"],
        seqs["X_valid"], seqs["y_valid"],
    )
    device = next(mdl.parameters()).device

    preds, actuals, dts = [], [], []
    for split_key in ("valid", "test"):
        X = torch.from_numpy(seqs[f"X_{split_key}"]).to(device)
        with torch.no_grad():
            preds.append(mdl(X).cpu().numpy())
        actuals.append(seqs[f"y_{split_key}"])
        dts.extend(seqs[f"dt_{split_key}"])

    pred_arr = np.concatenate(preds)
    actual_arr = np.concatenate(actuals)
    dt_arr = pd.DatetimeIndex(dts)

    pred_df = pd.DataFrame({
        "forecast_origin_datetime": dt_arr - pd.to_timedelta(horizon, "h"),
        "target_datetime": dt_arr,
        "horizon": horizon,
        "model": "lstm",
        "target": "price",
        "forecast": pred_arr.astype(float),
        "actual": actual_arr.astype(float),
    })
    acc_df = _build_acc(pred_df, "lstm", horizon)

    # Neg classifier uses tabular eval (same time period)
    feat_cols, train, valid_tab, test_tab = _prep_tabular(df, split, horizon)
    eval_tab = pd.concat([valid_tab, test_tab]).reset_index(drop=True)
    neg_df, clf_m = _neg_clf(train, eval_tab, feat_cols, rf_params, horizon)
    # Keep only rows whose datetimes overlap with LSTM coverage
    neg_df = neg_df[neg_df["target_datetime"].isin(dt_arr)].reset_index(drop=True)

    return pred_df, acc_df, neg_df, {**clf_m, "mae": float(acc_df["mae"].iloc[0])}


# ══════════════════════════════════════════════════════════════════════
# TCN pipeline
# ══════════════════════════════════════════════════════════════════════

def _tcn_pipeline(
    df: pd.DataFrame,
    split: dict,
    rf_params: dict,
    horizon: int,
    **_,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not _TORCH:
        raise ImportError("torch not installed. Run: pip install torch")
    from src.forecast_benchmark import (
        _prep_sequences, _TCNNet, _train_torch, SEQ_LEN,
    )

    seqs = _prep_sequences(df, split, SEQ_LEN)
    mdl = _train_torch(
        _TCNNet(seqs["n_features"]),
        seqs["X_train"], seqs["y_train"],
        seqs["X_valid"], seqs["y_valid"],
    )
    device = next(mdl.parameters()).device

    preds, actuals, dts = [], [], []
    for split_key in ("valid", "test"):
        X = torch.from_numpy(seqs[f"X_{split_key}"]).to(device)
        with torch.no_grad():
            preds.append(mdl(X).cpu().numpy())
        actuals.append(seqs[f"y_{split_key}"])
        dts.extend(seqs[f"dt_{split_key}"])

    pred_arr = np.concatenate(preds)
    actual_arr = np.concatenate(actuals)
    dt_arr = pd.DatetimeIndex(dts)

    pred_df = pd.DataFrame({
        "forecast_origin_datetime": dt_arr - pd.to_timedelta(horizon, "h"),
        "target_datetime": dt_arr,
        "horizon": horizon,
        "model": "tcn",
        "target": "price",
        "forecast": pred_arr.astype(float),
        "actual": actual_arr.astype(float),
    })
    acc_df = _build_acc(pred_df, "tcn", horizon)

    feat_cols, train, valid_tab, test_tab = _prep_tabular(df, split, horizon)
    eval_tab = pd.concat([valid_tab, test_tab]).reset_index(drop=True)
    neg_df, clf_m = _neg_clf(train, eval_tab, feat_cols, rf_params, horizon)
    neg_df = neg_df[neg_df["target_datetime"].isin(dt_arr)].reset_index(drop=True)

    return pred_df, acc_df, neg_df, {**clf_m, "mae": float(acc_df["mae"].iloc[0])}


# ══════════════════════════════════════════════════════════════════════
# Seasonal Naive pipeline (Lago et al. 2021 reference baseline)
# ══════════════════════════════════════════════════════════════════════

def _naive_seasonal_pipeline(
    df: pd.DataFrame,
    split: dict,
    rf_params: dict,
    horizon: int,
    lag_hours: int = 168,
    **_,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Seasonal Naive: p̂(t+h) = p(t+h - 168h) — same hour, 7 days earlier.

    This is the canonical EPF benchmark from Lago et al. (2021). All other
    models report rMAE = MAE_model / MAE_naive against this baseline.
    """
    work = df.copy()
    work["target_price"] = work["price"].shift(-horizon)
    work["naive_forecast"] = work["price"].shift(lag_hours - horizon)

    fit = (work[["datetime", "price", "target_price", "naive_forecast"]]
           .dropna()
           .reset_index(drop=True))
    n = len(fit)
    n_train = int(n * split["train_ratio"])
    n_valid = int(n * (split["train_ratio"] + split.get("valid_ratio", 0.15)))
    eval_data = fit.iloc[n_train:].reset_index(drop=True)

    pred = eval_data["naive_forecast"].to_numpy()
    pred_df = _make_pred_df(eval_data, pred, "naive_seasonal", horizon)
    acc_df = _build_acc(pred_df, "naive_seasonal", horizon)

    target_dt = (pd.to_datetime(eval_data["datetime"])
                 + pd.to_timedelta(horizon, "h"))
    p_neg = (eval_data["naive_forecast"] < 0).astype(float).to_numpy()
    neg_df = pd.DataFrame({"target_datetime": target_dt, "p_negative_price": p_neg})

    return pred_df, acc_df, neg_df, {
        "mae": float(acc_df["mae"].iloc[0]),
        "rmse": float(acc_df["rmse"].iloc[0]),
        "lag_hours": lag_hours,
    }


def compute_rmae(model_mae: float, naive_mae: float) -> float:
    """Relative MAE = MAE_model / MAE_naive. <1.0 means better than naive."""
    return float(model_mae) / float(naive_mae + 1e-9)
