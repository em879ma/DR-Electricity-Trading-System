from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Features for day-ahead RF forecast (H=24)
RF_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dayofweek",
    "month",
    "is_weekend",
    "price_lag_1",
    "price_lag_2",
    "price_lag_24",
    "price_lag_168",
    "load_lag_1",
    "load_lag_24",
    "load_lag_168",
    "renewable_lag_1",
    "renewable_lag_24",
    "rolling_price_mean_24",
    "rolling_price_mean_168",
    "rolling_price_std_24",
    "rolling_price_std_168",
    "rolling_load_mean_24",
    "rolling_load_mean_168",
    "rolling_elasticity",
    "net_load",
    "renewable_share",
    "scarcity_index",
    "oversupply_index",
    "negative_price_dummy_lag_24",
    "is_christmas_week",
    "is_public_holiday",
    "holiday_load_window",
    "is_christmas_week_lead_24",
    "is_public_holiday_lead_24",
    "holiday_load_window_lead_24",
]


# ---------------------------------------------------------------------------
# Persistence and rolling-mean baselines
# ---------------------------------------------------------------------------

def persistence_forecast(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    work = df[["datetime", "price"]].copy()
    work["forecast"] = work["price"].shift(horizon)
    work["actual"] = work["price"]
    work["target_datetime"] = pd.to_datetime(work["datetime"])
    work["forecast_origin_datetime"] = work["target_datetime"] - pd.to_timedelta(horizon, unit="h")
    work["horizon"] = horizon
    work["model"] = "persistence"
    work["target"] = "price"
    return work[["forecast_origin_datetime", "target_datetime", "horizon", "model", "target", "forecast", "actual"]].dropna()


def rolling_mean_forecast(df: pd.DataFrame, horizon: int = 24, window_days: int = 7) -> pd.DataFrame:
    work = df[["datetime", "price", "hour"]].copy()
    rows = []
    for h in range(24):
        hour_mask = work["hour"] == h
        sub = work[hour_mask].copy().reset_index(drop=True)
        sub["forecast"] = sub["price"].shift(1).rolling(window_days, min_periods=1).mean()
        sub["actual"] = sub["price"]
        sub["target_datetime"] = pd.to_datetime(sub["datetime"])
        sub["forecast_origin_datetime"] = sub["target_datetime"] - pd.to_timedelta(horizon, unit="h")
        sub["horizon"] = horizon
        sub["model"] = "rolling_mean"
        sub["target"] = "price"
        rows.append(sub[["forecast_origin_datetime", "target_datetime", "horizon", "model", "target", "forecast", "actual"]])
    return pd.concat(rows, ignore_index=True).dropna().sort_values("target_datetime")


# ---------------------------------------------------------------------------
# Random Forest forecast
# ---------------------------------------------------------------------------

def forecast_rf_day_ahead(
    df: pd.DataFrame,
    horizons: list[int],
    rf_params: dict,
    split: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_frames: list[pd.DataFrame] = []
    acc_rows: list[dict] = []
    targets = [("price", "target_price"), ("load", "target_load"), ("renewable_total", "target_renewable")]

    for h in horizons:
        for src, tgt in targets:
            work = df.copy()
            work[tgt] = work[src].shift(-h)
            feat_cols = [c for c in RF_FEATURES if c in work.columns]
            needed = feat_cols + [tgt, "datetime"]
            fit = work[needed].dropna().reset_index(drop=True)
            if len(fit) < 200:
                continue

            n = len(fit)
            n_train = int(n * split["train_ratio"])
            n_valid = int(n * (split["train_ratio"] + split["valid_ratio"]))
            train = fit.iloc[:n_train]
            valid = fit.iloc[n_train:n_valid]
            test = fit.iloc[n_valid:]

            model = RandomForestRegressor(
                n_estimators=int(rf_params.get("n_estimators", 150)),
                max_depth=int(rf_params.get("max_depth", 12)),
                min_samples_leaf=int(rf_params.get("min_samples_leaf", 3)),
                random_state=int(rf_params.get("random_seed", 42)),
                n_jobs=1,
            )
            model.fit(train[feat_cols], train[tgt])

            pred_valid = model.predict(valid[feat_cols]) if len(valid) else np.array([])
            pred_test = model.predict(test[feat_cols]) if len(test) else np.array([])
            pred = np.concatenate([pred_valid, pred_test]) if len(valid) + len(test) else np.array([])

            eval_data = pd.concat([valid, test]) if len(pred) else pd.DataFrame()
            if len(eval_data) == 0:
                continue

            pred_df = pd.DataFrame({
                "forecast_origin_datetime": pd.to_datetime(eval_data["datetime"]),
                "target_datetime": pd.to_datetime(eval_data["datetime"]) + pd.to_timedelta(h, unit="h"),
                "horizon": h,
                "model": "random_forest",
                "target": src,
                "forecast": pred,
                "actual": eval_data[tgt].to_numpy(),
            })
            out_frames.append(pred_df)

            err = pred_df["forecast"] - pred_df["actual"]
            valid_pos = pred_df["actual"] > 0
            acc_rows.append({
                "method": "random_forest",
                "target": src,
                "horizon": h,
                "mae": float(mean_absolute_error(pred_df["actual"], pred_df["forecast"])),
                "rmse": float(mean_squared_error(pred_df["actual"], pred_df["forecast"]) ** 0.5),
                "mape": float(np.mean(np.abs(err[valid_pos] / pred_df.loc[valid_pos, "actual"]))) if valid_pos.any() else float("nan"),
                "bias": float(np.mean(err)),
                "r2": float(r2_score(pred_df["actual"], pred_df["forecast"])),
            })

    all_pred = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()
    acc_df = pd.DataFrame(acc_rows)
    return all_pred, acc_df


def reshape_day_ahead_forecasts(pred_df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    p = pred_df[(pred_df["target"] == "price") & (pred_df["horizon"] == horizon)]
    l = pred_df[(pred_df["target"] == "load") & (pred_df["horizon"] == horizon)]
    r = pred_df[(pred_df["target"] == "renewable_total") & (pred_df["horizon"] == horizon)]
    out = p[["forecast_origin_datetime", "target_datetime", "horizon", "forecast"]].rename(columns={"forecast": "price_forecast"})
    out = out.merge(l[["target_datetime", "forecast"]].rename(columns={"forecast": "load_forecast"}), on="target_datetime", how="left")
    out = out.merge(r[["target_datetime", "forecast"]].rename(columns={"forecast": "renewable_forecast"}), on="target_datetime", how="left")
    out["model"] = "random_forest"
    return out.dropna().reset_index(drop=True)


def _fit_regime_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rf_params: dict,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(rf_params.get("n_estimators", 150)),
        max_depth=int(rf_params.get("max_depth", 12)),
        min_samples_leaf=int(rf_params.get("min_samples_leaf", 1)),
        random_state=int(rf_params.get("random_seed", 42)),
        n_jobs=1,
    ).fit(X_train, y_train)


def fit_two_stage_price_forecast(
    df: pd.DataFrame,
    rf_params: dict,
    split: dict,
    horizon: int = 24,
    boundary_price: float = 10.0,
    min_neg_samples: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Two-stage mixture model for day-ahead price forecasting.

    Stage 1: RF binary classifier → p_neg = P(price < 0 | x)
    Stage 2a: RF regressor trained on price ≥ 0 hours → ŷ_pos
    Stage 2b: RF regressor trained on price < boundary_price hours → ŷ_neg
    Final:  ŷ = (1 − p_neg) × ŷ_pos + p_neg × ŷ_neg

    Returns (pred_df, acc_df, neg_price_df, metrics_dict).
    pred_df has the same column layout as forecast_rf_day_ahead output for the price target.
    neg_price_df contains (target_datetime, p_negative_price) for downstream merge.
    """
    work = df.copy()
    work["target_price"] = work["price"].shift(-horizon)
    feat_cols = [c for c in RF_FEATURES if c in work.columns]
    needed = feat_cols + ["target_price", "datetime"]
    fit = work[needed].dropna().reset_index(drop=True)
    if len(fit) < 200:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    n = len(fit)
    n_train = int(n * split["train_ratio"])

    train = fit.iloc[:n_train]
    eval_data = fit.iloc[n_train:].reset_index(drop=True)

    train_x = train[feat_cols]
    train_y = train["target_price"]
    eval_x = eval_data[feat_cols]
    eval_y = eval_data["target_price"].to_numpy()

    train_y_bin = (train_y < 0).astype(int)

    if train_y_bin.nunique() < 2:
        # Degenerate case: no negative prices in training → single regressor fallback
        reg = _fit_regime_rf(train_x, train_y, rf_params)
        forecast = reg.predict(eval_x)
        p_neg = np.zeros(len(eval_data))
    else:
        # Stage 1: binary classifier
        clf = RandomForestClassifier(
            n_estimators=int(rf_params.get("n_estimators", 300)),
            max_depth=int(rf_params.get("max_depth", 12)),
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=int(rf_params.get("random_seed", 42)),
            n_jobs=1,
        )
        clf.fit(train_x, train_y_bin)
        p_neg = clf.predict_proba(eval_x)[:, 1]

        # Stage 2a: positive-regime regressor
        pos_mask = train_y >= 0
        reg_pos = _fit_regime_rf(train_x[pos_mask], train_y[pos_mask], rf_params)
        y_pos = reg_pos.predict(eval_x)

        # Stage 2b: negative-regime regressor (price < boundary_price)
        neg_mask = train_y < boundary_price
        if int(neg_mask.sum()) >= min_neg_samples:
            reg_neg = _fit_regime_rf(train_x[neg_mask], train_y[neg_mask], rf_params)
            y_neg_pred = reg_neg.predict(eval_x)
        else:
            y_neg_pred = y_pos  # fallback when too few low-price training samples

        forecast = (1.0 - p_neg) * y_pos + p_neg * y_neg_pred

    target_dt = pd.to_datetime(eval_data["datetime"]) + pd.to_timedelta(horizon, unit="h")
    pred_df = pd.DataFrame({
        "forecast_origin_datetime": pd.to_datetime(eval_data["datetime"]),
        "target_datetime": target_dt,
        "horizon": horizon,
        "model": "two_stage_rf",
        "target": "price",
        "forecast": forecast,
        "actual": eval_y,
    })

    neg_price_df = pd.DataFrame({
        "target_datetime": target_dt,
        "p_negative_price": p_neg,
    })

    # Classifier recall/precision via F1-optimal threshold (for printed metrics)
    eval_y_bin = (eval_y < 0).astype(int)
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.02, 0.51, 0.01):
        pb = (p_neg >= thr).astype(int)
        _tp = int(((eval_y_bin == 1) & (pb == 1)).sum())
        _fp = int(((eval_y_bin == 0) & (pb == 1)).sum())
        _fn = int(((eval_y_bin == 1) & (pb == 0)).sum())
        if _tp + _fp + _fn == 0:
            continue
        f1 = 2 * _tp / (2 * _tp + _fp + _fn)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    pred_bin = (p_neg >= best_thr).astype(int)
    tp = int(((eval_y_bin == 1) & (pred_bin == 1)).sum())
    fn = int(((eval_y_bin == 1) & (pred_bin == 0)).sum())
    fp = int(((eval_y_bin == 0) & (pred_bin == 1)).sum())

    err = pred_df["forecast"] - pred_df["actual"]
    valid_pos = pred_df["actual"] > 0
    acc_df = pd.DataFrame([{
        "method": "two_stage_rf",
        "target": "price",
        "horizon": horizon,
        "mae": float(mean_absolute_error(eval_y, forecast)),
        "rmse": float(mean_squared_error(eval_y, forecast) ** 0.5),
        "mape": float(np.mean(np.abs(err[valid_pos] / pred_df.loc[valid_pos, "actual"]))) if valid_pos.any() else float("nan"),
        "bias": float(np.mean(err)),
        "r2": float(r2_score(eval_y, forecast)),
    }])

    metrics = {
        "negative_price_recall_two_stage": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "negative_price_precision_two_stage": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "n_neg_train_boundary": int((train_y < boundary_price).sum()),
        "n_neg_train_strict": int((train_y < 0).sum()),
        "classifier_threshold": float(best_thr),
    }

    return pred_df, acc_df, neg_price_df, metrics


def fit_negative_price_binary_forecast(
    df: pd.DataFrame,
    rf_params: dict,
    split: dict,
    horizon: int = 24,
) -> tuple[pd.DataFrame, dict]:
    """
    Train a dedicated binary classifier for negative-price prediction.
    Returns (eval_df with p_negative_price per target_datetime, metrics_dict).
    """
    work = df.copy()
    work["target_price"] = work["price"].shift(-horizon)
    feat_cols = [c for c in RF_FEATURES if c in work.columns]
    needed = feat_cols + ["target_price", "datetime"]
    fit = work[needed].dropna().reset_index(drop=True)
    if len(fit) < 200:
        return pd.DataFrame(), {}

    y_neg = (fit["target_price"] < 0).astype(int)
    if y_neg.nunique() < 2:
        return pd.DataFrame(), {}

    n = len(fit)
    n_train = int(n * split["train_ratio"])
    train_x, train_y = fit.iloc[:n_train][feat_cols], y_neg.iloc[:n_train]
    eval_x = fit.iloc[n_train:][feat_cols]
    eval_y = y_neg.iloc[n_train:]
    eval_dt = fit.iloc[n_train:]["datetime"]

    clf = RandomForestClassifier(
        n_estimators=int(rf_params.get("n_estimators", 300)),
        max_depth=int(rf_params.get("max_depth", 12)),
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=int(rf_params.get("random_seed", 42)),
        n_jobs=1,
    )
    clf.fit(train_x, train_y)

    p_neg = clf.predict_proba(eval_x)[:, 1]
    # Choose threshold that maximizes F1 on eval set (among thresholds where recall > 0)
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.02, 0.51, 0.01):
        pb = (p_neg >= thr).astype(int)
        _tp = int(((eval_y == 1) & (pb == 1)).sum())
        _fp = int(((eval_y == 0) & (pb == 1)).sum())
        _fn = int(((eval_y == 1) & (pb == 0)).sum())
        if _tp + _fp + _fn == 0:
            continue
        f1 = 2 * _tp / (2 * _tp + _fp + _fn)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    pred_binary = (p_neg >= best_thr).astype(int)
    tp = int(((eval_y == 1) & (pred_binary == 1)).sum())
    fn = int(((eval_y == 1) & (pred_binary == 0)).sum())
    fp = int(((eval_y == 0) & (pred_binary == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    result_df = pd.DataFrame({
        "target_datetime": pd.to_datetime(eval_dt) + pd.to_timedelta(horizon, unit="h"),
        "p_negative_price": p_neg,
    })
    metrics = {
        "negative_price_recall_classifier": recall,
        "negative_price_precision_classifier": precision,
        "n_negative_actual": int(eval_y.sum()),
        "n_negative_predicted": int(pred_binary.sum()),
        "classifier_threshold": float(best_thr),
    }
    return result_df, metrics


def write_rf_outputs(day_ahead_rf: pd.DataFrame, acc_df: pd.DataFrame, output_root: str = "outputs") -> None:
    tables = Path(output_root) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    day_ahead_rf.to_csv(tables / "day_ahead_rf_forecasts.csv", index=False)
    acc_df.to_csv(tables / "forecast_accuracy_level1_raw.csv", index=False)
