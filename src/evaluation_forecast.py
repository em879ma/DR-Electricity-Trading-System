from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_forecast_accuracy(
    pred_df: pd.DataFrame,
    horizon: int = 24,
    target: str = "price",
) -> pd.DataFrame:
    """
    Level 1: Forecast accuracy evaluation.
    pred_df must have columns: target, horizon, forecast (or forecast_calibrated), actual.
    Returns one row per (model, target, horizon) with: MAE, RMSE, Bias, R2,
    high_price_recall, high_price_precision, negative_price_recall, negative_price_precision.
    """
    subset = pred_df[(pred_df["target"] == target) & (pred_df["horizon"] == horizon)].dropna(
        subset=["forecast", "actual"]
    )
    if subset.empty:
        return pd.DataFrame()

    rows = []
    for model_name, grp in subset.groupby("model"):
        rows.append(_forecast_metrics(model_name, target, horizon, grp["actual"], grp["forecast"]))

    # Also evaluate calibrated if present
    if "forecast_calibrated" in subset.columns:
        cal = subset.dropna(subset=["forecast_calibrated"])
        if not cal.empty:
            for model_name, grp in cal.groupby("model"):
                row = _forecast_metrics(f"{model_name}_calibrated", target, horizon, grp["actual"], grp["forecast_calibrated"])
                # Override negative_price_recall/precision with binary classifier if available
                if "p_negative_price" in grp.columns and grp["p_negative_price"].notna().any():
                    neg_actual = grp["actual"] < 0
                    p_clf = grp["p_negative_price"].fillna(0)
                    # Use F1-maximizing threshold from classifier training
                    best_f1, best_thr = 0.0, 0.5
                    for thr in np.arange(0.02, 0.51, 0.01):
                        pb = p_clf >= thr
                        _tp = int((neg_actual & pb).sum())
                        _fp = int((~neg_actual & pb).sum())
                        _fn = int((neg_actual & ~pb).sum())
                        if _tp + _fp + _fn == 0:
                            continue
                        f1 = 2 * _tp / (2 * _tp + _fp + _fn)
                        if f1 > best_f1:
                            best_f1, best_thr = f1, thr
                    neg_pred_clf = p_clf >= best_thr
                    tp = int((neg_actual & neg_pred_clf).sum())
                    fn = int((neg_actual & ~neg_pred_clf).sum())
                    fp = int((~neg_actual & neg_pred_clf).sum())
                    row["negative_price_recall"] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
                    row["negative_price_precision"] = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
                rows.append(row)

    return pd.DataFrame(rows)


def _forecast_metrics(model_name: str, target: str, horizon: int, y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    mask = y_true.notna() & y_pred.notna()
    yt, yp = y_true[mask], y_pred[mask]

    if len(yt) < 5:
        return {"model": model_name, "target": target, "horizon": horizon}

    p90 = float(yt.quantile(0.9))
    high_actual = yt >= p90
    high_pred = yp >= p90
    neg_actual = yt < 0
    neg_pred = yp < 0

    tp_high = int((high_actual & high_pred).sum())
    fp_high = int((~high_actual & high_pred).sum())
    fn_high = int((high_actual & ~high_pred).sum())
    tp_neg = int((neg_actual & neg_pred).sum())
    fp_neg = int((~neg_actual & neg_pred).sum())
    fn_neg = int((neg_actual & ~neg_pred).sum())

    return {
        "model": model_name,
        "target": target,
        "horizon": horizon,
        "n_obs": int(len(yt)),
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(mean_squared_error(yt, yp) ** 0.5),
        "Bias": float(np.mean(yp - yt)),
        "R2": float(r2_score(yt, yp)),
        "high_price_recall": tp_high / (tp_high + fn_high) if (tp_high + fn_high) > 0 else float("nan"),
        "high_price_precision": tp_high / (tp_high + fp_high) if (tp_high + fp_high) > 0 else float("nan"),
        "negative_price_recall": tp_neg / (tp_neg + fn_neg) if (tp_neg + fn_neg) > 0 else float("nan"),
        "negative_price_precision": tp_neg / (tp_neg + fp_neg) if (tp_neg + fp_neg) > 0 else float("nan"),
    }


def evaluate_all_horizons(
    pred_df: pd.DataFrame,
    target: str = "price",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    if horizons is None:
        horizons = sorted(pred_df["horizon"].unique().tolist())
    parts = []
    for h in horizons:
        parts.append(evaluate_forecast_accuracy(pred_df, horizon=h, target=target))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
