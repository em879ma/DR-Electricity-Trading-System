from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calibrate_forecasts(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calibrate raw RF forecasts using linear regression on validation split.
    Returns (calibrated_long_df, calibration_summary_df).
    Linear calibration: actual = a + b * raw_forecast
    Also computes mean-bias correction; keeps whichever gives lower validation MAE.
    """
    if pred_df.empty:
        return pred_df.copy(), pd.DataFrame()

    calibrated_parts: list[pd.DataFrame] = []
    rows: list[dict] = []

    for target, g in pred_df.groupby("target"):
        work = g.sort_values("target_datetime").reset_index(drop=True).copy()
        n = len(work)
        split = max(int(n * 0.5), 1)
        val = work.iloc[:split]
        test = work.iloc[split:]
        if len(val) < 5:
            val = work
            test = work

        raw_mae = float(mean_absolute_error(val["actual"], val["forecast"]))
        raw_rmse = float(mean_squared_error(val["actual"], val["forecast"]) ** 0.5)
        raw_bias = float(np.mean(val["forecast"] - val["actual"]))

        # Linear calibration: actual = a + b * raw
        lr = LinearRegression()
        lr.fit(val[["forecast"]], val["actual"])
        work["forecast_linear_cal"] = lr.predict(work[["forecast"]])

        # Mean-bias correction
        bias_correction = float(np.mean(val["forecast"] - val["actual"]))
        work["forecast_bias_cal"] = work["forecast"] - bias_correction

        # Evaluate both on validation split
        cal_val = work.iloc[:len(val)]
        mae_linear = float(mean_absolute_error(cal_val["actual"], cal_val["forecast_linear_cal"]))
        mae_bias = float(mean_absolute_error(cal_val["actual"], cal_val["forecast_bias_cal"]))

        # Keep the better calibration
        if mae_linear <= mae_bias:
            work["forecast_calibrated"] = work["forecast_linear_cal"]
            cal_method = "linear"
            cal_intercept = float(lr.intercept_)
            cal_slope = float(lr.coef_[0])
            cal_bias_val = float(np.mean(cal_val["forecast_linear_cal"] - cal_val["actual"]))
            cal_mae = mae_linear
            cal_rmse = float(mean_squared_error(cal_val["actual"], cal_val["forecast_linear_cal"]) ** 0.5)
        else:
            work["forecast_calibrated"] = work["forecast_bias_cal"]
            cal_method = "bias_correction"
            cal_intercept = float(-bias_correction)
            cal_slope = 1.0
            cal_bias_val = float(np.mean(cal_val["forecast_bias_cal"] - cal_val["actual"]))
            cal_mae = mae_bias
            cal_rmse = float(mean_squared_error(cal_val["actual"], cal_val["forecast_bias_cal"]) ** 0.5)

        rows.append({
            "target": target,
            "calibration_method": cal_method,
            "calibration_intercept": cal_intercept,
            "calibration_slope": cal_slope,
            "raw_mae": raw_mae,
            "raw_rmse": raw_rmse,
            "raw_bias": raw_bias,
            "calibrated_mae": cal_mae,
            "calibrated_rmse": cal_rmse,
            "calibrated_bias": cal_bias_val,
        })
        calibrated_parts.append(work)

    calibrated = pd.concat(calibrated_parts, ignore_index=True)
    summary = pd.DataFrame(rows)
    return calibrated, summary


def reshape_calibrated_day_ahead_forecasts(pred_df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    p = pred_df[(pred_df["target"] == "price") & (pred_df["horizon"] == horizon)]
    l = pred_df[(pred_df["target"] == "load") & (pred_df["horizon"] == horizon)]
    r = pred_df[(pred_df["target"] == "renewable_total") & (pred_df["horizon"] == horizon)]
    out = p[["forecast_origin_datetime", "target_datetime", "horizon", "forecast_calibrated"]].rename(
        columns={"forecast_calibrated": "price_forecast"}
    )
    out = out.merge(
        l[["target_datetime", "forecast_calibrated"]].rename(columns={"forecast_calibrated": "load_forecast"}),
        on="target_datetime", how="left",
    )
    out = out.merge(
        r[["target_datetime", "forecast_calibrated"]].rename(columns={"forecast_calibrated": "renewable_forecast"}),
        on="target_datetime", how="left",
    )
    out["model"] = "rf_calibrated"
    return out.dropna().reset_index(drop=True)


def slice_calibrated_day_ahead_block(
    pred_df: pd.DataFrame, origin_dt: pd.Timestamp, horizon: int = 24
) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    o = pd.Timestamp(origin_dt)
    p = pred_df[(pred_df["target"] == "price") & (pred_df["horizon"] == horizon) & (pred_df["forecast_origin_datetime"] == o)]
    l = pred_df[(pred_df["target"] == "load") & (pred_df["horizon"] == horizon) & (pred_df["forecast_origin_datetime"] == o)]
    r = pred_df[(pred_df["target"] == "renewable_total") & (pred_df["horizon"] == horizon) & (pred_df["forecast_origin_datetime"] == o)]
    if p.empty:
        return pd.DataFrame()
    out = p[["forecast_origin_datetime", "target_datetime", "horizon", "forecast_calibrated"]].rename(
        columns={"forecast_calibrated": "price_forecast"}
    )
    out = out.merge(l[["target_datetime", "forecast_calibrated"]].rename(columns={"forecast_calibrated": "load_forecast"}), on="target_datetime", how="left")
    out = out.merge(r[["target_datetime", "forecast_calibrated"]].rename(columns={"forecast_calibrated": "renewable_forecast"}), on="target_datetime", how="left")
    out["model"] = "rf_calibrated"
    return out.dropna().reset_index(drop=True)


def write_calibration_outputs(
    calibrated_long: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_root: str = "outputs",
) -> None:
    tables = Path(output_root) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(tables / "forecast_calibration_summary.csv", index=False)
