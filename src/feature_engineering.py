from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["datetime"])
    out["hour"] = ts.dt.hour
    out["dayofweek"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    for lag in [1, 2, 3, 24, 48, 168]:
        out[f"price_lag_{lag}"] = out["price"].shift(lag)
    for lag in [1, 2, 24, 48, 168]:
        out[f"load_lag_{lag}"] = out["load"].shift(lag)
    out["renewable_lag_1"] = out["renewable_total"].shift(1)
    out["renewable_lag_24"] = out["renewable_total"].shift(24)

    out["rolling_price_mean_24"] = out["price"].rolling(24, min_periods=6).mean()
    out["rolling_price_mean_168"] = out["price"].rolling(168, min_periods=24).mean()
    out["rolling_price_std_24"] = out["price"].rolling(24, min_periods=6).std()
    out["rolling_price_std_168"] = out["price"].rolling(168, min_periods=24).std()
    out["rolling_load_mean_24"] = out["load"].rolling(24, min_periods=6).mean()
    out["rolling_load_mean_168"] = out["load"].rolling(168, min_periods=24).mean()

    p90 = out["price"].quantile(0.9)
    out["high_price_dummy"] = (out["price"] >= p90).astype(int)
    out["negative_price_dummy"] = (out["price"] < 0).astype(int)
    out["negative_price_dummy_lag_24"] = out["negative_price_dummy"].shift(24)
    out["price_spike"] = (out["price"] - out["rolling_price_mean_168"]).clip(lower=0.0)

    # System-structure features
    renewable_total = out.get("renewable_total", out.get("wind_generation", pd.Series(0.0, index=out.index)) + out.get("solar_generation", pd.Series(0.0, index=out.index)))
    load = out["load"].astype(float)
    wind = out.get("wind_generation", pd.Series(0.0, index=out.index)).astype(float)
    solar = out.get("solar_generation", pd.Series(0.0, index=out.index)).astype(float)

    out["net_load"] = load - renewable_total
    out["renewable_share"] = renewable_total / (load + 1e-6)
    out["wind_share"] = wind / (load + 1e-6)
    out["solar_share"] = solar / (load + 1e-6)

    if "available_capacity" in out.columns and "imports" in out.columns:
        out["scarcity_index"] = load / (out["available_capacity"].astype(float) + out["imports"].astype(float) + 1e-6)
        out["oversupply_index"] = renewable_total + out["imports"].astype(float) - load
    else:
        # proxies when capacity/import data unavailable
        out["scarcity_index"] = load / (renewable_total + 1e-6)
        out["oversupply_index"] = renewable_total - load
        out["scarcity_proxy"] = out["scarcity_index"]
        out["oversupply_proxy"] = out["oversupply_index"]

    net_load_p10 = float(out["net_load"].quantile(0.10))
    ren_share_p90 = float(out["renewable_share"].quantile(0.90))
    out["low_net_load_dummy"] = (out["net_load"] <= net_load_p10).astype(int)
    out["high_renewable_dummy"] = (out["renewable_share"] >= ren_share_p90).astype(int)

    # Holiday/low-load calendar features (German market)
    month = ts.dt.month
    day = ts.dt.day
    # German public holidays in Q4: Oct 3 (Unity Day), Dec 25 (Christmas), Dec 26 (Boxing Day)
    german_holidays = (
        ((month == 10) & (day == 3)) |
        ((month == 12) & (day == 25)) |
        ((month == 12) & (day == 26))
    )
    out["is_public_holiday"] = german_holidays.astype(int)
    # Christmas–New Year window (Dec 24–Jan 2): anomalous industrial load drop
    out["is_christmas_week"] = (
        ((month == 12) & (day >= 24)) | ((month == 1) & (day <= 2))
    ).astype(int)
    # Extended low-load holiday window (Dec 21–31 + Jan 1–2): covers Dec 21–22 negative prices
    out["holiday_load_window"] = (
        ((month == 12) & (day >= 21)) |
        ((month == 1) & (day <= 2)) |
        ((month == 10) & (day == 3))
    ).astype(int)
    # Lead-24 versions: describe the TARGET hour for day-ahead forecasting
    out["is_public_holiday_lead_24"] = out["is_public_holiday"].shift(-24)
    out["is_christmas_week_lead_24"] = out["is_christmas_week"].shift(-24)
    out["holiday_load_window_lead_24"] = out["holiday_load_window"].shift(-24)

    out = out.bfill().ffill()
    return out
