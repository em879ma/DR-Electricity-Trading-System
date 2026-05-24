from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


OPSD_DEFAULT_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)


@dataclass(frozen=True)
class DataConfig:
    country_code: str = "DE"
    start: str = "2018-01-01"
    end: str = "2018-12-31 23:00:00"
    local_cache_path: str = "data/raw/opsd_time_series_60min_singleindex.csv"
    source_url: str = OPSD_DEFAULT_URL
    random_seed: int = 42


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _download_opsd_csv(config: DataConfig) -> pd.DataFrame:
    cache_path = Path(config.local_cache_path)
    _ensure_parent(cache_path)
    if cache_path.exists():
        return pd.read_csv(cache_path)
    # Check legacy location before downloading
    legacy = Path("data/opsd-time_series-2020-10-06/time_series_60min_singleindex.csv")
    if legacy.exists():
        import shutil
        shutil.copy(legacy, cache_path)
        return pd.read_csv(cache_path)
    df = pd.read_csv(config.source_url)
    df.to_csv(cache_path, index=False)
    return df


def _select_or_default(df: pd.DataFrame, col: str, fallback: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.full(len(df), fallback), index=df.index)


def _select_first_available(df: pd.DataFrame, candidates: list[str], fallback: float = np.nan) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.full(len(df), fallback), index=df.index)


def _build_import_proxy(price: pd.Series, wind: pd.Series, solar: pd.Series) -> pd.Series:
    renewable_share = (wind.fillna(0.0) + solar.fillna(0.0)).clip(lower=0.0)
    renewable_norm = renewable_share / renewable_share.replace(0.0, np.nan).median()
    price_norm = price / price.replace(0.0, np.nan).median()
    proxy = (1.2 - 0.2 * price_norm.fillna(1.0) + 0.1 * renewable_norm.fillna(1.0)).clip(lower=0.6)
    return proxy.rename("import_index")


def _synthetic_data(config: DataConfig) -> pd.DataFrame:
    idx = pd.date_range(config.start, config.end, freq="h")
    rng = np.random.default_rng(config.random_seed)
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    base_load = 48_000 + 4_500 * np.sin(2 * np.pi * hour / 24) + 2_800 * np.cos(2 * np.pi * doy / 365)
    wind = 8_500 + 1_700 * np.sin(2 * np.pi * (hour + 6) / 24) + rng.normal(0, 700, len(idx))
    solar = np.maximum(0, 7_000 * np.sin(np.pi * (hour - 6) / 12)) + rng.normal(0, 350, len(idx))
    import_index = np.clip(1.0 + rng.normal(0, 0.08, len(idx)), 0.7, 1.3)
    price = (
        20
        + 0.0016 * base_load
        - 0.0012 * (wind + solar)
        - 3.0 * (import_index - 1.0)
        + rng.normal(0, 5.0, len(idx))
    )
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "price": np.clip(price, 5, None),
            "load": np.clip(base_load + rng.normal(0, 1200, len(idx)), 1000, None),
            "wind_generation": np.clip(wind, 0, None),
            "solar_generation": np.clip(solar, 0, None),
            "import_index": import_index,
            "data_source": "synthetic_fallback",
        }
    )
    return df


def load_market_data(config: DataConfig) -> pd.DataFrame:
    try:
        raw = _download_opsd_csv(config)
        raw["timestamp"] = pd.to_datetime(raw.get("utc_timestamp", raw.get("cet_cest_timestamp")))
        raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp")
        cc = config.country_code.upper()
        price = _select_first_available(
            raw,
            [
                f"{cc}_price_day_ahead",
                "DE_LU_price_day_ahead",  # OPSD common naming for Germany-Luxembourg zone
            ],
        )
        load = _select_or_default(raw, f"{cc}_load_actual_entsoe_transparency")
        wind = _select_or_default(raw, f"{cc}_wind_generation_actual")
        solar = _select_or_default(raw, f"{cc}_solar_generation_actual")

        df = pd.DataFrame(
            {
                "timestamp": raw["timestamp"],
                "price": price,
                "load": load,
                "wind_generation": wind,
                "solar_generation": solar,
            }
        )
        df = df[(df["timestamp"] >= config.start) & (df["timestamp"] <= config.end)].copy()
        df = df.dropna(subset=["price", "load"])
        df["wind_generation"] = df["wind_generation"].interpolate(limit_direction="both").fillna(0.0)
        df["solar_generation"] = df["solar_generation"].interpolate(limit_direction="both").fillna(0.0)
        df["import_index"] = _build_import_proxy(df["price"], df["wind_generation"], df["solar_generation"])
        df["data_source"] = "opsd"
        if len(df) < 24 * 30:
            return _synthetic_data(config)
        return df.reset_index(drop=True)
    except Exception:
        return _synthetic_data(config)
