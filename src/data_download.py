from __future__ import annotations

from pathlib import Path

import pandas as pd


OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)
OPSD_LOCAL = Path("data/raw/opsd_time_series_60min_singleindex.csv")

EMBER_LOCAL = Path("data/raw/ember_yearly_electricity_data.csv")


def download_opsd(force: bool = False) -> Path:
    OPSD_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if OPSD_LOCAL.exists() and not force:
        print(f"OPSD already cached at {OPSD_LOCAL}")
        return OPSD_LOCAL
    # Check legacy location
    legacy = Path("data/opsd-time_series-2020-10-06/time_series_60min_singleindex.csv")
    if legacy.exists() and not force:
        import shutil
        shutil.copy(legacy, OPSD_LOCAL)
        print(f"Copied from legacy location to {OPSD_LOCAL}")
        return OPSD_LOCAL
    print(f"Downloading OPSD time series from {OPSD_URL} ...")
    df = pd.read_csv(OPSD_URL)
    df.to_csv(OPSD_LOCAL, index=False)
    print(f"Saved to {OPSD_LOCAL} ({len(df):,} rows)")
    return OPSD_LOCAL


def download_ember(force: bool = False) -> Path | None:
    EMBER_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if EMBER_LOCAL.exists() and not force:
        print(f"Ember data already cached at {EMBER_LOCAL}")
        return EMBER_LOCAL
    # Ember does not have a stable direct CSV download URL without registration.
    # Try a known public mirror URL; skip gracefully on failure.
    ember_url = "https://ember-energy.org/app/uploads/2024/05/yearly_full_release_long_format.csv"
    try:
        print(f"Attempting Ember download from {ember_url} ...")
        df = pd.read_csv(ember_url)
        df.to_csv(EMBER_LOCAL, index=False)
        print(f"Saved Ember data to {EMBER_LOCAL} ({len(df):,} rows)")
        return EMBER_LOCAL
    except Exception as exc:
        print(f"Ember download failed ({exc}). Skipping — energy-mix analysis will use proxies.")
        return None


def load_opsd_raw() -> pd.DataFrame:
    if not OPSD_LOCAL.exists():
        download_opsd()
    return pd.read_csv(OPSD_LOCAL)


def load_ember_raw() -> pd.DataFrame | None:
    if not EMBER_LOCAL.exists():
        result = download_ember()
        if result is None:
            return None
    try:
        return pd.read_csv(EMBER_LOCAL)
    except Exception:
        return None
