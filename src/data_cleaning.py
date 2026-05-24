from __future__ import annotations

from pathlib import Path

import pandas as pd


def clean_market_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={"timestamp": "datetime"})
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    out["renewable_total"] = (
        out.get("renewable_total", None)
        if "renewable_total" in out.columns
        else out.get("wind_generation", pd.Series(0.0, index=out.index)) + out.get("solar_generation", pd.Series(0.0, index=out.index))
    )

    # Import data: prefer real imports column; otherwise use proxy flag
    if "imports" in out.columns:
        out["imports"] = pd.to_numeric(out["imports"], errors="coerce").fillna(0.0)
        out["has_import_data"] = True
        out["import_index_proxy"] = out["imports"]
    elif "import_index" in out.columns:
        out["imports"] = pd.to_numeric(out["import_index"], errors="coerce").fillna(1.0)
        out["has_import_data"] = False
        out["import_index_proxy"] = out["imports"]
    else:
        out["imports"] = 1.0
        out["has_import_data"] = False
        out["import_index_proxy"] = 1.0

    # Available capacity: proxy if real data absent
    if "available_capacity" in out.columns:
        out["has_capacity_data"] = True
        out["available_capacity_proxy"] = out["available_capacity"]
    else:
        out["has_capacity_data"] = False
        out["available_capacity"] = (
            out["load"].rolling(24, min_periods=1).max()
            + 0.5 * out["renewable_total"].rolling(24, min_periods=1).mean()
            + 1000.0 * pd.to_numeric(out["imports"], errors="coerce").fillna(1.0)
        )
        out["available_capacity_proxy"] = out["available_capacity"]

    # Add country column for multi-country support
    if "country" not in out.columns:
        out["country"] = "DE"

    return out


def save_market_panel(df: pd.DataFrame, path: str = "data/processed/market_panel.csv") -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
