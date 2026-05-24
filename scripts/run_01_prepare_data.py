"""
Run 01: Load, clean, and prepare market panel with energy-mix features.
Saves data/processed/market_panel.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import DataConfig, load_market_data
from src.data_cleaning import clean_market_panel
from src.feature_engineering import engineer_features
from src.data_download import load_ember_raw
from src.energy_mix import build_ember_annual_features, merge_ember_to_panel


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_cfg = DataConfig(
        country_code=cfg["data"]["country_code"],
        start=cfg["data"]["start"],
        end=cfg["data"]["end"],
        local_cache_path=cfg["data"].get(
            "local_cache_path",
            "data/raw/opsd_time_series_60min_singleindex.csv"
        ),
    )

    print("Loading OPSD market data...")
    raw = load_market_data(data_cfg)
    print(f"  Loaded {len(raw)} rows (source: {raw['data_source'].iloc[0] if 'data_source' in raw.columns else 'unknown'})")

    print("Cleaning market panel...")
    panel = clean_market_panel(raw)

    print("Engineering features...")
    feat = engineer_features(panel)

    print("Merging Ember annual energy-mix data (if available)...")
    ember_raw = load_ember_raw()
    if ember_raw is not None:
        ember_features = build_ember_annual_features(ember_raw, country=data_cfg.country_code)
        feat = merge_ember_to_panel(feat, ember_features)
        print(f"  Merged Ember data: {len(ember_features)} years")
    else:
        print("  Ember data not available — energy-mix annual features skipped.")

    out_path = Path("data/processed/market_panel.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out_path, index=False)
    print(f"Saved market panel: {out_path} ({len(feat)} rows, {feat.shape[1]} cols)")
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
