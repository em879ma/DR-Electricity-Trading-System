"""
Run 02: Estimate rolling elasticity, baseline demand, and responsive demand.
Saves data/processed/market_panel_with_dr.csv.

Note: rolling_elasticity is a rolling OLS slope (feature), NOT causal elasticity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import json
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.elasticity import estimate_rolling_elasticity, estimate_dynamic_elasticity, write_elasticity_outputs
from src.demand_response import estimate_baseline_demand, construct_responsive_demand, write_market_panel_with_dr
from src.counterfactual_price_model import add_structural_price_features


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out_root = cfg["output"]["root"]

    panel_path = Path("data/processed/market_panel.csv")
    if not panel_path.exists():
        raise FileNotFoundError(f"Run 01 first. Missing: {panel_path}")

    print("Loading market panel...")
    feat = pd.read_csv(panel_path, parse_dates=["datetime"])

    print("Estimating rolling elasticity (feature, not causal elasticity)...")
    # rolling_elasticity expects 'price' and 'load' columns and timestamp index named 'timestamp'
    feat_for_elast = feat.rename(columns={"datetime": "timestamp"})
    feat["rolling_elasticity"] = estimate_rolling_elasticity(feat_for_elast).values
    print(f"  rolling_elasticity range: [{feat['rolling_elasticity'].min():.3f}, {feat['rolling_elasticity'].max():.3f}]")

    print("Estimating dynamic elasticity parameters...")
    params, elasticity_table = estimate_dynamic_elasticity(feat)
    write_elasticity_outputs(params, elasticity_table, output_root=out_root)
    print(f"  epsilon_0={params['epsilon_0']:.4f}, R2={params.get('r_squared', float('nan')):.3f}")

    print("Estimating baseline demand...")
    dr_df = estimate_baseline_demand(feat)

    print("Constructing responsive demand...")
    dr_df = construct_responsive_demand(dr_df, params)

    print("Adding structural price features...")
    dr_df = add_structural_price_features(dr_df)

    write_market_panel_with_dr(dr_df)
    print(f"Saved market_panel_with_dr.csv ({len(dr_df)} rows)")

    tables_dir = Path(out_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    elasticity_table.to_csv(tables_dir / "elasticity_parameter_estimates.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
