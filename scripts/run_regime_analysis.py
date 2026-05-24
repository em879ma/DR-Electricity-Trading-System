"""
Standalone: compute energy-mix regime summary from market_panel_with_dr.csv.

Renewable regime is determined from hourly renewable_share (no Ember needed).
Outputs: outputs/tables/energy_mix_regime_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.energy_mix import build_energy_mix_regime_summary


def main() -> None:
    print("Loading market panel...")
    panel = pd.read_csv(
        PROJECT_ROOT / "data/processed/market_panel_with_dr.csv",
        parse_dates=["datetime"],
    )
    print(f"  {len(panel)} rows, {panel.shape[1]} columns")
    print(f"  Period: {panel['datetime'].min()} – {panel['datetime'].max()}")

    # Compute energy structure features (not in old panel)
    panel["renewable_share"] = panel["renewable_total"] / (panel["load"] + 1e-6)
    panel["scarcity_index"] = (
        panel["load"] / (panel["available_capacity"] + panel["imports"] + 1e-6)
    )
    panel["oversupply_index"] = (
        panel["renewable_total"] + panel["imports"] - panel["load"]
    )

    # Merge DR schedule for q_DA and expected counterfactual price
    sched_path = PROJECT_ROOT / "outputs/tables/day_ahead_optimal_dr_schedule.csv"
    if sched_path.exists():
        sched = pd.read_csv(sched_path, parse_dates=["target_datetime"])
        sched = sched.rename(columns={
            "target_datetime": "datetime",
            "expected_counterfactual_price": "counterfactual_price",
        })[["datetime", "q_DA_optimal", "counterfactual_price"]]

        # align timezone awareness
        p_tz = panel["datetime"].dt.tz
        s_tz = sched["datetime"].dt.tz
        if p_tz is not None and s_tz is None:
            sched["datetime"] = sched["datetime"].dt.tz_localize("UTC")
        elif p_tz is None and s_tz is not None:
            sched["datetime"] = sched["datetime"].dt.tz_localize(None)

        panel = panel.merge(sched, on="datetime", how="left")
        panel["q_DA_optimal"] = panel["q_DA_optimal"].fillna(0.0)
        n_sched = sched["q_DA_optimal"].notna().sum()
        print(f"  Merged DR schedule: {n_sched} optimization hours (rest q_DA=0)")
    else:
        panel["q_DA_optimal"] = 0.0
        panel["counterfactual_price"] = float("nan")
        print("  No DR schedule found — q_DA set to 0")

    out_path = PROJECT_ROOT / "outputs/tables/energy_mix_regime_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    regime_df = build_energy_mix_regime_summary(panel, output_path=out_path)

    print("\n=== Energy-Mix Regime Summary ===")
    print(regime_df.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
