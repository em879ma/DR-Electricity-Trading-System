from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_residual_history(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    out["residual"] = out["actual"] - out["forecast"]
    return out.dropna(subset=["residual"])


def generate_bootstrap_scenarios(
    day_ahead_rf: pd.DataFrame,
    residual_history: pd.DataFrame,
    n_scenarios: int = 100,
    block_bootstrap: bool = True,
    block_length: int = 24,
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, float | str | int]] = []
    target_names = {"price_forecast": "price", "load_forecast": "load", "renewable_forecast": "renewable_total"}
    residual_map = {
        tgt: residual_history[residual_history["target"] == base]["residual"].dropna().to_numpy()
        for tgt, base in target_names.items()
    }

    # Use all available forecast rows, grouped by forecast day (24-hour blocks)
    all_dates = pd.to_datetime(day_ahead_rf["forecast_origin_datetime"]).dt.date.unique()
    all_dates = sorted(all_dates)

    for day in all_dates:
        day_mask = pd.to_datetime(day_ahead_rf["forecast_origin_datetime"]).dt.date == day
        base = day_ahead_rf[day_mask].copy().reset_index(drop=True)
        if base.empty:
            continue
        n_hours = len(base)
        for sid in range(1, n_scenarios + 1):
            if block_bootstrap:
                starts = {k: rng.integers(0, max(len(v) - block_length, 1)) for k, v in residual_map.items()}
                seq = {
                    k: (v[starts[k] : starts[k] + block_length] if len(v) >= block_length else rng.choice(v, size=n_hours, replace=True))
                    for k, v in residual_map.items()
                }
            else:
                seq = {k: rng.choice(v, size=n_hours, replace=True) for k, v in residual_map.items()}
            for i, row in base.iterrows():
                rows.append(
                    {
                        "scenario_id": sid,
                        "scenario_probability": 1.0 / n_scenarios,
                        "forecast_origin_datetime": row["forecast_origin_datetime"],
                        "target_datetime": row["target_datetime"],
                        "horizon": row["horizon"],
                        "price_scenario": row["price_forecast"] + seq["price_forecast"][i % len(seq["price_forecast"])],
                        "load_scenario": row["load_forecast"] + seq["load_forecast"][i % len(seq["load_forecast"])],
                        "renewable_scenario": row["renewable_forecast"] + seq["renewable_forecast"][i % len(seq["renewable_forecast"])],
                    }
                )
    return pd.DataFrame(rows)


def write_scenarios(df: pd.DataFrame, output_root: str = "outputs") -> None:
    out = Path(output_root) / "scenarios"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "bootstrap_scenarios.csv", index=False)
    df.to_csv(out / "bootstrap_scenarios_calibrated.csv", index=False)
