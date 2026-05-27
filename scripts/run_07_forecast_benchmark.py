"""
Run 07: Forecast model benchmark.
Trains Ridge / XGBoost / LSTM / TCN / Two-Stage RF on the same H=24 price task,
evaluates on the held-out test set, and saves figures + tables to outputs/benchmark/.

This script is fully standalone — it does NOT modify any outputs from runs 01-06.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.forecast_benchmark import run_benchmark, MODEL_LABELS


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    panel_path = Path("data/processed/market_panel_with_dr.csv")
    if not panel_path.exists():
        raise FileNotFoundError(f"Run scripts 01-02 first. Missing: {panel_path}")

    print("Loading market panel...")
    df = pd.read_csv(panel_path, parse_dates=["datetime"])
    print(f"  {len(df)} rows, {df['datetime'].min()} – {df['datetime'].max()}")

    out_dir = Path(cfg["output"]["root"]) / "benchmark"
    print(f"\nRunning benchmark → {out_dir}/")
    metrics_df = run_benchmark(
        df,
        split=cfg["forecast"]["split"],
        rf_params=cfg["forecast"]["rf"],
        output_dir=out_dir,
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS — Test Set (H=24, price)")
    print("=" * 70)
    disp = metrics_df[["model", "mae", "rmse", "r2", "neg_recall_reg",
                         "neg_f1_reg", "peak_recall", "peak_f1"]].copy()
    disp["model"] = disp["model"].map(lambda m: MODEL_LABELS.get(m, m))
    disp = disp.set_index("model")
    disp.columns = ["MAE", "RMSE", "R²", "Neg Recall", "Neg F1", "Peak Recall", "Peak F1"]
    print(disp.round(4).to_string())

    print("\n--- Regime MAE (€/MWh) ---")
    reg = metrics_df[["model", "mae_negative", "mae_low", "mae_normal",
                        "mae_high", "mae_peak"]].copy()
    reg["model"] = reg["model"].map(lambda m: MODEL_LABELS.get(m, m))
    reg = reg.set_index("model")
    reg.columns = ["Negative", "Low", "Normal", "High", "Peak"]
    print(reg.round(2).to_string())

    print("\n--- Pinball Loss ---")
    pb = metrics_df[["model", "pinball_q10", "pinball_q25", "pinball_q50",
                       "pinball_q75", "pinball_q90"]].copy()
    pb["model"] = pb["model"].map(lambda m: MODEL_LABELS.get(m, m))
    pb = pb.set_index("model")
    pb.columns = ["q10", "q25", "q50", "q75", "q90"]
    print(pb.round(3).to_string())

    print(f"\nOutputs saved to {out_dir}/")
    print("  tables/benchmark_metrics.csv")
    print("  tables/benchmark_summary.csv")
    print("  figures/07a_overall_accuracy.png")
    print("  figures/07b_regime_mae.png")
    print("  figures/07c_radar.png")
    print("  figures/07d_detection.png")
    print("  figures/07e_error_distribution.png")
    print("  figures/07f_pinball_loss.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
