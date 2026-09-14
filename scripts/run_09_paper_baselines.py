"""
Run 09: Paper-aligned baselines and statistical significance tests.

Adds two reference baselines on top of run_07's benchmark:
  - Seasonal Naive (p̂_t = p_{t-168h}) — canonical EPF reference (Lago 2021)
  - LEAR (24 LassoCV models per target hour) — strong low-complexity baseline

Then computes:
  - rMAE for every model = MAE_model / MAE_naive
  - Pairwise Diebold-Mariano tests (squared loss, HAC variance)
  - Giacomini-White joint test across all 24 hours

Outputs:
  outputs/benchmark/tables/benchmark_with_paper_baselines.csv
  outputs/benchmark/tables/predictions_naive_seasonal.csv
  outputs/benchmark/tables/predictions_lear.csv
  outputs/benchmark/tables/dm_test_pairwise.csv
  outputs/benchmark/tables/gw_test_joint.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.forecaster import fit_price_forecast, compute_rmae
from src.lear import fit_lear_pipeline
from src.stat_tests import (
    diebold_mariano_test, giacomini_white_test, pairwise_dm_table,
)


def _load_existing_predictions(bench_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "ridge":         "predictions_ridge.csv",
        "xgboost":       "predictions_xgboost.csv",
        "lstm":          "predictions_lstm.csv",
        "tcn":           "predictions_tcn.csv",
        "two_stage_rf":  "predictions_two_stage_rf.csv",
    }
    out = {}
    for name, fname in files.items():
        p = bench_dir / fname
        if not p.exists():
            print(f"  Skipping {name}: file missing")
            continue
        df = pd.read_csv(p, parse_dates=["target_datetime"])
        out[name] = df
    return out


def _align_on_common_index(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-join all model predictions on target_datetime."""
    base = None
    for name, df in preds.items():
        keep = df[["target_datetime", "actual", "forecast"]].rename(
            columns={"forecast": f"yhat_{name}"}
        )
        if base is None:
            base = keep
        else:
            base = base.merge(
                keep.drop(columns=["actual"]),
                on="target_datetime", how="inner",
            )
    return base.dropna().reset_index(drop=True)


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out_root = Path(cfg["output"]["root"])
    bench_dir = out_root / "benchmark" / "tables"
    bench_dir.mkdir(parents=True, exist_ok=True)

    panel_path = Path("data/processed/market_panel_with_dr.csv")
    if not panel_path.exists():
        raise FileNotFoundError(f"Run run_02 first. Missing: {panel_path}")

    print("Loading market panel...")
    df = pd.read_csv(panel_path, parse_dates=["datetime"])
    split = cfg["forecast"]["split"]
    rf_params = cfg["forecast"]["rf"]

    # ── Seasonal Naive ────────────────────────────────────────────────
    print("Fitting Seasonal Naive (lag=168h)...")
    naive_pred_df, naive_acc_df, _, naive_metrics = fit_price_forecast(
        df, model_name="naive_seasonal", split=split, rf_params=rf_params,
    )
    naive_mae = float(naive_acc_df["mae"].iloc[0])
    print(f"  Naive MAE = {naive_mae:.3f}")

    keep = ["target_datetime", "actual", "forecast"]
    naive_pred_df[keep].to_csv(bench_dir / "predictions_naive_seasonal.csv", index=False)

    # ── LEAR ──────────────────────────────────────────────────────────
    print("Fitting LEAR (24 LassoCV models)...")
    lear_pred_df, lear_acc_df, _, lear_metrics = fit_lear_pipeline(
        df, split=split, rf_params=rf_params,
    )
    lear_mae = float(lear_acc_df["mae"].iloc[0])
    print(f"  LEAR MAE = {lear_mae:.3f}  (active hour-models: {lear_metrics['n_active_hour_models']}/24)")
    lear_pred_df[keep].to_csv(bench_dir / "predictions_lear.csv", index=False)

    # ── Regime ensemble (v2.2) ────────────────────────────────────────
    print("Fitting regime ensemble (Ridge + XGBoost gated by regime classifier)...")
    ens_pred_df, ens_acc_df, _, _ = fit_price_forecast(
        df, model_name="regime_ensemble", split=split, rf_params=rf_params,
        feature_set=cfg["forecast"].get("feature_set", "full"),
    )
    print(f"  Regime ensemble MAE = {float(ens_acc_df['mae'].iloc[0]):.3f}")
    ens_pred_df[keep].to_csv(bench_dir / "predictions_regime_ensemble.csv", index=False)

    # ── Combine all model predictions on common index ─────────────────
    print("Aligning all model predictions on common target_datetime index...")
    existing = _load_existing_predictions(bench_dir)
    existing["naive_seasonal"] = naive_pred_df[keep]
    existing["lear"] = lear_pred_df[keep]
    existing["regime_ensemble"] = ens_pred_df[keep]
    aligned = _align_on_common_index(existing)
    print(f"  Common eval rows: {len(aligned):,}")

    # ── rMAE table ────────────────────────────────────────────────────
    print("Computing rMAE (vs Seasonal Naive)...")
    rows = []
    naive_mae_aligned = float(np.mean(np.abs(
        aligned["yhat_naive_seasonal"] - aligned["actual"]
    )))
    for name in existing:
        err = aligned[f"yhat_{name}"] - aligned["actual"]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        rows.append({
            "model": name,
            "mae": mae,
            "rmse": rmse,
            "rmae_vs_naive": compute_rmae(mae, naive_mae_aligned),
            "n_eval": len(aligned),
        })
    summary = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    summary.to_csv(bench_dir / "benchmark_with_paper_baselines.csv", index=False)
    print(summary.to_string(index=False))

    # ── Pairwise Diebold-Mariano tests ────────────────────────────────
    print("\nRunning pairwise Diebold-Mariano tests (squared loss)...")
    err_dict = {
        name: (aligned[f"yhat_{name}"] - aligned["actual"]).to_numpy()
        for name in existing
    }
    dm_df = pairwise_dm_table(err_dict, loss="squared")
    dm_df.to_csv(bench_dir / "dm_test_pairwise.csv", index=False)
    # Show interesting rows: XGBoost vs others
    xgb_rows = dm_df[(dm_df["model_1"] == "xgboost") | (dm_df["model_2"] == "xgboost")]
    print(xgb_rows.to_string(index=False))

    # ── Joint Giacomini-White test ────────────────────────────────────
    print("\nRunning multivariate Giacomini-White tests (all 24h joint)...")
    aligned["date"] = aligned["target_datetime"].dt.date
    aligned["hour"] = aligned["target_datetime"].dt.hour

    gw_rows = []
    model_names = list(existing.keys())
    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1:]:
            e1_long = aligned[f"yhat_{m1}"] - aligned["actual"]
            e2_long = aligned[f"yhat_{m2}"] - aligned["actual"]
            tmp = aligned[["date", "hour"]].copy()
            tmp["e1"] = e1_long.to_numpy()
            tmp["e2"] = e2_long.to_numpy()
            E1 = tmp.pivot_table(index="date", columns="hour",
                                 values="e1", aggfunc="mean").reindex(columns=range(24))
            E2 = tmp.pivot_table(index="date", columns="hour",
                                 values="e2", aggfunc="mean").reindex(columns=range(24))
            chi2, p = giacomini_white_test(E1.to_numpy(), E2.to_numpy())
            if not np.isfinite(chi2):
                winner = "n/a"
            elif p < 0.05:
                winner = m1 if E1.mean().mean() ** 2 < E2.mean().mean() ** 2 else m2
            else:
                winner = "tie"
            gw_rows.append({
                "model_1": m1, "model_2": m2,
                "chi2_stat": chi2, "p_value": p, "winner": winner,
            })
    gw_df = pd.DataFrame(gw_rows)
    gw_df.to_csv(bench_dir / "gw_test_joint.csv", index=False)
    print(gw_df[(gw_df["model_1"] == "xgboost") | (gw_df["model_2"] == "xgboost")].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
