"""
Run 12: XGBoost hyperparameter search via Optuna.

Goal: address the v2.1 finding that XGBoost's default settings give rMAE=1.13
(worse than Seasonal Naive). The default config uses no L1/L2 regularisation
and no minimum-split-loss constraint, which likely causes overfitting in the
low/negative-price regimes where Ridge wins.

Search space (100 trials, TPE sampler):
  reg_alpha            ∈ [0, 10]        L1 regularisation
  reg_lambda           ∈ [0, 10]        L2 regularisation
  gamma                ∈ [0, 5]         minimum loss reduction per split
  min_child_weight     ∈ [1, 20]        minimum sum-of-Hessian per leaf
  max_depth            ∈ [3, 10]
  learning_rate        ∈ [0.01, 0.1]
  subsample            ∈ [0.6, 1.0]
  colsample_bytree     ∈ [0.6, 1.0]
  n_estimators         fixed 800, early-stopping 30 rounds on valid

Objective: validation-set MAE (no calibration step; we want raw model quality).

Output: outputs/diagnostic/xgb_best_params.yaml — drop-in for config.yaml
under `forecast.xgb_params`. src/forecaster.py::_xgboost_pipeline already
honours the override via the `xgb_params` kwarg, so no code change needed
to use the tuned model.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

import xgboost as xgb
import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.forecaster import _prep_tabular

OUT_DIR = PROJECT_ROOT / "outputs" / "diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _objective_factory(train, valid, feat_cols):
    Xtr = train[feat_cols].to_numpy()
    ytr = train["target_price"].to_numpy()
    Xva = valid[feat_cols].to_numpy()
    yva = valid["target_price"].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=800,
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 10.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            random_state=42, n_jobs=1,
        )
        mdl = xgb.XGBRegressor(
            **params, early_stopping_rounds=30, eval_metric="rmse",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        yhat = mdl.predict(Xva)
        return float(mean_absolute_error(yva, yhat))

    return objective


def main(config_path: str = "config.yaml", n_trials: int = 100) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    df = pd.read_csv("data/processed/market_panel_with_dr.csv", parse_dates=["datetime"])

    feat_cols, train, valid, test = _prep_tabular(df, cfg["forecast"]["split"], 24)
    print(f"Train n={len(train)}  Valid n={len(valid)}  Test n={len(test)}")
    print(f"Searching XGBoost params over {n_trials} trials...")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(_objective_factory(train, valid, feat_cols),
                   n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_val_mae = study.best_value
    print(f"\nBest valid MAE: {best_val_mae:.3f}")
    print(f"Best params: {best}")

    # Validate best params on test set
    Xtr = train[feat_cols].to_numpy(); ytr = train["target_price"].to_numpy()
    Xva = valid[feat_cols].to_numpy(); yva = valid["target_price"].to_numpy()
    Xte = test[feat_cols].to_numpy();  yte = test["target_price"].to_numpy()

    mdl = xgb.XGBRegressor(
        **best, n_estimators=800, random_state=42, n_jobs=1,
        early_stopping_rounds=30, eval_metric="rmse",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    test_mae = float(mean_absolute_error(yte, mdl.predict(Xte)))
    print(f"Best params test MAE: {test_mae:.3f}")

    # Compare with default
    default_mdl = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1,
        early_stopping_rounds=20, eval_metric="rmse",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default_mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    default_test_mae = float(mean_absolute_error(yte, default_mdl.predict(Xte)))
    print(f"Default config test MAE: {default_test_mae:.3f}  (Δ = {test_mae - default_test_mae:+.3f})")

    # Dump best params
    out = {
        "best_params": {**best, "n_estimators": 800, "random_state": 42, "n_jobs": 1},
        "valid_mae": best_val_mae,
        "test_mae_tuned": test_mae,
        "test_mae_default": default_test_mae,
        "n_trials": n_trials,
    }
    (OUT_DIR / "xgb_best_params.yaml").write_text(
        yaml.safe_dump(out, sort_keys=False), encoding="utf-8"
    )
    print(f"\nSaved tuned params to {OUT_DIR / 'xgb_best_params.yaml'}")
    print("\nTo activate, copy under config.yaml as:")
    print("  forecast:")
    print("    xgb_params:")
    for k, v in best.items():
        print(f"      {k}: {v}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()
    main(args.config, args.n_trials)
