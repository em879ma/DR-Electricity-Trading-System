"""
Run 11: Forecast model diagnostic.

Purpose: figure out *why* XGBoost loses to Seasonal Naive (rMAE=1.13) and where
the residual is concentrated, so we can prune features / tune regularisation /
build a regime ensemble in a targeted way.

Outputs (all under outputs/diagnostic/):
  01_shap_importance.png        — SHAP global importance for XGBoost
  01_ridge_coef.png             — Ridge coefficient × stdev (signed importance)
  02_feature_correlation.png    — Spearman feature correlation heatmap
  03_learning_curve.png         — train/valid MAE vs n_estimators
  04_residual_by_regime.png     — bias & MAE by observed-price quintile
  05_calibration_effect.png     — raw vs calibrated MAE
  forecast_diagnostic_report.md — auto-summary
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.forecaster import _prep_tabular

import xgboost as xgb

try:
    import shap
    _SHAP = True
except ImportError:
    _SHAP = False
    print("[warn] shap not installed — skipping SHAP plot")


OUT_DIR = PROJECT_ROOT / "outputs" / "diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _fit_xgb(train, valid, feat_cols, n_estimators=500, learning_rate=0.05,
             max_depth=6, reg_alpha=0.0, reg_lambda=1.0, gamma=0.0):
    params = dict(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=reg_alpha, reg_lambda=reg_lambda, gamma=gamma,
        random_state=42, n_jobs=1,
    )
    mdl = xgb.XGBRegressor(**params, early_stopping_rounds=20, eval_metric="rmse")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mdl.fit(
            train[feat_cols].to_numpy(), train["target_price"].to_numpy(),
            eval_set=[(valid[feat_cols].to_numpy(), valid["target_price"].to_numpy())],
            verbose=False,
        )
    return mdl


def _plot_shap(model, X_sample, feat_cols, path):
    if not _SHAP:
        return None
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=feat_cols).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(mean_abs) + 1))
    mean_abs.plot.barh(ax=ax, color="#1F3A68")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("XGBoost feature importance (mean |SHAP|, valid set)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return mean_abs


def _plot_ridge_coef(train, valid, feat_cols, path):
    sc = StandardScaler()
    Xtr = sc.fit_transform(train[feat_cols])
    ridge = Ridge(alpha=10.0).fit(Xtr, train["target_price"].to_numpy())
    importance = pd.Series(ridge.coef_, index=feat_cols).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(importance) + 1))
    colors = ["#C0392B" if v < 0 else "#27AE60" for v in importance.values]
    importance.plot.barh(ax=ax, color=colors)
    ax.set_xlabel("Standardised Ridge coefficient")
    ax.set_title("Ridge feature signed importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return importance


def _plot_correlation(train, feat_cols, path, k=15):
    corr = train[feat_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(feat_cols))); ax.set_yticks(range(len(feat_cols)))
    ax.set_xticklabels(feat_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(feat_cols, fontsize=7)
    ax.set_title("Spearman feature correlation (train set)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

    # Return top-k redundant pairs
    pairs = []
    for i, c1 in enumerate(feat_cols):
        for c2 in feat_cols[i+1:]:
            r = corr.loc[c1, c2]
            if abs(r) > 0.85:
                pairs.append((c1, c2, float(r)))
    pairs.sort(key=lambda x: -abs(x[2]))
    return pairs[:k]


def _plot_learning_curve(train, valid, feat_cols, path):
    n_trees_grid = list(range(50, 801, 50))
    rows = []
    for n in n_trees_grid:
        mdl = _fit_xgb(train, valid, feat_cols, n_estimators=n)
        tr_mae = mean_absolute_error(train["target_price"],
                                     mdl.predict(train[feat_cols].to_numpy()))
        va_mae = mean_absolute_error(valid["target_price"],
                                     mdl.predict(valid[feat_cols].to_numpy()))
        rows.append({"n_trees": n, "train_mae": tr_mae, "valid_mae": va_mae,
                     "best_iter": int(getattr(mdl, "best_iteration", n))})
    lc = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(lc["n_trees"], lc["train_mae"], "-o", color="#1F3A68", label="Train")
    ax.plot(lc["n_trees"], lc["valid_mae"], "-o", color="#E85D04", label="Valid")
    ax.set_xlabel("n_estimators"); ax.set_ylabel("MAE")
    ax.set_title("XGBoost learning curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return lc


def _plot_residual_by_regime(model, X_eval, y_eval, path):
    yhat = model.predict(X_eval)
    res = yhat - y_eval
    quintiles = pd.qcut(y_eval, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    df = pd.DataFrame({"price": y_eval, "residual": res, "yhat": yhat,
                       "quintile": quintiles})
    grouped = df.groupby("quintile", observed=True).agg(
        n=("price", "size"),
        price_mean=("price", "mean"),
        bias=("residual", "mean"),
        mae=("residual", lambda r: float(np.mean(np.abs(r)))),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(grouped["quintile"].astype(str), grouped["bias"],
                color=["#C0392B" if b < 0 else "#27AE60" for b in grouped["bias"]])
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_title("Residual bias by observed-price quintile")
    axes[0].set_ylabel("Mean(yhat − y)"); axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(grouped["quintile"].astype(str), grouped["mae"], color="#1F3A68")
    axes[1].set_title("MAE by observed-price quintile")
    axes[1].set_ylabel("MAE (€/MWh)"); axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return grouped


def _plot_calibration_effect(path):
    cal_path = PROJECT_ROOT / "outputs" / "tables" / "rf_calibrated_forecasts.csv"
    if not cal_path.exists():
        return None
    df = pd.read_csv(cal_path)
    df = df[df["target"] == "price"].copy()
    if df.empty:
        return None
    df["err_raw"] = (df["forecast"] - df["actual"]).abs()
    df["err_cal"] = (df["forecast_calibrated"] - df["actual"]).abs()
    quintiles = pd.qcut(df["actual"], 5, labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")
    df["quintile"] = quintiles
    g = df.groupby("quintile", observed=True).agg(
        raw_mae=("err_raw", "mean"),
        cal_mae=("err_cal", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["raw_mae"], width=0.4, label="Raw",      color="#1F3A68")
    ax.bar(x + 0.2, g["cal_mae"], width=0.4, label="Calibrated", color="#E85D04")
    ax.set_xticks(x); ax.set_xticklabels(g["quintile"].astype(str))
    ax.set_ylabel("MAE"); ax.set_title("Effect of calibration by price quintile")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return g


def main(config_path: str = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    panel_path = Path("data/processed/market_panel_with_dr.csv")
    df = pd.read_csv(panel_path, parse_dates=["datetime"])

    feat_cols, train, valid, test = _prep_tabular(df, cfg["forecast"]["split"], 24)
    eval_data = pd.concat([valid, test]).reset_index(drop=True)
    print(f"Features ({len(feat_cols)}): {feat_cols[:6]}…")
    print(f"Train n={len(train)}  Valid n={len(valid)}  Test n={len(test)}")

    # ── Fit a reference XGBoost (same hyperparameters as src/forecaster.py)
    print("Fitting reference XGBoost...")
    mdl = _fit_xgb(train, valid, feat_cols, n_estimators=500)
    print(f"  best_iteration = {mdl.best_iteration}")

    # ── SHAP
    print("Computing SHAP importance...")
    shap_path = OUT_DIR / "01_shap_importance.png"
    sample = valid[feat_cols].sample(n=min(800, len(valid)), random_state=42)
    shap_imp = _plot_shap(mdl, sample, feat_cols, shap_path)

    # ── Ridge coef
    print("Computing Ridge signed importance...")
    ridge_imp = _plot_ridge_coef(train, valid, feat_cols, OUT_DIR / "01_ridge_coef.png")

    # ── Correlation matrix + redundant pairs
    print("Spearman feature correlation...")
    redundant = _plot_correlation(train, feat_cols, OUT_DIR / "02_feature_correlation.png")

    # ── Learning curve
    print("Learning curve...")
    lc = _plot_learning_curve(train, valid, feat_cols, OUT_DIR / "03_learning_curve.png")

    # ── Residual by regime (on test set, model already fitted)
    print("Residual by regime...")
    test_grouped = _plot_residual_by_regime(
        mdl, test[feat_cols].to_numpy(), test["target_price"].to_numpy(),
        OUT_DIR / "04_residual_by_regime.png",
    )

    # ── Calibration effect
    print("Calibration effect...")
    cal_g = _plot_calibration_effect(OUT_DIR / "05_calibration_effect.png")

    # ── Auto-summary
    print("Writing diagnostic_report.md...")
    md_lines = [
        "# Forecast diagnostic report (v2.1 baseline)",
        "",
        f"- Train rows: **{len(train):,}**",
        f"- Valid rows: **{len(valid):,}**",
        f"- Test rows: **{len(test):,}**",
        f"- Features: **{len(feat_cols)}**",
        f"- XGBoost best iteration on valid: **{mdl.best_iteration}**",
        "",
        "## Learning curve",
        "",
        "| n_trees | Train MAE | Valid MAE |",
        "|---------|-----------|-----------|",
    ]
    for _, r in lc.iterrows():
        md_lines.append(f"| {int(r['n_trees'])} | {r['train_mae']:.3f} | {r['valid_mae']:.3f} |")

    md_lines += ["", "## Residual by observed-price quintile (test set)", "",
                 "| Quintile | n | Avg price | Bias | MAE |",
                 "|----------|---|-----------|------|-----|"]
    for _, r in test_grouped.iterrows():
        md_lines.append(
            f"| {r['quintile']} | {int(r['n'])} | {r['price_mean']:.2f} | "
            f"{r['bias']:.3f} | {r['mae']:.3f} |"
        )

    if redundant:
        md_lines += ["", "## Top redundant feature pairs (|Spearman| > 0.85)", "",
                     "| Feature 1 | Feature 2 | ρ |",
                     "|-----------|-----------|----|"]
        for a, b, r in redundant:
            md_lines.append(f"| {a} | {b} | {r:+.3f} |")

    if shap_imp is not None:
        md_lines += ["", "## SHAP bottom-10 features (likely prunable)", ""]
        for name, val in shap_imp.head(10).items():
            md_lines.append(f"- `{name}`  |SHAP| = {val:.3f}")

    if cal_g is not None:
        md_lines += ["", "## Calibration effect (MAE)", "",
                     "| Quintile | Raw MAE | Calibrated MAE | Δ |",
                     "|----------|---------|----------------|---|"]
        for _, r in cal_g.iterrows():
            d = r['cal_mae'] - r['raw_mae']
            md_lines.append(f"| {r['quintile']} | {r['raw_mae']:.3f} | "
                            f"{r['cal_mae']:.3f} | {d:+.3f} |")

    md_lines += ["", "## Next-step recommendations (auto)", ""]
    if redundant:
        md_lines.append(f"- Prune at least {min(8, len(redundant))} highly-correlated features.")
    if lc["valid_mae"].iloc[-1] > lc["valid_mae"].min():
        md_lines.append("- Valid MAE bottoms out before n_estimators reaches 800 → cap trees.")
    bias_extremes = test_grouped["bias"].abs()
    if bias_extremes.iloc[0] > bias_extremes.iloc[2] * 2 or bias_extremes.iloc[-1] > bias_extremes.iloc[2] * 2:
        md_lines.append("- Bias is concentrated in the extreme quintiles (Q1/Q5) → regime ensemble worth trying.")
    md_lines.append("- Run `scripts/run_12_xgb_tuning.py` to search reg_alpha / reg_lambda / gamma.")

    (OUT_DIR / "forecast_diagnostic_report.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )
    print(f"\nDone. Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
