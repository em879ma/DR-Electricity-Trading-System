from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Structural features for counterfactual price model
# ---------------------------------------------------------------------------

STRUCTURAL_FEATURES = [
    "net_load",
    "scarcity_index",
    "import_share",
    "low_net_load_dummy",
    "high_renewable_dummy",
]


def add_structural_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "imports" not in out.columns:
        out["imports"] = out.get("import_index", 0.0)
    out["imports"] = pd.to_numeric(out["imports"], errors="coerce").fillna(0.0)
    load = out["load"].astype(float)
    ren = out["renewable_total"].astype(float)
    imp = out["imports"]

    out["net_load"] = load - ren
    out["renewable_share"] = ren / (load + 1e-6)
    out["wind_share"] = out.get("wind_generation", pd.Series(0.0, index=out.index)).astype(float) / (load + 1e-6)
    out["solar_share"] = out.get("solar_generation", pd.Series(0.0, index=out.index)).astype(float) / (load + 1e-6)
    out["import_share"] = imp / (load + 1e-6)

    if "available_capacity" in out.columns:
        out["scarcity_index"] = load / (out["available_capacity"].astype(float) + imp + 1e-6)
        out["oversupply_index"] = ren + imp - load
    else:
        out["scarcity_index"] = load / (ren + 1e-6)
        out["oversupply_index"] = ren - load

    net_load_p10 = float(out["net_load"].quantile(0.10))
    ren_share_p90 = float(out["renewable_share"].quantile(0.90))
    out["low_net_load_dummy"] = (out["net_load"] <= net_load_p10).astype(int)
    out["high_renewable_dummy"] = (out["renewable_share"] >= ren_share_p90).astype(int)
    out["net_load_p10"] = net_load_p10
    out["renewable_share_p90"] = ren_share_p90
    return out


def build_refit_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    hour = pd.to_datetime(df["datetime"]).dt.hour
    dow = pd.to_datetime(df["datetime"]).dt.dayofweek
    month = pd.to_datetime(df["datetime"]).dt.month
    dum = pd.get_dummies(
        pd.DataFrame({"hour": hour, "dayofweek": dow, "month": month}).astype("category"),
        drop_first=True,
    )
    base = df[STRUCTURAL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x = pd.concat([base, dum.astype(float)], axis=1).fillna(0.0)
    return x, list(x.columns)


def update_counterfactual_row(
    base_row: pd.Series,
    q: float,
    load_s: float,
    ren_s: float,
    *,
    demand_col: str = "demand_base",
    min_load_ratio: float = 0.70,
) -> pd.Series:
    out = base_row.copy()
    db = float(out[demand_col])
    min_allowed = min_load_ratio * db
    d_cf = max(db - q, min_allowed)
    out[demand_col] = d_cf
    out["load"] = float(load_s) - q
    out["renewable_total"] = float(ren_s)
    if "net_load" in out.index:
        out["net_load"] = out["load"] - float(ren_s)
    if "scarcity_index" in out.index:
        cap = float(out.get("available_capacity", out.get("renewable_total", 1.0)))
        imp = float(out.get("imports", 0.0))
        out["scarcity_index"] = d_cf / (cap + imp + 1e-6)
    if "renewable_share" in out.index:
        out["renewable_share"] = float(ren_s) / (float(out["load"]) + 1e-6)
    if "import_share" in out.index:
        out["import_share"] = float(out.get("imports", 0.0)) / (float(out["load"]) + 1e-6)
    if "oversupply_index" in out.index:
        out["oversupply_index"] = float(ren_s) + float(out.get("imports", 0.0)) - float(out["load"])
    net_p10 = float(out.get("net_load_p10", out.get("net_load", 0.0)))
    ren_p90 = float(out.get("renewable_share_p90", out.get("renewable_share", 0.0)))
    if "low_net_load_dummy" in out.index:
        out["low_net_load_dummy"] = int(float(out.get("net_load", 0.0)) <= net_p10)
    if "high_renewable_dummy" in out.index:
        out["high_renewable_dummy"] = int(float(out.get("renewable_share", 0.0)) >= ren_p90)
    return out


# ---------------------------------------------------------------------------
# Fit interpretable Ridge + RF counterfactual price models
# ---------------------------------------------------------------------------

def fit_counterfactual_price_models(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Fit Ridge (interpretable) + RF counterfactual price models on structural features.
    Also fits a negative-price classifier.
    Returns (model_pack, eval_df, coef_df).
    """
    panel = add_structural_price_features(df.copy())
    y = pd.to_numeric(panel["price"], errors="coerce")
    valid = y.notna()
    panel = panel.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    x_mat, colnames = build_refit_feature_matrix(panel)

    ridge = Ridge(alpha=2.5, random_state=42)
    ridge.fit(x_mat.to_numpy(dtype=float), y.to_numpy(dtype=float))

    rf = RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=1)
    rf.fit(x_mat, y)

    yhat_r = ridge.predict(x_mat.to_numpy(dtype=float))
    yhat_f = rf.predict(x_mat)

    def _metrics(name: str, pred: np.ndarray) -> dict:
        return {
            "model": name,
            "rmse": float(mean_squared_error(y, pred) ** 0.5),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
            "bias": float(np.mean(pred - y)),
        }

    eval_df = pd.DataFrame([_metrics("ridge", yhat_r), _metrics("random_forest", yhat_f)])
    coef_df = pd.DataFrame({
        "feature": ["intercept"] + colnames,
        "coefficient": [float(ridge.intercept_)] + list(ridge.coef_),
    })

    # Negative-price classifier
    y_neg = (y < 0).astype(int)
    clf_rf = None
    clf_lr = None
    if int(y_neg.nunique()) >= 2:
        clf_rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=1, class_weight="balanced")
        clf_rf.fit(x_mat, y_neg)
        clf_lr = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)
        clf_lr.fit(x_mat.to_numpy(dtype=float), y_neg.to_numpy(dtype=int))

    model_pack = {
        "ridge": ridge,
        "random_forest": rf,
        "feature_columns": colnames,
        "negative_price_classifier_rf": clf_rf,
        "negative_price_classifier_lr": clf_lr,
    }
    return model_pack, eval_df, coef_df


def predict_counterfactual_price(model_pack: dict, df: pd.DataFrame) -> pd.Series:
    panel = add_structural_price_features(df.copy())
    x, _ = build_refit_feature_matrix(panel)
    cols = model_pack["feature_columns"]
    for c in cols:
        if c not in x.columns:
            x[c] = 0.0
    x = x[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.Series(model_pack["random_forest"].predict(x), index=df.index, name="counterfactual_price")


def predict_negative_price_probability(model_pack: dict, df: pd.DataFrame) -> pd.Series:
    clf = model_pack.get("negative_price_classifier_rf")
    if clf is None:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index, name="p_negative_price")
    panel = add_structural_price_features(df.copy())
    x, _ = build_refit_feature_matrix(panel)
    cols = model_pack["feature_columns"]
    for c in cols:
        if c not in x.columns:
            x[c] = 0.0
    x = x[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    proba = clf.predict_proba(x)[:, 1]
    return pd.Series(proba, index=df.index, name="p_negative_price")


def demand_sensitivity_analysis(
    df: pd.DataFrame,
    model_pack: dict,
    ratios: tuple = (0.05, 0.10, 0.15, 0.20, 0.30),
) -> pd.DataFrame:
    work = df.dropna(subset=["demand_base", "price"]).copy().reset_index(drop=True)
    price_base = predict_counterfactual_price(model_pack, work).to_numpy(dtype=float)
    p90 = float(np.quantile(price_base, 0.9))
    high_mask = price_base >= p90
    rows = []
    for r in ratios:
        w = work.copy()
        w["demand_base"] = w["demand_base"].astype(float) * (1.0 - r)
        w["load"] = w["load"].astype(float) * (1.0 - r)
        price_test = predict_counterfactual_price(model_pack, w).to_numpy(dtype=float)
        diff = price_base - price_test
        avg_ch = float(np.mean(diff))
        hp_ch = float(np.mean(diff[high_mask])) if high_mask.any() else float("nan")
        rows.append({
            "demand_reduction_ratio": r,
            "avg_price_change": avg_ch,
            "high_price_price_change": hp_ch,
            "max_price_change": float(np.max(diff)),
            "percent_avg_price_change": float(avg_ch / (np.mean(price_base) + 1e-9) * 100.0),
            "mechanism_valid_avg_positive": bool(avg_ch > 0),
            "mechanism_valid_high_gt_avg": bool(hp_ch > avg_ch),
        })
    return pd.DataFrame(rows)


def write_counterfactual_model_outputs(
    eval_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    output_root: str = "outputs",
) -> None:
    tables = Path(output_root) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(tables / "counterfactual_price_model_evaluation.csv", index=False)
    coef_df.to_csv(tables / "counterfactual_price_model_coefficients.csv", index=False)
    sensitivity_df.to_csv(tables / "counterfactual_demand_sensitivity.csv", index=False)
