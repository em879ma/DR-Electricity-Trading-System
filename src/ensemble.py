"""
Regime-aware ensemble for day-ahead price forecasting.

Motivation: v2.1 benchmark showed XGBoost wins the High/Peak regime
(MAE 4.19 vs Ridge 5.80) but loses badly in Negative/Low (MAE 46.8 vs Ridge
41.3). A single model is a Pareto compromise — using each model where it is
strongest should beat both.

Architecture (3 stages):

  Stage 0 — regime classifier
    GradientBoostingClassifier predicts regime ∈ {neg, low, normal, high}
    based on the panel features. Thresholds come from the train-set
    price distribution (p25, p75, p90).

  Stage 1 — per-regime sub-models
    - Negative: Ridge      (smoother extrapolation in sparse regime)
    - Low:      Ridge      (Ridge wins in low/normal per v2.1 regime MAE)
    - Normal:   blend(Ridge, XGBoost, 0.5/0.5)
    - High:     XGBoost    (XGBoost wins in high/peak regime)

  Stage 2 — combine
    Final prediction is the sub-model output for the predicted regime.
    Optionally soft-blend with regime probabilities if `soft_assign=True`.

The module exposes a single class `RegimeAwareEnsemble` plus a pipeline
function `fit_regime_ensemble_pipeline()` that fits the `forecaster.py`
contract — pluggable directly via `fit_price_forecast(model_name="regime_ensemble")`.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False

from src.forecasting import RF_FEATURES


# ──────────────────────────────────────────────────────────────────────────────
# Regime labeling
# ──────────────────────────────────────────────────────────────────────────────

def _regime_thresholds(train_y: np.ndarray) -> dict[str, float]:
    pos = train_y[train_y >= 0]
    return {
        "neg_boundary":    0.0,
        "low_boundary":    float(np.quantile(pos, 0.25)) if len(pos) > 10 else 30.0,
        "high_boundary":   float(np.quantile(pos, 0.90)) if len(pos) > 10 else 80.0,
    }


def _label_regime(y: np.ndarray, thr: dict[str, float]) -> np.ndarray:
    out = np.empty(len(y), dtype=int)
    out[:] = 1  # default = low
    out[y < thr["neg_boundary"]] = 0           # negative
    out[(y >= thr["neg_boundary"]) & (y < thr["low_boundary"])] = 1   # low
    out[(y >= thr["low_boundary"]) & (y < thr["high_boundary"])] = 2  # normal
    out[y >= thr["high_boundary"]] = 3         # high
    return out


REGIME_NAMES = ["neg", "low", "normal", "high"]


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble class
# ──────────────────────────────────────────────────────────────────────────────

class RegimeAwareEnsemble:
    def __init__(
        self,
        ridge_alpha: float = 10.0,
        soft_assign: bool = False,
        random_state: int = 42,
    ) -> None:
        if not _XGB:
            raise ImportError("xgboost is required for RegimeAwareEnsemble.")
        self.ridge_alpha = ridge_alpha
        self.soft_assign = soft_assign
        self.random_state = random_state
        self.thr_: dict[str, float] = {}
        self.classifier_: GradientBoostingClassifier | None = None
        self.scaler_: StandardScaler | None = None
        self.ridge_: Ridge | None = None
        self.xgb_: xgb.XGBRegressor | None = None
        self.feat_cols_: list[str] = []
        # Per-regime blend weights: (w_ridge, w_xgb)
        self.regime_blend_: dict[int, tuple[float, float]] = {
            0: (1.0, 0.0),  # neg → Ridge only
            1: (1.0, 0.0),  # low → Ridge only
            2: (0.5, 0.5),  # normal → equal blend
            3: (0.0, 1.0),  # high → XGBoost only
        }

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame,
            feat_cols: list[str]) -> "RegimeAwareEnsemble":
        self.feat_cols_ = list(feat_cols)
        Xtr = train[feat_cols].to_numpy()
        ytr = train["target_price"].to_numpy()
        Xva = valid[feat_cols].to_numpy()
        yva = valid["target_price"].to_numpy()

        # Regime thresholds from train set
        self.thr_ = _regime_thresholds(ytr)
        regime_tr = _label_regime(ytr, self.thr_)

        # Regime classifier
        self.classifier_ = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=self.random_state,
        )
        self.classifier_.fit(Xtr, regime_tr)

        # Ridge (full panel, scaled)
        self.scaler_ = StandardScaler().fit(Xtr)
        self.ridge_ = Ridge(alpha=self.ridge_alpha).fit(
            self.scaler_.transform(Xtr), ytr,
        )

        # XGBoost (full panel)
        self.xgb_ = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=self.random_state, n_jobs=1,
            early_stopping_rounds=20, eval_metric="rmse",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.xgb_.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            arr = X[self.feat_cols_].to_numpy()
        else:
            arr = np.asarray(X)
        ridge_yhat = self.ridge_.predict(self.scaler_.transform(arr))
        xgb_yhat = self.xgb_.predict(arr)

        if self.soft_assign:
            proba = self.classifier_.predict_proba(arr)  # (n, 4)
            # Each column is regime probability; weighted blend
            yhat = np.zeros(len(arr))
            for r in range(4):
                w_r, w_x = self.regime_blend_[r]
                yhat += proba[:, r] * (w_r * ridge_yhat + w_x * xgb_yhat)
            return yhat
        else:
            pred_regime = self.classifier_.predict(arr)
            yhat = np.empty(len(arr))
            for r in range(4):
                mask = pred_regime == r
                if not mask.any():
                    continue
                w_r, w_x = self.regime_blend_[r]
                yhat[mask] = w_r * ridge_yhat[mask] + w_x * xgb_yhat[mask]
            return yhat


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline entry point — matches src/forecaster.py contract
# ──────────────────────────────────────────────────────────────────────────────

def fit_regime_ensemble_pipeline(
    df: pd.DataFrame,
    split: dict,
    rf_params: dict,
    horizon: int = 24,
    soft_assign: bool = False,
    feature_set: str = "full",
    **_,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    from src.forecaster import _prep_tabular, _make_pred_df, _build_acc, _neg_clf

    feat_cols, train, valid, test = _prep_tabular(df, split, horizon, feature_set)
    eval_data = pd.concat([valid, test]).reset_index(drop=True)

    model = RegimeAwareEnsemble(soft_assign=soft_assign).fit(train, valid, feat_cols)
    pred = model.predict(eval_data)

    pred_df = _make_pred_df(eval_data, pred, "regime_ensemble", horizon)
    acc_df = _build_acc(pred_df, "regime_ensemble", horizon)
    neg_df, clf_m = _neg_clf(train, eval_data, feat_cols, rf_params, horizon)

    # Per-regime breakdown
    yt = eval_data["target_price"].to_numpy()
    regime_actual = _label_regime(yt, model.thr_)
    by_regime = []
    for r, name in enumerate(REGIME_NAMES):
        mask = regime_actual == r
        if not mask.any():
            continue
        by_regime.append({
            "regime": name,
            "n": int(mask.sum()),
            "mae": float(mean_absolute_error(yt[mask], pred[mask])),
            "rmse": float(mean_squared_error(yt[mask], pred[mask]) ** 0.5),
        })

    metrics = {
        **clf_m,
        "mae": float(acc_df["mae"].iloc[0]),
        "rmse": float(acc_df["rmse"].iloc[0]),
        "regime_thresholds": model.thr_,
        "regime_breakdown": by_regime,
        "soft_assign": soft_assign,
    }
    return pred_df, acc_df, neg_df, metrics
