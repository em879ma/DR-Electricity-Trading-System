"""
Forecast model benchmark for day-ahead (H=24) electricity price prediction.
Compares: Ridge, XGBoost, LSTM, TCN, Two-Stage RF.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---- project import ----
from src.forecasting import RF_FEATURES, fit_two_stage_price_forecast

# ---- optional deps ----
try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False


# ============================================================
# Constants
# ============================================================

HORIZON = 24
SEQ_LEN = 48  # rolling-window length for LSTM/TCN (hours)
SEQ_FEATURES = [
    "price", "load", "renewable_total",
    "net_load", "renewable_share",
    "hour_sin", "hour_cos",
]

MODEL_COLORS = {
    "two_stage_rf": "#1f77b4",
    "ridge":        "#ff7f0e",
    "xgboost":      "#2ca02c",
    "lstm":         "#d62728",
    "tcn":          "#9467bd",
}
MODEL_LABELS = {
    "two_stage_rf": "Two-Stage RF",
    "ridge":        "Ridge",
    "xgboost":      "XGBoost",
    "lstm":         "LSTM",
    "tcn":          "TCN",
}


# ============================================================
# Data preparation helpers
# ============================================================

def _split_bounds(df: pd.DataFrame, split: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (valid_start_target_dt, test_start_target_dt)."""
    work = df.copy()
    work["_tp"] = work["price"].shift(-HORIZON)
    feat = [c for c in RF_FEATURES if c in work.columns]
    fit = work[feat + ["_tp", "datetime"]].dropna().sort_values("datetime").reset_index(drop=True)
    n = len(fit)
    n_train = int(n * split["train_ratio"])
    n_valid = int(n * (split["train_ratio"] + split.get("valid_ratio", 0.15)))
    valid_start = pd.to_datetime(fit.iloc[n_train]["datetime"]) + pd.to_timedelta(HORIZON, "h")
    test_start = pd.to_datetime(fit.iloc[n_valid]["datetime"]) + pd.to_timedelta(HORIZON, "h")
    return valid_start, test_start


def _prep_tabular(df: pd.DataFrame, split: dict):
    """Returns (feat_cols, train_df, valid_df, test_df)."""
    work = df.copy()
    work["target_price"] = work["price"].shift(-HORIZON)
    feat_cols = [c for c in RF_FEATURES if c in work.columns]
    fit = work[feat_cols + ["target_price", "datetime"]].dropna().reset_index(drop=True)
    n = len(fit)
    n_train = int(n * split["train_ratio"])
    n_valid = int(n * (split["train_ratio"] + split.get("valid_ratio", 0.15)))
    return feat_cols, fit.iloc[:n_train], fit.iloc[n_train:n_valid], fit.iloc[n_valid:]


def _prep_sequences(df: pd.DataFrame, split: dict, seq_len: int = SEQ_LEN) -> dict:
    """Build (X, y, target_datetimes) arrays for sequence models."""
    ts = df.sort_values("datetime").reset_index(drop=True)
    seq_cols = [c for c in SEQ_FEATURES if c in ts.columns]

    n = len(ts)
    n_train = int(n * split["train_ratio"])
    n_valid = int(n * (split["train_ratio"] + split.get("valid_ratio", 0.15)))

    train_vals = ts.iloc[:n_train][seq_cols].to_numpy(float)
    feat_mean = train_vals.mean(axis=0)
    feat_std = train_vals.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    norm = ((ts[seq_cols].to_numpy(float) - feat_mean) / feat_std).astype(np.float32)
    prices = ts["price"].to_numpy(float)
    datetimes = pd.to_datetime(ts["datetime"])

    xs, ys, tgt_dts = [], [], []
    for t in range(seq_len - 1, n - HORIZON):
        xs.append(norm[t - seq_len + 1: t + 1])
        ys.append(prices[t + HORIZON])
        tgt_dts.append(datetimes.iloc[t + HORIZON])

    X = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.float32)
    tgt_dts = pd.DatetimeIndex(tgt_dts)

    # Split by position of forecast origin in original ts
    orig_pos = np.arange(seq_len - 1, n - HORIZON)
    tr = orig_pos < n_train
    va = (orig_pos >= n_train) & (orig_pos < n_valid)
    te = orig_pos >= n_valid

    return dict(
        X_train=X[tr], y_train=y[tr], dt_train=tgt_dts[tr],
        X_valid=X[va], y_valid=y[va], dt_valid=tgt_dts[va],
        X_test=X[te],  y_test=y[te],  dt_test=tgt_dts[te],
        n_features=len(seq_cols),
    )


# ============================================================
# Ridge
# ============================================================

def fit_ridge(df: pd.DataFrame, split: dict, alpha: float = 10.0) -> dict:
    feat_cols, train, valid, test = _prep_tabular(df, split)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[feat_cols])
    X_te = scaler.transform(test[feat_cols])

    model = Ridge(alpha=alpha)
    model.fit(X_tr, train["target_price"])
    pred = model.predict(X_te)

    tgt_dt = pd.to_datetime(test["datetime"]) + pd.to_timedelta(HORIZON, "h")
    return dict(model="ridge", forecast=pred,
                actual=test["target_price"].to_numpy(),
                target_datetime=tgt_dt.to_numpy())


# ============================================================
# XGBoost
# ============================================================

def fit_xgboost(df: pd.DataFrame, split: dict) -> dict:
    if not _XGB:
        warnings.warn("xgboost not installed; skipping XGBoost.")
        return {}
    feat_cols, train, valid, test = _prep_tabular(df, split)

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1,
        early_stopping_rounds=20, eval_metric="rmse",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            train[feat_cols].to_numpy(), train["target_price"].to_numpy(),
            eval_set=[(valid[feat_cols].to_numpy(), valid["target_price"].to_numpy())],
            verbose=False,
        )

    pred = model.predict(test[feat_cols].to_numpy())
    tgt_dt = pd.to_datetime(test["datetime"]) + pd.to_timedelta(HORIZON, "h")
    return dict(model="xgboost", forecast=pred,
                actual=test["target_price"].to_numpy(),
                target_datetime=tgt_dt.to_numpy())


# ============================================================
# PyTorch models
# ============================================================

if _TORCH:
    class _LSTMNet(nn.Module):
        def __init__(self, n_feat: int, hidden: int = 64, n_layers: int = 2, drop: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True,
                                dropout=drop if n_layers > 1 else 0.0)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    class _TCNBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, drop: float = 0.2):
            super().__init__()
            pad = (kernel - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(drop)
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x):  # (B, C, T)
            h = self.drop(self.relu(self.conv(x)[:, :, :x.shape[2]]))
            return h + self.shortcut(x)

    class _TCNNet(nn.Module):
        def __init__(self, n_feat: int, n_ch: int = 32, kernel: int = 3,
                     dilations: list | None = None, drop: float = 0.2):
            super().__init__()
            dilations = dilations or [1, 2, 4, 8, 16, 32]
            layers, in_ch = [], n_feat
            for d in dilations:
                layers.append(_TCNBlock(in_ch, n_ch, kernel, d, drop))
                in_ch = n_ch
            self.net = nn.Sequential(*layers)
            self.fc = nn.Linear(n_ch, 1)

        def forward(self, x):  # x: (B, T, F)
            h = self.net(x.permute(0, 2, 1))  # (B, n_ch, T)
            return self.fc(h.mean(dim=2)).squeeze(-1)


def _train_torch(model, X_tr, y_tr, X_va, y_va,
                 epochs: int = 60, batch: int = 64, lr: float = 1e-3, patience: int = 10):
    if not _TORCH:
        return model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=batch, shuffle=True)
    X_va_t = torch.from_numpy(X_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss, best_state, no_imp = float("inf"), None, 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va_t), y_va_t).item()
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, no_imp = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def fit_lstm(df: pd.DataFrame, split: dict) -> dict:
    if not _TORCH:
        warnings.warn("torch not installed; skipping LSTM.")
        return {}
    seqs = _prep_sequences(df, split)
    model = _LSTMNet(seqs["n_features"])
    model = _train_torch(model, seqs["X_train"], seqs["y_train"],
                         seqs["X_valid"], seqs["y_valid"])
    device = next(model.parameters()).device
    with torch.no_grad():
        pred = model(torch.from_numpy(seqs["X_test"]).to(device)).cpu().numpy()
    return dict(model="lstm", forecast=pred.astype(float),
                actual=seqs["y_test"].astype(float),
                target_datetime=seqs["dt_test"])


def fit_tcn(df: pd.DataFrame, split: dict) -> dict:
    if not _TORCH:
        warnings.warn("torch not installed; skipping TCN.")
        return {}
    seqs = _prep_sequences(df, split)
    model = _TCNNet(seqs["n_features"])
    model = _train_torch(model, seqs["X_train"], seqs["y_train"],
                         seqs["X_valid"], seqs["y_valid"])
    device = next(model.parameters()).device
    with torch.no_grad():
        pred = model(torch.from_numpy(seqs["X_test"]).to(device)).cpu().numpy()
    return dict(model="tcn", forecast=pred.astype(float),
                actual=seqs["y_test"].astype(float),
                target_datetime=seqs["dt_test"])


def fit_two_stage(df: pd.DataFrame, split: dict, rf_params: dict, test_start: pd.Timestamp) -> dict:
    pred_df, _, _, _ = fit_two_stage_price_forecast(df, rf_params, split, horizon=HORIZON)
    if pred_df.empty:
        return {}
    # filter to test period only (same as other models)
    tgt_dt = pd.to_datetime(pred_df["target_datetime"])
    mask = tgt_dt >= test_start
    sub = pred_df[mask]
    return dict(model="two_stage_rf",
                forecast=sub["forecast"].to_numpy(),
                actual=sub["actual"].to_numpy(),
                target_datetime=tgt_dt[mask].to_numpy())


# ============================================================
# Metrics
# ============================================================

def compute_metrics(result: dict) -> dict:
    """
    Comprehensive metrics for a single model's test-set predictions.
    Covers overall accuracy, regime-specific MAE, neg-price and peak detection,
    pinball (quantile) loss, and bias.
    """
    y = np.asarray(result["actual"], dtype=float)
    yhat = np.asarray(result["forecast"], dtype=float)
    err = yhat - y
    model = result["model"]

    p90 = np.percentile(y, 90)
    p75 = np.percentile(y, 75)
    p25 = np.percentile(y, 25)

    # Regime masks
    neg_m   = y < 0
    low_m   = y < p25
    norm_m  = (y >= p25) & (y < p75)
    high_m  = (y >= p75) & (y < p90)
    peak_m  = y >= p90
    pos_m   = y > 0

    def safe_mae(mask):
        return float(np.mean(np.abs(err[mask]))) if mask.any() else float("nan")

    def pinball(q):
        e = y - yhat
        return float(np.mean(np.where(e >= 0, q * e, (q - 1) * e)))

    # Negative-price regression detection (forecast goes negative)
    neg_tp  = int(((yhat < 0) & neg_m).sum())
    neg_fp  = int(((yhat < 0) & ~neg_m).sum())
    neg_fn  = int(((yhat >= 0) & neg_m).sum())
    neg_act = int(neg_m.sum())
    neg_pred = int((yhat < 0).sum())

    # Peak-price detection (forecast ≥ P90 threshold)
    pk_tp  = int(((yhat >= p90) & peak_m).sum())
    pk_fp  = int(((yhat >= p90) & ~peak_m).sum())
    pk_fn  = int(((yhat < p90) & peak_m).sum())

    return {
        "model": model,
        # Overall
        "mae":  float(mean_absolute_error(y, yhat)),
        "rmse": float(mean_squared_error(y, yhat) ** 0.5),
        "bias": float(err.mean()),
        "r2":   float(r2_score(y, yhat)),
        "mape": float(np.mean(np.abs(err[pos_m] / y[pos_m])) * 100) if pos_m.any() else float("nan"),
        # Negative-price detection (regression)
        "neg_recall_reg":  neg_tp / (neg_tp + neg_fn) if (neg_tp + neg_fn) > 0 else float("nan"),
        "neg_precision_reg": neg_tp / (neg_tp + neg_fp) if (neg_tp + neg_fp) > 0 else float("nan"),
        "neg_f1_reg": (2*neg_tp / (2*neg_tp + neg_fp + neg_fn)) if (2*neg_tp + neg_fp + neg_fn) > 0 else float("nan"),
        "n_neg_actual": neg_act,
        "n_neg_pred_reg": neg_pred,
        # Peak detection
        "peak_recall":    pk_tp / (pk_tp + pk_fn) if (pk_tp + pk_fn) > 0 else float("nan"),
        "peak_precision": pk_tp / (pk_tp + pk_fp) if (pk_tp + pk_fp) > 0 else float("nan"),
        "peak_f1": (2*pk_tp / (2*pk_tp + pk_fp + pk_fn)) if (2*pk_tp + pk_fp + pk_fn) > 0 else float("nan"),
        # Regime MAE
        "mae_negative": safe_mae(neg_m),
        "mae_low":      safe_mae(low_m),
        "mae_normal":   safe_mae(norm_m),
        "mae_high":     safe_mae(high_m),
        "mae_peak":     safe_mae(peak_m),
        # Pinball (quantile) loss — lower is better
        "pinball_q10": pinball(0.10),
        "pinball_q25": pinball(0.25),
        "pinball_q50": pinball(0.50),
        "pinball_q75": pinball(0.75),
        "pinball_q90": pinball(0.90),
    }


# ============================================================
# Figures
# ============================================================

_FIG_DPI = 150
_PALETTE = MODEL_COLORS


def _model_order(metrics_df: pd.DataFrame) -> list[str]:
    order = ["two_stage_rf", "ridge", "xgboost", "lstm", "tcn"]
    return [m for m in order if m in metrics_df["model"].values]


def plot_overall_accuracy(metrics_df: pd.DataFrame, save_path: str | Path) -> None:
    order = _model_order(metrics_df)
    df = metrics_df.set_index("model").loc[order]
    labels = [MODEL_LABELS.get(m, m) for m in order]
    colors = [_PALETTE.get(m, "#888") for m in order]
    x = np.arange(len(order))
    w = 0.28

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title, unit in zip(
        axes,
        ["mae", "rmse", "r2"],
        ["MAE", "RMSE", "R²"],
        ["€/MWh", "€/MWh", ""],
    ):
        vals = df[col].to_numpy()
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(unit, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002 * max(vals),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Overall Forecast Accuracy — Test Set (H=24)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_regime_mae(metrics_df: pd.DataFrame, save_path: str | Path) -> None:
    order = _model_order(metrics_df)
    df = metrics_df.set_index("model").loc[order]
    regimes = ["mae_negative", "mae_low", "mae_normal", "mae_high", "mae_peak"]
    regime_labels = ["Negative\n(price<0)", "Low\n(<P25)", "Normal\n(P25–P75)", "High\n(P75–P90)", "Peak\n(>P90)"]
    n_reg = len(regimes)
    x = np.arange(n_reg)
    w = 0.14
    offsets = np.linspace(-(len(order) - 1) / 2 * w, (len(order) - 1) / 2 * w, len(order))

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, m in enumerate(order):
        vals = [df.loc[m, r] if not np.isnan(df.loc[m, r]) else 0 for r in regimes]
        ax.bar(x + offsets[i], vals, width=w, color=_PALETTE.get(m, "#888"),
               label=MODEL_LABELS.get(m, m), edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels, fontsize=10)
    ax.set_ylabel("MAE (€/MWh)", fontsize=10)
    ax.set_title("Regime-Specific Forecast Error", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_radar(metrics_df: pd.DataFrame, save_path: str | Path) -> None:
    """Radar chart: normalize each dimension so higher = better."""
    order = _model_order(metrics_df)
    df = metrics_df.set_index("model").loc[order]

    dims = {
        "Low MAE":       ("mae",          True),
        "Low RMSE":      ("rmse",         True),
        "High R²":       ("r2",           False),
        "Neg Recall":    ("neg_recall_reg", False),
        "Peak Recall":   ("peak_recall",  False),
        "Low Neg-MAE":   ("mae_negative", True),
        "Low Peak-MAE":  ("mae_peak",     True),
        "Low Pinball-Q90": ("pinball_q90", True),
    }
    labels = list(dims.keys())
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    normalized = {}
    for dim_label, (col, lower_is_better) in dims.items():
        vals = df[col].replace([np.inf, -np.inf], np.nan).fillna(df[col].median())
        if lower_is_better:
            mn, mx = vals.min(), vals.max()
            norm = 1 - (vals - mn) / (mx - mn + 1e-9)
        else:
            mn, mx = vals.min(), vals.max()
            norm = (vals - mn) / (mx - mn + 1e-9)
        normalized[dim_label] = norm

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="grey")

    for m in order:
        vals_norm = [normalized[dl][m] for dl in labels] + [normalized[labels[0]][m]]
        ax.plot(angles, vals_norm, "-o", color=_PALETTE.get(m, "#888"),
                linewidth=1.8, markersize=4, label=MODEL_LABELS.get(m, m))
        ax.fill(angles, vals_norm, alpha=0.07, color=_PALETTE.get(m, "#888"))

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.set_title("Multi-Dimensional Model Comparison\n(outer = better)", fontsize=11, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_neg_peak_detection(metrics_df: pd.DataFrame, save_path: str | Path) -> None:
    order = _model_order(metrics_df)
    df = metrics_df.set_index("model").loc[order]
    labels = [MODEL_LABELS.get(m, m) for m in order]
    x = np.arange(len(order))
    w = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1, ax2 = axes

    for ax, pairs, title in [
        (ax1,
         [("neg_recall_reg", "Recall"), ("neg_precision_reg", "Precision"), ("neg_f1_reg", "F1")],
         "Negative-Price Detection (Regression)"),
        (ax2,
         [("peak_recall", "Recall"), ("peak_precision", "Precision"), ("peak_f1", "F1")],
         "Peak-Price Detection (≥P90)"),
    ]:
        offsets = [-w, 0, w]
        bar_colors = ["#5470c6", "#91cc75", "#fac858"]
        for (col, lbl), off, bc in zip(pairs, offsets, bar_colors):
            vals = df[col].fillna(0).to_numpy()
            bars = ax.bar(x + off, vals, width=w, label=lbl, color=bc,
                          edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if v > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle("Extreme-Price Event Detection (Test Set)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_error_distribution(results: dict[str, dict], save_path: str | Path) -> None:
    """Violin + box plot of prediction errors per model."""
    order = [m for m in ["two_stage_rf", "ridge", "xgboost", "lstm", "tcn"] if m in results]
    errors = {m: np.asarray(results[m]["forecast"]) - np.asarray(results[m]["actual"]) for m in order}

    fig, ax = plt.subplots(figsize=(11, 4.5))
    parts = ax.violinplot([errors[m] for m in order], positions=range(len(order)),
                          showmedians=True, showextrema=False)
    for i, (body, m) in enumerate(zip(parts["bodies"], order)):
        body.set_facecolor(_PALETTE.get(m, "#888"))
        body.set_alpha(0.5)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax.boxplot([errors[m] for m in order], positions=range(len(order)),
               widths=0.12, patch_artist=False, showfliers=False,
               medianprops=dict(color="black", linewidth=0))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in order], fontsize=10)
    ax.set_ylabel("Forecast Error (€/MWh)", fontsize=10)
    ax.set_title("Error Distribution — Test Set (H=24)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pinball_loss(metrics_df: pd.DataFrame, save_path: str | Path) -> None:
    """Line chart of pinball loss across quantiles — lower is better."""
    order = _model_order(metrics_df)
    df = metrics_df.set_index("model").loc[order]
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    cols = ["pinball_q10", "pinball_q25", "pinball_q50", "pinball_q75", "pinball_q90"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for m in order:
        vals = df.loc[m, cols].to_numpy()
        ax.plot(quantiles, vals, "-o", color=_PALETTE.get(m, "#888"),
                label=MODEL_LABELS.get(m, m), linewidth=2, markersize=6)

    ax.set_xlabel("Quantile τ", fontsize=10)
    ax.set_ylabel("Pinball Loss (€/MWh)", fontsize=10)
    ax.set_title("Quantile (Pinball) Loss — Tail Calibration", fontsize=11, fontweight="bold")
    ax.set_xticks(quantiles)
    ax.set_xticklabels([f"q{int(q*100)}" for q in quantiles])
    ax.legend(fontsize=9, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main benchmark runner
# ============================================================

def run_benchmark(df: pd.DataFrame, split: dict, rf_params: dict,
                  output_dir: str | Path = "outputs/benchmark") -> pd.DataFrame:
    """
    Run all models, compute metrics, save tables and figures.
    Returns the full metrics DataFrame.
    """
    out = Path(output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    _, test_start = _split_bounds(df, split)
    results: dict[str, dict] = {}

    print("  [1/5] Ridge...")
    r = fit_ridge(df, split)
    if r:
        results["ridge"] = r

    print("  [2/5] XGBoost...")
    r = fit_xgboost(df, split)
    if r:
        results["xgboost"] = r

    print("  [3/5] LSTM...")
    r = fit_lstm(df, split)
    if r:
        results["lstm"] = r

    print("  [4/5] TCN...")
    r = fit_tcn(df, split)
    if r:
        results["tcn"] = r

    print("  [5/5] Two-Stage RF...")
    r = fit_two_stage(df, split, rf_params, test_start)
    if r:
        results["two_stage_rf"] = r

    # --- Align all models to common test datetimes ---
    # Use intersection of all model target_datetimes
    all_dt = [set(pd.DatetimeIndex(v["target_datetime"]).normalize()) for v in results.values()]
    # Actually keep individual; just ensure actual prices are same
    # (They come from same df, same target period)

    # --- Compute metrics ---
    metrics_rows = [compute_metrics(v) for v in results.values()]
    metrics_df = pd.DataFrame(metrics_rows)

    # --- Save tables ---
    metrics_df.to_csv(out / "tables" / "benchmark_metrics.csv", index=False)

    summary_cols = ["model", "mae", "rmse", "bias", "r2",
                    "neg_recall_reg", "neg_precision_reg", "neg_f1_reg",
                    "peak_recall", "peak_precision", "peak_f1",
                    "mae_negative", "mae_peak"]
    metrics_df[summary_cols].to_csv(out / "tables" / "benchmark_summary.csv", index=False)

    # Save per-model predictions for inspection
    for name, res in results.items():
        pd.DataFrame({
            "target_datetime": res["target_datetime"],
            "actual": res["actual"],
            "forecast": res["forecast"],
        }).to_csv(out / "tables" / f"predictions_{name}.csv", index=False)

    # --- Generate figures ---
    print("  Generating figures...")
    plot_overall_accuracy(metrics_df, out / "figures" / "07a_overall_accuracy.png")
    plot_regime_mae(metrics_df, out / "figures" / "07b_regime_mae.png")
    plot_radar(metrics_df, out / "figures" / "07c_radar.png")
    plot_neg_peak_detection(metrics_df, out / "figures" / "07d_detection.png")
    plot_error_distribution(results, out / "figures" / "07e_error_distribution.png")
    plot_pinball_loss(metrics_df, out / "figures" / "07f_pinball_loss.png")

    return metrics_df
