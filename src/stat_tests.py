"""
Statistical significance tests for forecast comparison.

Reference: Lago et al. (2021), §3.3 (DM test) and Giacomini & White (2006).

Two tests:
  - Diebold-Mariano (DM): per-hour comparison of two forecasts' loss
    differentials. Null hypothesis H0: E[L(e1) - L(e2)] = 0. We use
    squared-error loss; HAC variance via Newey-West.
  - Giacomini-White (GW): multivariate joint test across all 24 hours.
    Treats each day's 24-h error vector as one observation; tests whether
    the mean loss-differential vector is zero (Hotelling-type χ² statistic).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    loss: str = "squared",
    h: int = 1,
) -> tuple[float, float]:
    """
    Diebold-Mariano test on two error series.

    Parameters
    ----------
    e1, e2 : 1-D arrays of forecast errors (yhat - y) for two models.
    loss   : 'squared' or 'absolute'.
    h      : forecast horizon for Newey-West HAC lag truncation.

    Returns
    -------
    (dm_stat, p_value). p<0.05 → reject H0 of equal predictive accuracy.
    Negative dm_stat → model 1 is more accurate.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    mask = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[mask], e2[mask]
    n = len(e1)
    if n < 10:
        return float("nan"), float("nan")

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss '{loss}'")

    mean_d = float(np.mean(d))
    # Newey-West HAC variance with truncation lag = h-1
    gamma0 = float(np.var(d, ddof=0))
    var_d = gamma0
    for lag in range(1, h):
        cov = float(np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d)))
        var_d += 2.0 * (1.0 - lag / h) * cov

    if var_d <= 0:
        return float("nan"), float("nan")

    dm = mean_d / np.sqrt(var_d / n)
    # Harvey small-sample correction
    k = (n + 1 - 2 * h + h * (h - 1) / n) / n
    dm = dm * np.sqrt(max(k, 1e-9))
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def giacomini_white_test(
    E1: np.ndarray,
    E2: np.ndarray,
    loss: str = "squared",
) -> tuple[float, float]:
    """
    Multivariate Giacomini-White test across H hours jointly.

    Parameters
    ----------
    E1, E2 : (N_days, H) arrays of forecast errors. Each row is one day,
             each column is one of the H target hours.
    loss   : 'squared' or 'absolute'.

    Returns
    -------
    (chi2_stat, p_value). p<0.05 → reject H0 of joint equal accuracy.
    """
    E1 = np.asarray(E1, dtype=float)
    E2 = np.asarray(E2, dtype=float)
    if E1.shape != E2.shape:
        raise ValueError("E1 and E2 must have same shape")

    if loss == "squared":
        D = E1 ** 2 - E2 ** 2
    elif loss == "absolute":
        D = np.abs(E1) - np.abs(E2)
    else:
        raise ValueError(f"Unknown loss '{loss}'")

    # drop days with any missing
    mask = np.isfinite(D).all(axis=1)
    D = D[mask]
    n, hcols = D.shape
    if n < hcols + 5:
        return float("nan"), float("nan")

    mean_d = D.mean(axis=0)
    cov_d = np.cov(D, rowvar=False, ddof=0)
    # ridge for numerical stability
    cov_d += 1e-8 * np.eye(hcols)
    try:
        inv = np.linalg.inv(cov_d)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")

    chi2 = float(n * mean_d @ inv @ mean_d)
    p = float(1.0 - stats.chi2.cdf(chi2, df=hcols))
    return chi2, p


def pairwise_dm_table(
    error_dict: dict[str, np.ndarray],
    loss: str = "squared",
) -> pd.DataFrame:
    """
    Build a pairwise DM-test table for multiple models.

    Parameters
    ----------
    error_dict : {model_name: error_array}. All arrays must share the same
                 length and ordering (same evaluation set).
    loss       : 'squared' or 'absolute'.

    Returns
    -------
    DataFrame with columns (model_1, model_2, dm_stat, p_value, winner).
    """
    names = list(error_dict.keys())
    rows = []
    for i, m1 in enumerate(names):
        for m2 in names[i + 1:]:
            dm, p = diebold_mariano_test(error_dict[m1], error_dict[m2], loss=loss)
            if not np.isfinite(dm):
                winner = "n/a"
            elif p < 0.05:
                winner = m1 if dm < 0 else m2
            else:
                winner = "tie"
            rows.append({
                "model_1": m1, "model_2": m2,
                "dm_stat": dm, "p_value": p,
                "winner": winner,
            })
    return pd.DataFrame(rows)


def daily_error_matrix(
    pred_df: pd.DataFrame,
    target_col: str = "target_datetime",
    actual_col: str = "actual",
    forecast_col: str = "forecast",
) -> np.ndarray:
    """
    Reshape a long forecast frame into a (n_days, 24) error matrix.

    Returns NaN-padded matrix for the GW test. Rows with any NaN are
    dropped inside the test.
    """
    work = pred_df.copy()
    work[target_col] = pd.to_datetime(work[target_col])
    work["err"] = work[forecast_col] - work[actual_col]
    work["date"] = work[target_col].dt.date
    work["hour"] = work[target_col].dt.hour
    pivot = work.pivot_table(index="date", columns="hour",
                             values="err", aggfunc="mean")
    pivot = pivot.reindex(columns=range(24))
    return pivot.to_numpy()
