from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ember yearly data loader and feature builder
# ---------------------------------------------------------------------------

EMBER_FUEL_MAP = {
    "Wind": "wind",
    "Solar": "solar",
    "Hydro": "hydro",
    "Coal": "coal",
    "Gas": "gas",
    "Nuclear": "nuclear",
    "Other Renewables": "other_renewables",
    "Other Fossil": "other_fossil",
    "Bioenergy": "bioenergy",
}


def load_ember_annual(ember_path: str | Path) -> pd.DataFrame | None:
    p = Path(ember_path)
    if not p.exists():
        return None
    try:
        raw = pd.read_csv(p)
    except Exception:
        return None

    # Detect column layout — Ember long format has: area, year, variable, unit, value
    col_lower = {c.lower(): c for c in raw.columns}
    country_col = col_lower.get("area") or col_lower.get("country") or col_lower.get("entity")
    year_col = col_lower.get("year")
    var_col = col_lower.get("variable") or col_lower.get("category")
    val_col = col_lower.get("value") or col_lower.get("generation_twh")

    if not all([country_col, year_col, var_col, val_col]):
        return None

    df = raw[[country_col, year_col, var_col, val_col]].copy()
    df.columns = ["country", "year", "variable", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def build_ember_annual_features(df_ember: pd.DataFrame, country: str = "Germany") -> pd.DataFrame:
    sub = df_ember[df_ember["country"].str.contains(country, case=False, na=False)].copy()
    if sub.empty:
        return pd.DataFrame()

    years = sorted(sub["year"].dropna().unique())
    rows = []
    for yr in years:
        yr_data = sub[sub["year"] == yr]
        val = yr_data.set_index("variable")["value"].to_dict()

        def _get(*keys: str) -> float:
            for k in keys:
                for vk, vv in val.items():
                    if k.lower() in str(vk).lower():
                        return float(vv) if pd.notna(vv) else float("nan")
            return float("nan")

        total_gen = _get("Total Generation", "Electricity Generation")
        wind_gen = _get("Wind")
        solar_gen = _get("Solar")
        hydro_gen = _get("Hydro")
        coal_gen = _get("Coal")
        gas_gen = _get("Gas")
        nuclear_gen = _get("Nuclear")
        fossil_gen = _get("Fossil", "Total Fossil")
        renewables_gen = _get("Renewables", "Total Renewables")
        demand = _get("Demand", "Consumption")
        imports = _get("Net Import", "Import")

        def _share(num: float, den: float) -> float:
            if np.isnan(num) or np.isnan(den) or den == 0:
                return float("nan")
            return float(num / den)

        rows.append(
            {
                "year": int(yr),
                "total_generation_twh": total_gen,
                "renewable_generation_share_yearly": _share(renewables_gen, total_gen),
                "fossil_generation_share_yearly": _share(fossil_gen, total_gen),
                "wind_share_yearly": _share(wind_gen, total_gen),
                "solar_share_yearly": _share(solar_gen, total_gen),
                "hydro_share_yearly": _share(hydro_gen, total_gen),
                "coal_share_yearly": _share(coal_gen, total_gen),
                "gas_share_yearly": _share(gas_gen, total_gen),
                "nuclear_share_yearly": _share(nuclear_gen, total_gen),
                "import_share_yearly": _share(imports, demand),
                "demand_twh": demand,
            }
        )
    return pd.DataFrame(rows)


def merge_ember_to_panel(panel: pd.DataFrame, ember_features: pd.DataFrame) -> pd.DataFrame:
    if ember_features.empty:
        return panel
    out = panel.copy()
    out["year"] = pd.to_datetime(out["datetime"]).dt.year
    out = out.merge(ember_features, on="year", how="left")
    return out


# ---------------------------------------------------------------------------
# Energy-structure regime analysis
# ---------------------------------------------------------------------------

def build_energy_mix_regime_summary(
    hourly: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Group by renewable-share regime and compute per-regime statistics.

    hourly must have columns: datetime, price, observed_price or price,
    renewable_share, q_DA_optimal (or q_DA), counterfactual_price, demand_base,
    rolling_elasticity (optional).
    """
    h = hourly.copy()

    price_col = "observed_price" if "observed_price" in h.columns else "price"
    q_col = "q_DA_optimal" if "q_DA_optimal" in h.columns else "q_DA"

    if "renewable_share" not in h.columns:
        if "renewable_total" in h.columns and "load" in h.columns:
            h["renewable_share"] = h["renewable_total"] / (h["load"] + 1e-6)
        else:
            h["renewable_share"] = float("nan")

    p33 = float(h["renewable_share"].quantile(0.33))
    p66 = float(h["renewable_share"].quantile(0.66))
    p90_scarcity = float(h["scarcity_index"].quantile(0.90)) if "scarcity_index" in h.columns else None
    p90_oversupply = float(h["oversupply_index"].quantile(0.90)) if "oversupply_index" in h.columns else None

    def _regime(rs: float) -> str:
        if rs < p33:
            return "low_renewable"
        if rs < p66:
            return "medium_renewable"
        return "high_renewable"

    h["regime"] = h["renewable_share"].map(_regime)

    # Add special regimes
    h.loc[h[price_col] < 0, "regime"] = "negative_price"
    if p90_scarcity is not None:
        h.loc[h["scarcity_index"] >= p90_scarcity, "regime"] = "scarcity"
    if p90_oversupply is not None:
        h.loc[h["oversupply_index"] >= p90_oversupply, "regime"] = "oversupply"

    def _agg(grp: pd.DataFrame) -> dict:
        p90_thr = float(hourly[price_col].quantile(0.9)) if len(hourly) else 0.0
        q_vals = grp[q_col] if q_col in grp.columns else pd.Series([float("nan")])
        cf_price = grp["counterfactual_price"] if "counterfactual_price" in grp.columns else pd.Series([float("nan")])
        obs_price = grp[price_col]
        db = grp["demand_base"] if "demand_base" in grp.columns else pd.Series([1.0] * len(grp))
        elast = grp["rolling_elasticity"] if "rolling_elasticity" in grp.columns else pd.Series([float("nan")])

        obs_cost = (obs_price * db).sum()
        cf_cost = (cf_price * (db - q_vals)).sum() if "counterfactual_price" in grp.columns else float("nan")

        return {
            "num_hours": len(grp),
            "avg_price": float(obs_price.mean()),
            "negative_price_frequency": float((obs_price < 0).mean()),
            "high_price_frequency": float((obs_price >= p90_thr).mean()),
            "avg_DR_quantity": float(q_vals.mean()),
            "avg_price_reduction": float((obs_price - cf_price).mean()) if "counterfactual_price" in grp.columns else float("nan"),
            "avg_cost_reduction": float(obs_cost - cf_cost) if not np.isnan(cf_cost) else float("nan"),
            "estimated_elasticity_mean": float(elast.mean()),
        }

    rows = []
    for regime_name, grp in h.groupby("regime", observed=True):
        rec = _agg(grp)
        rec["regime"] = regime_name
        rows.append(rec)

    result = pd.DataFrame(rows)[
        ["regime", "num_hours", "avg_price", "negative_price_frequency",
         "high_price_frequency", "avg_DR_quantity", "avg_price_reduction",
         "avg_cost_reduction", "estimated_elasticity_mean"]
    ].sort_values("regime").reset_index(drop=True)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result
