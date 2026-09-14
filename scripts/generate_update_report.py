"""
Generate a concise update report (Word document) summarising all changes
made from v1.0 to the current version of the DR Compass pipeline.

Output: outputs/DR_Update_Report.docx
"""
from __future__ import annotations
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

OUT = Path("outputs/DR_Update_Report.docx")

# ── colour palette ────────────────────────────────────────────────────────────
ORANGE   = RGBColor(0xE8, 0x5D, 0x04)
DARK     = RGBColor(0x1A, 0x1A, 0x2E)
GREY_BG  = RGBColor(0xF5, 0xF5, 0xF5)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
GREEN    = RGBColor(0x2D, 0x6A, 0x4F)
RED      = RGBColor(0xD6, 0x29, 0x28)

def set_cell_bg(cell, hex_color: str):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)

def set_cell_border(cell, **kwargs):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top","bottom","left","right"):
        tag = OxmlElement(f"w:{edge}")
        tag.set(qn("w:val"), kwargs.get(edge, "none"))
        tag.set(qn("w:sz"), kwargs.get("sz", "4"))
        tag.set(qn("w:color"), kwargs.get("color", "auto"))
        tcBorders.append(tag)
    tcPr.append(tcBorders)

def para_format(para, space_before=0, space_after=6):
    para.paragraph_format.space_before = Pt(space_before)
    para.paragraph_format.space_after  = Pt(space_after)

def heading(doc, text, level=1, color=DARK):
    p = doc.add_heading(text, level=level)
    p.runs[0].font.color.rgb = color
    para_format(p, space_before=14 if level == 1 else 8, space_after=4)
    return p

def body(doc, text, bold=False, color=None, size=10.5):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = color
    para_format(p, space_after=4)
    return p

def bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    run.font.size = Pt(10.5)
    para_format(p, space_after=2)
    return p

def kv(doc, key, value, key_color=ORANGE):
    p = doc.add_paragraph()
    r1 = p.add_run(f"{key}: ")
    r1.bold = True
    r1.font.size = Pt(10.5)
    r1.font.color.rgb = key_color
    r2 = p.add_run(value)
    r2.font.size = Pt(10.5)
    para_format(p, space_after=2)
    return p

def divider(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "E85D04")
    pBdr.append(bottom)
    pPr.append(pBdr)
    para_format(p, space_before=4, space_after=8)

def make_table(doc, headers, rows, col_widths=None):
    t = doc.add_table(rows=1+len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    # header row
    hrow = t.rows[0]
    for i, h in enumerate(headers):
        cell = hrow.cells[i]
        cell.text = h
        set_cell_bg(cell, "1A1A2E")
        for run in cell.paragraphs[0].runs:
            run.font.color.rgb = WHITE
            run.bold = True
            run.font.size = Pt(9.5)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    # data rows
    for ri, row in enumerate(rows):
        tr = t.rows[ri+1]
        bg = "F9F9F9" if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate(row):
            cell = tr.cells[ci]
            cell.text = str(val)
            set_cell_bg(cell, bg)
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(9.5)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    return t

# ─────────────────────────────────────────────────────────────────────────────
# Build document
# ─────────────────────────────────────────────────────────────────────────────
doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# Default font
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10.5)

# ── Cover ─────────────────────────────────────────────────────────────────────
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("DR Compass — Pipeline Update Report")
r.font.size = Pt(22)
r.bold = True
r.font.color.rgb = DARK
para_format(p, space_before=12, space_after=4)

p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = p2.add_run("Changes from v1.0 (Random Forest Baseline) → v2.0 (XGBoost + Ablation-Validated Pipeline)")
r2.font.size = Pt(11)
r2.font.color.rgb = RGBColor(0x55,0x55,0x55)
para_format(p2, space_after=2)

p3 = doc.add_paragraph()
p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
r3 = p3.add_run(f"Generated {datetime.date.today().strftime('%B %d, %Y')}")
r3.font.size = Pt(10)
r3.font.color.rgb = RGBColor(0x88,0x88,0x88)
para_format(p3, space_after=16)

divider(doc)

# ── Executive Summary ─────────────────────────────────────────────────────────
heading(doc, "Executive Summary", level=1, color=ORANGE)
body(doc,
    "Starting from a Random Forest baseline (v1.0), four major workstreams were completed: "
    "(1) a Two-Stage Mixture Model to improve negative-price forecasting, "
    "(2) a multi-model benchmark study that identified XGBoost as the best forecaster, "
    "(3) a six-factor ablation study that validated each design choice with quantitative evidence, "
    "and (4) a unified model interface enabling config-driven model switching. "
    "The net result: calibrated price-forecast MAE improved from 13.59 → 12.58 €/MWh (−7.4%), "
    "total market cost reduction increased to €411M (11.6% price reduction), "
    "and all 8/8 economic mechanism checks continue to pass.",
    size=10.5)

divider(doc)

# ── Section 1: v1.0 Baseline ─────────────────────────────────────────────────
heading(doc, "1  Version 1.0 Baseline (Starting Point)", level=1)
body(doc,
    "The initial release implemented a fully automated Day-Ahead Demand Response (DR) pipeline "
    "for the German electricity market (OPSD DE 2018, 8 760 hourly observations). "
    "The forecasting model was a single Random Forest regressor trained on 32 engineered features, "
    "producing H=24 day-ahead price, load, and renewable-total forecasts. "
    "A separate RF binary classifier detected negative-price hours, and a block-bootstrap scenario "
    "generator (48 scenarios, block length 24 h) provided uncertainty quantification.", size=10.5)

body(doc, "Key v1.0 metrics:", bold=True)
make_table(doc,
    ["Metric", "Value"],
    [
        ["Calibrated MAE — price (H=24)", "13.59 €/MWh"],
        ["Neg-price recall (classifier)", "0.238"],
        ["Scenario P10–P90 coverage", "—  (not yet measured)"],
        ["Avg price reduction (DR)", "~5.7 €/MWh"],
        ["L4 mechanism checks passed", "8 / 8"],
    ],
    col_widths=[3.2, 2.0])
doc.add_paragraph()

divider(doc)

# ── Section 2: Two-Stage Mixture Model ───────────────────────────────────────
heading(doc, "2  Change 1 — Two-Stage Mixture Price Forecaster", level=1)

heading(doc, "2.1  Motivation", level=2)
body(doc,
    "The single RF regressor mixed ~98.8% positive-price and ~1.2% negative-price training hours. "
    "Because RF averages across all examples it systematically predicted positive values for "
    "negative-price hours, limiting DR guardrail accuracy.", size=10.5)

heading(doc, "2.2  Design", level=2)
body(doc,
    "A three-component mixture was introduced in src/forecasting.py "
    "(function fit_two_stage_price_forecast):", size=10.5)
bullet(doc, "Stage 1 — RF Binary Classifier → p_neg = P(price < 0 | features)")
bullet(doc, "Stage 2a — RF Regressor trained on price ≥ 0 hours → ŷ_pos")
bullet(doc, "Stage 2b — RF Regressor trained on price < 10 €/MWh hours → ŷ_neg  "
            "(boundary of 10 expands training set from ~27 to ~100–150 samples)")
bullet(doc, "Final forecast: ŷ = (1 − p_neg) × ŷ_pos + p_neg × ŷ_neg  (soft blend)")

heading(doc, "2.3  Result", level=2)
make_table(doc,
    ["Metric", "Single RF (v1.0)", "Two-Stage RF", "Change"],
    [
        ["Calibrated MAE (H=24)", "13.59 €/MWh", "13.46 €/MWh", "−0.13 (−1.0%)"],
        ["Neg-price recall", "0.238", "0.238", "unchanged"],
        ["Neg-price precision", "0.179", "0.179", "unchanged"],
        ["Classifier threshold τ*", "—", "0.09", "F1-optimal"],
        ["L4 checks passed", "8/8", "8/8", "maintained"],
    ],
    col_widths=[2.5, 1.5, 1.5, 1.8])
doc.add_paragraph()
body(doc,
    "The soft-blend architecture maintained overall accuracy while providing "
    "per-hour negative-price probability (p_neg) as an additional downstream signal. "
    "Recall did not improve because only 6 strictly-negative-price hours appear in the "
    "training fold — an irreducible data constraint at this dataset scale.", size=10.5)

divider(doc)

# ── Section 3: Forecast Benchmark ────────────────────────────────────────────
heading(doc, "3  Change 2 — Multi-Model Forecast Benchmark", level=1)

heading(doc, "3.1  Motivation", level=2)
body(doc,
    "It was unclear whether Random Forest was the optimal model architecture. "
    "A standalone benchmark study (scripts/run_07_forecast_benchmark.py, src/forecast_benchmark.py) "
    "was introduced to compare five model families on identical data splits, "
    "evaluating not only MAE but negative-price detection, peak-price stability, "
    "regime-specific MAE, pinball loss, and error distributions.", size=10.5)

heading(doc, "3.2  Models Compared", level=2)
bullet(doc, "Ridge — L2-regularised linear regression (baseline)")
bullet(doc, "Two-Stage RF — mixture model from Section 2")
bullet(doc, "XGBoost — gradient-boosted trees (500 estimators, early stopping)")
bullet(doc, "LSTM — 2-layer PyTorch LSTM, hidden=64, sequence length 48 h")
bullet(doc, "TCN — dilated causal Conv1d, receptive field ≥ 127 h")

heading(doc, "3.3  Results", level=2)
make_table(doc,
    ["Model", "MAE (€/MWh)", "RMSE", "R²", "Peak Recall", "q90 Pinball"],
    [
        ["Ridge",          "12.20", "15.24", "0.307", "0.091", "8.29"],
        ["XGBoost",        "9.95",  "14.90", "0.338", "0.303", "3.36  ✓ best"],
        ["Two-Stage RF",   "11.29", "15.47", "0.286", "0.364", "4.51"],
        ["TCN",            "10.71", "15.63", "0.287", "0.0",   "4.32"],
        ["LSTM",           "12.89", "18.78", "−0.03", "0.0",   "5.17"],
    ],
    col_widths=[1.6, 1.3, 1.1, 1.0, 1.2, 1.4])
doc.add_paragraph()
body(doc,
    "XGBoost achieved the lowest MAE (9.95 €/MWh on the evaluation set), best tail calibration "
    "(q90 pinball 3.36), and highest peak-price recall (0.303). "
    "LSTM underperformed (R² = −0.03) due to the limited 3-month dataset being too small for "
    "sequence models. TCN showed better data-efficiency than LSTM but still trailed XGBoost. "
    "A detailed Word report (outputs/benchmark/Forecast_Benchmark_Report.docx) was generated "
    "covering 11 chapters including regime-specific MAE and error distribution analysis.", size=10.5)

divider(doc)

# ── Section 4: Switch to XGBoost ─────────────────────────────────────────────
heading(doc, "4  Change 3 — Unified Model Interface & Switch to XGBoost", level=1)

heading(doc, "4.1  Unified Interface (src/forecaster.py)", level=2)
body(doc,
    "A model-agnostic dispatcher fit_price_forecast(df, model_name, split, rf_params) "
    "was created so any model can be selected via config.yaml without changing pipeline code. "
    "All five model backends return the same four-tuple: "
    "(pred_df, acc_df, neg_price_df, metrics).", size=10.5)
make_table(doc,
    ["config.yaml  forecast.model", "Model invoked"],
    [
        ['"xgboost"',      "XGBoost regressor with early stopping"],
        ['"two_stage_rf"', "Two-Stage RF mixture (Section 2)"],
        ['"ridge"',        "Ridge with StandardScaler"],
        ['"lstm"',         "PyTorch 2-layer LSTM"],
        ['"tcn"',          "PyTorch dilated TCN"],
    ],
    col_widths=[2.4, 3.4])
doc.add_paragraph()

heading(doc, "4.2  Production switch to XGBoost", level=2)
body(doc,
    "config.yaml was updated to forecast.model: \"xgboost\" and "
    "scripts/run_03_forecast_calibration_scenarios.py was updated to call "
    "fit_price_forecast() instead of the hard-coded two-stage function. "
    "All downstream scripts (run_04 through run_06) required no changes.", size=10.5)
make_table(doc,
    ["Metric", "Two-Stage RF", "XGBoost (current)", "Change"],
    [
        ["Calibrated MAE (H=24)",    "13.46 €/MWh", "12.58 €/MWh", "−0.88 (−6.5%)"],
        ["Calibrated RMSE",          "—",           "16.51 €/MWh", ""],
        ["R² (calibrated)",          "—",           "0.376",       ""],
        ["Neg-price recall",         "0.238",       "0.238",       "maintained"],
        ["Avg price reduction (DR)", "5.7 €/MWh",   "5.68 €/MWh",  "stable"],
        ["Total cost reduction",     "~€410M",      "€411M",       "stable"],
        ["L4 checks passed",         "8/8",         "8/8",         "maintained"],
    ],
    col_widths=[2.5, 1.5, 1.8, 1.8])
doc.add_paragraph()

divider(doc)

# ── Section 5: Ablation Study ─────────────────────────────────────────────────
heading(doc, "5  Change 4 — Ablation Study (src/ablation.py)", level=1)
body(doc,
    "A systematic ablation study (scripts/run_08_ablation_study.py) was conducted to validate "
    "each design decision with quantitative evidence. Six factors were tested against the "
    "XGBoost + block-bootstrap baseline.", size=10.5)

heading(doc, "5.1  Ablation Overview", level=2)
make_table(doc,
    ["#", "Ablation", "What changed", "Key result"],
    [
        ["1", "No Scenarios",       "Point forecast → DR (1 scenario)", "Cost reduction −27.5%  ★ most critical"],
        ["2", "IID Bootstrap",      "iid sampling instead of 24h blocks", "P10-P90 coverage −4.6 pp (72.7→69.4%)"],
        ["3", "No Holiday Features","Remove 7 holiday/dummy columns",    "MAE +2.7% (12.18→12.50)"],
        ["4", "No Sin/Cos Encoding","Replace hour_sin/cos with integer",  "MAE +2.9% (12.18→12.53)"],
        ["5", "No Elasticity",      "Remove rolling_elasticity feature",  "MAE +2.2% (12.18→12.44)"],
        ["6", "No Guardrail",       "Disable neg-price guardrail",        "21 neg-price DR hours triggered"],
    ],
    col_widths=[0.3, 1.5, 2.0, 2.8])
doc.add_paragraph()

heading(doc, "5.2  Detailed Findings", level=2)

body(doc, "Ablation 1 — Scenario Module (most impactful)", bold=True, color=ORANGE)
body(doc,
    "Replacing the 48-scenario fan with a single point-forecast scenario reduced average price "
    "reduction from 5.68 to 2.32 €/MWh (−59%) and total cost reduction from €411M to €298M "
    "(−27.5%). The scenario module is by far the most valuable architectural component: "
    "uncertainty quantification fundamentally changes which hours receive DR and by how much.", size=10.5)

body(doc, "Ablation 2 — Block Bootstrap vs IID Sampling", bold=True, color=ORANGE)
body(doc,
    "IID sampling (drawing residuals independently) reduced P10-P90 coverage from 72.7% to 69.4% "
    "(-4.6 percentage points) and high-price tail coverage from 53.0% to 54.5%. "
    "The 24-hour block length preserves intraday residual autocorrelation, which is essential "
    "for realistic scenario generation in an hourly electricity market.", size=10.5)

body(doc, "Ablations 3–5 — Feature Engineering (moderate impact)", bold=True, color=ORANGE)
make_table(doc,
    ["Feature group removed", "MAE Δ (€/MWh)", "MAE Δ (%)", "RMSE Δ (%)", "R² Δ (%)"],
    [
        ["Holiday / Christmas / neg-price dummy (7 features)", "+0.33", "+2.7%", "+4.2%", "−19.8%"],
        ["Hour sin/cos encoding (replaced with integer hour)", "+0.36", "+2.9%", "+1.0%",  "−4.5%"],
        ["Rolling elasticity (1 feature)",                    "+0.27", "+2.2%", "+1.1%",  "−5.1%"],
    ],
    col_widths=[2.9, 1.2, 1.0, 1.0, 1.0])
doc.add_paragraph()
body(doc,
    "All three feature groups contribute positively. Cyclic sin/cos encoding has the largest "
    "individual MAE impact (+2.9%), confirming that continuous hour representation captures "
    "the smooth intraday price cycle better than a discontinuous integer. "
    "The holiday features have the largest R² degradation (−19.8%), indicating they are "
    "critical for tail accuracy during anomalous demand periods. "
    "The elasticity feature, while the smallest contributor to MAE, "
    "represents the demand-side signal that links forecast accuracy to DR decision quality.", size=10.5)

body(doc, "Ablation 6 — Negative-Price Guardrail (mechanism validation)", bold=True, color=ORANGE)
body(doc,
    "With guardrail disabled (low_price_probability_threshold = 1.1), the optimizer triggered "
    "DR reductions in 21 negative-price hours — hours where reducing demand actually increases "
    "market costs (demand reduction raises price above zero). "
    "The guardrail correctly blocked all 21 events in the baseline (0 negative-price DR hours). "
    "This validates the L4 mechanism check 'negative_price_guardrail_blocks_dr' and "
    "demonstrates the economic necessity of the guardrail component.", size=10.5)

divider(doc)

# ── Section 6: Final Pipeline State ──────────────────────────────────────────
heading(doc, "6  Final Pipeline State (v2.0)", level=1)

heading(doc, "6.1  New and Modified Files", level=2)
make_table(doc,
    ["File", "Status", "Description"],
    [
        ["src/forecasting.py",                  "Modified", "Added fit_two_stage_price_forecast(), _fit_regime_rf()"],
        ["src/forecaster.py",                   "New",      "Unified fit_price_forecast() dispatcher for all 5 models"],
        ["src/forecast_benchmark.py",           "New",      "Full benchmark: LSTM, TCN, XGBoost, Ridge, Two-Stage RF"],
        ["src/ablation.py",                     "New",      "6 ablation functions, XGBoost retraining helpers"],
        ["src/data_preprocessing_agent.py",     "New",      "Claude API agent for CSV preprocessing"],
        ["scripts/run_03_*.py",                 "Modified", "Now calls fit_price_forecast() with config-driven model"],
        ["scripts/run_07_forecast_benchmark.py","New",      "Standalone benchmark runner"],
        ["scripts/run_08_ablation_study.py",    "New",      "Ablation runner + 4 comparison figures"],
        ["scripts/generate_benchmark_report.py","New",      "11-chapter Word report for benchmark"],
        ["scripts/generate_update_report.py",   "New",      "This report generator"],
        ["config.yaml",                         "Modified", "Added forecast.model: xgboost, forecast.two_stage block"],
        ["api/main.py",                         "New",      "FastAPI layer: /upload, /run_pipeline, /get_schedule"],
    ],
    col_widths=[2.6, 0.9, 3.1])
doc.add_paragraph()

heading(doc, "6.2  End-to-End Performance (v2.0 Baseline)", level=2)
make_table(doc,
    ["Level", "Metric", "Value"],
    [
        ["L1 Forecast", "XGBoost calibrated MAE (H=24)",    "12.58 €/MWh"],
        ["L1 Forecast", "XGBoost calibrated RMSE",          "16.51 €/MWh"],
        ["L1 Forecast", "R² (calibrated)",                   "0.376"],
        ["L1 Forecast", "Negative-price recall (classifier)","0.238  (threshold τ*=0.09)"],
        ["L2 Scenarios","P10–P90 coverage",                  "72.7%"],
        ["L2 Scenarios","P05–P95 coverage",                  "84.5%"],
        ["L2 Scenarios","Scenario bias",                     "+3.35 €/MWh"],
        ["L3 Decision", "Avg price reduction",               "5.68 €/MWh  (−11.6%)"],
        ["L3 Decision", "High-price hours eliminated",       "65 / 66  (98.5%)"],
        ["L3 Decision", "Total market cost reduction",       "€411M"],
        ["L3 Decision", "Realized DR profit",                "€183M"],
        ["L3 Decision", "Constraint violation rate",         "0%"],
        ["L4 Mechanism","All mechanism checks passed",       "8 / 8"],
    ],
    col_widths=[1.5, 3.0, 2.0])
doc.add_paragraph()

divider(doc)

# ── Section 7: Key Conclusions ───────────────────────────────────────────────
heading(doc, "7  Key Conclusions", level=1)

bullet(doc,
    "XGBoost outperforms Random Forest, Ridge, LSTM, and TCN on this dataset "
    "(MAE −7.4% vs RF, best pinball tail loss). "
    "LSTM/TCN require substantially more data than one year of hourly observations.")
bullet(doc,
    "The scenario module is the single most valuable component: removing it "
    "cuts cost reduction by 27.5% and average price reduction by 59%. "
    "Point-forecast-based DR optimization is materially inferior.")
bullet(doc,
    "Block bootstrap (24 h blocks) meaningfully outperforms IID sampling (+4.6 pp coverage). "
    "Preserving intraday residual structure matters for electricity markets.")
bullet(doc,
    "All three tested feature groups (holiday dummies, sin/cos hour encoding, price elasticity) "
    "contribute positively. Removing any single group costs 2–3% MAE.")
bullet(doc,
    "The negative-price guardrail is economically necessary: without it the optimizer "
    "mistakenly triggers DR in 21 negative-price hours, inverting the cost signal.")
bullet(doc,
    "The unified forecaster interface (src/forecaster.py) allows market-specific "
    "model selection via a single config flag, supporting the planned B2B SaaS roadmap.")

divider(doc)

# ── Footer note ───────────────────────────────────────────────────────────────
p_foot = doc.add_paragraph()
p_foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
r_foot = p_foot.add_run(
    "DR Compass · v2.0 · Data: OPSD Germany 2018 · "
    "Model: XGBoost (n_estimators=500, max_depth=6, lr=0.05)"
)
r_foot.font.size = Pt(8.5)
r_foot.font.color.rgb = RGBColor(0xAA, 0xAA, 0xAA)

# ── Save ─────────────────────────────────────────────────────────────────────
doc.save(OUT)
print(f"Saved: {OUT}  ({OUT.stat().st_size // 1024} KB)")
