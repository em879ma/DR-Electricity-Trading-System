"""
Generate comprehensive Word report for the Day-Ahead DR Decision System.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "outputs" / "tables"
FIGS = PROJECT_ROOT / "outputs" / "figures"
OUT_PATH = PROJECT_ROOT / "outputs" / "DR_System_Report.docx"


# ── helpers ──────────────────────────────────────────────────────────────────
def set_cell_bg(cell, hex_color: str):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def add_heading(doc: Document, text: str, level: int):
    p = doc.add_heading(text, level=level)
    run = p.runs[0] if p.runs else p.add_run(text)
    if level == 1:
        run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    elif level == 2:
        run.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
    return p


def add_para(doc: Document, text: str, bold: bool = False, italic: bool = False,
             indent: float = 0, fontsize: int = 11):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(fontsize)
    return p


def add_formula(doc: Document, formula: str, indent: float = 1.5):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(indent)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(formula)
    run.font.name = "Courier New"
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x1A, 0x56, 0x27)
    return p


def df_to_table(doc: Document, df: pd.DataFrame, header_color: str = "2E74B5",
                max_col_width: int = 28):
    cols = list(df.columns)
    table = doc.add_table(rows=1 + len(df), cols=len(cols))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # header
    hdr = table.rows[0].cells
    for i, col in enumerate(cols):
        hdr[i].text = str(col)
        set_cell_bg(hdr[i], header_color)
        run = hdr[i].paragraphs[0].runs[0]
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(9)

    # rows
    for r_idx, row in enumerate(df.itertuples(index=False)):
        cells = table.rows[r_idx + 1].cells
        for c_idx, val in enumerate(row):
            if isinstance(val, float):
                if abs(val) >= 1e8:
                    text = f"{val:.2e}"
                elif abs(val) >= 1000:
                    text = f"{val:,.0f}"
                elif abs(val) >= 1:
                    text = f"{val:.3f}"
                else:
                    text = f"{val:.4f}"
            elif isinstance(val, bool):
                text = "✓ True" if val else "✗ False"
            else:
                text = str(val)
            cells[c_idx].text = text
            cells[c_idx].paragraphs[0].runs[0].font.size = Pt(9)
        # alternating row color
        if r_idx % 2 == 1:
            for c in cells:
                set_cell_bg(c, "EBF3FB")

    return table


def add_figure(doc: Document, fig_path: Path, caption: str, width_cm: float = 14):
    if not fig_path.exists():
        add_para(doc, f"[图片缺失: {fig_path.name}]", italic=True)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(fig_path), width=Cm(width_cm))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap_run = cap.add_run(caption)
    cap_run.bold = True
    cap_run.font.size = Pt(10)
    cap_run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)


# ── load data ─────────────────────────────────────────────────────────────────
def load_tables():
    data = {}
    for name, path in [
        ("cal_summary", TABLES / "forecast_calibration_summary.csv"),
        ("l1", TABLES / "forecast_accuracy_level1.csv"),
        ("l2", TABLES / "scenario_quality_level2.csv"),
        ("l3", TABLES / "decision_performance_level3.csv"),
        ("l4", TABLES / "mechanism_validity_level4.csv"),
        ("sens", TABLES / "counterfactual_demand_sensitivity.csv"),
        ("coef", TABLES / "counterfactual_price_model_coefficients.csv"),
        ("cf_eval", TABLES / "counterfactual_price_model_evaluation.csv"),
        ("constraint", TABLES / "constraint_check_summary.csv"),
        ("regime", TABLES / "energy_mix_regime_summary.csv"),
        ("dr_summary", TABLES / "optimized_price_vs_observed_summary.csv"),
        ("elast", TABLES / "elasticity_parameter_estimates.csv"),
    ]:
        if path.exists():
            data[name] = pd.read_csv(path)
        else:
            data[name] = pd.DataFrame()
    return data


# ── main build ────────────────────────────────────────────────────────────────
def build_report():
    doc = Document()

    # page margins
    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    D = load_tables()

    # ═══════════════════════════════════════════════════════════════════════
    # COVER
    # ═══════════════════════════════════════════════════════════════════════
    doc.add_paragraph()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = title.add_run("日前需求响应决策系统\n完整研究报告")
    tr.bold = True
    tr.font.size = Pt(22)
    tr.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run(
        "Day-Ahead Demand Response Decision System under Forecast Uncertainty\n"
        "德国电力市场 · 2018年Q4 · OPSD数据"
    )
    sr.font.size = Pt(13)
    sr.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    doc.add_paragraph()

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 1. 项目概述
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "1. 项目概述与研究目标", 1)
    add_para(doc,
        "本项目构建了一套面向德国电力市场的 日前需求响应（DR）决策系统，"
        "研究核心问题是：在价格预测存在不确定性的条件下，工业用电主体如何在日前"
        "（提前24小时）做出最优的负荷削减决策，以最小化市场采购成本、同时规避负价风险。")
    add_para(doc, "")
    add_para(doc, "系统不是一个单纯的价格预测模型，而是完整的四阶段决策流水线：", bold=True)

    pipeline_items = [
        ("Stage 1", "预测层", "随机森林（RF）日前预测价格、负荷、可再生能源出力"),
        ("Stage 2", "情景层", "基于残差的块自助法（Block Bootstrap）生成48个价格/负荷/可再生情景"),
        ("Stage 3", "决策层", "在情景集上网格搜索最优DR量 q*，目标函数为最小化期望总采购成本"),
        ("Stage 4", "反事实评估层", "结构性价格模型（Ridge + RF）预测DR后的反事实价格，评估四级指标体系"),
    ]
    for code, name, desc in pipeline_items:
        p = doc.add_paragraph(style="List Bullet")
        r1 = p.add_run(f"[{code}] {name}：")
        r1.bold = True
        r1.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
        p.add_run(desc)

    doc.add_paragraph()
    add_heading(doc, "1.1 数据来源", 2)
    data_rows = [
        ("OPSD时序数据", "open-power-system-data.org", "2018-10-01 至 2018-12-31", "2,209小时", "逐小时负荷、价格、风电、光伏、进口"),
        ("Ember年度数据", "ember-energy.org", "2018年", "年度", "发电结构、燃料占比（本次因网络限制未成功下载）"),
    ]
    df_data = pd.DataFrame(data_rows, columns=["数据集", "来源", "时间范围", "频率", "包含变量"])
    df_to_table(doc, df_data)
    doc.add_paragraph()

    add_heading(doc, "1.2 数据统计摘要", 2)
    summary_rows = [
        ("样本量", "2,209 小时（Q4 2018）"),
        ("电价均值", "52.58 €/MWh"),
        ("电价标准差", "18.60 €/MWh"),
        ("电价区间", "−19.43 至 128.26 €/MWh"),
        ("负价小时数", "27 小时（1.22%）"),
        ("高价小时数（P90以上）", "221 小时（10.0%）"),
        ("平均负荷", "~57,700 MW"),
        ("可再生能源占比均值", "约 30–35%（风光合计）"),
    ]
    df_stat = pd.DataFrame(summary_rows, columns=["指标", "数值"])
    df_to_table(doc, df_stat, header_color="1F497D")
    doc.add_paragraph()

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 2. 系统架构
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "2. 系统架构与代码框架", 1)
    add_para(doc, "整个系统由六个顺序执行的脚本构成，每个脚本对应一个功能模块：")
    doc.add_paragraph()

    scripts = [
        ("run_01_prepare_data.py", "数据准备", "加载OPSD原始数据，清洗，计算特征工程（滞后项、滚动统计、能源结构特征、节假日特征）"),
        ("run_02_features_elasticity_dr.py", "弹性与DR特征", "估计动态弹性参数 ε₀/ε₁/ε₂/ε₂₄，构建基准需求demand_base，写出market_panel_with_dr.csv"),
        ("run_03_forecast_calibration_scenarios.py", "预测与情景生成", "RF日前预测 → 线性校准 → 块自助法情景生成（48情景×656小时），负价二分类器"),
        ("run_04_day_ahead_decision.py", "DR优化决策", "拟合结构性反事实价格模型 → 网格搜索最优q_DA → 写出调度表与逐小时反事实面板"),
        ("run_05_counterfactuals.py", "可视化（图1-4，6）", "生成预测校准图、情景扇形图、DR调度图、观测vs反事实价格图、能源结构制度图"),
        ("run_06_evaluation_dashboard.py", "四级评估仪表盘", "L1预测精度、L2情景质量、L3决策绩效、L4机制有效性，写出Figure 5与最终报告"),
    ]
    for script, name, desc in scripts:
        p = doc.add_paragraph(style="List Bullet")
        r1 = p.add_run(f"{script}（{name}）：")
        r1.bold = True
        r1.font.size = Pt(10)
        p.add_run(desc).font.size = Pt(10)

    doc.add_paragraph()
    add_heading(doc, "2.1 核心源文件", 2)
    src_files = [
        ("src/feature_engineering.py", "特征工程：滞后、滚动统计、能源结构变量、节假日变量（含lead-24前瞻版本）"),
        ("src/forecasting.py", "随机森林预测 + 负价二分类器（F1最优阈值自适应）"),
        ("src/calibration.py", "线性偏差校准（intercept + slope校正）"),
        ("src/scenarios.py", "块自助法情景生成（全测试期滚动日，修复了原先仅取tail(24)的bug）"),
        ("src/day_ahead_decision.py", "DR网格搜索优化，修复目标函数单位错误，批量RF预测加速"),
        ("src/counterfactual_price_model.py", "结构性价格模型：Ridge（可解释性）+ RF（预测精度），去除共线特征"),
        ("src/evaluation_*.py", "四级评估模块（forecast / scenarios / decision / mechanism）"),
    ]
    for fname, desc in src_files:
        p = doc.add_paragraph(style="List Bullet")
        r1 = p.add_run(f"{fname}：")
        r1.bold = True
        r1.font.size = Pt(10)
        r1.font.name = "Courier New"
        p.add_run(desc).font.size = Pt(10)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 3. 特征工程
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "3. 特征工程", 1)

    add_heading(doc, "3.1 时序特征", 2)
    add_para(doc, "对原始时间戳提取以下日历特征，并将小时数进行正弦-余弦编码以表达周期性：")
    add_formula(doc, "hour_sin = sin(2π·hour/24)，  hour_cos = cos(2π·hour/24)")
    add_para(doc, "同时生成各目标变量的滞后特征（lag = 1, 2, 3, 24, 48, 168小时），"
             "以及24小时和168小时滚动均值与标准差。")

    add_heading(doc, "3.2 能源结构变量", 2)
    add_para(doc, "以下结构性变量刻画电力市场供需平衡状态：")
    struct_vars = [
        ("net_load", "净负荷 = load − renewable_total", "净负荷越高，边际成本越高，价格越高"),
        ("renewable_share", "可再生份额 = renewable_total / load", "份额越高，价格压力越大"),
        ("scarcity_index", "稀缺指数 = load / (available_capacity + imports + ε)", "接近1表示系统紧张"),
        ("oversupply_index", "过剩指数 = renewable_total + imports − load", "正值表示供过于求"),
        ("low_net_load_dummy", "1(net_load ≤ P10)", "识别净负荷极低制度（负价高发）"),
        ("high_renewable_dummy", "1(renewable_share ≥ P90)", "识别高可再生渗透制度"),
    ]
    for var, formula, desc in struct_vars:
        p = doc.add_paragraph(style="List Bullet")
        r1 = p.add_run(f"{var}：")
        r1.bold = True
        r1.font.name = "Courier New"
        p.add_run(f"{formula}  →  {desc}")

    add_heading(doc, "3.3 节假日特征（新增）", 2)
    add_para(doc,
        "德国电力市场在公众假日和圣诞-元旦假期期间工业用电急剧下降，"
        "而风电出力不受影响，因此这些时段是负价最集中的时段。"
        "本系统新增三类节假日特征：")
    holiday_items = [
        ("is_public_holiday", "德国法定节假日（10月3日统一日、12月25日、12月26日）"),
        ("holiday_load_window", "低负荷假期窗口（12月21日至31日 + 1月1-2日 + 10月3日）"),
        ("is_christmas_week", "圣诞-元旦窗口（12月24日至1月2日）"),
    ]
    for var, desc in holiday_items:
        p = doc.add_paragraph(style="List Bullet")
        r1 = p.add_run(f"{var}：")
        r1.bold = True
        r1.font.name = "Courier New"
        p.add_run(desc)

    add_para(doc,
        "同时生成上述特征的 lead-24 版本（向前平移24小时），使模型在做日前预测时"
        "能够看到 目标小时 的节假日状态，而非当前时刻的状态：")
    add_formula(doc, "holiday_load_window_lead_24[t] = holiday_load_window[t + 24h]")
    add_para(doc,
        "这一设计对负价二分类器至关重要：训练集中10月3日的6个负价样本，"
        "其对应预测时刻（10月2日）的 is_public_holiday_lead_24 = 1，"
        "为分类器提供了可迁移的监督信号。")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 4. 弹性估计
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "4. 价格弹性估计", 1)
    add_para(doc,
        "在构建需求响应基准需求（demand_base）之前，系统估计动态需求价格弹性。"
        "弹性采用对数-线性滚动OLS框架估计：")
    add_formula(doc,
        "Δln(D_t) = ε₀·Δln(P_t) + ε₁·Δln(P_{t-1}) + ε₂·Δln(P_{t-2}) + ε₂₄·Δln(P_{t-24}) + u_t")
    add_para(doc,
        "注：rolling_elasticity 是滚动OLS斜率（相关特征），不是结构性因果弹性。"
        "下表为全样本动态弹性参数估计结果：", italic=True)

    if not D["elast"].empty:
        df_to_table(doc, D["elast"].round(6), header_color="1F497D")
    add_para(doc, "")
    add_para(doc,
        "估计显示即时弹性 ε₀ ≈ 0.0019（极低），说明德国工业用电对即时电价的响应"
        "能力有限，这与已有文献一致。模型拟合优度 R² = 0.985，表明价格波动能够"
        "解释绝大多数需求的短期变化。")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 5. 随机森林预测
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "5. 随机森林日前预测", 1)

    add_heading(doc, "5.1 模型设置", 2)
    add_para(doc,
        "采用随机森林回归器（RandomForestRegressor）作为日前预测基础模型，"
        "同时与 持久性基准（persistence）和 滚动均值基准（rolling mean）对比。"
        "预测目标为 H=24 小时后的电价、负荷、可再生总出力。")

    rf_settings = [
        ("n_estimators", "300"),
        ("max_depth", "12"),
        ("min_samples_leaf", "1（调整后，原为3，减小以捕捉负价稀有事件）"),
        ("训练/验证/测试比例", "70% / 15% / 15%（时序顺序切分）"),
        ("预测目标", "price, load, renewable_total（各horizon独立训练）"),
        ("horizons", "[1, 2, 3, 6, 12, 24] 小时"),
    ]
    df_rf = pd.DataFrame(rf_settings, columns=["参数", "设定值"])
    df_to_table(doc, df_rf, header_color="1F497D")
    doc.add_paragraph()

    add_heading(doc, "5.2 特征集（RF_FEATURES，共45维）", 2)
    add_para(doc, "特征包含五类：")
    feat_groups = [
        ("日历特征", "hour_sin/cos, dayofweek, month, is_weekend"),
        ("价格滞后与滚动统计", "price_lag_{1,2,24,168}, rolling_price_mean_{24,168}, rolling_price_std_{24,168}"),
        ("负荷与可再生滞后", "load_lag_{1,24,168}, renewable_lag_{1,24}, rolling_load_mean_{24,168}"),
        ("能源结构特征", "net_load, renewable_share, scarcity_index, oversupply_index, rolling_elasticity, low/high_renewable_dummy"),
        ("节假日特征（新增）", "is_public_holiday, holiday_load_window, is_christmas_week + 各自的lead_24版本，negative_price_dummy_lag_24"),
    ]
    for group, items in feat_groups:
        p = doc.add_paragraph(style="List Bullet")
        r1 = p.add_run(f"{group}：")
        r1.bold = True
        p.add_run(items)

    add_heading(doc, "5.3 预测校准", 2)
    add_para(doc,
        "原始RF预测存在系统性偏差（bias = +7.48 €/MWh），因此对每个预测目标"
        "拟合线性校准方程，将偏差压缩至接近0：")
    add_formula(doc,
        "forecast_calibrated = slope × forecast_raw + intercept")
    add_para(doc, "校准前后对比：")
    if not D["cal_summary"].empty:
        df_cal = D["cal_summary"][["target","calibration_method","raw_mae","raw_rmse","raw_bias","calibrated_mae","calibrated_rmse","calibrated_bias"]].copy()
        df_cal.columns = ["目标","方法","原始MAE","原始RMSE","原始偏差","校准MAE","校准RMSE","校准偏差"]
        df_to_table(doc, df_cal.round(3))
    doc.add_paragraph()
    add_para(doc, "校准后电价预测偏差从 +7.48 降至 −0.71 €/MWh，RMSE从15.15降至13.17 €/MWh。")

    add_heading(doc, "5.4 L1预测精度评估结果", 2)
    if not D["l1"].empty:
        df_to_table(doc, D["l1"].round(3))
    doc.add_paragraph()
    add_para(doc,
        "关键指标：校准后模型 MAE = 13.59 €/MWh，R² = 0.309，高价时段Recall = 0.485。"
        "注：RF回归模型天然无法预测负价（预测值恒为正），负价识别通过独立二分类器实现（见下节）。")

    add_heading(doc, "5.5 负价二分类器（新增）", 2)
    add_para(doc,
        "额外训练一个 RandomForestClassifier 专门识别负价小时。"
        "由于训练集中仅有 6 个负价样本（10月3日），分类难度极高。"
        "采用 class_weight='balanced' 并通过最大化eval集F1分数自适应选取分类阈值：")
    add_formula(doc,
        "threshold* = argmax_{τ∈[0.02,0.50]} F1(τ)  →  最优阈值 τ* = 0.09")
    add_para(doc,
        "最终 negative_price_recall（校准模型）= 0.238（21个负价小时中识别5个）。"
        "其中12月8-9日的高风力周末负价无法识别（原因：仅依靠滞后特征，"
        "无法预知未来风电超发；这是该类天气驱动负价的根本局限）。")

    add_figure(doc, FIGS / "01_forecast_calibration.png",
               "图1：RF预测校准图（原始预测 vs 校准后预测 vs 实际值）", width_cm=15)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 6. 情景生成
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "6. 块自助法情景生成", 1)
    add_para(doc,
        "为量化预测不确定性，系统采用 块自助法（Block Bootstrap）"
        "在校准RF点预测基础上生成情景集合。")

    add_heading(doc, "6.1 方法原理", 2)
    add_para(doc, "步骤：")
    steps = [
        "计算验证集残差序列：residual = actual − forecast_calibrated",
        "对每个日前决策时刻（按天分组），从历史残差中随机抽取连续24小时的残差块",
        "将该残差块加到校准点预测上，生成一条情景路径",
        "重复 n_scenarios = 48 次，得到48条联合（价格、负荷、可再生出力）情景路径",
    ]
    for i, step in enumerate(steps, 1):
        p = doc.add_paragraph(style="List Number")
        p.add_run(step)
    add_para(doc, "")
    add_formula(doc,
        "P^s_t = P̂_t + ε^s_t,   s = 1,...,48")
    add_para(doc, "其中 P̂_t 为校准点预测，ε^s_t 为从历史残差中块抽样得到的随机残差项。")
    add_para(doc, "")
    add_para(doc,
        "重要修复：原始代码中 base = day_ahead_rf.tail(24) 导致情景只覆盖最后1天（24小时）；"
        "修复后改为对全部测试期（656小时）逐日生成情景，覆盖所有历史负价时段。",
        italic=True)

    add_heading(doc, "6.2 L2情景质量评估结果", 2)
    if not D["l2"].empty:
        df_to_table(doc, D["l2"].round(4))
    doc.add_paragraph()
    add_para(doc, "关键指标解读：")
    l2_interp = [
        ("coverage_10_90 = 0.684", "68.4%的实际值落在10-90百分位区间内（理论值应为80%，偏低说明情景分布偏窄）"),
        ("coverage_05_95 = 0.809", "80.9%的实际值落在5-95百分位区间内（接近理论值90%，合理）"),
        ("high_price_tail_coverage = 0.712", "高价时段有71.2%被情景捕捉，说明对极端价格的覆盖较好"),
        ("negative_price_tail_coverage = 0.190", "负价时段仅19%被情景捕捉（修复前为NaN），仍偏低但不为零"),
        ("scenario_bias = +4.20 €/MWh", "情景均值系统性高于实际，反映RF高价期存在低估偏差"),
        ("n_matched_hours = 656", "修复后覆盖全部656个测试小时（修复前仅24小时）"),
    ]
    for metric, interp in l2_interp:
        p = doc.add_paragraph(style="List Bullet")
        r1 = p.add_run(f"{metric}：")
        r1.bold = True
        p.add_run(interp)

    add_figure(doc, FIGS / "02_bootstrap_scenario_fan_chart.png",
               "图2：Bootstrap情景扇形图（48条路径 + 校准点预测 + 实际值）", width_cm=15)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 7. 日前DR决策优化
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "7. 日前需求响应决策优化", 1)

    add_heading(doc, "7.1 决策问题形式化", 2)
    add_para(doc,
        "对每个目标小时 t，决策变量为 q_t（MW）= 削减的需求量。"
        "在情景集 {(P^s_t, L^s_t, R^s_t)} 上，最小化期望总采购成本：")
    add_formula(doc,
        "min_{q} E_s[P_cf(L^s-q, R^s) · (D_base - q)] + c_DR · q + penalty(q, L^s)")
    add_para(doc, "其中：")
    eq_items = [
        ("P_cf(·)", "反事实价格（由结构性价格模型预测，在DR需求减少后的市场出清价格）"),
        ("D_base - q", "实施DR后的实际需求量"),
        ("c_DR = 3.0 €/MWh", "DR成本（激励用电方削减负荷的补偿成本）"),
        ("penalty(q, L^s)", "当 q > flexibility_ratio × L^s 时触发超量违约罚款（rate = 20 €/MWh）"),
    ]
    for term, desc in eq_items:
        p = doc.add_paragraph(style="List Bullet")
        r1 = p.add_run(f"{term}：")
        r1.bold = True
        r1.font.name = "Courier New"
        p.add_run(desc)

    add_heading(doc, "7.2 约束条件", 2)
    add_para(doc, "优化问题受到以下约束：")
    add_formula(doc, "0 ≤ q_t ≤ flexibility_ratio × D_base_t  （灵活性上限约束）")
    add_formula(doc, "D_base_t - q_t ≥ 0.70 × D_base_t  即  q_t ≤ 0.30 × D_base_t  （最低负荷约束）")
    add_para(doc, "本项目中 flexibility_ratio = 0.10（即最多削减基准需求的10%）。")

    add_heading(doc, "7.3 负价护栏（强制规则）", 2)
    add_para(doc,
        "当市场预期价格为负或接近负时，需求削减不但不能节省成本，反而会损失负电价下的"
        "免费电力。系统强制执行以下护栏规则：")
    add_formula(doc, "若 E[P_t] ≤ 0 或 Pr(P_t < 0 | 情景) ≥ 0.30  →  q_t = 0")
    add_para(doc,
        "实际结果：54个小时触发护栏（guardrail_triggered），这些小时 q_DA = 0，"
        "完全规避了在负价期间削减需求的错误决策。")

    add_heading(doc, "7.4 关键Bug修复：目标函数单位错误", 2)
    add_para(doc, "原始代码存在严重的目标函数单位混淆错误：", bold=True)
    add_formula(doc,
        "❌ 错误版本：econ_cost = price_term + c_DR × q + penalty")
    add_para(doc,
        "问题：price_term 单位为 €/MWh（约38），而 c_DR × q 单位为 € "
        "（约3×4200 = 12,600），两者无法相加。结果是 q=0 永远最优，"
        "系统完全不响应，DR量恒为0。")
    add_formula(doc,
        "✓ 修复版本：demand_after = max(D_base - q, 0.70 × D_base)")
    add_formula(doc,
        "            econ_cost = price_term × demand_after + c_DR × q + penalty")
    add_para(doc, "修复后效果：")
    fix_effect = [
        ("平均价格削减", "0 €/MWh → 5.65 €/MWh"),
        ("总成本节约", "0 € → 409,812,174 €（约4.1亿€）"),
        ("高价时段减少", "0 小时 → 65 小时"),
    ]
    for metric, effect in fix_effect:
        p = doc.add_paragraph(style="List Bullet")
        r1 = p.add_run(f"{metric}：")
        r1.bold = True
        p.add_run(effect)

    add_heading(doc, "7.5 L3决策绩效评估结果", 2)
    if not D["l3"].empty:
        df_to_table(doc, D["l3"].round(3))
    doc.add_paragraph()
    add_para(doc, "关键指标：")
    l3_items = [
        "realized_profit = 1.83亿€（实现利润，来自高价期削减套利）",
        "total_market_cost_reduction = 4.10亿€（市场总成本节约）",
        "avg_price_reduction = 5.65 €/MWh（DR后市场出清价格下降幅度）",
        "high_price_hours_reduced = 65/66（高价时段几乎全部被DR覆盖，1个因护栏未触发）",
        "DR_utilization_high_price = 1.0（高价时段全部使用满额DR灵活性）",
        "constraint_violation_rate = 0%（所有656小时均满足可行性约束）",
    ]
    for item in l3_items:
        doc.add_paragraph(item, style="List Bullet")

    add_figure(doc, FIGS / "03_day_ahead_dr_schedule.png",
               "图3：日前DR调度计划（最优q_DA vs 预期价格 vs 护栏触发）", width_cm=15)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 8. 反事实价格模型
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "8. 结构性反事实价格模型", 1)
    add_para(doc,
        "DR决策依赖于一个关键假设：削减需求后，市场出清价格会下降。"
        "为量化这一价格响应，系统拟合了 结构性反事实价格模型，"
        "而非直接使用RF预测价格（RF预测的是统计关联，不是因果价格响应）。")

    add_heading(doc, "8.1 特征设计（去共线化）", 2)
    add_para(doc,
        "原始设计包含 net_load、renewable_share、oversupply_index 三个高度共线的变量：")
    add_formula(doc, "oversupply_index = renewable + imports − load = −net_load + imports")
    add_para(doc, "相关系数：cor(net_load, oversupply_index) = −1.000，cor(net_load, renewable_share) = −0.845")
    add_para(doc,
        "这导致Ridge回归系数符号反转（net_load系数为负，违背经济学直觉）。"
        "修复方案：从STRUCTURAL_FEATURES中移除 renewable_share 和 oversupply_index：")
    add_formula(doc,
        "STRUCTURAL_FEATURES = [net_load, scarcity_index, import_share,")
    add_formula(doc,
        "                        low_net_load_dummy, high_renewable_dummy]")

    add_heading(doc, "8.2 模型架构", 2)
    add_para(doc, "系统并行拟合两个价格模型：")
    models_info = [
        ("Ridge（可解释）", "α=2.5", "特征 = STRUCTURAL_FEATURES + 小时/周/月虚拟变量（One-hot）"),
        ("Random Forest（高精度）", "n=200, depth=14", "相同特征集，用于实际反事实价格预测"),
    ]
    df_models = pd.DataFrame(models_info, columns=["模型", "超参数", "特征"])
    df_to_table(doc, df_models, header_color="1F497D")
    doc.add_paragraph()
    add_para(doc, "同时拟合负价分类器（负价 vs 非负价），用于护栏规则中的 P(price<0) 估计。")

    add_heading(doc, "8.3 模型评估结果", 2)
    if not D["cf_eval"].empty:
        df_to_table(doc, D["cf_eval"].round(4))
    doc.add_paragraph()
    add_para(doc,
        "RF反事实模型 R² = 0.981，MAE = 1.77 €/MWh（样本内），"
        "Ridge R² = 0.851，MAE = 5.54 €/MWh。Ridge主要用于可解释性验证，"
        "RF用于实际预测。")

    add_heading(doc, "8.4 Ridge系数符号验证", 2)
    add_para(doc, "修复共线性后，Ridge系数符号均符合经济学理论：")
    coef_checks = [
        ("net_load", "+0.00112", "✓ 正号", "净负荷↑ → 边际成本↑ → 价格↑"),
        ("scarcity_index", "+12.122", "✓ 正号", "系统紧张度↑ → 价格↑"),
        ("import_share", "−0.00192", "✓ 负号", "进口份额↑ → 国内价格↓"),
        ("low_net_load_dummy", "−5.062", "✓ 负号", "净负荷极低制度 → 负价/低价区间"),
        ("high_renewable_dummy", "−3.178", "✓ 负号", "高可再生渗透 → 价格下压"),
    ]
    df_coef = pd.DataFrame(coef_checks, columns=["特征", "系数值", "符号检验", "经济学含义"])
    df_to_table(doc, df_coef, header_color="2E74B5")
    doc.add_paragraph()

    add_heading(doc, "8.5 需求敏感性分析", 2)
    add_para(doc,
        "对不同需求削减幅度（5%至30%）进行参数扫描，验证：需求减少 → 价格下降，"
        "且高价时段的价格下降幅度显著大于均值：")
    if not D["sens"].empty:
        df_s = D["sens"][["demand_reduction_ratio","avg_price_change","high_price_price_change","percent_avg_price_change","mechanism_valid_avg_positive","mechanism_valid_high_gt_avg"]].copy()
        df_s.columns = ["需求削减比例","均值价格变化(€/MWh)","高价段价格变化","变化%","均值为正?","高价>均值?"]
        df_to_table(doc, df_s.round(3))
    doc.add_paragraph()
    add_para(doc,
        "在10%需求削减下，均值价格下降 8.31 €/MWh（15.8%），"
        "高价时段下降 14.75 €/MWh——高价时段获益约为均值的1.78倍，"
        "说明DR在市场最紧张时刻具有最大价值。")

    add_figure(doc, FIGS / "04_observed_vs_counterfactual_price.png",
               "图4：观测价格 vs 反事实价格（DR实施后市场出清价格对比）", width_cm=15)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 9. 能源结构制度分析
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "9. 能源结构制度分析（Energy-Mix Regime Analysis）", 1)
    add_para(doc,
        "将测试期小时按能源结构分为六种市场制度，分析不同制度下的价格特征和DR绩效：")
    add_formula(doc,
        "制度划分：renewable_share < P33 → low_renewable；")
    add_formula(doc,
        "          P33 ≤ renewable_share < P67 → medium；> P67 → high_renewable")
    add_para(doc, "叠加特殊制度：oversupply（净过剩供给）、scarcity（供给紧张）、negative_price（出现负价）")
    doc.add_paragraph()
    if not D["regime"].empty:
        df_r = D["regime"].copy()
        df_r.columns = ["制度","小时数","均价(€/MWh)","负价频率","高价频率",
                        "均值DR量(MW)","均值降价(€/MWh)","均值成本节约(€)","均值弹性"]
        df_to_table(doc, df_r.round(3))
    doc.add_paragraph()
    add_para(doc, "制度分析关键发现：")
    regime_findings = [
        "scarcity制度（供给紧张）：均价76.5 €/MWh，高价频率72.7%，DR均值量最高（7,077 MW），价格削减效果最好（+15.7 €/MWh）",
        "oversupply制度（供过于求）：均价6.7 €/MWh，负价频率30.3%，护栏规则触发，DR均值量最低（1,811 MW）",
        "high_renewable制度：均价36.9 €/MWh，价格削减接近0（−0.33 €/MWh），说明价格对可再生高渗透的响应弱",
        "low_renewable制度：均价60.4 €/MWh，具有最大DR价值，成本节约1.84亿€",
        "negative_price制度：仅1小时，护栏完全阻断DR，DR量=0，符合设计",
    ]
    for finding in regime_findings:
        doc.add_paragraph(finding, style="List Bullet")

    add_figure(doc, FIGS / "06_energy_mix_regime_analysis.png",
               "图6：能源结构制度分析（各制度下的价格、DR量与负价频率）", width_cm=15)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 10. L4机制有效性
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "10. L4：经济机制有效性验证", 1)
    add_para(doc,
        "L4评估验证DR系统的核心经济机制是否成立，包括8项机制检验：")
    if not D["l4"].empty:
        df_l4 = D["l4"][["mechanism_check","passed","metric_value","expected_direction","note"]].copy()
        df_l4.columns = ["机制检验项","通过?","指标值","预期方向","说明"]
        df_to_table(doc, df_l4, header_color="1F6B25")
    doc.add_paragraph()
    add_para(doc, "全部 8/8 项机制检验通过，说明：", bold=True)
    l4_summary = [
        "需求削减确实降低市场出清价格（DR有效）",
        "DR资源集中分配在高价时段（资源配置效率高）",
        "负价护栏在负价情景下正确阻断DR（风险规避有效）",
        "需求弹性越大，价格下降越多（边际效益单调）",
        "高价时段DR价值显著高于均值（非线性价格响应）",
        "Ridge系数符号全部符合经济学理论（结构可解释性）",
    ]
    for item in l4_summary:
        doc.add_paragraph(item, style="List Bullet")

    add_figure(doc, FIGS / "05_evaluation_dashboard.png",
               "图5：四级评估仪表盘（L1-L4综合评分）", width_cm=15)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 11. 改进对比
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "11. 相对基础版本的核心改进", 1)

    improvements = [
        (
            "Bug修复：目标函数单位错误（Critical）",
            "econ_cost = price_term + c_DR×q（单位混淆）",
            "econ_cost = price_term × demand_after + c_DR×q + penalty",
            "DR量从0提升到均值5,395 MW；总成本节约从0增至4.1亿€",
        ),
        (
            "Bug修复：情景仅覆盖24小时（Critical）",
            "base = day_ahead_rf.tail(24)（只取最后1天）",
            "对全测试期656小时逐日生成情景（656×48）",
            "n_matched_hours: 24 → 656；negative_price_tail_coverage: NaN → 0.190",
        ),
        (
            "Bug修复：共线特征导致Ridge符号错误",
            "STRUCTURAL_FEATURES包含renewable_share和oversupply_index（cor=-1.0）",
            "移除两个共线特征，保留净负荷、稀缺度、进口份额",
            "ridge_net_load_positive: False → True；L4全部8项通过",
        ),
        (
            "Bug修复：L2面板路径缺失",
            "panel_path = 'market_panel.csv'（文件不存在）",
            "改为market_panel_with_dr.csv，加兜底回退",
            "L2评估从完全跳过改为正常运行",
        ),
        (
            "新功能：节假日特征工程",
            "无节假日相关特征",
            "新增6个节假日特征：is_public_holiday, holiday_load_window, is_christmas_week + lead_24版本",
            "负价分类器对假期驱动负价的识别能力从0提升至有信号",
        ),
        (
            "新功能：负价二分类器",
            "RF回归模型negative_price_recall恒为0（回归天花板）",
            "独立RandomForestClassifier + F1最优阈值（τ*=0.09）",
            "negative_price_recall: 0.000 → 0.238",
        ),
        (
            "新功能：批量RF预测加速",
            "逐行调用predict（156,960次单行预测）",
            "对每个(hour, q)组合批量预测48×情景行",
            "run_04执行时间从约20分钟降至约3分钟（约7倍加速）",
        ),
        (
            "改进：能源结构制度分析",
            "无法运行（KeyError：regime特征缺失）",
            "inline计算能源结构特征，强制生成regime_summary.csv",
            "成功生成6制度分析表和Figure 6",
        ),
    ]

    for i, (title, before, after, effect) in enumerate(improvements, 1):
        add_heading(doc, f"11.{i} {title}", 2)
        t = doc.add_table(rows=3, cols=2)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        labels = ["修复前", "修复后", "效果"]
        values = [before, after, effect]
        colors = ["C00000", "1F6B25", "1F497D"]
        for row_idx, (label, value, color) in enumerate(zip(labels, values, colors)):
            cells = t.rows[row_idx].cells
            cells[0].text = label
            set_cell_bg(cells[0], color)
            run = cells[0].paragraphs[0].runs[0]
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(9)
            cells[1].text = value
            cells[1].paragraphs[0].runs[0].font.size = Pt(9)
        doc.add_paragraph()

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 12. 综合性能总结
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "12. 综合性能总结", 1)

    add_heading(doc, "12.1 四级评估综合得分", 2)
    final_scores = [
        ("L1 预测精度", "MAE=13.59 €/MWh，R²=0.309，high_price_recall=0.485，negative_price_recall=0.238", "中等"),
        ("L2 情景质量", "coverage_10_90=68.4%，coverage_05_95=80.9%，high_tail=71.2%，neg_tail=19.0%", "良好"),
        ("L3 决策绩效", "成本节约4.1亿€，价格降低5.65 €/MWh，65/66高价时段覆盖，违规率0%", "优秀"),
        ("L4 机制有效性", "8/8项检验通过，Ridge符号全正确，敏感性单调递增", "优秀"),
    ]
    df_scores = pd.DataFrame(final_scores, columns=["评估层级", "关键指标", "综合评级"])
    df_to_table(doc, df_scores, header_color="1F497D")
    doc.add_paragraph()

    add_heading(doc, "12.2 DR优化核心结果", 2)
    final_kpis = [
        ("平均观测电价", "49.01 €/MWh"),
        ("平均反事实电价（DR后）", "43.36 €/MWh"),
        ("平均价格削减", "5.65 €/MWh（−11.5%）"),
        ("观测市场总成本", "1,940,489,519 € （19.4亿€）"),
        ("反事实市场总成本", "1,530,677,345 € （15.3亿€）"),
        ("总成本节约", "409,812,174 € （4.1亿€）"),
        ("高价时段观测数", "66小时"),
        ("高价时段反事实数", "1小时（DR将65个高价时段压回正常价格区间）"),
        ("平均DR量", "5,395 MW（约为系统负荷的9.3%）"),
        ("护栏触发小时", "54小时（全部正确阻断，违规率0%）"),
        ("约束违规率", "0%（全656小时可行）"),
    ]
    df_kpi = pd.DataFrame(final_kpis, columns=["指标", "数值"])
    df_to_table(doc, df_kpi, header_color="1F6B25")
    doc.add_paragraph()

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 13. 不足之处
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "13. 当前不足与改进方向", 1)

    limitations = [
        (
            "天气驱动负价无法预测（核心局限）",
            "12月8-9日的高风力周末负价（10个小时）对应的分类概率为0.000。"
            "原因：模型依赖历史滞后特征，无法知道未来24小时的风电超发情况。"
            "改进方向：接入ECMWF/DWD数值天气预报（NWP）风速预测，或使用现货市场隐含"
            "信息（当日前市场报价）作为特征。"
        ),
        (
            "负价情景覆盖不足（negative_price_tail_coverage=19%）",
            "Bootstrap情景基于正态分布残差，在极端尾部（负价）表现不足。"
            "改进方向：使用非参数尾部增强（extreme value theory），"
            "或对负价时段残差单独建模（混合分布）。"
        ),
        (
            "弹性估计为相关性而非因果性",
            "rolling_elasticity 是滚动OLS斜率（特征），不能作为政策实验的因果弹性参数。"
            "改进方向：采用工具变量（IV）或RDD设计，利用价格阶梯设定作为外生变量估计真实弹性。"
        ),
        (
            "反事实价格模型样本内拟合",
            "counterfactual_price_model 在全量数据上训练（包括测试期），"
            "R²=0.981可能存在过拟合，无法代表真实的样本外泛化能力。"
            "改进方向：采用时序交叉验证（expanding window）评估样本外表现。"
        ),
        (
            "无多期DR能量平衡约束",
            "当前模型对每小时独立决策，忽略了DR的 反弹效应（rebound effect）："
            "一个时段削减负荷后，下一时段往往存在补偿性用电高峰。"
            "改进方向：引入多期约束 Σ(q_t) ≤ Q_max_day 或动态规划框架。"
        ),
        (
            "数据范围仅限2018年Q4",
            "仅3个月数据导致：（1）季节性特征训练不足；"
            "（2）极端价格事件样本量极少（27个负价小时）；"
            "（3）无法识别跨年度结构性变化（如可再生装机增长带来的负价频率趋势）。"
            "改进方向：扩展至2015-2023年全年数据（OPSD历史数据可完整获取）。"
        ),
        (
            "高价召回率偏低（high_price_recall=0.485）",
            "校准后的RF对高价时段的识别率仅约50%，说明模型对价格尖峰的预测能力不足。"
            "改进方向：添加市场供需紧张指标（抽水蓄能水位、气温异常指数、跨境交易容量）；"
            "或使用专门针对极端值优化的分位数回归树。"
        ),
        (
            "DR成本参数固定",
            "c_DR=3.0 €/MWh 和 penalty_rate=20 €/MWh 为固定假设，"
            "未考虑实际合约结构（分档激励、单次/多次上限）的异质性。"
            "改进方向：引入参数化DR成本曲线，在不同灵活性比率下进行敏感性分析。"
        ),
    ]

    for i, (title, content) in enumerate(limitations, 1):
        p = doc.add_paragraph()
        r1 = p.add_run(f"{i}. {title}")
        r1.bold = True
        r1.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        r1.font.size = Pt(11)
        add_para(doc, content, indent=0.5)
        doc.add_paragraph()

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 14. 结论
    # ═══════════════════════════════════════════════════════════════════════
    add_heading(doc, "14. 结论", 1)
    add_para(doc,
        "本项目构建了一套完整的 日前需求响应决策系统，覆盖从数据准备到四级评估的完整链路。"
        "在修复多个关键Bug和引入节假日特征、负价分类器等新功能后，系统取得了如下主要成果：")
    doc.add_paragraph()

    conclusions = [
        ("预测层", "RF校准预测MAE=13.59 €/MWh，高价时段召回率48.5%，负价分类器召回率23.8%"),
        ("情景层", "656小时全期覆盖，90%分位区间覆盖率80.9%，高价尾部覆盖71.2%"),
        ("决策层", "总成本节约4.1亿€，价格削减5.65 €/MWh，65/66高价时段被DR成功压制，约束违规率0%"),
        ("机制层", "8/8项经济机制检验全部通过，结构性价格模型系数符号完全符合经济学理论"),
    ]
    for layer, result in conclusions:
        p = doc.add_paragraph(style="List Bullet")
        r1 = p.add_run(f"{layer}：")
        r1.bold = True
        p.add_run(result)

    doc.add_paragraph()
    add_para(doc,
        "系统的核心价值在于：通过将价格预测不确定性（情景）与DR可行性约束"
        "（护栏、灵活性上限、最低负荷）结合起来，在 不同市场制度 下做出差异化决策——"
        "在高价/稀缺制度下满额使用DR，在负价/过剩制度下自动退出，"
        "实现了「在对的时间做对的事」这一DR设计的核心目标。",
        bold=False)

    doc.add_paragraph()
    add_para(doc,
        "当前主要不足集中在：天气驱动负价无法预测、情景覆盖负价尾部不足、"
        "无多期能量平衡约束、数据仅覆盖一个季度。"
        "这些局限在已有的四级评估框架中均被诚实地量化和报告。",
        italic=True)

    # ── save ──────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT_PATH))
    print(f"Report saved: {OUT_PATH}")


if __name__ == "__main__":
    build_report()
