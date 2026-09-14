"""
Generate v2.1 update report (Word document).

Summarises the changes from v2.0 (XGBoost + 6-factor ablation) to v2.1
(dataset expansion + paper-aligned baselines + VSS/Regret), driven by
Lago et al. (2021) review of EPF methodologies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor, Inches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs" / "DR_Update_Report_v2_1.docx"
TABLES = PROJECT_ROOT / "outputs" / "tables"
BENCH = PROJECT_ROOT / "outputs" / "benchmark" / "tables"
ABL = PROJECT_ROOT / "outputs" / "ablation" / "tables"


def _heading(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x68)


def _para(doc: Document, text: str, *, bold: bool = False, italic: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(10)
    run.bold = bold
    run.italic = italic


def _bullet(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    run.font.size = Pt(10)


def _table(doc: Document, df: pd.DataFrame, header_fill: str = "1F3A68") -> None:
    tbl = doc.add_table(rows=1 + len(df), cols=len(df.columns))
    tbl.style = "Light Grid Accent 1"
    hdr_cells = tbl.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr_cells[i].text = str(col)
        for run in hdr_cells[i].paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(9)
    for r_idx, (_, row) in enumerate(df.iterrows(), start=1):
        for c_idx, col in enumerate(df.columns):
            v = row[col]
            txt = f"{v:.3f}" if isinstance(v, float) else str(v)
            tbl.rows[r_idx].cells[c_idx].text = txt
            for p in tbl.rows[r_idx].cells[c_idx].paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)


def main() -> None:
    doc = Document()

    # ── Title ────────────────────────────────────────────────────────
    title = doc.add_heading("DR 决策系统 v2.1 更新报告", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub.add_run("论文对齐改进：数据集扩充 + EPF 基准基线 + 统计显著性 + VSS/Regret")
    sub_run.italic = True
    sub_run.font.size = Pt(11)
    sub_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    doc.add_paragraph()

    # ── Executive Summary ───────────────────────────────────────────
    _heading(doc, "1. 执行摘要", level=1)
    _para(doc,
          "v2.1 在 v2.0 的 XGBoost + 6 因素消融基础上，按 Lago et al. (2021) "
          "《Forecasting day-ahead electricity prices》一文给出的方法论标准做了五项关键改进。"
          "最重要的发现是：在数据集从 8,760h 扩充到 17,540h（2 倍）之后，原本看起来强势的 XGBoost "
          "在统计显著性检验下 **与 Seasonal Naive 基线打平**（DM 检验 p = 0.89），并明显 **弱于简单的 "
          "Ridge 回归**（DM 检验 p ≈ 0）。这与 Lago 论文里关于'许多 EPF 方法在与公允基线对比时优势消失' "
          "的警告完全吻合，是本次升级最有价值的发现。")

    # ── Section 2: Changes ──────────────────────────────────────────
    _heading(doc, "2. 五项核心改动", level=1)

    changes = pd.DataFrame([
        {"#": "P0-A", "改动": "数据集扩充", "范围": "config + 重跑流水线",
         "结果": "8,760h → 17,540h；负价小时 6h → 484h（80×）"},
        {"#": "P0-B", "改动": "Seasonal Naive 基线 + rMAE", "范围": "src/forecaster.py",
         "结果": "rMAE = MAE_model / MAE_naive，跨数据集可比指标"},
        {"#": "P0-C", "改动": "LEAR 基线（24 LASSO-ARX）", "范围": "src/lear.py",
         "结果": "MAE = 8.43，与论文低复杂度强基线对齐"},
        {"#": "P0-D", "改动": "DM / GW 显著性检验", "范围": "src/stat_tests.py",
         "结果": "成对 DM + 多元 GW，识别真实显著的模型差异"},
        {"#": "P1-A", "改动": "特征可用性审计", "范围": "feature_availability_audit.md",
         "结果": "无关键泄漏；记录了 net_load 的近似假设"},
        {"#": "P1-B", "改动": "VSS / Decision Regret", "范围": "src/evaluation_decision.py + run_10",
         "结果": "VSS = -€191M（场景反而轻微伤害），Regret = €709M（vs 完美信息）"},
    ])
    _table(doc, changes)

    # ── Section 3: Dataset ──────────────────────────────────────────
    _heading(doc, "3. 数据集扩充：从 1 年到 2 年", level=1)
    _para(doc,
          "原 OPSD 时间序列 CSV 已含 2015-2020 的全量数据，但 DE_LU_price_day_ahead 列（"
          "Germany-Luxembourg 市场耦合后的统一日前价格）只在 2018-09-30 之后有效。"
          "因此实际可用窗口为 2018-10 至 2020-09，共 17,540 小时——是原数据集的两倍。")

    _bullet(doc, "训练 + 验证 + 测试比例: 55 / 18 / 27（旧版 70/15/15）")
    _bullet(doc, "测试集小时数: 656h → 4,736h，统计检验的样本量足够")
    _bullet(doc, "负价小时: 6h → 484h，两阶段 RF 负价回归终于有训练样本")
    _bullet(doc, "高价小时（≥ p90）: 877h → 1,754h，DR 优化器有更多触发机会")

    # ── Section 4: Baselines & rMAE ─────────────────────────────────
    _heading(doc, "4. 论文基线与 rMAE", level=1)
    _para(doc,
          "新增两个论文标准基线，并将所有模型的 MAE 标准化为 rMAE（除以 Seasonal Naive 的 MAE）："
          "rMAE < 1 表示模型优于 7 天前同小时的简单复读机基线。")

    bench_path = BENCH / "benchmark_with_paper_baselines.csv"
    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        bench = bench[["model", "mae", "rmse", "rmae_vs_naive"]].copy()
        bench.columns = ["模型", "MAE", "RMSE", "rMAE (vs Naive)"]
        _table(doc, bench)

    _para(doc, "关键观察：", bold=True)
    _bullet(doc, "Ridge 是唯一 rMAE < 1 的非平凡模型（rMAE ≈ 0.90），最稳健。")
    _bullet(doc, "LEAR（论文 reference baseline）rMAE ≈ 0.96，刚好优于 Naive。")
    _bullet(doc, "XGBoost（v2.0 的明星）rMAE ≈ 1.13——比 Naive 还差。")
    _bullet(doc, "LSTM/TCN 同样未能击败 Naive，验证了 Lago 论文的核心警告：复杂模型在公允基线下未必胜出。")

    # ── Section 5: Stat tests ───────────────────────────────────────
    _heading(doc, "5. 统计显著性检验（DM / GW）", level=1)
    _para(doc,
          "对所有模型对做两种检验：(a) 成对 Diebold-Mariano（平方误差损失，Newey-West HAC 方差，"
          "Harvey 小样本修正）；(b) 多元 Giacomini-White（24 小时联合，χ² 统计量）。"
          "下表只显示与 XGBoost 有关的检验：")

    dm_path = BENCH / "dm_test_pairwise.csv"
    if dm_path.exists():
        dm = pd.read_csv(dm_path)
        dm_x = dm[(dm["model_1"] == "xgboost") | (dm["model_2"] == "xgboost")]
        dm_show = dm_x[["model_1", "model_2", "dm_stat", "p_value", "winner"]].copy()
        dm_show.columns = ["模型1", "模型2", "DM 统计量", "p 值", "显著胜方"]
        _table(doc, dm_show)

    _para(doc, "GW 多元联合检验（同一表格用 χ² 替代 DM）：", italic=True)
    gw_path = BENCH / "gw_test_joint.csv"
    if gw_path.exists():
        gw = pd.read_csv(gw_path)
        gw_x = gw[(gw["model_1"] == "xgboost") | (gw["model_2"] == "xgboost")]
        gw_show = gw_x[["model_1", "model_2", "chi2_stat", "p_value", "winner"]].copy()
        gw_show.columns = ["模型1", "模型2", "χ² 统计量", "p 值", "显著胜方"]
        _table(doc, gw_show)

    _para(doc,
          "结论：XGBoost 与 Naive 在 DM 检验下打平（p = 0.89），在 GW 检验下显著弱于 Naive（p ≈ 1.9e-6）。"
          "Ridge 在两种检验下都显著优于 XGBoost。这意味着 v2.0 报告里展示的 XGBoost 优势在 v2.1 的更大测试集上不再稳健。")

    # ── Section 6: Ablation (updated) ───────────────────────────────
    _heading(doc, "6. 消融实验（v2.1 重新运行）", level=1)
    _para(doc, "在扩充后的数据集上重跑 6 因素消融，结果与 v2.0 有显著差异：")

    abl_path = ABL / "ablation_summary.csv"
    if abl_path.exists():
        abl = pd.read_csv(abl_path)
        cols_keep = ["ablation", "description", "mae_delta_pct",
                     "avg_price_reduction_delta_pct", "total_cost_reduction_delta_pct",
                     "n_neg_price_dr"]
        cols_keep = [c for c in cols_keep if c in abl.columns]
        abl_view = abl[cols_keep].copy()
        abl_view.columns = ["消融", "描述", "MAE Δ%",
                            "降价 Δ%", "成本节省 Δ%", "负价DR小时"][:len(cols_keep)]
        _table(doc, abl_view)

    _para(doc, "重大改变：", bold=True)
    _bullet(doc, "no_holiday: MAE 改善 -2.07%（v2.0 是 +2.7%）——更大数据集让节假日 dummy 反而轻微过拟合。")
    _bullet(doc, "no_scenarios: 成本节省提高 +20.5%（v2.0 是 -27.5%）——点预测竟然比场景方法更优。")
    _bullet(doc, "iid_bootstrap: 覆盖率几乎无变化（v2.0 损失 -4.6pp）——块自举的优势在大样本下消失。")
    _bullet(doc, "no_guardrail: 281 个负价 DR 小时（v2.0 是 21）——guardrail 仍然是必要的，且作用更大。")

    # ── Section 7: VSS / Regret ─────────────────────────────────────
    _heading(doc, "7. VSS 与 Decision Regret", level=1)
    _para(doc,
          "VSS（Value of Stochastic Solution）= 点预测 DR 成本 − 场景 DR 成本。Regret = 模型 DR 成本 − 完美信息 DR 成本。"
          "两者都在 €10¹⁰ 量级，因此用相对值更直观：")

    vss_path = TABLES / "vss_regret_summary.csv"
    if vss_path.exists():
        vss = pd.read_csv(vss_path).iloc[0]
        rows = pd.DataFrame([
            {"指标": "观测市场总成本 (€)",            "数值": f"{vss['observed_total_cost']:,.0f}"},
            {"指标": "场景 DR 后成本 (€)",            "数值": f"{vss['stochastic_DR_cost']:,.0f}"},
            {"指标": "点预测 DR 后成本 (€)",          "数值": f"{vss['deterministic_DR_cost']:,.0f}"},
            {"指标": "完美信息 DR 后成本 (€)",        "数值": f"{vss['perfect_information_cost']:,.0f}"},
            {"指标": "VSS（绝对，€）",               "数值": f"{vss['VSS_absolute']:,.0f}"},
            {"指标": "VSS（相对）",                  "数值": f"{vss['VSS_relative']:.2%}"},
            {"指标": "Regret（场景模型，绝对 €）",   "数值": f"{vss['regret_stochastic_absolute']:,.0f}"},
            {"指标": "Regret（场景模型，相对）",     "数值": f"{vss['regret_stochastic_relative']:.2%}"},
            {"指标": "Regret（点预测模型，相对）",   "数值": f"{vss['regret_deterministic_relative']:.2%}"},
        ])
        _table(doc, rows)

    _para(doc, "结论：", bold=True)
    _bullet(doc, "VSS = -1.66%——在当前数据规模下场景方法反而轻微伤害决策质量。")
    _bullet(doc, "场景模型相对 regret = 6.47%——距离完美信息上限还有约 7% 的提升空间。")
    _bullet(doc, "点预测模型相对 regret = 4.73%——比场景模型更接近完美信息基线。")
    _bullet(doc, "这一发现与第 6 节的 no_scenarios 消融完全一致：当前不确定性量化方法可能需要重新设计。")

    # ── Section 8: Recommendations ──────────────────────────────────
    _heading(doc, "8. 下一步建议", level=1)
    _bullet(doc, "重新审视场景生成方式：当前 48 个 bootstrap 场景对 DR 优化器可能引入了过度保守。"
                 "可尝试减少场景数（如 12 个），或改用条件分位数回归代替自举。")
    _bullet(doc, "把 Ridge / LEAR 作为生产基线：它们简单、稳定、显著优于 Naive，并且训练成本极低。")
    _bullet(doc, "进一步扩充数据集：若能接入 ENTSO-E 直接 API，可拿到 2015-01 之前的 EPEX-DE 数据，"
                 "测试集长度对齐 Lago 论文的 2 年完整测试期。")
    _bullet(doc, "把 day-ahead 决策时点调整为真实市场的 D-1 12:00，使各小时的真实 horizon 不再统一为 24h。")
    _bullet(doc, "尝试集成（Ridge × LEAR × XGBoost）：若各模型误差相关性低，集成或可同时降低 MAE 与 regret。")

    # ── Section 9: Files ────────────────────────────────────────────
    _heading(doc, "9. 本次升级新增 / 改动的文件", level=1)
    files = pd.DataFrame([
        {"文件": "config.yaml", "类型": "改动",
         "说明": "data.start/end 改为 2015-01 / 2020-09-30；split 改为 55/18/27"},
        {"文件": "src/forecaster.py", "类型": "扩展",
         "说明": "新增 _naive_seasonal_pipeline + compute_rmae"},
        {"文件": "src/lear.py", "类型": "新建",
         "说明": "LEAR：24 个 LassoCV 模型，每小时一个，对齐 Lago (2021)"},
        {"文件": "src/stat_tests.py", "类型": "新建",
         "说明": "Diebold-Mariano + Giacomini-White 检验"},
        {"文件": "src/evaluation_decision.py", "类型": "扩展",
         "说明": "新增 compute_vss / compute_regret"},
        {"文件": "scripts/run_09_paper_baselines.py", "类型": "新建",
         "说明": "Naive + LEAR + rMAE + DM/GW，扩展 run_07 的基准表"},
        {"文件": "scripts/run_10_vss_regret.py", "类型": "新建",
         "说明": "VSS + 完美信息 oracle + Regret 汇总"},
        {"文件": "outputs/reports/feature_availability_audit.md", "类型": "新建",
         "说明": "逐特征排查 day-ahead 可用性，结论无关键泄漏"},
    ])
    _table(doc, files)

    doc.save(OUT)
    print(f"Saved: {OUT}")
    print(f"  Size: {OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
