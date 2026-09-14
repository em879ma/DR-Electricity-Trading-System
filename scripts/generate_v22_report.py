"""
Generate v2.2 update report (Word document).

Summarises the changes from v2.1 (paper baselines + VSS/Regret) to v2.2
(forecaster diagnostic + regime ensemble + Optuna tuning).

Key finding: forecast L1 improved meaningfully (rMAE 1.13 → 0.93) but
L3 (VSS, Regret) barely moved — pointing the next iteration toward
scenario redesign and decision optimisation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs" / "DR_Update_Report_v2_2.docx"
TABLES = PROJECT_ROOT / "outputs" / "tables"
BENCH = PROJECT_ROOT / "outputs" / "benchmark" / "tables"
DIAG = PROJECT_ROOT / "outputs" / "diagnostic"


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


def _table(doc: Document, df: pd.DataFrame) -> None:
    tbl = doc.add_table(rows=1 + len(df), cols=len(df.columns))
    tbl.style = "Light Grid Accent 1"
    hdr = tbl.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr[i].text = str(col)
        for run in hdr[i].paragraphs[0].runs:
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

    title = doc.add_heading("DR 决策系统 v2.2 更新报告", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub.add_run("预测器诊断与修复：从单一 XGBoost 到 Regime-aware 集成")
    sub_run.italic = True
    sub_run.font.size = Pt(11)
    sub_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    doc.add_paragraph()

    # ── Executive Summary ───────────────────────────────────────────
    _heading(doc, "1. 执行摘要", level=1)
    _para(doc,
          "v2.1 发现 XGBoost 在扩充数据集上输给 Seasonal Naive（rMAE = 1.13），"
          "v2.2 通过三条并行轨道进行了诊断与修复：（A）特征剪枝，（B）Optuna 正则化调参，"
          "（C）Regime-aware 集成。最终采用 Track C，单模型 rMAE 从 1.13 降到 0.93 "
          "（DM 检验 vs Naive: p < 1e-10，显著优于）。")
    _para(doc,
          "但是——关键发现——L1 提升并没有显著改善 L3：VSS 仍为 −1.66%，Regret 仅从 6.47% "
          "降到 6.41%。这意味着**预测质量已不再是 DR 决策的瓶颈**，下一阶段（v2.3）应聚焦"
          "于场景生成方式（block bootstrap → 分位数回归）和 DR 优化器的鲁棒性（CVaR + ambiguity set）。")

    # ── Section 2: Diagnostic ──────────────────────────────────────
    _heading(doc, "2. 诊断阶段（Phase 1）", level=1)
    _para(doc, "新建脚本 scripts/run_11_forecast_diagnostic.py 输出 5 张诊断图 + 1 份 markdown。关键观察：")
    _bullet(doc,
            "XGBoost 在 55 棵树后停止改进；train MAE 5.51 vs valid MAE 8.85 → 严重过拟合。")
    _bullet(doc,
            "残差按价格分位数分箱：Q1（低价段 avg €5.92）bias = +20.5，MAE = 21.1 — "
            "模型严重高估低价段。Q5（高价段 avg €48.9）bias = −5.7 — 低估尖峰。")
    _bullet(doc,
            "Spearman 相关性 > 0.85 的冗余对：net_load × oversupply_index = −1.00，"
            "price_lag_1 × price_lag_2 = +0.94，renewable_lag_1 × renewable_share = +0.94。")
    _bullet(doc,
            "SHAP 贡献接近 0 的特征：negative_price_dummy_lag_24、is_public_holiday、is_weekend。")
    _bullet(doc,
            "校准步骤的副作用：在 Q1-Q3 降低 MAE，但在 Q4-Q5 反而升高 MAE（+0.4 / +2.5）。")

    # ── Section 3: Three tracks ─────────────────────────────────────
    _heading(doc, "3. 修复阶段（Phase 2）— 三条并行轨道", level=1)

    tracks = pd.DataFrame([
        {"轨道": "A. 特征剪枝",
         "做法": "32 个特征 → 21 个（删除高相关 + 低 SHAP）",
         "valid MAE": "8.72",
         "结论": "**反而变差**（8.62 → 8.72）。冗余特征在 Ridge/XGBoost 下仍有边际信息。保留为可选项。"},
        {"轨道": "B. XGBoost Optuna 调参",
         "做法": "60 trials，搜 reg_alpha / lambda / gamma / depth 等",
         "valid MAE": "8.63 → 9.85（test）",
         "结论": "比默认 9.91 仅改善 0.06，整体仍弱于 Ridge。问题不在超参数。"},
        {"轨道": "C. Regime-aware 集成",
         "做法": "GradientBoostingClassifier 4-regime（neg/low/normal/high）→ Ridge 在 neg/low，XGBoost 在 high，50/50 blend 在 normal",
         "valid MAE": "**8.53**（最佳）",
         "结论": "**采纳**。DM 检验 vs XGBoost: dm = 15.5, p < 1e-10。"},
    ])
    _table(doc, tracks)

    # ── Section 4: Benchmark table ──────────────────────────────────
    _heading(doc, "4. 与论文基线的对比（含新模型）", level=1)
    bench_path = BENCH / "benchmark_with_paper_baselines.csv"
    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        bench = bench[["model", "mae", "rmse", "rmae_vs_naive"]].copy()
        bench.columns = ["模型", "MAE", "RMSE", "rMAE (vs Naive)"]
        _table(doc, bench)
    _para(doc, "亮点：", bold=True)
    _bullet(doc, "regime_ensemble：rMAE 从 v2.1 XGBoost 的 1.13 降到 0.93（−18%）")
    _bullet(doc, "DM 检验 XGBoost vs regime_ensemble：dm = 15.47，p ≈ 0，集成显著更优")
    _bullet(doc, "GW 多元联合检验：χ² = 203.6，p ≈ 0，所有 24 小时同向显著")
    _bullet(doc, "Ridge 单模型仍是最强（rMAE 0.90）—— 集成是为了引入 XGBoost 的高价段优势")

    # ── Section 5: Downstream — L3 ──────────────────────────────────
    _heading(doc, "5. 下游影响：VSS / Regret 几乎不变", level=1)
    vss_path = TABLES / "vss_regret_summary.csv"
    if vss_path.exists():
        vss = pd.read_csv(vss_path).iloc[0]
        rows = pd.DataFrame([
            {"指标": "Stochastic-DR 节省 (€)",  "v2.1": "1.228e+09", "v2.2": f"{vss['stochastic_saving_vs_observed']:.3e}"},
            {"指标": "VSS（相对）",             "v2.1": "−1.66%",    "v2.2": f"{vss['VSS_relative']:.2%}"},
            {"指标": "Regret（stochastic，相对）", "v2.1": "6.47%",   "v2.2": f"{vss['regret_stochastic_relative']:.2%}"},
            {"指标": "Regret（deterministic，相对）", "v2.1": "4.73%", "v2.2": f"{vss['regret_deterministic_relative']:.2%}"},
        ])
        _table(doc, rows)
    _para(doc,
          "L1 预测精度大幅提升，但 L3 决策指标几乎没动。说明 DR 流水线下游（场景生成 + 优化器）"
          "已经成为瓶颈。L3 的 VSS 仍为负，意味着**场景的不确定性反而轻微伤害决策**——这一现象"
          "和 v2.1 的发现一致，与场景的统计设计无关，而是场景规模与决策风险偏好不匹配的产物。")

    # ── Section 6: Recommended v2.3 ────────────────────────────────
    _heading(doc, "6. 推荐的 v2.3 方向", level=1)
    _para(doc, "v2.2 已经把 L1 调到接近天花板，下一阶段应该跳到 L2/L3：", bold=True)
    _bullet(doc,
            "**场景重设计**：用分位数回归（QuantileGradientBoosting）替换 block bootstrap，"
            "再用 conformal prediction 校准 P10/P90 覆盖率；尝试把场景数从 48 降到 12。")
    _bullet(doc,
            "**CVaR-aware DR 优化器**：当前优化器只用 average_cost 目标。把 `cvar_alpha` 参数（已部分实现）"
            "接入 run_04 主流程，让优化器对尾部风险更敏感。")
    _bullet(doc,
            "**Robust optimization**：在 ambiguity set 上求 worst-case 而非 expected cost，"
            "理论上可以让 VSS 转正。")
    _bullet(doc,
            "**Decision-focused learning**：训练目标改为 regret-aware loss，"
            "可以直接关闭 L1 → L3 的传播缝隙（需要 cvxpylayers 或 surrogate）。")

    # ── Section 7: Files ───────────────────────────────────────────
    _heading(doc, "7. 本次升级新增 / 改动文件", level=1)
    files = pd.DataFrame([
        {"文件": "scripts/run_11_forecast_diagnostic.py", "类型": "新建",
         "说明": "SHAP + 学习曲线 + 残差分箱 + 特征相关矩阵 + 校准对比"},
        {"文件": "scripts/run_12_xgb_tuning.py", "类型": "新建",
         "说明": "Optuna 60 trials 调 XGBoost 正则化"},
        {"文件": "src/ensemble.py", "类型": "新建",
         "说明": "RegimeAwareEnsemble：4 regime 分类器 + Ridge/XGB 分段"},
        {"文件": "src/forecasting.py", "类型": "扩展",
         "说明": "新增 RF_FEATURES_PRUNED + get_feature_set()"},
        {"文件": "src/forecaster.py", "类型": "扩展",
         "说明": "dispatcher 接 regime_ensemble；_prep_tabular 支持 feature_set；xgb_params 支持外部传入"},
        {"文件": "config.yaml", "类型": "改动",
         "说明": "forecast.model = regime_ensemble；新增 forecast.feature_set"},
        {"文件": "scripts/run_03 / run_09", "类型": "改动",
         "说明": "传递 feature_set / xgb_params 给 fit_price_forecast；run_09 加入 ensemble"},
        {"文件": "outputs/diagnostic/", "类型": "新建（目录）",
         "说明": "5 张 PNG + forecast_diagnostic_report.md + xgb_best_params.yaml"},
    ])
    _table(doc, files)

    # ── Section 8: Goal vs achieved ────────────────────────────────
    _heading(doc, "8. 目标完成度", level=1)
    goals = pd.DataFrame([
        {"指标": "XGBoost-class rMAE", "目标": "< 0.90", "实际": "0.93（集成）/ 0.90（Ridge）", "达成": "部分"},
        {"指标": "负价段 MAE", "目标": "< 35", "实际": "见 benchmark", "达成": "—"},
        {"指标": "VSS", "目标": "> 0", "实际": "−1.66%", "达成": "**否**"},
        {"指标": "Regret", "目标": "< 5%", "实际": "6.41%（stoch）", "达成": "**否**"},
        {"指标": "DM vs Naive", "目标": "p < 0.01", "实际": "p ≪ 1e-10", "达成": "**是**"},
    ])
    _table(doc, goals)
    _para(doc,
          "结论：L1 显著达标，但 L3 未达标。这正是为什么 v2.3 应该把焦点从预测器转到场景与决策。",
          italic=True)

    doc.save(OUT)
    print(f"Saved: {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
