"""Generate the Problem 3 code-and-results delivery document."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.问题一.config import REPOSITORY_ROOT

from .config import Problem3Parameters


DEFAULT_DELIVERY_REPORT = REPOSITORY_ROOT / "docs" / "问题三_代码说明与结果交付.md"


def _candidate_table(summary: pd.DataFrame) -> list[str]:
    lines = [
        "| 外层总层数 | 总厚度/mm | 总质量/kg | 新增质量/kg | "
        "总成本/元 | $t_{15}$/min | 重量惩罚/s | 主评分/min | 安全评分/min |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            f"{int(row['outer_layer_count'])} | "
            f"{float(row['outer_thickness_mm']):.1f} | "
            f"{float(row['garment_mass_kg']):.6f} | "
            f"{float(row['added_garment_mass_kg']):.6f} | "
            f"{float(row['total_cost_yuan']):.6f} | "
            f"{float(row['t15_min']):.3f} | "
            f"{float(row['weight_penalty_s']):.3f} | "
            f"{float(row['standing_time_score_min']):.3f} | "
            f"{float(row['safe_score_min']):.3f} |"
        )
    return lines


def _validation_table(validation: pd.DataFrame) -> list[str]:
    lines = [
        "| 验证项 | 基准值/s | 复核值/s | 绝对差/s |",
        "|---|---:|---:|---:|",
    ]
    for _, row in validation.iterrows():
        lines.append(
            "| "
            f"{row['check']} | "
            f"{float(row['problem3_value_s']):.6f} | "
            f"{float(row['reference_value_s']):.6f} | "
            f"{float(row['absolute_difference_s']):.6g} |"
        )
    return lines


def _thermal_sensitivity_table(data: pd.DataFrame) -> list[str]:
    lines = [
        "| 参数 | 情景 | 核心降至 $35^\\circ\\mathrm C$ /min | 相对变化/% |",
        "|---|---:|---:|---:|",
    ]
    for _, row in data.iterrows():
        lines.append(
            "| "
            f"{row['parameter']} | {row['scenario']} | "
            f"{float(row['t_core_35_min']):.3f} | "
            f"{float(row['relative_change_pct']):+.3f} |"
        )
    return lines


def _selected_scenario_table(
    data: pd.DataFrame,
    scenario_column: str,
    scenario_format: str,
) -> list[str]:
    lines = [
        "| 情景参数 | 最优外层总层数 |",
        "|---:|---:|",
    ]
    selected = data[data["selected"]].copy()
    for _, row in selected.iterrows():
        lines.append(
            f"| {format(float(row[scenario_column]), scenario_format)} | "
            f"{int(row['outer_layer_count'])} |"
        )
    return lines


def write_delivery_report(
    config: Problem3Parameters,
    summary: pd.DataFrame,
    strict_summary: dict[str, float | str],
    validation: pd.DataFrame,
    sensitivity: dict[str, pd.DataFrame],
    output_dir: Path,
    destination: str | Path = DEFAULT_DELIVERY_REPORT,
) -> Path:
    """Write a deterministic Markdown handoff describing code and final results."""

    optimum_index = int(summary["standing_time_score_s"].idxmax())
    optimum = summary.loc[optimum_index]
    baseline = summary.iloc[0]
    report_path = Path(destination)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# 问题三代码说明与结果交付",
        "",
        "## 1. 交付结论",
        "",
        "本程序已在问题一六状态动态皮肤血流模型上完成外层涂层整数优化。"
        "100 kg 表示不包括人体自重的总外部承重上限，因此硬约束检查整套防护服质量；"
        "时间惩罚以问题一为基准，只作用于新增质量。",
        "",
        "$$",
        "J(N)=t_{15}(N)-20[m(N)-m(1)].",
        "$$",
        "",
        f"最优方案是外层总计 **{int(optimum['outer_layer_count'])} 层**，"
        f"即在原防护服上新增 **{int(optimum['added_outer_layer_count'])} 层**，"
        f"外层总厚度 **{float(optimum['outer_thickness_mm']):.1f} mm**。",
        "",
        f"- 问题一基准时间：{float(baseline['t15_min']):.3f} min；",
        f"- 最优方案纯传热时间：{float(optimum['t15_min']):.3f} min；",
        f"- 新增重量惩罚：{float(optimum['weight_penalty_s']):.3f} s；",
        f"- 题意主口径站立时间评分：{float(optimum['standing_time_score_min']):.3f} min；",
        f"- 相对问题一净增益：{float(optimum['net_gain_vs_problem1_min']):.3f} min；",
        f"- 核心 $35^\\circ\\mathrm C$ 安全修正评分：{float(optimum['safe_score_min']):.3f} min。",
        "",
        "## 2. 代码结构",
        "",
        "- `src/问题三/config.py`：决策、成本、质量、预算和承重约束；",
        "- `src/问题三/model.py`：多外层温度节点和人体—防护服 ODE；",
        "- `src/问题三/simulation.py`：皮肤 $15^\\circ\\mathrm C$、$10^\\circ\\mathrm C$ "
        "与核心 $35^\\circ\\mathrm C$ 事件、能量审计和候选方案评价；",
        "- `src/问题三/sensitivity.py`：热学、初温、重量、代谢、压迫和成本敏感性；",
        "- `src/问题三/reporting.py`：自动生成本交付文档；",
        "- `src/问题三/workflow.py`：统一运行枚举、严格复核、敏感性、CSV 导出和报告生成。",
        "",
        "## 3. 运行方式",
        "",
        "```powershell",
        "D:\\python\\python.exe -m src.问题三",
        "```",
        "",
        "单元测试：",
        "",
        "```powershell",
        "D:\\python\\python.exe -m unittest discover -s temp\\tests -p test_problem3.py -v",
        "```",
        "",
        "## 4. 候选方案结果",
        "",
        *_candidate_table(summary),
        "",
        "主目标按题意选择外层总层数；核心温度安全修正只作为附加风险指标。"
        "两种口径都选择四层，但长时间的 $t_{15}$ 不应直接称为医学安全时间。",
        "",
        "## 5. 约束与资源余量",
        "",
        f"- 防护服总质量：{float(optimum['garment_mass_kg']):.6f} kg；",
        f"- 100 kg 外部承重余量：{float(optimum['external_load_margin_kg']):.6f} kg；",
        f"- 总材料成本：{float(optimum['total_cost_yuan']):.6f} 元；",
        f"- 成本上限：{config.maximum_total_cost_yuan:.6f} 元；",
        f"- 直接材料预算余量：{config.maximum_total_cost_yuan - float(optimum['total_cost_yuan']):.6f} 元。",
        "",
        "## 6. 数值验证",
        "",
        *_validation_table(validation),
        "",
        f"最优方案严格容差复核的相对能量守恒误差为 "
        f"{float(strict_summary['relative_energy_balance_error']):.3e}。",
        "",
        "## 7. 敏感性结果",
        "",
        "### 7.1 主要热学参数",
        "",
        *_thermal_sensitivity_table(sensitivity["thermal"]),
        "",
        "### 7.2 重量惩罚系数",
        "",
        *_selected_scenario_table(
            sensitivity["weight_penalty"],
            "weight_penalty_s_per_kg",
            ".0f",
        ),
        "",
        "### 7.3 制造附加成本",
        "",
        *_selected_scenario_table(
            sensitivity["manufacturing_cost"],
            "manufacturing_overhead_rate",
            ".6f",
        ),
        "",
        "重量惩罚的题目值为 $20\\,\\mathrm{s/kg}$，距离最优层数发生变化的范围很远。"
        "相比之下，代谢率、内侧换热和未计入的制造附加成本是更重要的不确定性来源。",
        "",
        "## 8. 生成的结果文件",
        "",
        f"结果目录：`{output_dir.as_posix()}`",
        "",
        "- `candidate_summary.csv`：所有可行外层层数的汇总；",
        "- `optimal_solution.csv`：主目标最优解；",
        "- `optimal_solution_strict.csv`：最优解严格容差复核；",
        "- `candidate_timeseries.csv`：所有候选方案的温度和热流时序列；",
        "- `model_validation.csv`：与问题一及严格求解的数值一致性；",
        "- `sensitivity_*.csv`：各类敏感性和情景分析。",
        "",
        "## 9. 已知边界与后续建议",
        "",
        "- 负重代谢系数和压迫换热系数未由题目数据标定，故仅作情景分析；",
        "- 基准服装初始温度为 $37^\\circ\\mathrm C$，实际未预热服装应参考初温敏感性结果；",
        "- 主模型未包含风、辐射、结霜、水分迁移、冷桥和制造公差；",
        "- 若有实验数据，优先标定代谢率、内侧换热系数和层间接触热阻。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
