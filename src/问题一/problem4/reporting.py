"""CSV-backed Markdown and concise terminal reporting for Problem 4."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    view = frame.loc[:, columns].copy()
    for column in view.select_dtypes(include="number").columns:
        view[column] = view[column].map(lambda value: f"{value:.{digits}g}")
    headers = "| " + " | ".join(view.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in record) + " |"
        for record in view.itertuples(index=False, name=None)
    ]
    return "\n".join([headers, rule, *rows])


def write_delivery_report(
    report_path: Path,
    output_dir: Path,
    target_source: dict[str, object],
) -> None:
    """Generate the required Markdown by reading the actual exported CSV files."""

    search = pd.read_csv(output_dir / "problem4_search.csv")
    optimal = pd.read_csv(output_dir / "problem4_optimal_solution.csv").iloc[0]
    strict = pd.read_csv(output_dir / "problem4_optimal_solution_strict.csv").iloc[0]
    inverse = pd.read_csv(output_dir / "problem4_sensitivity_inverse.csv")
    fixed = pd.read_csv(output_dir / "problem4_sensitivity_fixed_design.csv")
    summary = pd.read_csv(output_dir / "problem4_sensitivity_summary.csv")
    target = pd.read_csv(output_dir / "problem4_target_sensitivity.csv")
    validation = pd.read_csv(output_dir / "problem4_model_validation.csv")
    worst = fixed.sort_values("margin_to_target_s").head(6)
    failed = fixed[~fixed["feasible"]]
    validation_failures = validation[~validation["passed"]]
    files = sorted(path.name for path in output_dir.glob("*.csv"))

    text = f"""# 问题四代码说明与结果交付

## 1. 任务与模型口径

问题四调用问题一六状态动态皮肤血流母模型，恢复 0.7/0.4/0.3 mm 三层结构、-40 ℃ 静止无风工况。唯一设计变量为 `lambda_pcm`；它只缩放 DSC 基线扣除后的纯相变峰，不改变基础比热、PCM 厚度、密度、导热率、相变温区或峰形。

目标来自 `{target_source['path']}` 的 `{target_source['field']}`，读取值为 {float(target_source['value_min']):.9f} min；与题示 734.815 min 相差 {float(target_source['difference_min']):.9g} min。

## 2. 代码结构

- `src/问题一/problem4/config.py`：问题四固定口径、路径和求解器配置。
- `src/问题一/problem4/pcm.py`：纯相变项倍率包装，复用问题一 DSC 可信实现。
- `src/问题一/problem4/solver.py`：连续事件求解与 Brent 标量反演。
- `src/问题一/problem4/sensitivity.py`：反演敏感性与固定设计情景定义。
- `src/问题一/problem4/workflow.py`：全流程、验证、CSV 输出。
- `src/问题一/problem4/reporting.py`：读取实际 CSV 生成本文档。

## 3. 运行方式

```bash
python -m src.问题一.problem4
```

自动测试：

```bash
python -m unittest discover -s temp/tests -p "test_problem4.py"
```

## 4. 主反演结果

- 主根 `lambda*` = {optimal['lambda_opt']:.10f}，至少提高 {optimal['improvement_pct']:.6f}%。
- 单位质量潜热：{optimal['original_latent_heat_kJ_kg']:.6f} → {optimal['improved_latent_heat_kJ_kg']:.6f} kJ/kg。
- PCM 总潜热：{optimal['original_total_latent_heat_kJ']:.6f} → {optimal['improved_total_latent_heat_kJ']:.6f} kJ。
- 目标/实际 `t15`：{optimal['target_t15_min']:.9f} / {optimal['achieved_t15_min']:.9f} min，误差 {optimal['t15_error_s']:.6g} s。
- `t10` = {optimal['t10_min']:.9f} min。
- 严格复核使用同一主根 {strict['lambda_opt']:.10f}；DOP853 细化根 = {strict['strict_refined_lambda']:.10f}，相对主根差 {strict['delta_lambda_vs_main']:.6g}。同一主根下严格结果与主结果 `t15` 差 {strict['delta_t15_vs_main_s']:.6g} s。

## 5. 粗搜索与单调性

{_markdown_table(search, ['lambda_pcm', 'improvement_pct', 't15_min', 't10_min', 'gap_to_target_s', 'feasible'])}

相邻粗搜索点的最小 `t15` 增量为 {search.loc[search['has_previous_search_point'], 'adjacent_t15_difference_s'].min():.6f} s，计算结果支持根区间内的单调性假设。

## 6. 数值验证

{_markdown_table(validation, ['validation_name', 'reference_value', 'computed_value', 'absolute_error', 'tolerance', 'passed'])}

验证状态：{'全部通过' if validation_failures.empty else f'{len(validation_failures)} 项未通过'}。主解最大绝对能量残差为 {optimal['max_abs_energy_balance_residual_J']:.6g} J，相对误差为 {optimal['relative_energy_balance_error']:.6g}；严格解相对误差为 {strict['relative_energy_balance_error']:.6g}。

## 7. 敏感性分析

### 7.1 反演结果敏感性

每个物理参数情景均重新执行完整 `lambda` 反演。完整结果见 `problem4_sensitivity_inverse.csv`；最不利的高倍率情景如下：

{_markdown_table(inverse.sort_values('lambda_opt', ascending=False).head(8), ['parameter', 'scenario', 'parameter_value', 'lambda_opt', 'improvement_pct', 't10_s', 'feasible'])}

### 7.2 固定设计稳健性

固定基准 `lambda*` 后，共有 {len(failed)} 个情景不能达到目标。裕量最小的情景如下：

{_markdown_table(worst, ['parameter', 'scenario', 'parameter_value', 't15_min', 'margin_to_target_min', 'feasible'])}

### 7.3 敏感度排序

{_markdown_table(summary, ['rank', 'parameter', 'normalized_sensitivity_lambda', 'abs_normalized_sensitivity_lambda', 'direction_interpretation'])}

正号表示参数增大会提高所需 PCM 倍率，负号表示参数增大会降低所需倍率。`tau_bl` 和初始皮肤温度按离散情景比较，不强行套用中心差分。

### 7.4 目标时间敏感性

{_markdown_table(target, ['target_factor', 'target_t15_min', 'lambda_opt', 'improvement_pct', 'achieved_t15_min', 't15_error_s'])}

## 8. 生成的数据文件

"""
    for name in files:
        text += f"- `results/outputs/problem4/{name}`\n"
    text += """

## 9. 已知边界

- 结果建立在问题一六状态人体热调节模型及其参数假设上。
- 173 s 是跨研究场景采用的血流响应参考参数。
- 10 K/min 来自 DSC 与附件参数的一致性反演；改变扫描速率时，ODE 内表观热容和报表潜热同步变化。
- 本题只改变潜热幅值；现实材料未必能在导热率、密度、基础比热和相变温区均不变时实现同等提升。
- `t15` 是题目比较指标，不是医学意义上的绝对安全暴露时间。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
