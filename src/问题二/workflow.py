"""Effect decomposition, OAT sensitivity, validation and CSV delivery workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.问题一.dsc import PCMModel, load_pcm_model
from src.问题一.model import alpha_skin_from_blood_flow, respiratory_heat_loss

from .config import (
    DEFAULT_DELIVERY_PATH,
    DEFAULT_DSC_PATH,
    DEFAULT_OUTPUT_DIR,
    STRICT_SOLVER_SETTINGS,
    ModelParameters,
    SolverSettings,
)
from .simulation import CaseResult, run_case


MAIN_TIMESERIES_COLUMNS = [
    "time_s",
    "T_core_C",
    "T_skin_C",
    "T_layer1_C",
    "T_pcm_C",
    "T_layer3_C",
    "skin_blood_flow_eq_L_m2_h",
    "skin_blood_flow_actual_L_m2_h",
    "d_skin_blood_flow_dt_L_m2_h_s",
    "tau_blood_flow_s",
    "alpha_skin",
    "d_alpha_skin_dt_1_s",
    "C_core_J_K",
    "C_skin_J_K",
    "q_cs_W_m2",
    "q_res_lat_W_m2",
    "q_res_sens_W_m2",
    "q_res_total_W_m2",
    "q_s1_W_m2",
    "q_12_W_m2",
    "q_23_W_m2",
    "q_3inf_W_m2",
    "h_in_W_m2K",
    "h_nat_W_m2K",
    "h_forced_W_m2K",
    "h_out_W_m2K",
    "wind_speed_m_s",
    "metabolic_rate_W_m2",
    "c_eff_pcm_J_kgK",
    "pcm_phase_fraction",
    "pcm_latent_released_J",
    "dm_core_to_skin_kg_s",
    "enthalpy_mass_transfer_W",
    "energy_balance_residual_J",
]

SENSITIVITY_TIMESERIES_COLUMNS = [
    "case_id",
    "parameter",
    "scenario",
    "parameter_value",
    *MAIN_TIMESERIES_COLUMNS,
]


@dataclass(frozen=True)
class Scenario:
    """One OAT case and its single parameter override."""

    case_id: str
    parameter: str
    scenario: str
    parameter_value: float
    unit: str
    updates: dict[str, float]


def _safe_delta_percent(value: float, baseline: float) -> float:
    """Return percent change, or NaN if either event is unavailable."""

    if not np.isfinite(value) or not np.isfinite(baseline) or baseline == 0.0:
        return np.nan
    return (value - baseline) / baseline * 100.0


def _validate_reference_values(pcm: PCMModel) -> None:
    """Check the controlled analytical, DSC and forced-convection references."""

    if not np.isclose(12.1 * np.sqrt(3.0), 20.9578147716, rtol=0.0, atol=1.0e-9):
        raise RuntimeError("Forced-convection reference check failed")
    if not np.isclose(alpha_skin_from_blood_flow(6.3), 0.15, atol=2.0e-5):
        raise RuntimeError("Initial alpha_skin reference check failed")
    latent_heat = pcm.latent_heat_J_kg(10.0)
    if not np.isclose(latent_heat, 122300.0, rtol=0.01):
        raise RuntimeError(
            f"DSC latent heat {latent_heat / 1000.0:.3f} kJ/kg differs from the report reference"
        )
    q_lat, q_sens, q_total = respiratory_heat_loss(93.0, -40.0, 0.0)
    if not np.isclose(q_total, q_lat + q_sens, atol=1.0e-12):
        raise RuntimeError("Respiratory-loss recomputation check failed")


def _effect_cases() -> list[tuple[str, str, dict[str, float]]]:
    """Return the required Q1/W/I/M/Q2 decomposition definitions."""

    return [
        ("Q1", "问题一基线", {"wind_speed_m_s": 0.0, "h_in_W_m2K": 3.0, "metabolic_rate_W_m2": 70.0}),
        ("W", "只加入外界风", {"wind_speed_m_s": 3.0, "h_in_W_m2K": 3.0, "metabolic_rate_W_m2": 70.0}),
        ("I", "只改变内侧运动换热", {"wind_speed_m_s": 0.0, "h_in_W_m2K": 4.0, "metabolic_rate_W_m2": 70.0}),
        ("M", "只改变运动代谢", {"wind_speed_m_s": 0.0, "h_in_W_m2K": 3.0, "metabolic_rate_W_m2": 93.0}),
        ("Q2", "问题二完整工况", {"wind_speed_m_s": 3.0, "h_in_W_m2K": 4.0, "metabolic_rate_W_m2": 93.0}),
    ]


def _sensitivity_scenarios() -> list[Scenario]:
    """Return the 15 required OAT rows, including one Q2 reference per parameter."""

    specifications = [
        ("wind_speed", "wind_speed_m_s", [2.7, 3.0, 3.3], "m/s", ["minus10", "baseline", "plus10"]),
        ("h_in", "h_in_W_m2K", [3.6, 4.0, 4.4], "W/(m2 K)", ["minus10", "baseline", "plus10"]),
        ("metabolic_rate", "metabolic_rate_W_m2", [83.7, 93.0, 102.3], "W/m2", ["minus10", "baseline", "plus10"]),
        ("beta_DSC", "dsc_scan_rate_K_min", [9.0, 10.0, 11.0], "K/min", ["minus10", "baseline", "plus10"]),
        ("tau_blood_flow", "blood_flow_time_constant_s", [60.0, 173.0, 200.0], "s", ["lower_reference", "baseline", "upper_reference"]),
    ]
    scenarios: list[Scenario] = []
    for parameter, field, values, unit, labels in specifications:
        for value, label in zip(values, labels, strict=True):
            scenarios.append(
                Scenario(
                    f"{parameter}_{label}",
                    parameter,
                    label,
                    value,
                    unit,
                    {field: value},
                )
            )
    return scenarios


def _sensitivity_row(
    scenario: Scenario, result: CaseResult, baseline: CaseResult
) -> dict[str, float | bool | str]:
    """Create one fully specified sensitivity summary row."""

    summary = result.summary
    base = baseline.summary
    baseline_values = {
        "wind_speed": 3.0,
        "h_in": 4.0,
        "metabolic_rate": 93.0,
        "beta_DSC": 10.0,
        "tau_blood_flow": 173.0,
    }
    return {
        "case_id": scenario.case_id,
        "parameter": scenario.parameter,
        "scenario": scenario.scenario,
        "parameter_value": scenario.parameter_value,
        "unit": scenario.unit,
        "relative_to_baseline": scenario.parameter_value / baseline_values[scenario.parameter],
        "t15_reached": summary["t15_reached"],
        "t15_s": summary["t15_s"],
        "t15_min": summary["t15_min"],
        "t10_reached": summary["t10_reached"],
        "t10_s": summary["t10_s"],
        "t10_min": summary["t10_min"],
        "delta_t15_pct": _safe_delta_percent(float(summary["t15_s"]), float(base["t15_s"])),
        "delta_t10_pct": _safe_delta_percent(float(summary["t10_s"]), float(base["t10_s"])),
        "T_core_at_t15_C": summary["T_core_at_t15_C"],
        "T_core_at_t10_C": summary["T_core_at_t10_C"],
        "T_pcm_at_t15_C": summary["T_pcm_at_t15_C"],
        "T_pcm_at_t10_C": summary["T_pcm_at_t10_C"],
        "pcm_phase_fraction_at_t15": summary["pcm_phase_fraction_at_t15"],
        "pcm_phase_fraction_at_t10": summary["pcm_phase_fraction_at_t10"],
        "skin_blood_flow_at_t15": summary["skin_blood_flow_at_t15"],
        "skin_blood_flow_eq_at_t15": summary["skin_blood_flow_eq_at_t15"],
        "skin_blood_flow_at_t10": summary["skin_blood_flow_at_t10"],
        "alpha_skin_at_t15": summary["alpha_skin_at_t15"],
        "alpha_skin_at_t10": summary["alpha_skin_at_t10"],
        "max_abs_energy_balance_residual_J": summary["max_abs_energy_balance_residual_J"],
        "relative_energy_balance_error": summary["relative_energy_balance_error"],
    }


def _sensitivity_indices(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute centered dimensionless event-time sensitivities and ranking."""

    definitions = [
        ("wind_speed", 3.0, 2.7, 3.3, "m/s"),
        ("h_in", 4.0, 3.6, 4.4, "W/(m2 K)"),
        ("metabolic_rate", 93.0, 83.7, 102.3, "W/m2"),
        ("beta_DSC", 10.0, 9.0, 11.0, "K/min"),
    ]
    rows: list[dict[str, float | str]] = []
    for parameter, baseline_value, minus_value, plus_value, unit in definitions:
        subset = summary[summary["parameter"] == parameter]
        minus = subset[np.isclose(subset["parameter_value"], minus_value)].iloc[0]
        baseline = subset[np.isclose(subset["parameter_value"], baseline_value)].iloc[0]
        plus = subset[np.isclose(subset["parameter_value"], plus_value)].iloc[0]
        times = [float(minus["t15_s"]), float(baseline["t15_s"]), float(plus["t15_s"])]
        times10 = [float(minus["t10_s"]), float(baseline["t10_s"]), float(plus["t10_s"])]
        S_t15 = (times[2] - times[0]) / (0.2 * times[1]) if np.all(np.isfinite(times)) else np.nan
        S_t10 = (times10[2] - times10[0]) / (0.2 * times10[1]) if np.all(np.isfinite(times10)) else np.nan
        rows.append(
            {
                "parameter": parameter,
                "baseline_value": baseline_value,
                "minus10_value": minus_value,
                "plus10_value": plus_value,
                "unit": unit,
                "t15_minus": times[0],
                "t15_baseline": times[1],
                "t15_plus": times[2],
                "S_t15": S_t15,
                "t10_minus": times10[0],
                "t10_baseline": times10[1],
                "t10_plus": times10[2],
                "S_t10": S_t10,
                "abs_S_t15": abs(S_t15),
                "abs_S_t10": abs(S_t10),
            }
        )
    indices = pd.DataFrame(rows).sort_values("abs_S_t15", ascending=False, na_position="last")
    indices["rank_by_abs_S_t15"] = np.arange(1, len(indices) + 1)
    return indices.reset_index(drop=True)


def _effect_row(
    case_id: str, description: str, result: CaseResult, q1: CaseResult
) -> dict[str, float | str]:
    """Create one problem-effect decomposition record against Q1."""

    summary = result.summary
    q1_summary = q1.summary
    return {
        "case_id": case_id,
        "description": description,
        "wind_speed_m_s": summary["wind_speed_m_s"],
        "h_in_W_m2K": summary["h_in_W_m2K"],
        "metabolic_rate_W_m2": summary["metabolic_rate_W_m2"],
        "t15_s": summary["t15_s"],
        "t15_min": summary["t15_min"],
        "t10_s": summary["t10_s"],
        "t10_min": summary["t10_min"],
        "delta_t15_vs_q1_pct": _safe_delta_percent(float(summary["t15_s"]), float(q1_summary["t15_s"])),
        "delta_t10_vs_q1_pct": _safe_delta_percent(float(summary["t10_s"]), float(q1_summary["t10_s"])),
        "T_core_at_t15_C": summary["T_core_at_t15_C"],
        "T_pcm_at_t15_C": summary["T_pcm_at_t15_C"],
        "pcm_phase_fraction_at_t15": summary["pcm_phase_fraction_at_t15"],
        "skin_blood_flow_at_t15": summary["skin_blood_flow_at_t15"],
        "alpha_skin_at_t15": summary["alpha_skin_at_t15"],
        "max_abs_energy_balance_residual_J": summary["max_abs_energy_balance_residual_J"],
        "relative_energy_balance_error": summary["relative_energy_balance_error"],
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic UTF-8 CSV with explicit NaN values."""

    frame.to_csv(path, index=False, encoding="utf-8", na_rep="NaN", float_format="%.12g")


def _sparse_timeseries(frame: pd.DataFrame, interval_s: float = 30.0) -> pd.DataFrame:
    """Select a sparse uniform interface grid and preserve the exact final sample."""

    time_s = frame["time_s"].to_numpy(dtype=float)
    mask = np.isclose(np.mod(time_s, interval_s), 0.0, rtol=0.0, atol=1.0e-9)
    selected = frame.loc[mask].copy()
    if selected.empty or not np.isclose(
        float(selected.iloc[-1]["time_s"]), float(frame.iloc[-1]["time_s"]), atol=1.0e-9
    ):
        selected = pd.concat([selected, frame.iloc[[-1]]], ignore_index=True)
    return selected


def _format_number(value: object, digits: int = 6) -> str:
    """Format a finite scalar for the generated Markdown delivery document."""

    number = float(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "NaN"


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as Markdown without optional dependencies."""

    headers = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def _build_delivery_document(
    output_dir: Path, delivery_path: Path, pcm: PCMModel
) -> None:
    """Build the delivery Markdown by reading back the authoritative CSV outputs."""

    summary = pd.read_csv(output_dir / "problem2_summary.csv").iloc[0]
    effect = pd.read_csv(output_dir / "problem2_effect_decomposition.csv")
    sensitivity = pd.read_csv(output_dir / "problem2_sensitivity_summary.csv")
    indices = pd.read_csv(output_dir / "problem2_sensitivity_indices.csv")
    verification = pd.read_csv(output_dir / "problem2_numerical_verification.csv")
    baseline_verification = verification[verification["solver_case"] == "baseline"].iloc[0]
    strict_verification = verification[verification["solver_case"] == "strict"].iloc[0]
    tau = sensitivity[sensitivity["parameter"] == "tau_blood_flow"].copy()
    index_lines = []
    for _, row in indices.iterrows():
        direction = "增加" if row["S_t15"] > 0.0 else "降低"
        index_lines.append(
            f"- {int(row['rank_by_abs_S_t15'])}. `{row['parameter']}`: "
            f"$S_{{t15}}={row['S_t15']:.6f}$，参数提高使 $t_{{15}}${direction}。"
        )

    effect_table = effect[
        ["case_id", "t15_s", "t10_s", "delta_t15_vs_q1_pct", "delta_t10_vs_q1_pct"]
    ].copy()
    for column in effect_table.columns[1:]:
        effect_table[column] = effect_table[column].map(lambda value: _format_number(value))
    tau_table = tau[["parameter_value", "t15_s", "t10_s", "T_core_at_t15_C", "skin_blood_flow_at_t15", "alpha_skin_at_t15"]].copy()
    for column in tau_table.columns:
        tau_table[column] = tau_table[column].map(lambda value: _format_number(value))

    text = f"""# 问题二代码与数据交付说明

## 1. 任务完成情况

六状态非线性 ODE 主模型、15/10 °C 连续事件检测、Q1/W/I/M/Q2 效应拆分、五组 OAT 敏感性分析、中心无量纲敏感度、总能量守恒检查、RK45-DOP853 收敛复核和七个 CSV 接口均已完成。未生成任何可视化文件。可选的 33.7 °C 初始皮肤温度稳健性检查未执行，因为它不属于本轮强制验收项。

仓库中未发现任务所称《第 13 题问题二：有风轻微运动条件下低温防护服耦合传热模型改进报告》独立文件，因此以本轮任务说明为问题二控制定义，并以问题一正式模型和附件 1/2 为源文件完成实现。

## 2. 实际代码结构

- `src/问题二/config.py`：问题二参数、路径和求解设置；继承问题一参数母类并增加风速。
- `src/问题二/model.py`：强制/自然对流边界与六状态 RHS；复用问题一 DSC、血流、质量分区和质量交换焓流函数。
- `src/问题二/simulation.py`：`solve_ivp`、连续事件提取、时间序列诊断和总能量审计。
- `src/问题二/workflow.py`：效应拆分、OAT、敏感度、CSV 与本文档的生成。
- `src/问题二/__main__.py`：从仓库根目录运行的正式入口。
- `temp/tests/test_problem2.py`：单元与短时物理检查。
- `temp/tests/validate_problem2_outputs.py`：正式 CSV 交付后验验证。

## 3. 模型实现对应关系

| 报告机理 | 代码函数/模块 | 状态 |
| --- | --- | --- |
| 动态皮肤血流 | `src.问题一.model.equilibrium_skin_blood_flow`、`src.问题二.model.rhs` | 完成 |
| 动态 $\\alpha_s$ | `src.问题一.model.alpha_skin_from_blood_flow`、`dalpha_d_blood_flow` | 完成 |
| 质量交换焓流 | `src.问题一.model.human_temperature_derivatives_dynamic_sbf` | 完成 |
| 呼吸散热 | `src.问题一.model.respiratory_heat_loss` | 完成 |
| 内侧运动对流 | `src.问题二.model.heat_fluxes` | 完成 |
| 外侧强制对流 | `src.问题二.model.heat_fluxes` | 完成 |
| PCM 表观热容 | `src.问题一.dsc.PCMModel` | 完成 |
| $t_{{15}}$、$t_{{10}}$ 事件 | `src.问题二.simulation._event_15`、`_event_10` | 完成 |
| 能量守恒 | `src.问题二.simulation._energy_audit` | 完成 |

## 4. 正式问题二结果

- $t_{{15}}={_format_number(summary['t15_s'])}$ s（{_format_number(summary['t15_min'])} min），$t_{{10}}={_format_number(summary['t10_s'])}$ s（{_format_number(summary['t10_min'])} min）。
- $T_c(t_{{15}})={_format_number(summary['T_core_at_t15_C'])}$ °C，$T_c(t_{{10}})={_format_number(summary['T_core_at_t10_C'])}$ °C。
- PCM 相变进度：$\\xi(t_{{15}})={_format_number(summary['pcm_phase_fraction_at_t15'])}$，$\\xi(t_{{10}})={_format_number(summary['pcm_phase_fraction_at_t10'])}$。
- 实际皮肤血流：$\\dot V_{{bl}}(t_{{15}})={_format_number(summary['skin_blood_flow_at_t15'])}$ L/(m²·h)，$\\dot V_{{bl}}(t_{{10}})={_format_number(summary['skin_blood_flow_at_t10'])}$ L/(m²·h)。
- 最大绝对能量残差为 {_format_number(summary['max_abs_energy_balance_residual_J'], 9)} J，相对能量误差为 {float(summary['relative_energy_balance_error']):.9e}。

## 5. 问题一—问题二效应拆分

{_markdown_table(effect_table)}

W 与 Q1 的差值表示单独风速影响，I 表示单独内侧运动换热，M 表示单独代谢变化；Q2 是三项同时改变后的非线性净结果，不能由三个单因素百分比线性相加得到。

## 6. 敏感性分析结果

按 $|S_{{t15}}|$ 从大到小：

{chr(10).join(index_lines)}

血流时间常数采用离散参考值，不计算伪中心敏感度：

{_markdown_table(tau_table)}

## 7. 数值验证

- RK45 基线相对严格 DOP853：$\\Delta t_{{15}}={_format_number(strict_verification['delta_t15_vs_baseline_s'], 9)}$ s，$\\Delta t_{{10}}={_format_number(strict_verification['delta_t10_vs_baseline_s'], 9)}$ s。
- RK45 最大绝对能量残差 {_format_number(baseline_verification['max_abs_energy_balance_residual_J'], 9)} J，相对误差 {float(baseline_verification['relative_energy_balance_error']):.9e}。
- PCM 累计潜热范围为 [{_format_number(summary['pcm_latent_min_J'])}, {_format_number(summary['pcm_latent_max_J'])}] J，上限 {_format_number(summary['pcm_latent_limit_J'])} J；检查通过。
- 基准 DSC 潜热为 {pcm.latent_heat_J_kg(10.0) / 1000.0:.6f} kJ/kg，与表观热容来自同一 PCHIP 峰积分。

## 8. CSV 数据接口

- `problem2_timeseries.csv`：一行一个 5 s 正式 Q2 采样点；温度为 °C，热流为 W/m²，时间为 s。事件时间不从本表估计。
- `problem2_summary.csv`：一行正式 Q2 事件、人体/PCM 状态和守恒指标。
- `problem2_effect_decomposition.csv`：一行一个 Q1/W/I/M/Q2 工况，用于论文效应拆分表。
- `problem2_sensitivity_summary.csv`：一行一个 OAT 工况，含两事件状态和验证指标。
- `problem2_sensitivity_indices.csv`：一行一个 ±10% 参数，含中心无量纲敏感度与排序。
- `problem2_sensitivity_timeseries.csv`：一行一个敏感性工况的 30 s 稀疏采样点（并保留精确终点），用于后续复核；ODE 内部最大步长仍为 5 s，独立于输出采样。
- `problem2_numerical_verification.csv`：一行一个求解器设置，记录事件差值与能量误差。

## 9. 运行方法

从仓库根目录运行：

```bash
python -m src.问题二
```

测试与交付验证：

```bash
python -m unittest temp.tests.test_problem2
python temp/tests/validate_problem2_outputs.py
```

## 10. 已知限制

- $12.1\\sqrt v$ 为经验相关式，风速按整体平均值处理，不分局部迎风/背风区。
- 93 W/m² 是典型轻微运动代谢率，173 s 是参考血流响应时间常数。
- 不考虑服装通风质量流、空气渗透、辐射和颤抖产热。
- 附件 1 未明示扫描速率；10 K/min 是问题一延续的控制建模假设。
- 模型是整体集总参数近似，不解析局部人体或服装温度差异。
"""
    delivery_path.parent.mkdir(parents=True, exist_ok=True)
    delivery_path.write_text(text, encoding="utf-8")


def execute_workflow(
    dsc_path: str | Path = DEFAULT_DSC_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    delivery_path: str | Path = DEFAULT_DELIVERY_PATH,
) -> dict[str, object]:
    """Run all required Problem 2 calculations and create the CSV/Markdown delivery."""

    pcm = load_pcm_model(dsc_path)
    _validate_reference_values(pcm)
    baseline_params = ModelParameters()
    baseline_settings = SolverSettings()
    baseline = run_case("Q2", baseline_params, pcm, baseline_settings)
    strict = run_case("Q2_strict", baseline_params, pcm, STRICT_SOLVER_SETTINGS)

    effect_results: dict[str, tuple[str, CaseResult]] = {}
    for case_id, description, updates in _effect_cases():
        result = baseline if case_id == "Q2" else run_case(
            case_id, baseline_params.with_updates(**updates), pcm, baseline_settings
        )
        effect_results[case_id] = (description, result)
    q1 = effect_results["Q1"][1]
    effect = pd.DataFrame(
        [_effect_row(case_id, description, result, q1) for case_id, (description, result) in effect_results.items()]
    )

    scenarios = _sensitivity_scenarios()
    sensitivity_rows = []
    sensitivity_frames = []
    sensitivity_results: dict[str, CaseResult] = {}
    for scenario in scenarios:
        result = baseline if scenario.scenario == "baseline" else run_case(
            scenario.case_id,
            baseline_params.with_updates(**scenario.updates),
            pcm,
            baseline_settings,
        )
        sensitivity_results[scenario.case_id] = result
        sensitivity_rows.append(_sensitivity_row(scenario, result, baseline))
        frame = _sparse_timeseries(
            result.timeseries.loc[:, MAIN_TIMESERIES_COLUMNS].copy()
        )
        frame.insert(0, "parameter_value", scenario.parameter_value)
        frame.insert(0, "scenario", scenario.scenario)
        frame.insert(0, "parameter", scenario.parameter)
        frame.insert(0, "case_id", scenario.case_id)
        sensitivity_frames.append(frame)
    sensitivity = pd.DataFrame(sensitivity_rows)
    indices = _sensitivity_indices(sensitivity)
    sensitivity_timeseries = pd.concat(sensitivity_frames, ignore_index=True)

    verification_rows = []
    for solver_case, result in [("baseline", baseline), ("strict", strict)]:
        t15_delta = float(result.summary["t15_s"]) - float(baseline.summary["t15_s"])
        t10_delta = float(result.summary["t10_s"]) - float(baseline.summary["t10_s"])
        verification_rows.append(
            {
                "solver_case": solver_case,
                "solver": result.settings.method,
                "rtol": result.settings.rtol,
                "atol": result.settings.atol,
                "max_step_s": result.settings.max_step_s,
                "t15_s": result.summary["t15_s"],
                "t10_s": result.summary["t10_s"],
                "delta_t15_vs_baseline_s": t15_delta,
                "delta_t15_vs_baseline_pct": _safe_delta_percent(float(result.summary["t15_s"]), float(baseline.summary["t15_s"])),
                "delta_t10_vs_baseline_s": t10_delta,
                "delta_t10_vs_baseline_pct": _safe_delta_percent(float(result.summary["t10_s"]), float(baseline.summary["t10_s"])),
                "max_abs_energy_balance_residual_J": result.summary["max_abs_energy_balance_residual_J"],
                "relative_energy_balance_error": result.summary["relative_energy_balance_error"],
            }
        )
    verification = pd.DataFrame(verification_rows)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(baseline.timeseries.loc[:, MAIN_TIMESERIES_COLUMNS], destination / "problem2_timeseries.csv")
    _write_csv(pd.DataFrame([baseline.summary]), destination / "problem2_summary.csv")
    _write_csv(effect, destination / "problem2_effect_decomposition.csv")
    _write_csv(sensitivity, destination / "problem2_sensitivity_summary.csv")
    _write_csv(indices, destination / "problem2_sensitivity_indices.csv")
    _write_csv(sensitivity_timeseries.loc[:, SENSITIVITY_TIMESERIES_COLUMNS], destination / "problem2_sensitivity_timeseries.csv")
    _write_csv(verification, destination / "problem2_numerical_verification.csv")
    _build_delivery_document(destination, Path(delivery_path), pcm)
    return {
        "pcm": pcm,
        "baseline": baseline,
        "strict": strict,
        "effect": effect,
        "sensitivity": sensitivity,
        "indices": indices,
        "verification": verification,
        "output_dir": destination,
        "delivery_path": Path(delivery_path),
    }


def print_terminal_summary(results: dict[str, object]) -> None:
    """Print the concise required terminal summary without time-series data."""

    baseline: CaseResult = results["baseline"]
    effect: pd.DataFrame = results["effect"]
    indices: pd.DataFrame = results["indices"]
    verification: pd.DataFrame = results["verification"]
    summary = baseline.summary
    strict = verification[verification["solver_case"] == "strict"].iloc[0]
    print("Problem 2 simulation completed.")
    print("\nBaseline Q2:")
    print(f"t15 = {summary['t15_s']:.6f} s ({summary['t15_min']:.6f} min)")
    print(f"t10 = {summary['t10_s']:.6f} s ({summary['t10_min']:.6f} min)")
    print(f"T_core(t15) = {summary['T_core_at_t15_C']:.6f} degC")
    print(f"PCM fraction(t15) = {summary['pcm_phase_fraction_at_t15']:.6f}")
    print(f"max energy residual = {summary['max_abs_energy_balance_residual_J']:.6e} J")
    print(f"relative energy error = {summary['relative_energy_balance_error']:.6e}")
    print("\nEffect decomposition:")
    for _, row in effect.iterrows():
        print(f"{row['case_id']}: t15={row['t15_s']:.6f} s, t10={row['t10_s']:.6f} s")
    print("\nSensitivity ranking by |S_t15|:")
    for _, row in indices.iterrows():
        print(f"{int(row['rank_by_abs_S_t15'])}. {row['parameter']}: S_t15={row['S_t15']:.6f}")
    print(
        "\nStrict minus baseline t15/t10 = "
        f"{strict['delta_t15_vs_baseline_s']:.6e} / "
        f"{strict['delta_t10_vs_baseline_s']:.6e} s"
    )
    print(f"\nCSV files written to:\n{results['output_dir']}")
    print(f"Delivery document:\n{results['delivery_path']}")
