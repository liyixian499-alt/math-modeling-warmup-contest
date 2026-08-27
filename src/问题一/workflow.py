"""Sensitivity, convergence, CSV export and terminal reporting workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_DSC_PATH, DEFAULT_OUTPUT_DIR, ModelParameters
from .dsc import PCMModel, load_pcm_model
from .model import alpha_skin_from_blood_flow, respiratory_heat_loss
from .simulation import CaseResult, run_case


MAIN_TIMESERIES_COLUMNS = [
    "time_s", "T_core_C", "T_skin_C", "T_layer1_C", "T_pcm_C", "T_layer3_C",
    "skin_blood_flow_L_m2_h", "alpha_skin", "alpha_dot_1_s", "C_core_J_K",
    "C_skin_J_K", "q_cs_W_m2", "q_res_lat_W_m2", "q_res_sens_W_m2",
    "q_res_total_W_m2", "q_s1_W_m2", "q_12_W_m2", "q_23_W_m2",
    "q_3inf_W_m2", "h_in_W_m2K", "h_out_W_m2K", "c_eff_pcm_J_kgK",
    "pcm_phase_fraction", "pcm_latent_released_J", "dm_core_to_skin_kg_s",
    "enthalpy_mass_transfer_W", "mass_redistribution_correction_W",
    "energy_balance_residual_J",
]

SENSITIVITY_TIMESERIES_COLUMNS = [
    "case_id", "parameter", "scenario", "parameter_value", "time_s", "T_core_C",
    "T_skin_C", "T_layer1_C", "T_pcm_C", "T_layer3_C",
    "skin_blood_flow_L_m2_h", "alpha_skin", "q_res_total_W_m2",
    "c_eff_pcm_J_kgK", "pcm_phase_fraction", "pcm_latent_released_J",
    "energy_balance_residual_J",
]


@dataclass(frozen=True)
class Scenario:
    """Metadata and parameter override for one OAT simulation."""

    case_id: str
    parameter: str
    scenario: str
    parameter_value: float
    unit: str
    updates: dict[str, float]


def _validate_reference_values(pcm: PCMModel) -> None:
    """Assert required analytical and source-data reference checks."""

    if not np.isclose(alpha_skin_from_blood_flow(6.3), 0.15, atol=2.0e-5):
        raise RuntimeError("alpha_skin reference check failed at blood flow 6.3")
    if not np.isclose(alpha_skin_from_blood_flow(0.5), 0.7283, atol=2.0e-4):
        raise RuntimeError("alpha_skin reference check failed at clipped blood flow 0.5")
    latent, sensible, total = respiratory_heat_loss(70.0, -40.0, 0.0)
    if not np.allclose([latent, sensible, total], [7.084, 7.252, 14.336], atol=1.0e-10):
        raise RuntimeError("Respiratory heat-loss reference check failed")
    latent_heat = pcm.latent_heat_J_kg(10.0)
    if not np.isclose(latent_heat, 122300.0, rtol=0.01):
        raise RuntimeError(
            f"DSC latent heat {latent_heat / 1000.0:.3f} kJ/kg is inconsistent with 122.3 kJ/kg"
        )


def _scenario_definitions() -> list[Scenario]:
    """Return the nine nonbaseline sensitivity cases required by the task."""

    return [
        Scenario("Tskin_33p7", "T_skin_initial", "alternative", 33.7, "degC", {"initial_skin_temperature_C": 33.7}),
        Scenario("M0_minus10", "M0", "minus10%", 63.0, "W/m2", {"metabolic_rate_W_m2": 63.0}),
        Scenario("M0_plus10", "M0", "plus10%", 77.0, "W/m2", {"metabolic_rate_W_m2": 77.0}),
        Scenario("beta_minus10", "beta_DSC", "minus10%", 9.0, "K/min", {"dsc_scan_rate_K_min": 9.0}),
        Scenario("beta_plus10", "beta_DSC", "plus10%", 11.0, "K/min", {"dsc_scan_rate_K_min": 11.0}),
        Scenario("hin_minus10", "h_in", "minus10%", 2.7, "W/(m2 K)", {"h_in_W_m2K": 2.7}),
        Scenario("hin_plus10", "h_in", "plus10%", 3.3, "W/(m2 K)", {"h_in_W_m2K": 3.3}),
        Scenario("hout_minus10", "lambda_out", "minus10%", 0.9, "1", {"lambda_out": 0.9}),
        Scenario("hout_plus10", "lambda_out", "plus10%", 1.1, "1", {"lambda_out": 1.1}),
    ]


def _safe_delta_percent(value: float, baseline: float) -> float:
    """Return percent change or NaN when either threshold was not reached."""

    if not np.isfinite(value) or not np.isfinite(baseline):
        return np.nan
    return (value - baseline) / baseline * 100.0


def _summary_row(
    result: CaseResult,
    parameter: str,
    scenario: str,
    parameter_value: float,
    unit: str,
    baseline: CaseResult,
) -> dict[str, float | str]:
    """Build one sensitivity-summary record from a solved case."""

    summary = result.summary
    baseline_summary = baseline.summary
    baseline_values = {"M0": 70.0, "beta_DSC": 10.0, "h_in": 3.0, "lambda_out": 1.0, "T_skin_initial": 37.0}
    relative = (
        parameter_value / baseline_values[parameter]
        if parameter in baseline_values and np.isfinite(parameter_value)
        else 1.0
    )
    return {
        "case_id": result.case_id,
        "parameter": parameter,
        "scenario": scenario,
        "parameter_value": parameter_value,
        "unit": unit,
        "relative_to_baseline": relative,
        "t15_s": summary["t15_s"],
        "t10_s": summary["t10_s"],
        "delta_t15_pct": _safe_delta_percent(summary["t15_s"], baseline_summary["t15_s"]),
        "delta_t10_pct": _safe_delta_percent(summary["t10_s"], baseline_summary["t10_s"]),
        "T_core_at_t15_C": summary["T_core_at_t15_C"],
        "T_core_at_t10_C": summary["T_core_at_t10_C"],
        "T_pcm_at_t15_C": summary["T_pcm_at_t15_C"],
        "T_pcm_at_t10_C": summary["T_pcm_at_t10_C"],
        "alpha_skin_at_t15": summary["alpha_skin_at_t15"],
        "alpha_skin_at_t10": summary["alpha_skin_at_t10"],
        "pcm_phase_fraction_at_t15": summary["pcm_phase_fraction_at_t15"],
        "pcm_phase_fraction_at_t10": summary["pcm_phase_fraction_at_t10"],
        "max_abs_energy_balance_residual_J": summary["max_abs_energy_balance_residual_J"],
        "relative_energy_balance_error": summary["relative_energy_balance_error"],
        "central_sensitivity_t15": np.nan,
        "central_sensitivity_t10": np.nan,
    }


def _sensitivity_indices(summary: pd.DataFrame, baseline: CaseResult) -> pd.DataFrame:
    """Compute centered dimensionless t15 and t10 sensitivities and ranks."""

    definitions = [
        ("M0", 70.0, 63.0, 77.0),
        ("beta_DSC", 10.0, 9.0, 11.0),
        ("h_in", 3.0, 2.7, 3.3),
        ("lambda_out", 1.0, 0.9, 1.1),
    ]
    rows = []
    for parameter, base_value, lower_value, upper_value in definitions:
        subset = summary[summary["parameter"] == parameter]
        lower = subset[np.isclose(subset["parameter_value"], lower_value)].iloc[0]
        upper = subset[np.isclose(subset["parameter_value"], upper_value)].iloc[0]
        S_t15 = (upper["t15_s"] - lower["t15_s"]) / (0.2 * baseline.summary["t15_s"])
        S_t10 = (upper["t10_s"] - lower["t10_s"]) / (0.2 * baseline.summary["t10_s"])
        rows.append(
            {
                "parameter": parameter,
                "baseline_value": base_value,
                "lower_value": lower_value,
                "upper_value": upper_value,
                "S_t15": S_t15,
                "S_t10": S_t10,
                "abs_S_t15": abs(S_t15),
                "abs_S_t10": abs(S_t10),
            }
        )
    indices = pd.DataFrame(rows)
    indices["rank_t15"] = indices["abs_S_t15"].rank(method="min", ascending=False).astype(int)
    indices["rank_t10"] = indices["abs_S_t10"].rank(method="min", ascending=False).astype(int)
    return indices


def execute_workflow(
    dsc_path: str | Path = DEFAULT_DSC_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """Run baseline, strict convergence and all OAT cases, then export six CSV files."""

    pcm = load_pcm_model(dsc_path)
    _validate_reference_values(pcm)
    baseline_params = ModelParameters()
    baseline = run_case("baseline", baseline_params, pcm)
    strict = run_case(
        "strict", baseline_params, pcm, rtol=1.0e-9, atol=1.0e-11, max_step_s=1.0
    )

    scenarios = _scenario_definitions()
    case_results: list[tuple[Scenario, CaseResult]] = []
    for scenario in scenarios:
        result = run_case(scenario.case_id, baseline_params.with_updates(**scenario.updates), pcm)
        case_results.append((scenario, result))

    sensitivity_rows = [
        _summary_row(baseline, "baseline", "baseline", np.nan, "mixed", baseline)
    ]
    for scenario, result in case_results:
        sensitivity_rows.append(
            _summary_row(
                result,
                scenario.parameter,
                scenario.scenario,
                scenario.parameter_value,
                scenario.unit,
                baseline,
            )
        )
    sensitivity_summary = pd.DataFrame(sensitivity_rows)
    indices = _sensitivity_indices(sensitivity_summary, baseline)
    for _, index_row in indices.iterrows():
        mask = sensitivity_summary["parameter"] == index_row["parameter"]
        sensitivity_summary.loc[mask, "central_sensitivity_t15"] = index_row["S_t15"]
        sensitivity_summary.loc[mask, "central_sensitivity_t10"] = index_row["S_t10"]

    long_frames = []
    baseline_long = baseline.timeseries.copy()
    baseline_long.insert(0, "parameter_value", np.nan)
    baseline_long.insert(0, "scenario", "baseline")
    baseline_long.insert(0, "parameter", "baseline")
    baseline_long.insert(0, "case_id", "baseline")
    long_frames.append(baseline_long)
    for scenario, result in case_results:
        frame = result.timeseries.copy()
        frame.insert(0, "parameter_value", scenario.parameter_value)
        frame.insert(0, "scenario", scenario.scenario)
        frame.insert(0, "parameter", scenario.parameter)
        frame.insert(0, "case_id", scenario.case_id)
        long_frames.append(frame)
    sensitivity_timeseries = pd.concat(long_frames, ignore_index=True)

    convergence = pd.DataFrame(
        [
            {
                "solver_case": "baseline",
                "method": baseline.solver_method,
                "rtol": baseline.rtol,
                "atol": baseline.atol,
                "max_step_s": baseline.max_step_s,
                "t15_s": baseline.summary["t15_s"],
                "t10_s": baseline.summary["t10_s"],
                "delta_t15_vs_strict_s": baseline.summary["t15_s"] - strict.summary["t15_s"],
                "delta_t10_vs_strict_s": baseline.summary["t10_s"] - strict.summary["t10_s"],
                "max_abs_energy_balance_residual_J": baseline.summary["max_abs_energy_balance_residual_J"],
                "relative_energy_balance_error": baseline.summary["relative_energy_balance_error"],
            },
            {
                "solver_case": "strict",
                "method": strict.solver_method,
                "rtol": strict.rtol,
                "atol": strict.atol,
                "max_step_s": strict.max_step_s,
                "t15_s": strict.summary["t15_s"],
                "t10_s": strict.summary["t10_s"],
                "delta_t15_vs_strict_s": 0.0,
                "delta_t10_vs_strict_s": 0.0,
                "max_abs_energy_balance_residual_J": strict.summary["max_abs_energy_balance_residual_J"],
                "relative_energy_balance_error": strict.summary["relative_energy_balance_error"],
            },
        ]
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    baseline.timeseries.loc[:, MAIN_TIMESERIES_COLUMNS].to_csv(
        destination / "main_timeseries.csv", index=False
    )
    pd.DataFrame([baseline.summary]).to_csv(destination / "main_summary.csv", index=False)
    sensitivity_summary.to_csv(destination / "sensitivity_summary.csv", index=False)
    sensitivity_timeseries.loc[:, SENSITIVITY_TIMESERIES_COLUMNS].to_csv(
        destination / "sensitivity_timeseries.csv", index=False
    )
    indices.to_csv(destination / "sensitivity_indices.csv", index=False)
    convergence.to_csv(destination / "solver_convergence.csv", index=False)

    return {
        "pcm": pcm,
        "baseline": baseline,
        "strict": strict,
        "sensitivity_summary": sensitivity_summary,
        "sensitivity_indices": indices,
        "solver_convergence": convergence,
        "output_dir": destination,
    }


def print_terminal_summary(results: dict[str, object]) -> None:
    """Print the requested concise numerical baseline and sensitivity summary."""

    pcm: PCMModel = results["pcm"]
    baseline: CaseResult = results["baseline"]
    convergence: pd.DataFrame = results["solver_convergence"]
    sensitivity: pd.DataFrame = results["sensitivity_summary"]
    indices: pd.DataFrame = results["sensitivity_indices"]
    summary = baseline.summary
    respiration = baseline.timeseries.iloc[0]

    def event_text(value: float) -> str:
        return f"{value:.6f} s ({value / 60.0:.6f} min)" if np.isfinite(value) else "not reached within simulation horizon"

    print("=== Problem 1 five-node model ===")
    print(f"DSC latent heat at 10 K/min: {pcm.latent_heat_J_kg(10.0) / 1000.0:.6f} kJ/kg")
    print(f"t15: {event_text(summary['t15_s'])}")
    print(f"t10: {event_text(summary['t10_s'])}")
    print(
        "Tcore at t15/t10: "
        f"{summary['T_core_at_t15_C']:.6f} / {summary['T_core_at_t10_C']:.6f} degC"
    )
    print(
        "PCM T, released fraction at t15: "
        f"{summary['T_pcm_at_t15_C']:.6f} degC, {summary['pcm_phase_fraction_at_t15']:.6f}"
    )
    print(
        "PCM T, released fraction at t10: "
        f"{summary['T_pcm_at_t10_C']:.6f} degC, {summary['pcm_phase_fraction_at_t10']:.6f}"
    )
    print(
        "alpha range: "
        f"{baseline.timeseries['alpha_skin'].min():.6f} to {baseline.timeseries['alpha_skin'].max():.6f}"
    )
    print(
        "respiration latent/sensible/total: "
        f"{respiration['q_res_lat_W_m2']:.6f} / {respiration['q_res_sens_W_m2']:.6f} / "
        f"{respiration['q_res_total_W_m2']:.6f} W/m2"
    )
    print(
        "energy max residual / relative error: "
        f"{summary['max_abs_energy_balance_residual_J']:.6e} J / "
        f"{summary['relative_energy_balance_error']:.6e}"
    )
    baseline_conv = convergence.iloc[0]
    print(
        "baseline minus strict t15/t10: "
        f"{baseline_conv['delta_t15_vs_strict_s']:.6e} / "
        f"{baseline_conv['delta_t10_vs_strict_s']:.6e} s"
    )
    print("--- Sensitivity cases ---")
    for _, row in sensitivity.iterrows():
        print(
            f"{row['case_id']}: t15={row['t15_s']:.6f} s ({row['delta_t15_pct']:.6f}%), "
            f"t10={row['t10_s']:.6f} s ({row['delta_t10_pct']:.6f}%)"
        )
    print("--- Continuous-parameter ranking by abs(S_t15) ---")
    for _, row in indices.sort_values("rank_t15").iterrows():
        print(
            f"{int(row['rank_t15'])}. {row['parameter']}: "
            f"S_t15={row['S_t15']:.6f}, S_t10={row['S_t10']:.6f}"
        )
    print(f"CSV output directory: {results['output_dir']}")
