"""ODE execution, event extraction and energy-balance auditing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_simpson, solve_ivp

from .config import ModelParameters
from .dsc import PCMModel
from .model import (
    DYNAMIC_SBF,
    MODEL_VARIANTS,
    QUASISTEADY_SBF,
    alpha_skin,
    alpha_skin_from_blood_flow,
    rhs,
    state_diagnostics,
)


@dataclass
class CaseResult:
    """Numerical solution, event summary and sampled diagnostics for one model case."""

    case_id: str
    params: ModelParameters
    model_variant: str
    model_name: str
    state_dimension: int
    timeseries: pd.DataFrame
    summary: dict[str, float | str]
    solver_method: str
    rtol: float
    atol: float
    max_step_s: float


def model_name(model_variant: str) -> str:
    """Return a readable name for an explicit model variant."""

    if model_variant == DYNAMIC_SBF:
        return "Model 6 - Dynamic SBF"
    if model_variant == QUASISTEADY_SBF:
        return "Model 5 - Quasi-steady SBF"
    raise ValueError(f"Unknown model variant: {model_variant}")


def _event_15(_time_s: float, state: np.ndarray) -> float:
    """Return the downward skin-temperature event function for 15 degC."""

    return float(state[1] - 15.0)


def _event_10(_time_s: float, state: np.ndarray) -> float:
    """Return the downward skin-temperature event function for 10 degC."""

    return float(state[1] - 10.0)


_event_15.direction = -1.0
_event_15.terminal = False
_event_10.direction = -1.0
_event_10.terminal = True


def _sample_times(end_time_s: float, step_s: float) -> np.ndarray:
    """Return a uniform time grid that always includes the exact final time."""

    grid = np.arange(0.0, end_time_s + 0.5 * step_s, step_s)
    grid = grid[grid <= end_time_s]
    if grid.size == 0 or not np.isclose(grid[-1], end_time_s, rtol=0.0, atol=1.0e-10):
        grid = np.append(grid, end_time_s)
    return grid


def _initial_state(params: ModelParameters, model_variant: str) -> np.ndarray:
    """Return the explicitly dimensioned initial state for the selected model."""

    temperatures = [
        params.initial_core_temperature_C,
        params.initial_skin_temperature_C,
        params.initial_layer1_temperature_C,
        params.initial_pcm_temperature_C,
        params.initial_layer3_temperature_C,
    ]
    if model_variant == DYNAMIC_SBF:
        return np.array(
            [*temperatures, params.initial_skin_blood_flow_L_m2_h], dtype=float
        )
    if model_variant == QUASISTEADY_SBF:
        return np.array(temperatures, dtype=float)
    raise ValueError(f"Unknown model variant: {model_variant}")


def _total_stored_energy_J(
    states: np.ndarray,
    params: ModelParameters,
    pcm: PCMModel,
    model_variant: str,
) -> np.ndarray:
    """Return total stored energy in J without assigning energy to the blood-flow state."""

    T_core, T_skin, T_layer1, T_pcm, T_layer3 = states[:5]
    if model_variant == DYNAMIC_SBF:
        alpha = np.asarray(alpha_skin_from_blood_flow(states[5]), dtype=float)
    elif model_variant == QUASISTEADY_SBF:
        alpha = np.asarray(alpha_skin(T_skin), dtype=float)
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
    reference = params.reference_temperature_C
    body = params.body_mass_kg * params.body_heat_capacity_J_kgK * (
        (1.0 - alpha) * (T_core - reference) + alpha * (T_skin - reference)
    )
    layer1 = params.layer1_capacity_J_K * (T_layer1 - reference)
    pcm_energy = params.pcm_mass_kg * np.asarray(
        pcm.specific_enthalpy_J_kg(T_pcm, params.dsc_scan_rate_K_min, reference),
        dtype=float,
    )
    layer3 = params.layer3_capacity_J_K * (T_layer3 - reference)
    return body + layer1 + pcm_energy + layer3


def _energy_audit(
    solution: object,
    params: ModelParameters,
    pcm: PCMModel,
    model_variant: str,
    output_times_s: np.ndarray,
    audit_step_s: float,
) -> tuple[np.ndarray, float, float]:
    """Integrate external heat exchange and return residuals and normalized error."""

    end_time = float(solution.t[-1])
    audit_times = _sample_times(end_time, audit_step_s)
    states = solution.sol(audit_times)
    q_external = np.empty_like(audit_times)
    for index, state in enumerate(states.T):
        diagnostics = state_diagnostics(state, params, pcm, model_variant)
        q_external[index] = params.heat_transfer_area_m2 * (
            params.metabolic_rate_W_m2
            - float(diagnostics["q_res_total_W_m2"])
            - float(diagnostics["q_3inf_W_m2"])
        )
    cumulative_external = cumulative_simpson(q_external, x=audit_times, initial=0.0)
    stored = _total_stored_energy_J(states, params, pcm, model_variant)
    residual = stored - stored[0] - cumulative_external
    absolute_external = cumulative_simpson(np.abs(q_external), x=audit_times, initial=0.0)
    stored_change_scale = float(np.max(np.abs(stored - stored[0])))
    scale = max(float(absolute_external[-1]), stored_change_scale, 1.0)
    max_absolute = float(np.max(np.abs(residual)))
    relative = max_absolute / scale
    return np.interp(output_times_s, audit_times, residual), max_absolute, relative


def extract_event_state(
    solution: object,
    event_index: int,
    params: ModelParameters,
    pcm: PCMModel,
    model_variant: str,
) -> dict[str, float]:
    """Return exact solve_ivp event time and selected diagnostics, or NaNs."""

    if len(solution.t_events[event_index]) == 0:
        return {
            "time_s": np.nan,
            "T_core_C": np.nan,
            "T_pcm_C": np.nan,
            "alpha_skin": np.nan,
            "pcm_phase_fraction": np.nan,
            "skin_blood_flow_actual_L_m2_h": np.nan,
            "skin_blood_flow_eq_L_m2_h": np.nan,
        }
    time_s = float(solution.t_events[event_index][0])
    state = np.asarray(solution.y_events[event_index][0], dtype=float)
    diagnostics = state_diagnostics(state, params, pcm, model_variant)
    return {
        "time_s": time_s,
        "T_core_C": float(state[0]),
        "T_pcm_C": float(state[3]),
        "alpha_skin": float(diagnostics["alpha_skin"]),
        "pcm_phase_fraction": float(diagnostics["pcm_phase_fraction"]),
        "skin_blood_flow_actual_L_m2_h": float(
            diagnostics["skin_blood_flow_actual_L_m2_h"]
        ),
        "skin_blood_flow_eq_L_m2_h": float(
            diagnostics["skin_blood_flow_eq_L_m2_h"]
        ),
    }


def run_case(
    case_id: str,
    params: ModelParameters,
    pcm: PCMModel,
    *,
    model_variant: str = DYNAMIC_SBF,
    method: str = "RK45",
    rtol: float = 1.0e-7,
    atol: float = 1.0e-9,
    max_step_s: float = 5.0,
    horizon_s: float = 86400.0,
    output_step_s: float = 5.0,
    audit_step_s: float = 0.25,
) -> CaseResult:
    """Solve one explicit five- or six-state case through the 10 degC event or horizon."""

    params.validate()
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {model_variant}")
    initial_state = _initial_state(params, model_variant)
    solution = solve_ivp(
        fun=lambda time, state: rhs(time, state, params, pcm, model_variant),
        t_span=(0.0, horizon_s),
        y0=initial_state,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step_s,
        events=(_event_15, _event_10),
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(f"Solver failed for {case_id}: {solution.message}")
    if solution.sol is None or not np.all(np.isfinite(solution.y)):
        raise RuntimeError(f"Solver returned invalid dense output for {case_id}")

    output_times = _sample_times(float(solution.t[-1]), output_step_s)
    states = solution.sol(output_times)
    rows: list[dict[str, float | str]] = []
    for time_s, state in zip(output_times, states.T, strict=True):
        diagnostics = state_diagnostics(state, params, pcm, model_variant)
        rows.append(
            {
                "time_s": float(time_s),
                "T_core_C": float(state[0]),
                "T_skin_C": float(state[1]),
                "T_layer1_C": float(state[2]),
                "T_pcm_C": float(state[3]),
                "T_layer3_C": float(state[4]),
                **diagnostics,
                "h_in_W_m2K": params.h_in_W_m2K,
            }
        )
    timeseries = pd.DataFrame(rows)
    residual, max_residual, relative_error = _energy_audit(
        solution, params, pcm, model_variant, output_times, audit_step_s
    )
    timeseries["energy_balance_residual_J"] = residual

    numeric = timeseries.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError(f"NaN or Inf found in sampled output for {case_id}")
    if not ((timeseries["alpha_skin"] > 0.0) & (timeseries["alpha_skin"] < 1.0)).all():
        raise RuntimeError(f"Skin mass fraction out of bounds for {case_id}")
    if not (timeseries["C_core_J_K"] > 0.0).all() or not (timeseries["C_skin_J_K"] > 0.0).all():
        raise RuntimeError(f"Nonpositive human heat capacity for {case_id}")
    if not timeseries["skin_blood_flow_eq_L_m2_h"].between(0.5, 90.0).all():
        raise RuntimeError(f"Equilibrium skin blood flow out of bounds for {case_id}")
    if not timeseries["skin_blood_flow_actual_L_m2_h"].between(0.5 - 1.0e-8, 90.0 + 1.0e-8).all():
        raise RuntimeError(f"Actual skin blood flow out of bounds for {case_id}")
    if model_variant == QUASISTEADY_SBF and not np.allclose(
        timeseries["skin_blood_flow_actual_L_m2_h"],
        timeseries["skin_blood_flow_eq_L_m2_h"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Quasi-steady actual and equilibrium blood flow must be identical")
    if not (timeseries["c_eff_pcm_J_kgK"] >= pcm.base_heat_capacity_J_kgK).all():
        raise RuntimeError(f"PCM apparent heat capacity below base value for {case_id}")
    if not timeseries["pcm_phase_fraction"].between(0.0, 1.0).all():
        raise RuntimeError(f"PCM phase fraction out of bounds for {case_id}")
    if not (timeseries["h_out_W_m2K"] > 0.0).all():
        raise RuntimeError(f"Nonpositive outside heat-transfer coefficient for {case_id}")
    latent_limit = params.pcm_mass_kg * pcm.latent_heat_J_kg(params.dsc_scan_rate_K_min)
    if timeseries["pcm_latent_released_J"].min() < -1.0e-9 or (
        timeseries["pcm_latent_released_J"].max() > latent_limit * (1.0 + 1.0e-12)
    ):
        raise RuntimeError(f"PCM latent release outside physical limits for {case_id}")

    event15 = extract_event_state(solution, 0, params, pcm, model_variant)
    event10 = extract_event_state(solution, 1, params, pcm, model_variant)
    effective_tau = params.blood_flow_time_constant_s if model_variant == DYNAMIC_SBF else 0.0
    summary: dict[str, float | str] = {
        "model_name": model_name(model_variant),
        "model_variant": model_variant,
        "state_dimension": len(initial_state),
        "tau_blood_flow_s": effective_tau,
        "skin_blood_flow_initial": float(
            timeseries.iloc[0]["skin_blood_flow_actual_L_m2_h"]
        ),
        "t15_s": event15["time_s"],
        "t15_min": event15["time_s"] / 60.0,
        "t10_s": event10["time_s"],
        "t10_min": event10["time_s"] / 60.0,
        "T_core_at_t15_C": event15["T_core_C"],
        "T_core_at_t10_C": event10["T_core_C"],
        "T_pcm_at_t15_C": event15["T_pcm_C"],
        "T_pcm_at_t10_C": event10["T_pcm_C"],
        "skin_blood_flow_at_t15": event15["skin_blood_flow_actual_L_m2_h"],
        "skin_blood_flow_eq_at_t15": event15["skin_blood_flow_eq_L_m2_h"],
        "skin_blood_flow_at_t10": event10["skin_blood_flow_actual_L_m2_h"],
        "skin_blood_flow_eq_at_t10": event10["skin_blood_flow_eq_L_m2_h"],
        "alpha_skin_at_t15": event15["alpha_skin"],
        "alpha_skin_at_t10": event10["alpha_skin"],
        "pcm_phase_fraction_at_t15": event15["pcm_phase_fraction"],
        "pcm_phase_fraction_at_t10": event10["pcm_phase_fraction"],
        "max_abs_energy_balance_residual_J": max_residual,
        "relative_energy_balance_error": relative_error,
        "M0_W_m2": params.metabolic_rate_W_m2,
        "T_skin_initial_C": params.initial_skin_temperature_C,
        "beta_DSC_K_min": params.dsc_scan_rate_K_min,
        "h_in_W_m2K": params.h_in_W_m2K,
        "lambda_out": params.lambda_out,
        "p_a_Torr": params.vapor_pressure_Torr,
    }
    return CaseResult(
        case_id,
        params,
        model_variant,
        model_name(model_variant),
        len(initial_state),
        timeseries,
        summary,
        method,
        rtol,
        atol,
        max_step_s,
    )
