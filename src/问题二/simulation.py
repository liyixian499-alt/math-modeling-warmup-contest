"""Problem 2 ODE execution, event extraction and energy-balance auditing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_simpson, solve_ivp

from src.问题一.dsc import PCMModel
from src.问题一.model import alpha_skin_from_blood_flow, respiratory_heat_loss

from .config import ModelParameters, SolverSettings
from .model import rhs, state_diagnostics


@dataclass
class CaseResult:
    """Numerical solution, exact events and sampled diagnostics for one case."""

    case_id: str
    params: ModelParameters
    settings: SolverSettings
    timeseries: pd.DataFrame
    summary: dict[str, float | bool | str]


def _event_15(_time_s: float, state: np.ndarray) -> float:
    """Detect a downward crossing of 15 degC by skin temperature."""

    return float(state[1] - 15.0)


def _event_10(_time_s: float, state: np.ndarray) -> float:
    """Detect a downward crossing of 10 degC by skin temperature."""

    return float(state[1] - 10.0)


_event_15.direction = -1.0
_event_15.terminal = False
_event_10.direction = -1.0
_event_10.terminal = True


def _sample_times(end_time_s: float, step_s: float) -> np.ndarray:
    """Return a uniform sampling grid containing the exact final time."""

    grid = np.arange(0.0, end_time_s + 0.5 * step_s, step_s)
    grid = grid[grid <= end_time_s]
    if grid.size == 0 or not np.isclose(grid[-1], end_time_s, rtol=0.0, atol=1.0e-10):
        grid = np.append(grid, end_time_s)
    return grid


def _initial_state(params: ModelParameters) -> np.ndarray:
    """Return the formal six-component initial state."""

    return np.array(
        [
            params.initial_core_temperature_C,
            params.initial_skin_temperature_C,
            params.initial_layer1_temperature_C,
            params.initial_pcm_temperature_C,
            params.initial_layer3_temperature_C,
            params.initial_skin_blood_flow_L_m2_h,
        ],
        dtype=float,
    )


def _total_stored_energy_J(
    states: np.ndarray, params: ModelParameters, pcm: PCMModel
) -> np.ndarray:
    """Return body plus clothing stored energy using the PCM enthalpy integral."""

    T_core, T_skin, T_layer1, T_pcm, T_layer3, actual_flow = states
    alpha = np.asarray(alpha_skin_from_blood_flow(actual_flow), dtype=float)
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
    output_times_s: np.ndarray,
    audit_step_s: float,
) -> tuple[np.ndarray, float, float, float]:
    """Return sampled, maximum, final and relative total-energy residuals."""

    audit_times = _sample_times(float(solution.t[-1]), audit_step_s)
    states = solution.sol(audit_times)
    q_external_W = np.empty_like(audit_times)
    for index, state in enumerate(states.T):
        diagnostics = state_diagnostics(state, params, pcm)
        q_external_W[index] = params.heat_transfer_area_m2 * (
            params.metabolic_rate_W_m2
            - float(diagnostics["q_res_total_W_m2"])
            - float(diagnostics["q_3inf_W_m2"])
        )
    cumulative_external_J = cumulative_simpson(q_external_W, x=audit_times, initial=0.0)
    stored_J = _total_stored_energy_J(states, params, pcm)
    residual_J = stored_J - stored_J[0] - cumulative_external_J
    absolute_external_J = cumulative_simpson(
        np.abs(q_external_W), x=audit_times, initial=0.0
    )
    scale_J = max(
        float(absolute_external_J[-1]),
        float(np.max(np.abs(stored_J - stored_J[0]))),
        1.0,
    )
    max_absolute_J = float(np.max(np.abs(residual_J)))
    return (
        np.interp(output_times_s, audit_times, residual_J),
        max_absolute_J,
        float(residual_J[-1]),
        max_absolute_J / scale_J,
    )


def _event_state(
    solution: object, event_index: int, params: ModelParameters, pcm: PCMModel
) -> dict[str, float]:
    """Return the dense solver's first event state, or explicit NaNs if absent."""

    if len(solution.t_events[event_index]) == 0:
        return {
            "time_s": np.nan,
            "T_core_C": np.nan,
            "T_pcm_C": np.nan,
            "pcm_phase_fraction": np.nan,
            "skin_blood_flow_actual_L_m2_h": np.nan,
            "skin_blood_flow_eq_L_m2_h": np.nan,
            "alpha_skin": np.nan,
        }
    time_s = float(solution.t_events[event_index][0])
    state = np.asarray(solution.y_events[event_index][0], dtype=float)
    diagnostics = state_diagnostics(state, params, pcm)
    return {
        "time_s": time_s,
        "T_core_C": float(state[0]),
        "T_pcm_C": float(state[3]),
        "pcm_phase_fraction": float(diagnostics["pcm_phase_fraction"]),
        "skin_blood_flow_actual_L_m2_h": float(
            diagnostics["skin_blood_flow_actual_L_m2_h"]
        ),
        "skin_blood_flow_eq_L_m2_h": float(
            diagnostics["skin_blood_flow_eq_L_m2_h"]
        ),
        "alpha_skin": float(diagnostics["alpha_skin"]),
    }


def run_case(
    case_id: str,
    params: ModelParameters,
    pcm: PCMModel,
    settings: SolverSettings | None = None,
) -> CaseResult:
    """Solve one formal six-state case through t10 or the configured horizon."""

    params.validate()
    settings = settings or SolverSettings()
    settings.validate()
    solution = solve_ivp(
        fun=lambda time, state: rhs(time, state, params, pcm),
        t_span=(0.0, settings.horizon_s),
        y0=_initial_state(params),
        method=settings.method,
        rtol=settings.rtol,
        atol=settings.atol,
        max_step=settings.max_step_s,
        events=(_event_15, _event_10),
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(f"Solver failed for {case_id}: {solution.message}")
    if solution.sol is None or not np.all(np.isfinite(solution.y)):
        raise RuntimeError(f"Solver returned invalid output for {case_id}")

    output_times = _sample_times(float(solution.t[-1]), settings.output_step_s)
    states = solution.sol(output_times)
    rows: list[dict[str, float | str]] = []
    for time_s, state in zip(output_times, states.T, strict=True):
        rows.append(
            {
                "time_s": float(time_s),
                "T_core_C": float(state[0]),
                "T_skin_C": float(state[1]),
                "T_layer1_C": float(state[2]),
                "T_pcm_C": float(state[3]),
                "T_layer3_C": float(state[4]),
                **state_diagnostics(state, params, pcm),
                "h_in_W_m2K": params.h_in_W_m2K,
                "wind_speed_m_s": params.wind_speed_m_s,
                "metabolic_rate_W_m2": params.metabolic_rate_W_m2,
            }
        )
    timeseries = pd.DataFrame(rows)
    residual, max_residual, final_residual, relative_error = _energy_audit(
        solution, params, pcm, output_times, settings.audit_step_s
    )
    timeseries["energy_balance_residual_J"] = residual
    numeric = timeseries.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError(f"NaN or Inf found in sampled output for {case_id}")
    if not timeseries["skin_blood_flow_eq_L_m2_h"].between(0.5, 90.0).all():
        raise RuntimeError(f"Equilibrium skin blood flow out of bounds for {case_id}")
    if not timeseries["skin_blood_flow_actual_L_m2_h"].between(
        0.5 - 1.0e-8, 90.0 + 1.0e-8
    ).all():
        raise RuntimeError(f"Actual skin blood flow out of bounds for {case_id}")
    if not ((timeseries["alpha_skin"] > 0.0) & (timeseries["alpha_skin"] < 1.0)).all():
        raise RuntimeError(f"Skin mass fraction out of bounds for {case_id}")
    if not (timeseries[["C_core_J_K", "C_skin_J_K"]] > 0.0).all().all():
        raise RuntimeError(f"Nonpositive human heat capacity for {case_id}")
    if not timeseries["pcm_phase_fraction"].between(0.0, 1.0).all():
        raise RuntimeError(f"PCM phase fraction out of bounds for {case_id}")
    latent_limit_J = params.pcm_mass_kg * pcm.latent_heat_J_kg(
        params.dsc_scan_rate_K_min
    )
    if timeseries["pcm_latent_released_J"].min() < -1.0e-9 or timeseries[
        "pcm_latent_released_J"
    ].max() > latent_limit_J * (1.0 + 1.0e-12):
        raise RuntimeError(f"PCM latent release out of bounds for {case_id}")
    if not (timeseries["h_out_W_m2K"] > 0.0).all():
        raise RuntimeError(f"Nonpositive outside convection coefficient for {case_id}")

    event15 = _event_state(solution, 0, params, pcm)
    event10 = _event_state(solution, 1, params, pcm)
    _, _, respiratory_total = respiratory_heat_loss(
        params.metabolic_rate_W_m2,
        params.air_temperature_C,
        params.vapor_pressure_Torr,
    )
    summary: dict[str, float | bool | str] = {
        "case_id": case_id,
        "state_dimension": 6,
        "wind_speed_m_s": params.wind_speed_m_s,
        "h_in_W_m2K": params.h_in_W_m2K,
        "h_forced_W_m2K": 12.1 * np.sqrt(params.wind_speed_m_s),
        "metabolic_rate_W_m2": params.metabolic_rate_W_m2,
        "respiratory_loss_W_m2": respiratory_total,
        "tau_blood_flow_s": params.blood_flow_time_constant_s,
        "beta_DSC_K_min": params.dsc_scan_rate_K_min,
        "t15_reached": bool(np.isfinite(event15["time_s"])),
        "t15_s": event15["time_s"],
        "t15_min": event15["time_s"] / 60.0,
        "t10_reached": bool(np.isfinite(event10["time_s"])),
        "t10_s": event10["time_s"],
        "t10_min": event10["time_s"] / 60.0,
        "T_core_at_t15_C": event15["T_core_C"],
        "T_core_at_t10_C": event10["T_core_C"],
        "T_pcm_at_t15_C": event15["T_pcm_C"],
        "T_pcm_at_t10_C": event10["T_pcm_C"],
        "pcm_phase_fraction_at_t15": event15["pcm_phase_fraction"],
        "pcm_phase_fraction_at_t10": event10["pcm_phase_fraction"],
        "skin_blood_flow_at_t15": event15["skin_blood_flow_actual_L_m2_h"],
        "skin_blood_flow_eq_at_t15": event15["skin_blood_flow_eq_L_m2_h"],
        "skin_blood_flow_at_t10": event10["skin_blood_flow_actual_L_m2_h"],
        "skin_blood_flow_eq_at_t10": event10["skin_blood_flow_eq_L_m2_h"],
        "alpha_skin_at_t15": event15["alpha_skin"],
        "alpha_skin_at_t10": event10["alpha_skin"],
        "max_abs_energy_balance_residual_J": max_residual,
        "final_energy_balance_residual_J": final_residual,
        "relative_energy_balance_error": relative_error,
        "pcm_latent_limit_J": latent_limit_J,
        "pcm_latent_min_J": float(timeseries["pcm_latent_released_J"].min()),
        "pcm_latent_max_J": float(timeseries["pcm_latent_released_J"].max()),
    }
    return CaseResult(case_id, params, settings, timeseries, summary)
