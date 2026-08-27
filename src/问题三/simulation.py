"""Numerical integration and objective evaluation for Problem 3 candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_simpson, solve_ivp

from src.问题一.dsc import PCMModel
from src.问题一.model import alpha_skin_from_blood_flow

from .config import Problem3Parameters
from .model import initial_state, rhs, state_diagnostics


@dataclass
class Problem3CaseResult:
    """Thermal solution and constrained standing-time result for one coating count."""

    outer_layer_count: int
    timeseries: pd.DataFrame
    summary: dict[str, float | str]
    solution: object


def _event_15(_time_s: float, state: np.ndarray) -> float:
    return float(state[1] - 15.0)


def _event_10(_time_s: float, state: np.ndarray) -> float:
    return float(state[1] - 10.0)


_event_15.direction = -1.0
_event_15.terminal = False
_event_10.direction = -1.0
_event_10.terminal = True


def _sample_times(end_time_s: float, step_s: float) -> np.ndarray:
    grid = np.arange(0.0, end_time_s + 0.5 * step_s, step_s)
    grid = grid[grid <= end_time_s]
    if grid.size == 0 or not np.isclose(grid[-1], end_time_s, rtol=0.0, atol=1.0e-10):
        grid = np.append(grid, end_time_s)
    return grid


def _stored_energy_J(
    states: np.ndarray,
    config: Problem3Parameters,
    pcm: PCMModel,
    outer_layer_count: int,
) -> np.ndarray:
    params = config.thermal
    reference = params.reference_temperature_C
    T_core = states[0]
    T_skin = states[1]
    T_inner = states[2]
    T_pcm = states[3]
    T_outer = states[4 : 4 + outer_layer_count]
    actual_flow = states[-1]
    alpha = np.asarray(alpha_skin_from_blood_flow(actual_flow), dtype=float)
    body = params.body_mass_kg * params.body_heat_capacity_J_kgK * (
        (1.0 - alpha) * (T_core - reference)
        + alpha * (T_skin - reference)
    )
    inner = params.layer1_capacity_J_K * (T_inner - reference)
    pcm_energy = params.pcm_mass_kg * np.asarray(
        pcm.specific_enthalpy_J_kg(
            T_pcm,
            params.dsc_scan_rate_K_min,
            reference,
        ),
        dtype=float,
    )
    outer = config.outer_coating_capacity_J_K * np.sum(
        T_outer - reference,
        axis=0,
    )
    return body + inner + pcm_energy + outer


def _energy_audit(
    solution: object,
    config: Problem3Parameters,
    pcm: PCMModel,
    outer_layer_count: int,
    output_times_s: np.ndarray,
    audit_step_s: float,
) -> tuple[np.ndarray, float, float]:
    params = config.thermal
    audit_times = _sample_times(float(solution.t[-1]), audit_step_s)
    states = solution.sol(audit_times)
    external_power = np.empty_like(audit_times)
    for index, state in enumerate(states.T):
        diagnostics = state_diagnostics(state, config, pcm, outer_layer_count)
        external_power[index] = params.heat_transfer_area_m2 * (
            params.metabolic_rate_W_m2
            - float(diagnostics["q_res_total_W_m2"])
            - float(diagnostics["q_outer_air_W_m2"])
        )
    cumulative_external = cumulative_simpson(
        external_power,
        x=audit_times,
        initial=0.0,
    )
    stored = _stored_energy_J(states, config, pcm, outer_layer_count)
    residual = stored - stored[0] - cumulative_external
    absolute_external = cumulative_simpson(
        np.abs(external_power),
        x=audit_times,
        initial=0.0,
    )
    scale = max(
        float(absolute_external[-1]),
        float(np.max(np.abs(stored - stored[0]))),
        1.0,
    )
    maximum = float(np.max(np.abs(residual)))
    return np.interp(output_times_s, audit_times, residual), maximum, maximum / scale


def _event_time(solution: object, event_index: int) -> float:
    if len(solution.t_events[event_index]) == 0:
        return float("nan")
    return float(solution.t_events[event_index][0])


def run_candidate(
    config: Problem3Parameters,
    pcm: PCMModel,
    outer_layer_count: int,
    *,
    method: str = "RK45",
    rtol: float = 1.0e-7,
    atol: float = 1.0e-9,
    max_step_s: float = 5.0,
    horizon_s: float = 86400.0,
    output_step_s: float = 30.0,
    audit_step_s: float = 1.0,
) -> Problem3CaseResult:
    """Solve one budget-feasible coating design and evaluate its objective."""

    config.validate()
    if not config.is_budget_feasible(outer_layer_count):
        raise ValueError("Candidate exceeds the permitted material budget")
    y0 = initial_state(config, outer_layer_count)
    solution = solve_ivp(
        fun=lambda time, state: rhs(
            time,
            state,
            config,
            pcm,
            outer_layer_count,
        ),
        t_span=(0.0, horizon_s),
        y0=y0,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step_s,
        events=(_event_15, _event_10),
        dense_output=True,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"Problem 3 solver failed: {solution.message}")
    if not np.all(np.isfinite(solution.y)):
        raise RuntimeError("Problem 3 solver returned NaN or Inf")

    output_times = _sample_times(float(solution.t[-1]), output_step_s)
    states = solution.sol(output_times)
    garment_mass = config.garment_mass_kg(outer_layer_count)
    load_time = config.load_capacity_time_s(outer_layer_count)
    rows: list[dict[str, float | str]] = []
    for time_s, state in zip(output_times, states.T, strict=True):
        diagnostics = state_diagnostics(state, config, pcm, outer_layer_count)
        row: dict[str, float | str] = {
            "outer_layer_count": outer_layer_count,
            "outer_thickness_mm": 0.3 * outer_layer_count,
            "time_s": float(time_s),
            "T_core_C": float(state[0]),
            "T_skin_C": float(state[1]),
            "T_layer1_C": float(state[2]),
            "T_pcm_C": float(state[3]),
            "T_outer_surface_C": float(state[3 + outer_layer_count]),
            "skin_blood_flow_eq_L_m2_h": float(
                diagnostics["skin_blood_flow_eq_L_m2_h"]
            ),
            "skin_blood_flow_actual_L_m2_h": float(
                diagnostics["skin_blood_flow_actual_L_m2_h"]
            ),
            "alpha_skin": float(diagnostics["alpha_skin"]),
            "q_outer_air_W_m2": float(diagnostics["q_outer_air_W_m2"]),
            "h_out_W_m2K": float(diagnostics["h_out_W_m2K"]),
            "c_eff_pcm_J_kgK": float(diagnostics["c_eff_pcm_J_kgK"]),
            "pcm_phase_fraction": float(diagnostics["pcm_phase_fraction"]),
            "garment_mass_kg": garment_mass,
            "allowable_load_kg": max(
                0.0,
                config.maximum_load_initial_kg
                - config.load_loss_rate_kg_s * float(time_s),
            ),
            "load_margin_kg": (
                config.maximum_load_initial_kg
                - config.load_loss_rate_kg_s * float(time_s)
                - garment_mass
            ),
        }
        for index, temperature in enumerate(
            state[4 : 4 + outer_layer_count],
            start=1,
        ):
            row[f"T_outer_{index}_C"] = float(temperature)
        rows.append(row)
    timeseries = pd.DataFrame(rows)
    residual, maximum_residual, relative_error = _energy_audit(
        solution,
        config,
        pcm,
        outer_layer_count,
        output_times,
        audit_step_s,
    )
    timeseries["energy_balance_residual_J"] = residual

    t15 = _event_time(solution, 0)
    t10 = _event_time(solution, 1)
    thermal_limit = t15 if np.isfinite(t15) else float("inf")
    weight_penalty = config.weight_penalty_s(outer_layer_count)
    objective_time = max(0.0, thermal_limit - weight_penalty)
    hard_constraint_time = min(thermal_limit, load_time)
    hard_constraint_active = (
        "thermal_15C" if thermal_limit <= load_time else "declining_load"
    )
    objective_state = np.asarray(solution.sol(objective_time), dtype=float)
    objective_diagnostics = state_diagnostics(
        objective_state,
        config,
        pcm,
        outer_layer_count,
    )
    summary: dict[str, float | str] = {
        "outer_layer_count": outer_layer_count,
        "added_outer_layer_count": outer_layer_count - 1,
        "outer_thickness_mm": 0.3 * outer_layer_count,
        "state_dimension": len(y0),
        "garment_mass_kg": garment_mass,
        "baseline_cost_yuan": config.baseline_cost_yuan,
        "added_cost_yuan": config.added_cost_yuan(outer_layer_count),
        "total_cost_yuan": config.total_cost_yuan(outer_layer_count),
        "maximum_total_cost_yuan": config.maximum_total_cost_yuan,
        "budget_utilization_pct": (
            config.added_cost_yuan(outer_layer_count)
            / (config.maximum_total_cost_yuan - config.baseline_cost_yuan)
            * 100.0
        ),
        "weight_penalty_s": weight_penalty,
        "weight_penalty_min": weight_penalty / 60.0,
        "load_capacity_time_s": load_time,
        "load_capacity_time_min": load_time / 60.0,
        "t15_s": t15,
        "t15_min": t15 / 60.0,
        "t10_s": t10,
        "t10_min": t10 / 60.0,
        "objective_mode": "soft_weight_penalty",
        "objective_standing_time_s": objective_time,
        "objective_standing_time_min": objective_time / 60.0,
        "hard_constraint_standing_time_s": hard_constraint_time,
        "hard_constraint_standing_time_min": hard_constraint_time / 60.0,
        "hard_constraint_active": hard_constraint_active,
        "T_skin_at_objective_time_C": float(objective_state[1]),
        "T_core_at_objective_time_C": float(objective_state[0]),
        "T_pcm_at_objective_time_C": float(objective_state[3]),
        "pcm_phase_fraction_at_objective_time": float(
            objective_diagnostics["pcm_phase_fraction"]
        ),
        "max_abs_energy_balance_residual_J": maximum_residual,
        "relative_energy_balance_error": relative_error,
        "solver_method": method,
        "rtol": rtol,
        "atol": atol,
        "max_step_s": max_step_s,
    }
    return Problem3CaseResult(
        outer_layer_count=outer_layer_count,
        timeseries=timeseries,
        summary=summary,
        solution=solution,
    )

