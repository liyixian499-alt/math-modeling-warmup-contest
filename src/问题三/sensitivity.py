"""Reproducible sensitivity analysis for the Problem 3 optimum."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from src.问题一.dsc import PCMModel

from .config import Problem3Parameters
from .model import initial_state, rhs


def _core_35_time_s(
    config: Problem3Parameters,
    pcm: PCMModel,
    outer_layer_count: int,
    *,
    rtol: float = 2.0e-7,
    atol: float = 2.0e-9,
    max_step_s: float = 10.0,
    horizon_s: float = 30000.0,
) -> float:
    """Integrate only until core temperature first reaches 35 degC."""

    def core_35_event(_time_s: float, state: np.ndarray) -> float:
        return float(state[0] - 35.0)

    core_35_event.direction = -1.0
    core_35_event.terminal = True
    solution = solve_ivp(
        fun=lambda time, state: rhs(
            time,
            state,
            config,
            pcm,
            outer_layer_count,
        ),
        t_span=(0.0, horizon_s),
        y0=initial_state(config, outer_layer_count),
        method="RK45",
        rtol=rtol,
        atol=atol,
        max_step=max_step_s,
        events=core_35_event,
    )
    if not solution.success:
        raise RuntimeError(f"Core-35 sensitivity solve failed: {solution.message}")
    if len(solution.t_events[0]) == 0:
        raise RuntimeError("Core temperature did not reach 35 degC in sensitivity horizon")
    return float(solution.t_events[0][0])


def _thermal_parameter_sensitivity(
    config: Problem3Parameters,
    pcm: PCMModel,
    outer_layer_count: int,
    baseline_time_s: float,
) -> pd.DataFrame:
    thermal = config.thermal
    cases: list[tuple[str, str, float, float, Problem3Parameters]] = []

    def add_scalar_cases(
        parameter: str,
        field_name: str,
        factors: tuple[float, float],
    ) -> None:
        baseline_value = float(getattr(thermal, field_name))
        for factor in factors:
            varied = thermal.with_updates(**{field_name: baseline_value * factor})
            cases.append(
                (
                    parameter,
                    f"{factor:.1f}x",
                    factor,
                    baseline_value * factor,
                    replace(config, thermal=varied),
                )
            )

    add_scalar_cases("metabolic_rate_W_m2", "metabolic_rate_W_m2", (0.9, 1.1))
    add_scalar_cases("h_in_W_m2K", "h_in_W_m2K", (0.9, 1.1))
    add_scalar_cases("lambda_out", "lambda_out", (0.9, 1.1))
    add_scalar_cases(
        "blood_flow_time_constant_s",
        "blood_flow_time_constant_s",
        (0.8, 1.2),
    )
    for factor in (0.9, 1.1):
        varied_layer = replace(
            thermal.layer3,
            conductivity_W_mK=thermal.layer3.conductivity_W_mK * factor,
        )
        cases.append(
            (
                "outer_conductivity_W_mK",
                f"{factor:.1f}x",
                factor,
                varied_layer.conductivity_W_mK,
                replace(config, thermal=thermal.with_updates(layer3=varied_layer)),
            )
        )

    rows: list[dict[str, float | str]] = []
    for parameter, scenario, factor, value, varied_config in cases:
        event_time = _core_35_time_s(varied_config, pcm, outer_layer_count)
        rows.append(
            {
                "parameter": parameter,
                "scenario": scenario,
                "factor": factor,
                "parameter_value": value,
                "t_core_35_s": event_time,
                "t_core_35_min": event_time / 60.0,
                "relative_change_pct": 100.0
                * (event_time - baseline_time_s)
                / baseline_time_s,
            }
        )
    return pd.DataFrame(rows)


def _initial_temperature_sensitivity(
    config: Problem3Parameters,
    pcm: PCMModel,
    outer_layer_count: int,
    baseline_time_s: float,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for temperature_C in (20.0, 25.0, 30.0, 37.0):
        if np.isclose(temperature_C, 37.0):
            event_time = baseline_time_s
        else:
            thermal = config.thermal.with_updates(
                initial_layer1_temperature_C=temperature_C,
                initial_pcm_temperature_C=temperature_C,
                initial_layer3_temperature_C=temperature_C,
            )
            event_time = _core_35_time_s(
                replace(config, thermal=thermal),
                pcm,
                outer_layer_count,
            )
        rows.append(
            {
                "clothing_initial_temperature_C": temperature_C,
                "t_core_35_s": event_time,
                "t_core_35_min": event_time / 60.0,
                "relative_change_pct": 100.0
                * (event_time - baseline_time_s)
                / baseline_time_s,
            }
        )
    return pd.DataFrame(rows)


def _load_effect_sensitivity(
    config: Problem3Parameters,
    pcm: PCMModel,
    outer_layer_count: int,
    baseline_time_s: float,
) -> pd.DataFrame:
    thermal = config.thermal
    added_mass = config.added_garment_mass_kg(outer_layer_count)
    rows: list[dict[str, float | str]] = []

    for kappa_W_kg in (0.0, 2.0, 5.0, 10.0):
        effective_metabolic_rate = (
            thermal.metabolic_rate_W_m2
            + kappa_W_kg * added_mass / thermal.heat_transfer_area_m2
        )
        if kappa_W_kg == 0.0:
            event_time = baseline_time_s
        else:
            varied = thermal.with_updates(
                metabolic_rate_W_m2=effective_metabolic_rate
            )
            event_time = _core_35_time_s(
                replace(config, thermal=varied),
                pcm,
                outer_layer_count,
            )
        rows.append(
            {
                "effect": "load_metabolism",
                "coefficient": kappa_W_kg,
                "coefficient_unit": "W/kg",
                "effective_parameter_value": effective_metabolic_rate,
                "effective_parameter_unit": "W/m2",
                "t_core_35_s": event_time,
                "t_core_35_min": event_time / 60.0,
                "relative_change_pct": 100.0
                * (event_time - baseline_time_s)
                / baseline_time_s,
            }
        )

    for gamma_per_kg in (0.0, 0.1, 0.2, 0.5):
        effective_h_in = thermal.h_in_W_m2K * (
            1.0 + gamma_per_kg * added_mass
        )
        if gamma_per_kg == 0.0:
            event_time = baseline_time_s
        else:
            varied = thermal.with_updates(h_in_W_m2K=effective_h_in)
            event_time = _core_35_time_s(
                replace(config, thermal=varied),
                pcm,
                outer_layer_count,
            )
        rows.append(
            {
                "effect": "compression_inner_heat_transfer",
                "coefficient": gamma_per_kg,
                "coefficient_unit": "1/kg",
                "effective_parameter_value": effective_h_in,
                "effective_parameter_unit": "W/(m2 K)",
                "t_core_35_s": event_time,
                "t_core_35_min": event_time / 60.0,
                "relative_change_pct": 100.0
                * (event_time - baseline_time_s)
                / baseline_time_s,
            }
        )
    return pd.DataFrame(rows)


def _weight_penalty_sensitivity(
    config: Problem3Parameters,
    candidate_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool]] = []
    for coefficient in (0.0, 20.0, 1000.0, 2000.0, 4000.0, 6000.0, 6100.0, 7000.0):
        scores = candidate_summary["t15_s"] - coefficient * candidate_summary[
            "added_garment_mass_kg"
        ]
        optimum_index = int(scores.idxmax())
        for index, candidate in candidate_summary.iterrows():
            score = float(scores.loc[index])
            rows.append(
                {
                    "weight_penalty_s_per_kg": coefficient,
                    "outer_layer_count": int(candidate["outer_layer_count"]),
                    "standing_time_score_s": score,
                    "standing_time_score_min": score / 60.0,
                    "selected": bool(index == optimum_index),
                    "is_problem_coefficient": bool(
                        np.isclose(coefficient, config.weight_penalty_s_per_kg)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _manufacturing_cost_sensitivity(
    config: Problem3Parameters,
    candidate_summary: pd.DataFrame,
) -> pd.DataFrame:
    direct_added = candidate_summary["added_cost_yuan"].to_numpy(dtype=float)
    rows: list[dict[str, float | int | bool]] = []
    threshold = (
        config.maximum_total_cost_yuan
        - float(
            candidate_summary.loc[
                candidate_summary["outer_layer_count"] == 4,
                "total_cost_yuan",
            ].iloc[0]
        )
    ) / float(
        candidate_summary.loc[
            candidate_summary["outer_layer_count"] == 4,
            "added_cost_yuan",
        ].iloc[0]
    )
    rates = (0.0, 0.05, 0.10, threshold, 0.15, 0.20)
    for overhead_rate in rates:
        effective_costs = (
            candidate_summary["total_cost_yuan"].to_numpy(dtype=float)
            + overhead_rate * direct_added
        )
        feasible = effective_costs <= config.maximum_total_cost_yuan + 1.0e-9
        feasible_scores = candidate_summary["standing_time_score_s"].where(
            feasible,
            -np.inf,
        )
        optimum_index = int(feasible_scores.idxmax())
        for position, (index, candidate) in enumerate(candidate_summary.iterrows()):
            rows.append(
                {
                    "manufacturing_overhead_rate": overhead_rate,
                    "outer_layer_count": int(candidate["outer_layer_count"]),
                    "effective_total_cost_yuan": float(effective_costs[position]),
                    "maximum_total_cost_yuan": config.maximum_total_cost_yuan,
                    "feasible": bool(feasible[position]),
                    "selected": bool(index == optimum_index),
                }
            )
    return pd.DataFrame(rows)


def run_sensitivity_analysis(
    config: Problem3Parameters,
    pcm: PCMModel,
    candidate_summary: pd.DataFrame,
    optimum_outer_layer_count: int,
    optimum_core_35_time_s: float,
) -> dict[str, pd.DataFrame]:
    """Run all quantitative and scenario sensitivities used in the report."""

    return {
        "thermal": _thermal_parameter_sensitivity(
            config,
            pcm,
            optimum_outer_layer_count,
            optimum_core_35_time_s,
        ),
        "initial_temperature": _initial_temperature_sensitivity(
            config,
            pcm,
            optimum_outer_layer_count,
            optimum_core_35_time_s,
        ),
        "load_effects": _load_effect_sensitivity(
            config,
            pcm,
            optimum_outer_layer_count,
            optimum_core_35_time_s,
        ),
        "weight_penalty": _weight_penalty_sensitivity(config, candidate_summary),
        "manufacturing_cost": _manufacturing_cost_sensitivity(
            config,
            candidate_summary,
        ),
    }

