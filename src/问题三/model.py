"""Variable-dimension thermal model for multiple outer insulation coatings."""

from __future__ import annotations

import numpy as np

from src.问题一.dsc import PCMModel
from src.问题一.model import (
    alpha_skin_from_blood_flow,
    dalpha_d_blood_flow,
    equilibrium_skin_blood_flow,
    human_temperature_derivatives_dynamic_sbf,
    respiratory_heat_loss,
)

from .config import Problem3Parameters


def state_dimension(outer_layer_count: int) -> int:
    """Return core, skin, inner, PCM, outer nodes and actual blood-flow dimension."""

    if outer_layer_count < 1:
        raise ValueError("At least one outer coating is required")
    return outer_layer_count + 5


def initial_state(config: Problem3Parameters, outer_layer_count: int) -> np.ndarray:
    """Return the initial state with one temperature node per outer coating."""

    thermal = config.thermal
    outer = [thermal.initial_layer3_temperature_C] * outer_layer_count
    return np.array(
        [
            thermal.initial_core_temperature_C,
            thermal.initial_skin_temperature_C,
            thermal.initial_layer1_temperature_C,
            thermal.initial_pcm_temperature_C,
            *outer,
            thermal.initial_skin_blood_flow_L_m2_h,
        ],
        dtype=float,
    )


def state_diagnostics(
    state: np.ndarray,
    config: Problem3Parameters,
    pcm: PCMModel,
    outer_layer_count: int,
) -> dict[str, float | str | np.ndarray]:
    """Return heat fluxes, human regulation and all state derivatives."""

    if len(state) != state_dimension(outer_layer_count):
        raise ValueError("State dimension does not match outer coating count")

    params = config.thermal
    T_core = float(state[0])
    T_skin = float(state[1])
    T_inner = float(state[2])
    T_pcm = float(state[3])
    T_outer = np.asarray(state[4 : 4 + outer_layer_count], dtype=float)
    actual_flow = float(state[-1])

    equilibrium_flow = float(equilibrium_skin_blood_flow(T_skin))
    flow_rate = (
        equilibrium_flow - actual_flow
    ) / params.blood_flow_time_constant_s
    alpha = float(alpha_skin_from_blood_flow(actual_flow))
    alpha_rate = float(dalpha_d_blood_flow(actual_flow)) * flow_rate

    G_cs = 5.28 + 1.163 * actual_flow
    q_cs = G_cs * (T_core - T_skin)
    r_s1 = 1.0 / params.h_in_W_m2K + params.layer1.thickness_m / (
        2.0 * params.layer1.conductivity_W_mK
    )
    r_12 = params.layer1.thickness_m / (
        2.0 * params.layer1.conductivity_W_mK
    ) + params.pcm_layer.thickness_m / (
        2.0 * params.pcm_layer.conductivity_W_mK
    )
    r_pcm_outer = params.pcm_layer.thickness_m / (
        2.0 * params.pcm_layer.conductivity_W_mK
    ) + params.layer3.thickness_m / (
        2.0 * params.layer3.conductivity_W_mK
    )
    r_outer_interface = params.layer3.thickness_m / params.layer3.conductivity_W_mK

    q_s1 = (T_skin - T_inner) / r_s1
    q_12 = (T_inner - T_pcm) / r_12
    q_pcm_outer = (T_pcm - T_outer[0]) / r_pcm_outer
    if outer_layer_count > 1:
        q_outer_interfaces = (T_outer[:-1] - T_outer[1:]) / r_outer_interface
    else:
        q_outer_interfaces = np.empty(0, dtype=float)

    difference_to_air = float(T_outer[-1] - params.air_temperature_C)
    h_out = params.lambda_out * 2.38 * max(abs(difference_to_air), 1.0e-12) ** 0.25
    r_outer_air = params.layer3.thickness_m / (
        2.0 * params.layer3.conductivity_W_mK
    ) + 1.0 / h_out
    q_outer_air = difference_to_air / r_outer_air

    q_res_lat, q_res_sens, q_res_total = respiratory_heat_loss(
        params.metabolic_rate_W_m2,
        params.air_temperature_C,
        params.vapor_pressure_Torr,
    )
    area = params.heat_transfer_area_m2
    F_core = area * (params.metabolic_rate_W_m2 - q_res_total - q_cs)
    F_skin = area * (q_cs - q_s1)
    human = human_temperature_derivatives_dynamic_sbf(
        T_core,
        T_skin,
        actual_flow,
        flow_rate,
        F_core,
        F_skin,
        params,
    )

    c_eff = float(
        pcm.effective_heat_capacity_J_kgK(T_pcm, params.dsc_scan_rate_K_min)
    )
    inner_rate = area * (q_s1 - q_12) / params.layer1_capacity_J_K
    pcm_rate = area * (q_12 - q_pcm_outer) / (params.pcm_mass_kg * c_eff)
    outer_rates = np.empty(outer_layer_count, dtype=float)
    for index in range(outer_layer_count):
        incoming = q_pcm_outer if index == 0 else float(q_outer_interfaces[index - 1])
        outgoing = q_outer_air if index == outer_layer_count - 1 else float(q_outer_interfaces[index])
        outer_rates[index] = (
            area * (incoming - outgoing) / config.outer_coating_capacity_J_K
        )

    mb_cb = params.body_mass_kg * params.body_heat_capacity_J_kgK
    phase_fraction = float(pcm.phase_fraction_released(T_pcm))
    latent_heat = pcm.latent_heat_J_kg(params.dsc_scan_rate_K_min)
    derivatives = np.array(
        [
            human.core_temperature_rate_C_s,
            human.skin_temperature_rate_C_s,
            inner_rate,
            pcm_rate,
            *outer_rates,
            flow_rate,
        ],
        dtype=float,
    )
    return {
        "skin_blood_flow_eq_L_m2_h": equilibrium_flow,
        "skin_blood_flow_actual_L_m2_h": actual_flow,
        "d_skin_blood_flow_dt_L_m2_h_s": flow_rate,
        "tau_blood_flow_s": params.blood_flow_time_constant_s,
        "alpha_skin": alpha,
        "alpha_dot_1_s": alpha_rate,
        "C_core_J_K": (1.0 - alpha) * mb_cb,
        "C_skin_J_K": alpha * mb_cb,
        "G_cs_W_m2K": G_cs,
        "q_cs_W_m2": q_cs,
        "q_res_lat_W_m2": q_res_lat,
        "q_res_sens_W_m2": q_res_sens,
        "q_res_total_W_m2": q_res_total,
        "q_s1_W_m2": q_s1,
        "q_12_W_m2": q_12,
        "q_pcm_outer_W_m2": q_pcm_outer,
        "q_outer_interfaces_W_m2": q_outer_interfaces,
        "q_outer_air_W_m2": q_outer_air,
        "h_out_W_m2K": h_out,
        "c_eff_pcm_J_kgK": c_eff,
        "pcm_phase_fraction": phase_fraction,
        "pcm_latent_released_J": params.pcm_mass_kg * latent_heat * phase_fraction,
        "dm_core_to_skin_kg_s": human.mass_core_to_skin_kg_s,
        "enthalpy_mass_transfer_W": human.enthalpy_mass_transfer_W,
        "mass_redistribution_correction_W": human.mass_redistribution_correction_W,
        "human_mass_branch": human.branch,
        "state_derivatives": derivatives,
    }


def rhs(
    _time_s: float,
    state: np.ndarray,
    config: Problem3Parameters,
    pcm: PCMModel,
    outer_layer_count: int,
) -> np.ndarray:
    """Return the variable-dimension Problem 3 state derivative."""

    return np.asarray(
        state_diagnostics(state, config, pcm, outer_layer_count)["state_derivatives"],
        dtype=float,
    )

