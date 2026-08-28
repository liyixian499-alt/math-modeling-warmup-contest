"""Problem 2 six-state heat-transfer equations with wind forcing."""

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

from .config import ModelParameters


CORE_SKIN_BASE_CONDUCTANCE_W_M2K = 5.28
CORE_SKIN_FLOW_COEFFICIENT_W_H_L_K = 1.163
NATURAL_CONVECTION_COEFFICIENT = 2.38
FORCED_CONVECTION_COEFFICIENT = 12.1


def heat_fluxes(state: np.ndarray, params: ModelParameters) -> dict[str, float]:
    """Return Problem 2 heat fluxes and outside convection coefficients."""

    if len(state) != 6:
        raise ValueError("Problem 2 requires the six-state vector [Tc, Ts, T1, T2, T3, Vbl]")
    T_core, T_skin, T_layer1, T_pcm, T_layer3, actual_flow = (
        float(value) for value in state
    )
    equilibrium_flow = float(equilibrium_skin_blood_flow(T_skin))
    flow_rate = (
        equilibrium_flow - actual_flow
    ) / params.blood_flow_time_constant_s
    conductance = (
        CORE_SKIN_BASE_CONDUCTANCE_W_M2K
        + CORE_SKIN_FLOW_COEFFICIENT_W_H_L_K * actual_flow
    )

    r_s1 = 1.0 / params.h_in_W_m2K + params.layer1.thickness_m / (
        2.0 * params.layer1.conductivity_W_mK
    )
    r_12 = params.layer1.thickness_m / (2.0 * params.layer1.conductivity_W_mK) + (
        params.pcm_layer.thickness_m / (2.0 * params.pcm_layer.conductivity_W_mK)
    )
    r_23 = params.pcm_layer.thickness_m / (2.0 * params.pcm_layer.conductivity_W_mK) + (
        params.layer3.thickness_m / (2.0 * params.layer3.conductivity_W_mK)
    )
    difference_to_air = T_layer3 - params.air_temperature_C
    h_nat = NATURAL_CONVECTION_COEFFICIENT * max(abs(difference_to_air), 1.0e-12) ** 0.25
    h_forced = FORCED_CONVECTION_COEFFICIENT * np.sqrt(params.wind_speed_m_s)
    h_out = max(h_nat, h_forced)
    r_3inf = params.layer3.thickness_m / (2.0 * params.layer3.conductivity_W_mK) + 1.0 / h_out
    respiratory_latent, respiratory_sensible, respiratory_total = respiratory_heat_loss(
        params.metabolic_rate_W_m2,
        params.air_temperature_C,
        params.vapor_pressure_Torr,
    )
    return {
        "skin_blood_flow_eq_L_m2_h": equilibrium_flow,
        "skin_blood_flow_actual_L_m2_h": actual_flow,
        "d_skin_blood_flow_dt_L_m2_h_s": flow_rate,
        "tau_blood_flow_s": params.blood_flow_time_constant_s,
        "G_cs_W_m2K": conductance,
        "q_cs_W_m2": conductance * (T_core - T_skin),
        "q_res_lat_W_m2": respiratory_latent,
        "q_res_sens_W_m2": respiratory_sensible,
        "q_res_total_W_m2": respiratory_total,
        "q_s1_W_m2": (T_skin - T_layer1) / r_s1,
        "q_12_W_m2": (T_layer1 - T_pcm) / r_12,
        "q_23_W_m2": (T_pcm - T_layer3) / r_23,
        "q_3inf_W_m2": difference_to_air / r_3inf,
        "h_nat_W_m2K": h_nat,
        "h_forced_W_m2K": h_forced,
        "h_out_W_m2K": h_out,
    }


def state_diagnostics(
    state: np.ndarray, params: ModelParameters, pcm: PCMModel
) -> dict[str, float | str]:
    """Return all state-dependent rates, capacities, fluxes and PCM diagnostics."""

    T_core, T_skin, _, T_pcm, _, actual_flow = (float(value) for value in state)
    flux = heat_fluxes(state, params)
    alpha = float(alpha_skin_from_blood_flow(actual_flow))
    flow_rate = float(flux["d_skin_blood_flow_dt_L_m2_h_s"])
    alpha_rate = float(dalpha_d_blood_flow(actual_flow)) * flow_rate
    area = params.heat_transfer_area_m2
    F_core = area * (
        params.metabolic_rate_W_m2
        - float(flux["q_res_total_W_m2"])
        - float(flux["q_cs_W_m2"])
    )
    F_skin = area * (float(flux["q_cs_W_m2"]) - float(flux["q_s1_W_m2"]))
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
    phase_fraction = float(pcm.phase_fraction_released(T_pcm))
    latent_heat = pcm.latent_heat_J_kg(params.dsc_scan_rate_K_min)
    mb_cb = params.body_mass_kg * params.body_heat_capacity_J_kgK
    return {
        **flux,
        "alpha_skin": alpha,
        "d_alpha_skin_dt_1_s": alpha_rate,
        "C_core_J_K": (1.0 - alpha) * mb_cb,
        "C_skin_J_K": alpha * mb_cb,
        "c_eff_pcm_J_kgK": c_eff,
        "pcm_phase_fraction": phase_fraction,
        "pcm_latent_released_J": params.pcm_mass_kg * latent_heat * phase_fraction,
        "dm_core_to_skin_kg_s": human.mass_core_to_skin_kg_s,
        "enthalpy_mass_transfer_W": human.enthalpy_mass_transfer_W,
        "mass_redistribution_correction_W": human.mass_redistribution_correction_W,
        "human_mass_branch": human.branch,
        "dT_core_dt_C_s": human.core_temperature_rate_C_s,
        "dT_skin_dt_C_s": human.skin_temperature_rate_C_s,
        "dT_layer1_dt_C_s": area
        * (float(flux["q_s1_W_m2"]) - float(flux["q_12_W_m2"]))
        / params.layer1_capacity_J_K,
        "dT_pcm_dt_C_s": area
        * (float(flux["q_12_W_m2"]) - float(flux["q_23_W_m2"]))
        / (params.pcm_mass_kg * c_eff),
        "dT_layer3_dt_C_s": area
        * (float(flux["q_23_W_m2"]) - float(flux["q_3inf_W_m2"]))
        / params.layer3_capacity_J_K,
    }


def rhs(_time_s: float, state: np.ndarray, params: ModelParameters, pcm: PCMModel) -> np.ndarray:
    """Return the derivatives of the formal Problem 2 six-state ODE."""

    diagnostics = state_diagnostics(state, params, pcm)
    return np.array(
        [
            diagnostics["dT_core_dt_C_s"],
            diagnostics["dT_skin_dt_C_s"],
            diagnostics["dT_layer1_dt_C_s"],
            diagnostics["dT_pcm_dt_C_s"],
            diagnostics["dT_layer3_dt_C_s"],
            diagnostics["d_skin_blood_flow_dt_L_m2_h_s"],
        ],
        dtype=float,
    )
