"""Human thermoregulation and clothing heat-transfer equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelParameters
from .dsc import PCMModel


@dataclass(frozen=True)
class HumanDerivativeResult:
    """Temperature derivatives and mass-redistribution diagnostics."""

    core_temperature_rate_C_s: float
    skin_temperature_rate_C_s: float
    alpha_rate_1_s: float
    mass_core_to_skin_kg_s: float
    enthalpy_mass_transfer_W: float
    mass_redistribution_correction_W: float
    branch: str


def skin_blood_flow(T_skin_C: float | np.ndarray) -> float | np.ndarray:
    """Return skin blood flow in L/(m2 h) from skin temperature in degC."""

    temperature = np.asarray(T_skin_C, dtype=float)
    cold_signal = np.maximum(0.0, 33.7 - temperature)
    flow = np.clip(6.3 / (1.0 + 0.5 * cold_signal), 0.5, 90.0)
    return float(flow) if flow.ndim == 0 else flow


def alpha_skin_from_blood_flow(flow_L_m2_h: float | np.ndarray) -> float | np.ndarray:
    """Return the dimensionless Gagge/Pierce skin mass fraction."""

    flow = np.asarray(flow_L_m2_h, dtype=float)
    alpha = 0.0417737 + 0.7451833 / (flow + 0.585417)
    return float(alpha) if alpha.ndim == 0 else alpha


def alpha_skin(T_skin_C: float | np.ndarray) -> float | np.ndarray:
    """Return skin mass fraction as an algebraic function of skin temperature in degC."""

    return alpha_skin_from_blood_flow(skin_blood_flow(T_skin_C))


def dalpha_dTskin(T_skin_C: float) -> float:
    """Return analytic d(alpha_skin)/d(T_skin) in 1/K with clip-aware branches."""

    if T_skin_C >= 33.7:
        return 0.0
    denominator = 1.0 + 0.5 * (33.7 - T_skin_C)
    raw_flow = 6.3 / denominator
    if raw_flow <= 0.5 or raw_flow >= 90.0:
        return 0.0
    dflow_dtemperature = 3.15 / denominator**2
    return -0.7451833 * dflow_dtemperature / (raw_flow + 0.585417) ** 2


def respiratory_heat_loss(
    metabolic_rate_W_m2: float, air_temperature_C: float, vapor_pressure_Torr: float
) -> tuple[float, float, float]:
    """Return latent, sensible and total respiratory losses in W/m2."""

    latent = 0.0023 * metabolic_rate_W_m2 * (44.0 - vapor_pressure_Torr)
    sensible = 0.0014 * metabolic_rate_W_m2 * (34.0 - air_temperature_C)
    return latent, sensible, latent + sensible


def heat_fluxes(state: np.ndarray, params: ModelParameters) -> dict[str, float]:
    """Return all model heat flux densities in W/m2 and h_out in W/(m2 K)."""

    T_core, T_skin, T_layer1, T_pcm, T_layer3 = (float(value) for value in state)
    flow = float(skin_blood_flow(T_skin))
    G_cs = 5.28 + 1.163 * flow
    q_cs = G_cs * (T_core - T_skin)

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
    h_out = params.lambda_out * 2.38 * max(abs(difference_to_air), 1.0e-12) ** 0.25
    r_3inf = params.layer3.thickness_m / (2.0 * params.layer3.conductivity_W_mK) + 1.0 / h_out

    latent, sensible, total = respiratory_heat_loss(
        params.metabolic_rate_W_m2, params.air_temperature_C, params.vapor_pressure_Torr
    )
    return {
        "skin_blood_flow_L_m2_h": flow,
        "G_cs_W_m2K": G_cs,
        "q_cs_W_m2": q_cs,
        "q_res_lat_W_m2": latent,
        "q_res_sens_W_m2": sensible,
        "q_res_total_W_m2": total,
        "q_s1_W_m2": (T_skin - T_layer1) / r_s1,
        "q_12_W_m2": (T_layer1 - T_pcm) / r_12,
        "q_23_W_m2": (T_pcm - T_layer3) / r_23,
        "q_3inf_W_m2": difference_to_air / r_3inf,
        "h_out_W_m2K": h_out,
    }


def human_temperature_derivatives(
    T_core_C: float,
    T_skin_C: float,
    F_core_W: float,
    F_skin_W: float,
    params: ModelParameters,
) -> HumanDerivativeResult:
    """Solve the conservative dynamic-mass human derivative branches in degC/s."""

    alpha = float(alpha_skin(T_skin_C))
    derivative = dalpha_dTskin(T_skin_C)
    mb_cb = params.body_mass_kg * params.body_heat_capacity_J_kgK
    C_skin = alpha * mb_cb
    C_core = (1.0 - alpha) * mb_cb
    if derivative == 0.0:
        core_rate = F_core_W / C_core
        skin_rate = F_skin_W / C_skin
        alpha_rate = 0.0
        branch = "zero_derivative"
    else:
        coupling = mb_cb * (T_core_C - T_skin_C) * derivative
        candidate_skin_rate = F_skin_W / (C_skin - coupling)
        candidate_alpha_rate = derivative * candidate_skin_rate
        if candidate_alpha_rate >= 0.0:
            core_rate = F_core_W / C_core
            skin_rate = candidate_skin_rate
            alpha_rate = candidate_alpha_rate
            branch = "core_to_skin"
        else:
            skin_rate = F_skin_W / C_skin
            alpha_rate = derivative * skin_rate
            core_rate = (
                F_core_W + mb_cb * (T_core_C - T_skin_C) * alpha_rate
            ) / C_core
            branch = "skin_to_core"

    mass_rate = params.body_mass_kg * alpha_rate
    upstream_temperature = T_core_C if mass_rate >= 0.0 else T_skin_C
    enthalpy_transfer = (
        params.body_heat_capacity_J_kgK
        * (upstream_temperature - params.reference_temperature_C)
        * mass_rate
    )
    correction = mb_cb * (T_core_C - T_skin_C) * alpha_rate
    return HumanDerivativeResult(
        core_temperature_rate_C_s=core_rate,
        skin_temperature_rate_C_s=skin_rate,
        alpha_rate_1_s=alpha_rate,
        mass_core_to_skin_kg_s=mass_rate,
        enthalpy_mass_transfer_W=enthalpy_transfer,
        mass_redistribution_correction_W=correction,
        branch=branch,
    )


def state_diagnostics(
    state: np.ndarray, params: ModelParameters, pcm: PCMModel
) -> dict[str, float | str]:
    """Return the complete set of instantaneous physical diagnostics and units implied by keys."""

    T_core, T_skin, _, T_pcm, _ = (float(value) for value in state)
    flux = heat_fluxes(state, params)
    alpha = float(alpha_skin(T_skin))
    mb_cb = params.body_mass_kg * params.body_heat_capacity_J_kgK
    F_core = params.heat_transfer_area_m2 * (
        params.metabolic_rate_W_m2 - flux["q_res_total_W_m2"] - flux["q_cs_W_m2"]
    )
    F_skin = params.heat_transfer_area_m2 * (
        flux["q_cs_W_m2"] - flux["q_s1_W_m2"]
    )
    human = human_temperature_derivatives(T_core, T_skin, F_core, F_skin, params)
    phase_fraction = float(pcm.phase_fraction_released(T_pcm))
    latent_heat = pcm.latent_heat_J_kg(params.dsc_scan_rate_K_min)
    return {
        **flux,
        "alpha_skin": alpha,
        "alpha_dot_1_s": human.alpha_rate_1_s,
        "C_core_J_K": (1.0 - alpha) * mb_cb,
        "C_skin_J_K": alpha * mb_cb,
        "c_eff_pcm_J_kgK": float(
            pcm.effective_heat_capacity_J_kgK(T_pcm, params.dsc_scan_rate_K_min)
        ),
        "pcm_phase_fraction": phase_fraction,
        "pcm_latent_released_J": params.pcm_mass_kg * latent_heat * phase_fraction,
        "dm_core_to_skin_kg_s": human.mass_core_to_skin_kg_s,
        "enthalpy_mass_transfer_W": human.enthalpy_mass_transfer_W,
        "mass_redistribution_correction_W": human.mass_redistribution_correction_W,
        "human_mass_branch": human.branch,
    }


def rhs(
    _time_s: float, state: np.ndarray, params: ModelParameters, pcm: PCMModel
) -> np.ndarray:
    """Return five-node temperature rates in degC/s for solve_ivp."""

    diagnostics = state_diagnostics(state, params, pcm)
    T_core, T_skin, _, _, _ = (float(value) for value in state)
    F_core = params.heat_transfer_area_m2 * (
        params.metabolic_rate_W_m2
        - float(diagnostics["q_res_total_W_m2"])
        - float(diagnostics["q_cs_W_m2"])
    )
    F_skin = params.heat_transfer_area_m2 * (
        float(diagnostics["q_cs_W_m2"]) - float(diagnostics["q_s1_W_m2"])
    )
    human = human_temperature_derivatives(T_core, T_skin, F_core, F_skin, params)
    area = params.heat_transfer_area_m2
    layer1_rate = area * (
        float(diagnostics["q_s1_W_m2"]) - float(diagnostics["q_12_W_m2"])
    ) / params.layer1_capacity_J_K
    pcm_rate = area * (
        float(diagnostics["q_12_W_m2"]) - float(diagnostics["q_23_W_m2"])
    ) / (params.pcm_mass_kg * float(diagnostics["c_eff_pcm_J_kgK"]))
    layer3_rate = area * (
        float(diagnostics["q_23_W_m2"]) - float(diagnostics["q_3inf_W_m2"])
    ) / params.layer3_capacity_J_K
    return np.array(
        [
            human.core_temperature_rate_C_s,
            human.skin_temperature_rate_C_s,
            layer1_rate,
            pcm_rate,
            layer3_rate,
        ],
        dtype=float,
    )
