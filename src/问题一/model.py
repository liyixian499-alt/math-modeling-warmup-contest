"""Human thermoregulation and clothing heat-transfer equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelParameters
from .dsc import PCMModel


DYNAMIC_SBF = "dynamic_sbf"
QUASISTEADY_SBF = "quasisteady_sbf"
MODEL_VARIANTS = frozenset({DYNAMIC_SBF, QUASISTEADY_SBF})


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


def equilibrium_skin_blood_flow(T_skin_C: float | np.ndarray) -> float | np.ndarray:
    """Return equilibrium skin blood flow in L/(m2 h) at skin temperature in degC."""

    temperature = np.asarray(T_skin_C, dtype=float)
    cold_signal = np.maximum(0.0, 33.7 - temperature)
    flow = np.clip(6.3 / (1.0 + 0.5 * cold_signal), 0.5, 90.0)
    return float(flow) if flow.ndim == 0 else flow


def skin_blood_flow(T_skin_C: float | np.ndarray) -> float | np.ndarray:
    """Backward-compatible alias for equilibrium skin blood flow in L/(m2 h)."""

    return equilibrium_skin_blood_flow(T_skin_C)


def d_equilibrium_blood_flow_dTskin(T_skin_C: float) -> float:
    """Return the clip-aware derivative of equilibrium blood flow versus skin temperature."""

    if T_skin_C >= 33.7:
        return 0.0
    denominator = 1.0 + 0.5 * (33.7 - T_skin_C)
    raw_flow = 6.3 / denominator
    if raw_flow <= 0.5 or raw_flow >= 90.0:
        return 0.0
    return 3.15 / denominator**2


def alpha_skin_from_blood_flow(flow_L_m2_h: float | np.ndarray) -> float | np.ndarray:
    """Return the dimensionless Gagge/Pierce skin mass fraction from actual blood flow."""

    flow = np.asarray(flow_L_m2_h, dtype=float)
    alpha = 0.0417737 + 0.7451833 / (flow + 0.585417)
    return float(alpha) if alpha.ndim == 0 else alpha


def dalpha_d_blood_flow(flow_L_m2_h: float | np.ndarray) -> float | np.ndarray:
    """Return d(alpha_skin)/d(blood_flow) in (m2 h)/L."""

    flow = np.asarray(flow_L_m2_h, dtype=float)
    derivative = -0.7451833 / (flow + 0.585417) ** 2
    return float(derivative) if derivative.ndim == 0 else derivative


def alpha_skin(T_skin_C: float | np.ndarray) -> float | np.ndarray:
    """Return quasi-steady skin mass fraction as a function of skin temperature."""

    return alpha_skin_from_blood_flow(equilibrium_skin_blood_flow(T_skin_C))


def dalpha_dTskin(T_skin_C: float) -> float:
    """Return clip-aware d(alpha_skin)/d(T_skin) for the quasi-steady model only."""

    flow = float(equilibrium_skin_blood_flow(T_skin_C))
    return float(dalpha_d_blood_flow(flow)) * d_equilibrium_blood_flow_dTskin(T_skin_C)


def respiratory_heat_loss(
    metabolic_rate_W_m2: float, air_temperature_C: float, vapor_pressure_Torr: float
) -> tuple[float, float, float]:
    """Return latent, sensible and total respiratory losses in W/m2."""

    latent = 0.0023 * metabolic_rate_W_m2 * (44.0 - vapor_pressure_Torr)
    sensible = 0.0014 * metabolic_rate_W_m2 * (34.0 - air_temperature_C)
    return latent, sensible, latent + sensible


def _blood_flow_terms(
    state: np.ndarray, params: ModelParameters, model_variant: str
) -> tuple[float, float, float, float]:
    """Return equilibrium flow, actual flow, actual-flow rate and effective tau."""

    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {model_variant}")
    T_skin = float(state[1])
    equilibrium = float(equilibrium_skin_blood_flow(T_skin))
    if model_variant == DYNAMIC_SBF:
        if len(state) != 6:
            raise ValueError("dynamic_sbf requires a six-state vector")
        actual = float(state[5])
        flow_rate = (equilibrium - actual) / params.blood_flow_time_constant_s
        tau = params.blood_flow_time_constant_s
    else:
        if len(state) != 5:
            raise ValueError("quasisteady_sbf requires a five-state vector")
        actual = equilibrium
        flow_rate = np.nan
        tau = 0.0
    return equilibrium, actual, flow_rate, tau


def heat_fluxes(
    state: np.ndarray, params: ModelParameters, model_variant: str = DYNAMIC_SBF
) -> dict[str, float]:
    """Return heat flux densities in W/m2 using actual blood flow for core-skin transfer."""

    T_core, T_skin, T_layer1, T_pcm, T_layer3 = (float(value) for value in state[:5])
    equilibrium_flow, actual_flow, flow_rate, tau = _blood_flow_terms(
        state, params, model_variant
    )
    G_cs = 5.28 + 1.163 * actual_flow
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
        "skin_blood_flow_eq_L_m2_h": equilibrium_flow,
        "skin_blood_flow_actual_L_m2_h": actual_flow,
        "skin_blood_flow_L_m2_h": actual_flow,
        "d_skin_blood_flow_dt_L_m2_h_s": flow_rate,
        "tau_blood_flow_s": tau,
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


def _mass_transfer_result(
    T_core_C: float,
    T_skin_C: float,
    F_core_W: float,
    F_skin_W: float,
    alpha: float,
    alpha_rate_1_s: float,
    params: ModelParameters,
) -> HumanDerivativeResult:
    """Apply upstream-enthalpy mass-transfer branches for a known alpha rate."""

    mb_cb = params.body_mass_kg * params.body_heat_capacity_J_kgK
    C_skin = alpha * mb_cb
    C_core = (1.0 - alpha) * mb_cb
    correction = mb_cb * (T_core_C - T_skin_C) * alpha_rate_1_s
    if alpha_rate_1_s >= 0.0:
        core_rate = F_core_W / C_core
        skin_rate = (F_skin_W + correction) / C_skin
        branch = "core_to_skin" if alpha_rate_1_s > 0.0 else "zero_derivative"
    else:
        skin_rate = F_skin_W / C_skin
        core_rate = (F_core_W + correction) / C_core
        branch = "skin_to_core"

    mass_rate = params.body_mass_kg * alpha_rate_1_s
    upstream_temperature = T_core_C if mass_rate >= 0.0 else T_skin_C
    enthalpy_transfer = (
        params.body_heat_capacity_J_kgK
        * (upstream_temperature - params.reference_temperature_C)
        * mass_rate
    )
    return HumanDerivativeResult(
        core_temperature_rate_C_s=core_rate,
        skin_temperature_rate_C_s=skin_rate,
        alpha_rate_1_s=alpha_rate_1_s,
        mass_core_to_skin_kg_s=mass_rate,
        enthalpy_mass_transfer_W=enthalpy_transfer,
        mass_redistribution_correction_W=correction,
        branch=branch,
    )


def human_temperature_derivatives_dynamic_sbf(
    T_core_C: float,
    T_skin_C: float,
    actual_flow_L_m2_h: float,
    flow_rate_L_m2_h_s: float,
    F_core_W: float,
    F_skin_W: float,
    params: ModelParameters,
) -> HumanDerivativeResult:
    """Return explicit human temperature rates for the six-state dynamic-SBF model."""

    alpha = float(alpha_skin_from_blood_flow(actual_flow_L_m2_h))
    alpha_rate = float(dalpha_d_blood_flow(actual_flow_L_m2_h)) * flow_rate_L_m2_h_s
    return _mass_transfer_result(
        T_core_C, T_skin_C, F_core_W, F_skin_W, alpha, alpha_rate, params
    )


def human_temperature_derivatives_quasisteady_sbf(
    T_core_C: float,
    T_skin_C: float,
    F_core_W: float,
    F_skin_W: float,
    params: ModelParameters,
) -> HumanDerivativeResult:
    """Solve the implicit conservative five-state quasi-steady-SBF branches."""

    alpha = float(alpha_skin(T_skin_C))
    derivative = dalpha_dTskin(T_skin_C)
    mb_cb = params.body_mass_kg * params.body_heat_capacity_J_kgK
    C_skin = alpha * mb_cb
    C_core = (1.0 - alpha) * mb_cb
    if derivative == 0.0:
        core_rate = F_core_W / C_core
        skin_rate = F_skin_W / C_skin
        alpha_rate = 0.0
    else:
        coupling = mb_cb * (T_core_C - T_skin_C) * derivative
        candidate_skin_rate = F_skin_W / (C_skin - coupling)
        candidate_alpha_rate = derivative * candidate_skin_rate
        if candidate_alpha_rate >= 0.0:
            core_rate = F_core_W / C_core
            skin_rate = candidate_skin_rate
            alpha_rate = candidate_alpha_rate
        else:
            skin_rate = F_skin_W / C_skin
            alpha_rate = derivative * skin_rate
            core_rate = (F_core_W + mb_cb * (T_core_C - T_skin_C) * alpha_rate) / C_core
    result = _mass_transfer_result(
        T_core_C, T_skin_C, F_core_W, F_skin_W, alpha, alpha_rate, params
    )
    return HumanDerivativeResult(
        core_temperature_rate_C_s=core_rate,
        skin_temperature_rate_C_s=skin_rate,
        alpha_rate_1_s=result.alpha_rate_1_s,
        mass_core_to_skin_kg_s=result.mass_core_to_skin_kg_s,
        enthalpy_mass_transfer_W=result.enthalpy_mass_transfer_W,
        mass_redistribution_correction_W=result.mass_redistribution_correction_W,
        branch=result.branch,
    )


def human_temperature_derivatives(
    T_core_C: float,
    T_skin_C: float,
    F_core_W: float,
    F_skin_W: float,
    params: ModelParameters,
) -> HumanDerivativeResult:
    """Backward-compatible wrapper for the five-state quasi-steady human model."""

    return human_temperature_derivatives_quasisteady_sbf(
        T_core_C, T_skin_C, F_core_W, F_skin_W, params
    )


def state_diagnostics(
    state: np.ndarray,
    params: ModelParameters,
    pcm: PCMModel,
    model_variant: str = DYNAMIC_SBF,
) -> dict[str, float | str]:
    """Return instantaneous diagnostics for an explicitly selected blood-flow model."""

    T_core, T_skin, _, T_pcm, _ = (float(value) for value in state[:5])
    flux = heat_fluxes(state, params, model_variant)
    actual_flow = float(flux["skin_blood_flow_actual_L_m2_h"])
    alpha = float(alpha_skin_from_blood_flow(actual_flow))
    mb_cb = params.body_mass_kg * params.body_heat_capacity_J_kgK
    F_core = params.heat_transfer_area_m2 * (
        params.metabolic_rate_W_m2 - flux["q_res_total_W_m2"] - flux["q_cs_W_m2"]
    )
    F_skin = params.heat_transfer_area_m2 * (
        flux["q_cs_W_m2"] - flux["q_s1_W_m2"]
    )
    if model_variant == DYNAMIC_SBF:
        human = human_temperature_derivatives_dynamic_sbf(
            T_core,
            T_skin,
            actual_flow,
            float(flux["d_skin_blood_flow_dt_L_m2_h_s"]),
            F_core,
            F_skin,
            params,
        )
    else:
        human = human_temperature_derivatives_quasisteady_sbf(
            T_core, T_skin, F_core, F_skin, params
        )
        flux["d_skin_blood_flow_dt_L_m2_h_s"] = (
            d_equilibrium_blood_flow_dTskin(T_skin)
            * human.skin_temperature_rate_C_s
        )

    phase_fraction = float(pcm.phase_fraction_released(T_pcm))
    latent_heat = pcm.latent_heat_J_kg(params.dsc_scan_rate_K_min)
    c_eff = float(pcm.effective_heat_capacity_J_kgK(T_pcm, params.dsc_scan_rate_K_min))
    area = params.heat_transfer_area_m2
    layer1_rate = area * (flux["q_s1_W_m2"] - flux["q_12_W_m2"]) / params.layer1_capacity_J_K
    pcm_rate = area * (flux["q_12_W_m2"] - flux["q_23_W_m2"]) / (
        params.pcm_mass_kg * c_eff
    )
    layer3_rate = area * (flux["q_23_W_m2"] - flux["q_3inf_W_m2"]) / params.layer3_capacity_J_K
    return {
        **flux,
        "alpha_skin": alpha,
        "alpha_dot_1_s": human.alpha_rate_1_s,
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
        "dT_layer1_dt_C_s": layer1_rate,
        "dT_pcm_dt_C_s": pcm_rate,
        "dT_layer3_dt_C_s": layer3_rate,
    }


def rhs_quasisteady_sbf(
    _time_s: float, state: np.ndarray, params: ModelParameters, pcm: PCMModel
) -> np.ndarray:
    """Return five temperature derivatives for the quasi-steady-SBF model."""

    diagnostics = state_diagnostics(state, params, pcm, QUASISTEADY_SBF)
    return np.array(
        [
            diagnostics["dT_core_dt_C_s"],
            diagnostics["dT_skin_dt_C_s"],
            diagnostics["dT_layer1_dt_C_s"],
            diagnostics["dT_pcm_dt_C_s"],
            diagnostics["dT_layer3_dt_C_s"],
        ],
        dtype=float,
    )


def rhs_dynamic_sbf(
    _time_s: float, state: np.ndarray, params: ModelParameters, pcm: PCMModel
) -> np.ndarray:
    """Return five temperature derivatives plus actual skin-blood-flow derivative."""

    diagnostics = state_diagnostics(state, params, pcm, DYNAMIC_SBF)
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


def rhs(
    time_s: float,
    state: np.ndarray,
    params: ModelParameters,
    pcm: PCMModel,
    model_variant: str = DYNAMIC_SBF,
) -> np.ndarray:
    """Dispatch to an explicitly named five- or six-state model variant."""

    if model_variant == DYNAMIC_SBF:
        return rhs_dynamic_sbf(time_s, state, params, pcm)
    if model_variant == QUASISTEADY_SBF:
        return rhs_quasisteady_sbf(time_s, state, params, pcm)
    raise ValueError(f"Unknown model variant: {model_variant}")
