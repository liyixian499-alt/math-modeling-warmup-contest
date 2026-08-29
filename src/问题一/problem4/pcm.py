"""Single-source PCM latent-peak scaling for Problem 4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsc import PCMModel


@dataclass(frozen=True)
class ScaledPCMModel:
    """Wrap a baseline-corrected PCM model and scale only its latent term."""

    base: PCMModel
    lambda_pcm: float

    def __post_init__(self) -> None:
        if self.lambda_pcm < 1.0 or not np.isfinite(self.lambda_pcm):
            raise ValueError("lambda_pcm must be finite and at least one")

    @property
    def base_heat_capacity_J_kgK(self) -> float:
        return self.base.base_heat_capacity_J_kgK

    @property
    def temperature_C(self) -> np.ndarray:
        return self.base.temperature_C

    @property
    def excess_heat_flow_mW_mg(self) -> np.ndarray:
        return self.lambda_pcm * self.base.excess_heat_flow_mW_mg

    @property
    def lower_temperature_C(self) -> float:
        return self.base.lower_temperature_C

    @property
    def upper_temperature_C(self) -> float:
        return self.base.upper_temperature_C

    def q_latent(self, temperature_C: float | np.ndarray) -> float | np.ndarray:
        return self.lambda_pcm * self.base.q_latent(temperature_C)

    def latent_heat_J_kg(self, beta_K_min: float) -> float:
        return self.lambda_pcm * self.base.latent_heat_J_kg(beta_K_min)

    def effective_heat_capacity_J_kgK(
        self, temperature_C: float | np.ndarray, beta_K_min: float
    ) -> float | np.ndarray:
        temperature = np.asarray(temperature_C, dtype=float)
        result = self.base_heat_capacity_J_kgK + (
            self.lambda_pcm
            * 60000.0
            * np.asarray(self.base.q_latent(temperature), dtype=float)
            / beta_K_min
        )
        return float(result) if result.ndim == 0 else result

    def phase_fraction_released(
        self, temperature_C: float | np.ndarray
    ) -> float | np.ndarray:
        return self.base.phase_fraction_released(temperature_C)

    def specific_enthalpy_J_kg(
        self,
        temperature_C: float | np.ndarray,
        beta_K_min: float,
        reference_temperature_C: float = 0.0,
    ) -> float | np.ndarray:
        temperature = np.asarray(temperature_C, dtype=float)
        sensible = self.base_heat_capacity_J_kgK * (
            temperature - reference_temperature_C
        )
        latent = self.latent_heat_J_kg(beta_K_min) * (
            1.0 - np.asarray(self.phase_fraction_released(temperature), dtype=float)
        )
        result = sensible + latent
        return float(result) if result.ndim == 0 else result
