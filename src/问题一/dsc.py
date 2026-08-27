"""DSC data loading and PCM apparent-heat-capacity functions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class PCMModel:
    """PCHIP representation of baseline-corrected DSC heat flow in mW/mg."""

    temperature_C: np.ndarray
    excess_heat_flow_mW_mg: np.ndarray
    interpolator: PchipInterpolator
    antiderivative: object
    integral_mW_mg_K: float
    base_heat_capacity_J_kgK: float = 2400.0

    @property
    def lower_temperature_C(self) -> float:
        """Lower DSC temperature bound in degC."""

        return float(self.temperature_C[0])

    @property
    def upper_temperature_C(self) -> float:
        """Upper DSC temperature bound in degC."""

        return float(self.temperature_C[-1])

    def q_latent(self, temperature_C: float | np.ndarray) -> float | np.ndarray:
        """Return nonnegative baseline-corrected DSC heat flow in mW/mg."""

        values = np.asarray(temperature_C, dtype=float)
        inside = (values >= self.lower_temperature_C) & (values <= self.upper_temperature_C)
        result = np.zeros_like(values, dtype=float)
        if np.any(inside):
            result[inside] = np.maximum(self.interpolator(values[inside]), 0.0)
        return float(result) if result.ndim == 0 else result

    def latent_heat_J_kg(self, beta_K_min: float) -> float:
        """Return latent heat in J/kg for a DSC scan rate in K/min."""

        if beta_K_min <= 0.0:
            raise ValueError("DSC scan rate must be positive")
        return 60000.0 * self.integral_mW_mg_K / beta_K_min

    def effective_heat_capacity_J_kgK(
        self, temperature_C: float | np.ndarray, beta_K_min: float
    ) -> float | np.ndarray:
        """Return PCM apparent heat capacity in J/(kg K)."""

        return self.base_heat_capacity_J_kgK + 60000.0 * self.q_latent(temperature_C) / beta_K_min

    def phase_fraction_released(self, temperature_C: float | np.ndarray) -> float | np.ndarray:
        """Return cooling-process latent-release progress xi between zero and one."""

        values = np.asarray(temperature_C, dtype=float)
        result = np.empty_like(values, dtype=float)
        result[values >= self.upper_temperature_C] = 0.0
        result[values <= self.lower_temperature_C] = 1.0
        inside = (values > self.lower_temperature_C) & (values < self.upper_temperature_C)
        if np.any(inside):
            upper_primitive = float(self.antiderivative(self.upper_temperature_C))
            result[inside] = (
                upper_primitive - self.antiderivative(values[inside])
            ) / self.integral_mW_mg_K
        result = np.clip(result, 0.0, 1.0)
        return float(result) if result.ndim == 0 else result

    def specific_enthalpy_J_kg(
        self,
        temperature_C: float | np.ndarray,
        beta_K_min: float,
        reference_temperature_C: float = 0.0,
    ) -> float | np.ndarray:
        """Return PCM specific enthalpy in J/kg at the configured reference temperature."""

        temperature = np.asarray(temperature_C, dtype=float)
        latent = self.latent_heat_J_kg(beta_K_min)
        result = self.base_heat_capacity_J_kgK * (temperature - reference_temperature_C)
        result = result + latent * (1.0 - self.phase_fraction_released(temperature))
        return float(result) if result.ndim == 0 else result


def load_pcm_model(path: str | Path, base_heat_capacity_J_kgK: float = 2400.0) -> PCMModel:
    """Load the first nonempty XLSX sheet and construct a baseline-corrected PCHIP DSC model."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"DSC raw data file not found: {source}")

    workbook = pd.ExcelFile(source)
    frame = None
    for sheet in workbook.sheet_names:
        candidate = pd.read_excel(source, sheet_name=sheet)
        if candidate.shape[0] > 0 and candidate.shape[1] >= 2:
            frame = candidate.iloc[:, :2].copy()
            break
    if frame is None:
        raise ValueError(f"No nonempty two-column DSC sheet found in {source}")

    frame.columns = ["temperature_C", "heat_flow_mW_mg"]
    frame["temperature_C"] = pd.to_numeric(frame["temperature_C"], errors="coerce")
    frame["heat_flow_mW_mg"] = pd.to_numeric(frame["heat_flow_mW_mg"], errors="coerce")
    frame = frame.dropna().groupby("temperature_C", as_index=False)["heat_flow_mW_mg"].mean()
    frame = frame.sort_values("temperature_C")
    if len(frame) < 3:
        raise ValueError("At least three valid DSC points are required")

    temperatures = frame["temperature_C"].to_numpy(dtype=float)
    magnitudes = np.abs(frame["heat_flow_mW_mg"].to_numpy(dtype=float))
    baseline = magnitudes[0] + (magnitudes[-1] - magnitudes[0]) * (
        temperatures - temperatures[0]
    ) / (temperatures[-1] - temperatures[0])
    excess = np.maximum(magnitudes - baseline, 0.0)
    interpolator = PchipInterpolator(temperatures, excess, extrapolate=False)
    antiderivative = interpolator.antiderivative()
    integral = float(interpolator.integrate(temperatures[0], temperatures[-1]))
    if not integral > 0.0:
        raise ValueError("Baseline-corrected DSC latent peak has nonpositive integral")

    return PCMModel(
        temperature_C=temperatures,
        excess_heat_flow_mW_mg=excess,
        interpolator=interpolator,
        antiderivative=antiderivative,
        integral_mW_mg_K=integral,
        base_heat_capacity_J_kgK=base_heat_capacity_J_kgK,
    )
