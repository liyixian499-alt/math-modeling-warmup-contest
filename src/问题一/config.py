"""Central parameter definitions for the Problem 1 thermal model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DSC_PATH = REPOSITORY_ROOT / "problem" / "attachments" / "附件1 放热能力数据.xlsx"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "outputs" / "problem1"


@dataclass(frozen=True)
class MaterialLayer:
    """A clothing layer's thickness, heat capacity, conductivity and density in SI units."""

    thickness_m: float
    heat_capacity_J_kgK: float
    conductivity_W_mK: float
    density_kg_m3: float


@dataclass(frozen=True)
class ModelParameters:
    """All configurable model inputs, with temperatures in degC and time in seconds."""

    body_mass_kg: float = 60.0
    body_heat_capacity_J_kgK: float = 3490.0
    heat_transfer_area_m2: float = 1.6521
    material_area_factor: float = 1.25
    air_temperature_C: float = -40.0
    metabolic_rate_W_m2: float = 70.0
    vapor_pressure_Torr: float = 0.0
    reference_temperature_C: float = 0.0
    initial_core_temperature_C: float = 37.0
    initial_skin_temperature_C: float = 37.0
    initial_layer1_temperature_C: float = 37.0
    initial_pcm_temperature_C: float = 37.0
    initial_layer3_temperature_C: float = 37.0
    h_in_W_m2K: float = 3.0
    lambda_out: float = 1.0
    dsc_scan_rate_K_min: float = 10.0
    blood_flow_time_constant_s: float = 173.0
    initial_skin_blood_flow_L_m2_h: float = 6.3
    layer1: MaterialLayer = MaterialLayer(0.7e-3, 4803.8, 0.068, 208.0)
    pcm_layer: MaterialLayer = MaterialLayer(0.4e-3, 2400.0, 0.060, 552.3)
    layer3: MaterialLayer = MaterialLayer(0.3e-3, 5463.2, 0.0527, 300.0)

    @property
    def material_area_m2(self) -> float:
        """Material inventory area in m2; this is used only for layer masses."""

        return self.material_area_factor * self.heat_transfer_area_m2

    @property
    def layer1_mass_kg(self) -> float:
        """Mass of clothing layer 1 in kg."""

        return self.layer1.density_kg_m3 * self.layer1.thickness_m * self.material_area_m2

    @property
    def pcm_mass_kg(self) -> float:
        """Mass of the PCM layer in kg."""

        return self.pcm_layer.density_kg_m3 * self.pcm_layer.thickness_m * self.material_area_m2

    @property
    def layer3_mass_kg(self) -> float:
        """Mass of clothing layer 3 in kg."""

        return self.layer3.density_kg_m3 * self.layer3.thickness_m * self.material_area_m2

    @property
    def layer1_capacity_J_K(self) -> float:
        """Sensible heat capacity of layer 1 in J/K."""

        return self.layer1_mass_kg * self.layer1.heat_capacity_J_kgK

    @property
    def layer3_capacity_J_K(self) -> float:
        """Sensible heat capacity of layer 3 in J/K."""

        return self.layer3_mass_kg * self.layer3.heat_capacity_J_kgK

    def with_updates(self, **updates: float) -> "ModelParameters":
        """Return a copied parameter set with the named fields replaced."""

        return replace(self, **updates)

    def validate(self) -> None:
        """Raise ValueError when a physical or numerical input is invalid."""

        positive = {
            "body_mass_kg": self.body_mass_kg,
            "body_heat_capacity_J_kgK": self.body_heat_capacity_J_kgK,
            "heat_transfer_area_m2": self.heat_transfer_area_m2,
            "material_area_factor": self.material_area_factor,
            "metabolic_rate_W_m2": self.metabolic_rate_W_m2,
            "h_in_W_m2K": self.h_in_W_m2K,
            "lambda_out": self.lambda_out,
            "dsc_scan_rate_K_min": self.dsc_scan_rate_K_min,
            "blood_flow_time_constant_s": self.blood_flow_time_constant_s,
            "initial_skin_blood_flow_L_m2_h": self.initial_skin_blood_flow_L_m2_h,
        }
        invalid = [name for name, value in positive.items() if not value > 0.0]
        if invalid:
            raise ValueError(f"Parameters must be positive: {invalid}")
        if not 0.5 <= self.initial_skin_blood_flow_L_m2_h <= 90.0:
            raise ValueError("Initial skin blood flow must be within [0.5, 90] L/(m2 h)")
