"""Configuration, cost, mass and added-load constraints for Problem 3."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.问题一.config import ModelParameters


@dataclass(frozen=True)
class Problem3Parameters:
    """Problem 3 design parameters built on the Problem 1 six-state model."""

    thermal: ModelParameters = field(default_factory=ModelParameters)
    baseline_outer_layer_count: int = 1
    maximum_external_load_kg: float = 100.0
    penalty_mass_increment_kg: float = 0.5
    penalty_time_increment_s: float = 10.0
    maximum_budget_increase_fraction: float = 0.5
    inner_price_yuan_per_kg: float = 1000.0
    outer_price_yuan_per_kg: float = 300.0
    pcm_price_yuan_per_m2: float = 10.0

    def validate(self) -> None:
        """Raise ValueError when a design, price or load parameter is invalid."""

        self.thermal.validate()
        if self.baseline_outer_layer_count != 1:
            raise ValueError("The supplied garment must start with exactly one outer coating")
        positive = {
            "maximum_external_load_kg": self.maximum_external_load_kg,
            "penalty_mass_increment_kg": self.penalty_mass_increment_kg,
            "penalty_time_increment_s": self.penalty_time_increment_s,
            "inner_price_yuan_per_kg": self.inner_price_yuan_per_kg,
            "outer_price_yuan_per_kg": self.outer_price_yuan_per_kg,
            "pcm_price_yuan_per_m2": self.pcm_price_yuan_per_m2,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"Problem 3 parameters must be positive: {invalid}")
        if self.maximum_budget_increase_fraction < 0.0:
            raise ValueError("Budget increase fraction cannot be negative")

    @property
    def weight_penalty_s_per_kg(self) -> float:
        """Time penalty per kilogram added relative to the Problem 1 garment."""

        return self.penalty_time_increment_s / self.penalty_mass_increment_kg

    @property
    def outer_coating_mass_kg(self) -> float:
        """Mass of one full-area 0.3 mm outer coating."""

        return self.thermal.layer3_mass_kg

    @property
    def outer_coating_capacity_J_K(self) -> float:
        """Heat capacity of one outer coating."""

        return self.thermal.layer3_capacity_J_K

    @property
    def baseline_garment_mass_kg(self) -> float:
        """Mass of the supplied Problem 1 garment."""

        return self.garment_mass_kg(self.baseline_outer_layer_count)

    @property
    def baseline_cost_yuan(self) -> float:
        """Original three-layer material cost."""

        inner = self.inner_price_yuan_per_kg * self.thermal.layer1_mass_kg
        pcm = self.pcm_price_yuan_per_m2 * self.thermal.material_area_m2
        outer = self.outer_price_yuan_per_kg * self.outer_coating_mass_kg
        return inner + pcm + outer

    @property
    def maximum_total_cost_yuan(self) -> float:
        """Original cost plus the permitted fractional increase."""

        return self.baseline_cost_yuan * (1.0 + self.maximum_budget_increase_fraction)

    @property
    def available_added_budget_yuan(self) -> float:
        """Maximum amount that may be added to the baseline material cost."""

        return self.maximum_total_cost_yuan - self.baseline_cost_yuan

    def garment_mass_kg(self, outer_layer_count: int) -> float:
        """Return garment mass, excluding the wearer's body mass."""

        self._validate_layer_count(outer_layer_count)
        return (
            self.thermal.layer1_mass_kg
            + self.thermal.pcm_mass_kg
            + outer_layer_count * self.outer_coating_mass_kg
        )

    def total_cost_yuan(self, outer_layer_count: int) -> float:
        """Return material cost for the requested total number of outer coatings."""

        self._validate_layer_count(outer_layer_count)
        additional = outer_layer_count - self.baseline_outer_layer_count
        return self.baseline_cost_yuan + (
            additional * self.outer_price_yuan_per_kg * self.outer_coating_mass_kg
        )

    def outer_thickness_mm(self, outer_layer_count: int) -> float:
        """Return total outer-coating thickness in millimetres."""

        self._validate_layer_count(outer_layer_count)
        return 1000.0 * outer_layer_count * self.thermal.layer3.thickness_m

    def added_garment_mass_kg(self, outer_layer_count: int) -> float:
        """Return mass added relative to the supplied Problem 1 garment."""

        return self.garment_mass_kg(outer_layer_count) - self.baseline_garment_mass_kg

    def added_cost_yuan(self, outer_layer_count: int) -> float:
        """Return cost added relative to the supplied one-coating garment."""

        return self.total_cost_yuan(outer_layer_count) - self.baseline_cost_yuan

    def is_budget_feasible(self, outer_layer_count: int) -> bool:
        """Return whether a coating count satisfies the 50% cost increase limit."""

        return self.total_cost_yuan(outer_layer_count) <= self.maximum_total_cost_yuan + 1.0e-12

    def is_load_feasible(self, outer_layer_count: int) -> bool:
        """Return whether total garment mass stays within the external-load limit."""

        return self.garment_mass_kg(outer_layer_count) <= (
            self.maximum_external_load_kg + 1.0e-12
        )

    def is_feasible(self, outer_layer_count: int) -> bool:
        """Return whether both material budget and added-load limits hold."""

        return self.is_budget_feasible(outer_layer_count) and self.is_load_feasible(
            outer_layer_count
        )

    def feasible_outer_layer_counts(self) -> tuple[int, ...]:
        """Return all consecutive coating counts allowed by every hard constraint."""

        counts: list[int] = []
        count = self.baseline_outer_layer_count
        while self.is_feasible(count):
            counts.append(count)
            count += 1
            if count > 10000:
                raise RuntimeError("Unbounded coating enumeration; check price parameters")
        return tuple(counts)

    def weight_penalty_s(self, outer_layer_count: int) -> float:
        """Return penalty for mass added relative to the Problem 1 garment."""

        return (
            self.weight_penalty_s_per_kg
            * self.added_garment_mass_kg(outer_layer_count)
        )

    def _validate_layer_count(self, outer_layer_count: int) -> None:
        if not isinstance(outer_layer_count, int) or outer_layer_count < 1:
            raise ValueError("outer_layer_count must be a positive integer")

