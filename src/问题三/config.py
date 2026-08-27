"""Configuration, cost, mass and load constraints for Problem 3."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.问题一.config import ModelParameters


@dataclass(frozen=True)
class Problem3Parameters:
    """Problem 3 design parameters built on the Problem 1 six-state model."""

    thermal: ModelParameters = field(default_factory=ModelParameters)
    baseline_outer_layer_count: int = 1
    maximum_load_initial_kg: float = 100.0
    load_loss_kg_per_10s: float = 0.5
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
            "maximum_load_initial_kg": self.maximum_load_initial_kg,
            "load_loss_kg_per_10s": self.load_loss_kg_per_10s,
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
    def load_loss_rate_kg_s(self) -> float:
        """Continuous decline rate of allowable external load in kg/s."""

        return self.load_loss_kg_per_10s / 10.0

    @property
    def outer_coating_mass_kg(self) -> float:
        """Mass of one full-area 0.3 mm outer coating."""

        return self.thermal.layer3_mass_kg

    @property
    def outer_coating_capacity_J_K(self) -> float:
        """Heat capacity of one outer coating."""

        return self.thermal.layer3_capacity_J_K

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

    def added_cost_yuan(self, outer_layer_count: int) -> float:
        """Return cost added relative to the supplied one-coating garment."""

        return self.total_cost_yuan(outer_layer_count) - self.baseline_cost_yuan

    def is_budget_feasible(self, outer_layer_count: int) -> bool:
        """Return whether a coating count satisfies the 50% cost increase limit."""

        return self.total_cost_yuan(outer_layer_count) <= self.maximum_total_cost_yuan + 1.0e-12

    def feasible_outer_layer_counts(self) -> tuple[int, ...]:
        """Return all consecutive coating counts allowed by the budget."""

        counts: list[int] = []
        count = self.baseline_outer_layer_count
        while self.is_budget_feasible(count):
            counts.append(count)
            count += 1
            if count > 10000:
                raise RuntimeError("Unbounded coating enumeration; check price parameters")
        return tuple(counts)

    def load_capacity_time_s(self, outer_layer_count: int) -> float:
        """Return when declining allowable load first equals garment mass."""

        mass = self.garment_mass_kg(outer_layer_count)
        return max(
            0.0,
            (self.maximum_load_initial_kg - mass) / self.load_loss_rate_kg_s,
        )

    def weight_penalty_s(self, outer_layer_count: int) -> float:
        """Return the time-equivalent penalty due to garment mass."""

        return self.garment_mass_kg(outer_layer_count) / self.load_loss_rate_kg_s

    def _validate_layer_count(self, outer_layer_count: int) -> None:
        if not isinstance(outer_layer_count, int) or outer_layer_count < 1:
            raise ValueError("outer_layer_count must be a positive integer")

