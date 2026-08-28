"""Central configuration for the Problem 2 thermal model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.问题一.config import ModelParameters as Problem1Parameters


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DSC_PATH = REPOSITORY_ROOT / "problem" / "attachments" / "附件1 放热能力数据.xlsx"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "outputs" / "problem2"
DEFAULT_DELIVERY_PATH = REPOSITORY_ROOT / "docs" / "problem2_delivery.md"


@dataclass(frozen=True)
class ModelParameters(Problem1Parameters):
    """Problem 1 mother-model parameters extended by the Problem 2 wind boundary."""

    metabolic_rate_W_m2: float = 93.0
    h_in_W_m2K: float = 4.0
    wind_speed_m_s: float = 3.0

    def validate(self) -> None:
        """Raise ValueError when a physical model input is invalid."""

        super().validate()
        if self.wind_speed_m_s < 0.0:
            raise ValueError("Wind speed must be nonnegative")


@dataclass(frozen=True)
class SolverSettings:
    """Numerical integration and sampling controls, all time quantities in seconds."""

    method: str = "RK45"
    rtol: float = 1.0e-7
    atol: float = 1.0e-9
    max_step_s: float = 5.0
    horizon_s: float = 86400.0
    output_step_s: float = 5.0
    audit_step_s: float = 0.5

    def validate(self) -> None:
        """Validate numerical controls before integration."""

        positive = {
            "rtol": self.rtol,
            "atol": self.atol,
            "max_step_s": self.max_step_s,
            "horizon_s": self.horizon_s,
            "output_step_s": self.output_step_s,
            "audit_step_s": self.audit_step_s,
        }
        invalid = [name for name, value in positive.items() if not value > 0.0]
        if invalid:
            raise ValueError(f"Solver settings must be positive: {invalid}")


STRICT_SOLVER_SETTINGS = SolverSettings(
    method="DOP853",
    rtol=1.0e-9,
    atol=1.0e-11,
    max_step_s=1.0,
    output_step_s=5.0,
    audit_step_s=0.25,
)
