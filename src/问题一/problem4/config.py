"""Immutable configuration for the Problem 4 inverse-design workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import DEFAULT_DSC_PATH, REPOSITORY_ROOT, ModelParameters


DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "outputs" / "problem4"
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "docs" / "问题四_代码说明与结果交付.md"
DEFAULT_PROBLEM1_SUMMARY = REPOSITORY_ROOT / "results" / "outputs" / "problem1" / "main_summary.csv"
DEFAULT_PROBLEM3_SOLUTION = REPOSITORY_ROOT / "results" / "outputs" / "problem3" / "optimal_solution.csv"


@dataclass(frozen=True)
class SolverSettings:
    method: str = "RK45"
    rtol: float = 1.0e-7
    atol: float = 1.0e-9
    max_step_s: float = 5.0
    horizon_s: float = 100000.0
    root_xtol: float = 1.0e-8
    maximum_lambda: float = 100.0


@dataclass(frozen=True)
class Problem4Config:
    """All paths, model inputs and numerical controls used by Problem 4."""

    thermal: ModelParameters = field(default_factory=ModelParameters)
    dsc_path: Path = DEFAULT_DSC_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    report_path: Path = DEFAULT_REPORT_PATH
    problem1_summary_path: Path = DEFAULT_PROBLEM1_SUMMARY
    problem3_solution_path: Path = DEFAULT_PROBLEM3_SOLUTION
    target_fallback_min: float = 734.815
    target_reference_min: float = 734.815
    target_readback_tolerance_min: float = 0.01
    coarse_lambdas: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 3.0)
    main_solver: SolverSettings = field(default_factory=SolverSettings)
    strict_solver: SolverSettings = field(
        default_factory=lambda: SolverSettings(
            method="DOP853",
            rtol=1.0e-9,
            atol=1.0e-11,
            max_step_s=1.0,
            root_xtol=1.0e-10,
        )
    )
    output_step_s: float = 10.0
    audit_step_s: float = 5.0
    strict_audit_step_s: float = 1.0

    def validate(self) -> None:
        self.thermal.validate()
        if self.thermal.layer3.thickness_m != 0.3e-3:
            raise ValueError("Problem 4 must use the original 0.3 mm outer layer")
        if self.thermal.pcm_layer.thickness_m != 0.4e-3:
            raise ValueError("Problem 4 must keep the PCM layer at 0.4 mm")
        if self.thermal.air_temperature_C != -40.0:
            raise ValueError("Problem 4 must use the Problem 1 -40 degC environment")
        if min(self.coarse_lambdas) < 1.0:
            raise ValueError("PCM multipliers must be at least one")
