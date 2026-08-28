"""Complete Problem 3 enumeration, validation and CSV export workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.问题一 import DYNAMIC_SBF, load_pcm_model
from src.问题一.config import DEFAULT_DSC_PATH, REPOSITORY_ROOT
from src.问题一.simulation import run_case as run_problem1_case

from .config import Problem3Parameters
from .simulation import Problem3CaseResult, run_candidate


DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "outputs" / "problem3"


def execute_workflow(
    dsc_path: str | Path = DEFAULT_DSC_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """Enumerate every budget-feasible coating design and export final results."""

    config = Problem3Parameters()
    config.validate()
    pcm = load_pcm_model(dsc_path)
    counts = config.feasible_outer_layer_counts()
    candidates = [run_candidate(config, pcm, count) for count in counts]
    summary = pd.DataFrame([candidate.summary for candidate in candidates])
    optimum_index = int(summary["objective_standing_time_s"].idxmax())
    optimum = candidates[optimum_index]

    strict = run_candidate(
        config,
        pcm,
        optimum.outer_layer_count,
        rtol=1.0e-9,
        atol=1.0e-11,
        max_step_s=1.0,
        output_step_s=30.0,
        audit_step_s=0.25,
    )
    problem1_reference = run_problem1_case(
        "problem3_regression_reference",
        config.thermal,
        pcm,
        model_variant=DYNAMIC_SBF,
        output_step_s=30.0,
        audit_step_s=1.0,
    )
    baseline = candidates[0]
    validation = pd.DataFrame(
        [
            {
                "check": "one_outer_node_reproduces_problem1",
                "problem3_value_s": baseline.summary["t15_s"],
                "reference_value_s": problem1_reference.summary["t15_s"],
                "absolute_difference_s": abs(
                    float(baseline.summary["t15_s"])
                    - float(problem1_reference.summary["t15_s"])
                ),
            },
            {
                "check": "optimal_candidate_solver_convergence",
                "problem3_value_s": optimum.summary["t15_s"],
                "reference_value_s": strict.summary["t15_s"],
                "absolute_difference_s": abs(
                    float(optimum.summary["t15_s"])
                    - float(strict.summary["t15_s"])
                ),
            },
        ]
    )
    if validation.iloc[0]["absolute_difference_s"] > 1.0e-6:
        raise RuntimeError("The one-coating Problem 3 model does not reproduce Problem 1")
    if not np.all(np.diff(summary["t15_s"].to_numpy(dtype=float)) >= -1.0e-6):
        raise RuntimeError("Thermal endurance unexpectedly decreases with added insulation")
    if not np.all(
        np.diff(summary["objective_standing_time_s"].to_numpy(dtype=float))
        >= -1.0e-6
    ):
        raise RuntimeError("Weight-adjusted objective unexpectedly decreases")
    if not (summary["total_cost_yuan"] <= config.maximum_total_cost_yuan + 1.0e-9).all():
        raise RuntimeError("An exported candidate violates the budget")

    all_timeseries = pd.concat(
        [candidate.timeseries for candidate in candidates],
        ignore_index=True,
        sort=False,
    )
    optimum_row = summary.loc[[optimum_index]].copy()
    optimum_row.insert(
        0,
        "selection_rule",
        "maximize_t15_minus_time_equivalent_weight_penalty",
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination / "candidate_summary.csv", index=False)
    optimum_row.to_csv(destination / "optimal_solution.csv", index=False)
    all_timeseries.to_csv(destination / "candidate_timeseries.csv", index=False)
    validation.to_csv(destination / "model_validation.csv", index=False)
    pd.DataFrame([strict.summary]).to_csv(
        destination / "optimal_solution_strict.csv",
        index=False,
    )
    return {
        "config": config,
        "pcm": pcm,
        "candidates": candidates,
        "summary": summary,
        "optimum": optimum,
        "strict": strict,
        "validation": validation,
        "output_dir": destination,
    }


def print_terminal_summary(results: dict[str, object]) -> None:
    """Print a concise candidate comparison and selected design."""

    summary: pd.DataFrame = results["summary"]
    optimum: Problem3CaseResult = results["optimum"]
    print("=== Problem 3 coating optimization ===")
    print(summary[
        [
            "outer_layer_count",
            "outer_thickness_mm",
            "garment_mass_kg",
            "total_cost_yuan",
            "t15_min",
            "weight_penalty_min",
            "objective_standing_time_min",
            "hard_constraint_standing_time_min",
        ]
    ].to_string(index=False))
    selected = optimum.summary
    print(
        "Selected design: "
        f"{selected['outer_layer_count']} total outer coating(s), "
        f"{selected['outer_thickness_mm']:.1f} mm, "
        f"weight-adjusted standing time {selected['objective_standing_time_min']:.6f} min"
    )
    print(f"CSV output directory: {results['output_dir']}")

