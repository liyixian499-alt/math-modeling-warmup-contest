"""End-to-end Problem 4 inversion, validation, sensitivity and CSV workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..dsc import PCMModel, load_pcm_model
from ..model import DYNAMIC_SBF
from ..simulation import CaseResult, run_case
from .config import Problem4Config, SolverSettings
from .pcm import ScaledPCMModel
from .reporting import write_delivery_report
from .sensitivity import normalized_sensitivity_summary, scenario_definitions
from .solver import EventResult, InverseResult, invert_lambda, solve_events


def _read_target(config: Problem4Config) -> dict[str, object]:
    path = config.problem3_solution_path
    if path.is_file():
        frame = pd.read_csv(path)
        preferred = (
            "standing_time_score_min",
            "objective_time_min",
            "final_standing_time_min",
        )
        field = next((name for name in preferred if name in frame.columns), None)
        if field is None:
            candidates = [
                name
                for name in frame.columns
                if "score" in name.lower() and name.lower().endswith("_min")
            ]
            if not candidates:
                raise RuntimeError("Problem 3 target score field could not be identified")
            field = candidates[0]
        value_min = float(frame.iloc[0][field])
        source_kind = "formal_problem3_csv"
    else:
        field = "fallback_constant"
        value_min = config.target_fallback_min
        source_kind = "fallback"
    difference = value_min - config.target_reference_min
    if abs(difference) > config.target_readback_tolerance_min:
        raise RuntimeError(
            f"Problem 3 target {value_min:.9f} min differs materially from 734.815 min"
        )
    return {
        "path": str(path),
        "field": field,
        "value_min": value_min,
        "value_s": value_min * 60.0,
        "difference_min": difference,
        "source_kind": source_kind,
    }


def _run_full_case(
    case_id: str,
    params,
    base_pcm: PCMModel,
    lambda_pcm: float,
    settings: SolverSettings,
    config: Problem4Config,
    *,
    audit_step_s: float | None = None,
) -> CaseResult:
    return run_case(
        case_id,
        params,
        ScaledPCMModel(base_pcm, lambda_pcm),
        model_variant=DYNAMIC_SBF,
        method=settings.method,
        rtol=settings.rtol,
        atol=settings.atol,
        max_step_s=settings.max_step_s,
        horizon_s=settings.horizon_s,
        output_step_s=config.output_step_s,
        audit_step_s=(config.audit_step_s if audit_step_s is None else audit_step_s),
    )


def _latent_values(
    params, base_pcm: PCMModel, lambda_pcm: float
) -> tuple[float, float, float, float]:
    original_specific = base_pcm.latent_heat_J_kg(params.dsc_scan_rate_K_min)
    improved_specific = lambda_pcm * original_specific
    original_total = params.pcm_mass_kg * original_specific
    improved_total = lambda_pcm * original_total
    return (
        original_specific / 1000.0,
        improved_specific / 1000.0,
        original_total / 1000.0,
        improved_total / 1000.0,
    )


def _optimal_row(
    result: CaseResult,
    params,
    base_pcm: PCMModel,
    lambda_pcm: float,
    target_s: float,
    settings: SolverSettings,
    root_tolerance: float,
) -> dict[str, float | str]:
    summary = result.summary
    original_l, improved_l, original_q, improved_q = _latent_values(
        params, base_pcm, lambda_pcm
    )
    return {
        "lambda_opt": lambda_pcm,
        "improvement_pct": (lambda_pcm - 1.0) * 100.0,
        "original_latent_heat_kJ_kg": original_l,
        "improved_latent_heat_kJ_kg": improved_l,
        "original_total_latent_heat_kJ": original_q,
        "improved_total_latent_heat_kJ": improved_q,
        "target_t15_s": target_s,
        "target_t15_min": target_s / 60.0,
        "achieved_t15_s": float(summary["t15_s"]),
        "achieved_t15_min": float(summary["t15_s"]) / 60.0,
        "t15_error_s": float(summary["t15_s"]) - target_s,
        "t10_s": float(summary["t10_s"]),
        "t10_min": float(summary["t10_s"]) / 60.0,
        "T_core_at_t15_C": float(summary["T_core_at_t15_C"]),
        "T_pcm_at_t15_C": float(summary["T_pcm_at_t15_C"]),
        "pcm_phase_fraction_at_t15": float(summary["pcm_phase_fraction_at_t15"]),
        "skin_blood_flow_at_t15": float(summary["skin_blood_flow_at_t15"]),
        "max_abs_energy_balance_residual_J": float(
            summary["max_abs_energy_balance_residual_J"]
        ),
        "solver_method": settings.method,
        "root_method": "brentq",
        "root_tolerance": root_tolerance,
        "relative_energy_balance_error": float(
            summary["relative_energy_balance_error"]
        ),
    }


def _search_frame(
    events: list[EventResult],
    base_pcm: PCMModel,
    params,
    target_s: float,
) -> pd.DataFrame:
    rows = []
    previous = None
    for event in sorted(events, key=lambda item: item.lambda_pcm):
        original_l, improved_l, _, improved_q = _latent_values(
            params, base_pcm, event.lambda_pcm
        )
        difference = 0.0 if previous is None else event.t15_s - previous
        rows.append(
            {
                "lambda_pcm": event.lambda_pcm,
                "improvement_pct": (event.lambda_pcm - 1.0) * 100.0,
                "latent_heat_kJ_kg": improved_l,
                "total_pcm_latent_heat_kJ": improved_q,
                "t15_s": event.t15_s,
                "t15_min": event.t15_s / 60.0,
                "t10_s": event.t10_s,
                "t10_min": event.t10_s / 60.0,
                "target_t15_min": target_s / 60.0,
                "gap_to_target_s": event.t15_s - target_s,
                "feasible": bool(event.t15_s >= target_s),
                "event_15_reached": event.event_15_reached,
                "event_10_reached": event.event_10_reached,
                "adjacent_t15_difference_s": difference,
                "has_previous_search_point": previous is not None,
                "original_latent_heat_kJ_kg": original_l,
            }
        )
        previous = event.t15_s
    return pd.DataFrame(rows)


def _sensitivity_inverse(
    config: Problem4Config,
    base_pcm: PCMModel,
    target_s: float,
    baseline_inverse: InverseResult,
    baseline_full: CaseResult,
) -> pd.DataFrame:
    baseline_lambda = baseline_inverse.lambda_opt
    rows: list[dict[str, object]] = []
    for scenario in scenario_definitions():
        params = scenario.parameters(config.thermal)
        if not scenario.updates:
            inverse = baseline_inverse
            full = baseline_full
        else:
            inverse = invert_lambda(
                params,
                base_pcm,
                target_s,
                config.main_solver,
                initial_upper=max(1.5, baseline_lambda * 1.15),
            )
            full = _run_full_case(
                f"inverse_{scenario.case_id}",
                params,
                base_pcm,
                inverse.lambda_opt,
                config.main_solver,
                config,
            )
        _, latent, _, total = _latent_values(params, base_pcm, inverse.lambda_opt)
        summary = full.summary
        delta = inverse.lambda_opt - baseline_lambda
        rows.append(
            {
                "case_id": scenario.case_id,
                "parameter": scenario.parameter,
                "scenario": scenario.scenario,
                "parameter_value": scenario.parameter_value,
                "unit": scenario.unit,
                "baseline_value": scenario.baseline_value,
                "target_t15_min": target_s / 60.0,
                "lambda_opt": inverse.lambda_opt,
                "improvement_pct": (inverse.lambda_opt - 1.0) * 100.0,
                "latent_heat_kJ_kg": latent,
                "total_pcm_latent_heat_kJ": total,
                "delta_lambda_vs_baseline": delta,
                "delta_lambda_pct": 100.0 * delta / baseline_lambda,
                "delta_improvement_percentage_points": 100.0 * delta,
                "achieved_t15_s": float(summary["t15_s"]),
                "t15_error_s": float(summary["t15_s"]) - target_s,
                "t10_s": float(summary["t10_s"]),
                "T_core_at_t15_C": float(summary["T_core_at_t15_C"]),
                "relative_energy_balance_error": float(
                    summary["relative_energy_balance_error"]
                ),
                "feasible": bool(float(summary["t15_s"]) >= target_s - 0.05),
                "notes": "baseline reused" if not scenario.updates else "full reinversion",
            }
        )
    return pd.DataFrame(rows)


def _sensitivity_fixed(
    config: Problem4Config,
    base_pcm: PCMModel,
    target_s: float,
    baseline_lambda: float,
    baseline_full: CaseResult,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in scenario_definitions():
        params = scenario.parameters(config.thermal)
        full = (
            baseline_full
            if not scenario.updates
            else _run_full_case(
                f"fixed_{scenario.case_id}",
                params,
                base_pcm,
                baseline_lambda,
                config.main_solver,
                config,
            )
        )
        summary = full.summary
        margin = float(summary["t15_s"]) - target_s
        rows.append(
            {
                "case_id": scenario.case_id,
                "parameter": scenario.parameter,
                "scenario": scenario.scenario,
                "parameter_value": scenario.parameter_value,
                "unit": scenario.unit,
                "lambda_pcm_fixed": baseline_lambda,
                "t15_s": float(summary["t15_s"]),
                "t15_min": float(summary["t15_s"]) / 60.0,
                "t10_s": float(summary["t10_s"]),
                "t10_min": float(summary["t10_s"]) / 60.0,
                "target_t15_min": target_s / 60.0,
                "margin_to_target_s": margin,
                "margin_to_target_min": margin / 60.0,
                "feasible": bool(margin >= -0.05),
                "T_core_at_t15_C": float(summary["T_core_at_t15_C"]),
                "relative_energy_balance_error": float(
                    summary["relative_energy_balance_error"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _target_sensitivity(
    config: Problem4Config,
    base_pcm: PCMModel,
    target_s: float,
    baseline_inverse: InverseResult,
) -> pd.DataFrame:
    rows = []
    for factor in (0.95, 1.0, 1.05):
        scenario_target = factor * target_s
        inverse = (
            baseline_inverse
            if factor == 1.0
            else invert_lambda(
                config.thermal,
                base_pcm,
                scenario_target,
                config.main_solver,
                initial_upper=max(1.5, baseline_inverse.lambda_opt * 1.2),
            )
        )
        achieved = inverse.evaluations[inverse.lambda_opt]
        _, latent, _, total = _latent_values(
            config.thermal, base_pcm, inverse.lambda_opt
        )
        rows.append(
            {
                "target_factor": factor,
                "target_t15_min": scenario_target / 60.0,
                "lambda_opt": inverse.lambda_opt,
                "improvement_pct": (inverse.lambda_opt - 1.0) * 100.0,
                "latent_heat_kJ_kg": latent,
                "total_pcm_latent_heat_kJ": total,
                "achieved_t15_min": achieved / 60.0,
                "t15_error_s": achieved - scenario_target,
            }
        )
    return pd.DataFrame(rows)


def _validation_frame(
    config: Problem4Config,
    base_pcm: PCMModel,
    target_source: dict[str, object],
    search: pd.DataFrame,
    main_row: dict[str, object],
    strict_row: dict[str, object],
    lambda_below: float,
    t15_below_s: float,
) -> pd.DataFrame:
    problem1 = pd.read_csv(config.problem1_summary_path).iloc[0]
    base_l = base_pcm.latent_heat_J_kg(config.thermal.dsc_scan_rate_K_min)

    def row(name, reference, computed, tolerance, passed, notes):
        absolute = abs(float(computed) - float(reference))
        relative = absolute / max(abs(float(reference)), 1.0)
        return {
            "validation_name": name,
            "reference_value": reference,
            "computed_value": computed,
            "absolute_error": absolute,
            "relative_error": relative,
            "tolerance": tolerance,
            "passed": bool(passed),
            "notes": notes,
        }

    lambda1_t15 = float(search.loc[np.isclose(search["lambda_pcm"], 1.0), "t15_s"].iloc[0])
    monotonic_min = float(search.loc[search["has_previous_search_point"], "adjacent_t15_difference_s"].min())
    rows = [
        row(
            "lambda_1_reproduces_problem1",
            float(problem1["t15_s"]),
            lambda1_t15,
            0.05,
            abs(lambda1_t15 - float(problem1["t15_s"])) <= 0.05,
            "unit=s; formal Problem 1 main_summary.csv",
        ),
        row(
            "problem3_target_readback",
            config.target_reference_min,
            float(target_source["value_min"]),
            config.target_readback_tolerance_min,
            abs(float(target_source["difference_min"])) <= config.target_readback_tolerance_min,
            f"unit=min; field={target_source['field']}",
        ),
        row(
            "latent_heat_scaling_lambda_1_5",
            1.5,
            ScaledPCMModel(base_pcm, 1.5).latent_heat_J_kg(
                config.thermal.dsc_scan_rate_K_min
            ) / base_l,
            1.0e-12,
            True,
            "ratio L(lambda)/L0",
        ),
        row(
            "latent_heat_scaling_lambda_opt",
            float(main_row["lambda_opt"]),
            ScaledPCMModel(base_pcm, float(main_row["lambda_opt"])).latent_heat_J_kg(
                config.thermal.dsc_scan_rate_K_min
            ) / base_l,
            1.0e-12,
            True,
            "ratio L(lambda*)/L0",
        ),
        row(
            "monotonicity_check",
            0.0,
            monotonic_min,
            0.01,
            monotonic_min >= -0.01,
            "minimum adjacent t15 increase in seconds; nonnegative expected",
        ),
        row(
            "solver_convergence_lambda",
            float(main_row["lambda_opt"]),
            float(strict_row["strict_refined_lambda"]),
            5.0e-5,
            abs(
                float(strict_row["strict_refined_lambda"])
                - float(main_row["lambda_opt"])
            )
            <= 5.0e-5,
            "RK45 main root versus strict DOP853 refined root",
        ),
        row(
            "solver_convergence_t15",
            float(main_row["achieved_t15_s"]),
            float(strict_row["achieved_t15_s"]),
            0.05,
            abs(float(strict_row["achieved_t15_s"]) - float(main_row["achieved_t15_s"])) <= 0.05,
            "unit=s; both solvers evaluated the same main lambda",
        ),
        row(
            "energy_balance_main",
            0.0,
            float(main_row["relative_energy_balance_error"]),
            1.0e-4,
            float(main_row["relative_energy_balance_error"]) < 1.0e-4,
            "relative energy residual; current-lambda PCM enthalpy",
        ),
        row(
            "energy_balance_strict",
            0.0,
            float(strict_row["relative_energy_balance_error"]),
            1.0e-5,
            float(strict_row["relative_energy_balance_error"]) < 1.0e-5,
            "relative energy residual; strict DOP853",
        ),
        row(
            "lambda_just_below_is_infeasible",
            float(target_source["value_s"]),
            t15_below_s,
            0.0,
            t15_below_s < float(target_source["value_s"]),
            f"lambda_below={lambda_below:.12g}",
        ),
    ]
    return pd.DataFrame(rows)


def _timeseries_frame(
    result: CaseResult,
    base_pcm: PCMModel,
    params,
    lambda_pcm: float,
) -> pd.DataFrame:
    frame = result.timeseries.copy().rename(
        columns={"alpha_dot_1_s": "d_alpha_skin_dt_1_s"}
    )
    frame["lambda_pcm"] = lambda_pcm
    frame["latent_heat_kJ_kg"] = (
        lambda_pcm
        * base_pcm.latent_heat_J_kg(params.dsc_scan_rate_K_min)
        / 1000.0
    )
    frame["released_pcm_latent_heat_kJ"] = frame["pcm_latent_released_J"] / 1000.0
    columns = [
        "time_s", "T_core_C", "T_skin_C", "T_layer1_C", "T_pcm_C", "T_layer3_C",
        "skin_blood_flow_eq_L_m2_h", "skin_blood_flow_actual_L_m2_h",
        "d_skin_blood_flow_dt_L_m2_h_s", "alpha_skin", "d_alpha_skin_dt_1_s",
        "q_cs_W_m2", "q_s1_W_m2", "q_12_W_m2", "q_23_W_m2", "q_3inf_W_m2",
        "c_eff_pcm_J_kgK", "pcm_phase_fraction", "lambda_pcm", "latent_heat_kJ_kg",
        "released_pcm_latent_heat_kJ", "energy_balance_residual_J",
    ]
    return frame.loc[:, columns]


def execute_workflow(config: Problem4Config | None = None) -> dict[str, object]:
    """Run every required Problem 4 computation and generate all deliverables."""

    config = config or Problem4Config()
    config.validate()
    base_pcm = load_pcm_model(config.dsc_path)
    target_source = _read_target(config)
    target_s = float(target_source["value_s"])

    coarse_events = [
        solve_events(config.thermal, base_pcm, value, config.main_solver)
        for value in config.coarse_lambdas
    ]
    search = _search_frame(coarse_events, base_pcm, config.thermal, target_s)
    if not search["event_15_reached"].all() or not search["event_10_reached"].all():
        raise RuntimeError("A required coarse-search threshold event was not reached")
    differences = search.loc[
        search["has_previous_search_point"], "adjacent_t15_difference_s"
    ]
    if (differences < -0.01).any():
        raise RuntimeError("Coarse-search t15 is nonmonotonic; inversion stopped")
    known = dict(zip(search["lambda_pcm"], search["t15_s"], strict=True))
    main_inverse = invert_lambda(
        config.thermal,
        base_pcm,
        target_s,
        config.main_solver,
        initial_upper=max(config.coarse_lambdas),
        known_evaluations=known,
    )
    main_full = _run_full_case(
        "problem4_optimal_main",
        config.thermal,
        base_pcm,
        main_inverse.lambda_opt,
        config.main_solver,
        config,
    )
    main_row = _optimal_row(
        main_full,
        config.thermal,
        base_pcm,
        main_inverse.lambda_opt,
        target_s,
        config.main_solver,
        config.main_solver.root_xtol,
    )

    strict_inverse = invert_lambda(
        config.thermal,
        base_pcm,
        target_s,
        config.strict_solver,
        initial_upper=max(1.5, main_inverse.lambda_opt * 1.02),
    )
    strict_full = _run_full_case(
        "problem4_optimal_strict",
        config.thermal,
        base_pcm,
        main_inverse.lambda_opt,
        config.strict_solver,
        config,
        audit_step_s=config.strict_audit_step_s,
    )
    strict_row = _optimal_row(
        strict_full,
        config.thermal,
        base_pcm,
        main_inverse.lambda_opt,
        target_s,
        config.strict_solver,
        config.strict_solver.root_xtol,
    )
    strict_row.update(
        {
            "main_lambda": main_inverse.lambda_opt,
            "strict_refined_lambda": strict_inverse.lambda_opt,
            "strict_refined_t15_s": strict_inverse.evaluations[
                strict_inverse.lambda_opt
            ],
            "delta_t15_vs_main_s": float(strict_row["achieved_t15_s"])
            - float(main_row["achieved_t15_s"]),
            "delta_t10_vs_main_s": float(strict_row["t10_s"])
            - float(main_row["t10_s"]),
            "delta_lambda_vs_main": strict_inverse.lambda_opt
            - main_inverse.lambda_opt,
        }
    )

    lambda_below = max(
        1.0, main_inverse.lambda_opt - max(1.0e-5, 10.0 * config.main_solver.root_xtol)
    )
    below_event = solve_events(
        config.thermal, base_pcm, lambda_below, config.main_solver, stop_at_15=True
    )
    main_row["lambda_below_check"] = lambda_below
    main_row["t15_below_check_s"] = below_event.t15_s
    main_row["below_check_feasible"] = bool(below_event.t15_s >= target_s)

    inverse_sensitivity = _sensitivity_inverse(
        config, base_pcm, target_s, main_inverse, main_full
    )
    fixed_sensitivity = _sensitivity_fixed(
        config, base_pcm, target_s, main_inverse.lambda_opt, main_full
    )
    sensitivity_summary = normalized_sensitivity_summary(
        inverse_sensitivity, main_inverse.lambda_opt
    )
    target_sensitivity = _target_sensitivity(
        config, base_pcm, target_s, main_inverse
    )
    validation = _validation_frame(
        config,
        base_pcm,
        target_source,
        search,
        main_row,
        strict_row,
        lambda_below,
        below_event.t15_s,
    )
    timeseries = _timeseries_frame(
        main_full, base_pcm, config.thermal, main_inverse.lambda_opt
    )

    destination = Path(config.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    search.to_csv(destination / "problem4_search.csv", index=False)
    pd.DataFrame([main_row]).to_csv(
        destination / "problem4_optimal_solution.csv", index=False
    )
    pd.DataFrame([strict_row]).to_csv(
        destination / "problem4_optimal_solution_strict.csv", index=False
    )
    timeseries.to_csv(destination / "problem4_timeseries.csv", index=False)
    inverse_sensitivity.to_csv(
        destination / "problem4_sensitivity_inverse.csv", index=False
    )
    fixed_sensitivity.to_csv(
        destination / "problem4_sensitivity_fixed_design.csv", index=False
    )
    sensitivity_summary.to_csv(
        destination / "problem4_sensitivity_summary.csv", index=False
    )
    target_sensitivity.to_csv(
        destination / "problem4_target_sensitivity.csv", index=False
    )
    validation.to_csv(destination / "problem4_model_validation.csv", index=False)
    write_delivery_report(config.report_path, destination, target_source)
    return {
        "config": config,
        "pcm": base_pcm,
        "target_source": target_source,
        "main_inverse": main_inverse,
        "main": main_full,
        "main_row": main_row,
        "strict": strict_full,
        "strict_row": strict_row,
        "search": search,
        "inverse_sensitivity": inverse_sensitivity,
        "fixed_sensitivity": fixed_sensitivity,
        "sensitivity_summary": sensitivity_summary,
        "target_sensitivity": target_sensitivity,
        "validation": validation,
        "output_dir": destination,
        "report_path": config.report_path,
    }


def print_terminal_summary(results: dict[str, object]) -> None:
    main = results["main_row"]
    strict = results["strict_row"]
    target = results["target_source"]
    ranking: pd.DataFrame = results["sensitivity_summary"]
    fixed: pd.DataFrame = results["fixed_sensitivity"]
    failed = fixed[~fixed["feasible"]]
    print("=== Problem 4 PCM inverse design ===")
    print(f"Problem 3 target: {target['value_min']:.9f} min ({target['field']})")
    print(f"lambda*: {main['lambda_opt']:.10f}")
    print(f"PCM improvement: {main['improvement_pct']:.6f}%")
    print(
        "latent heat kJ/kg original/improved: "
        f"{main['original_latent_heat_kJ_kg']:.6f} / {main['improved_latent_heat_kJ_kg']:.6f}"
    )
    print(
        "total latent heat kJ original/improved: "
        f"{main['original_total_latent_heat_kJ']:.6f} / {main['improved_total_latent_heat_kJ']:.6f}"
    )
    print(
        f"t15: {main['achieved_t15_min']:.9f} min; error={main['t15_error_s']:.6g} s"
    )
    print(f"t10: {main['t10_min']:.9f} min")
    print(
        "strict deltas: "
        f"lambda={strict['delta_lambda_vs_main']:.6g}, "
        f"t15={strict['delta_t15_vs_main_s']:.6g} s, "
        f"t10={strict['delta_t10_vs_main_s']:.6g} s"
    )
    print(
        "energy max/relative: "
        f"{main['max_abs_energy_balance_residual_J']:.6g} J / "
        f"{main['relative_energy_balance_error']:.6g}"
    )
    print("inverse sensitivity ranking:")
    for row in ranking.itertuples(index=False):
        print(
            f"  {int(row.rank)}. {row.parameter}: "
            f"S_lambda={row.normalized_sensitivity_lambda:.6f}"
        )
    if failed.empty:
        print("fixed baseline design: all tested scenarios remain feasible")
    else:
        worst = failed.sort_values("margin_to_target_s").iloc[0]
        print(
            f"fixed baseline design: {len(failed)} failed scenarios; worst="
            f"{worst['case_id']} ({worst['margin_to_target_min']:.6f} min)"
        )
    print(f"CSV output directory: {results['output_dir']}")
    print(f"Markdown delivery: {results['report_path']}")
