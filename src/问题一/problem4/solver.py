"""Continuous-event solves and scalar Brent inversion for Problem 4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ..config import ModelParameters
from ..dsc import PCMModel
from ..model import DYNAMIC_SBF, rhs
from .config import SolverSettings
from .pcm import ScaledPCMModel


@dataclass(frozen=True)
class EventResult:
    lambda_pcm: float
    t15_s: float
    t10_s: float
    event_15_reached: bool
    event_10_reached: bool


@dataclass
class InverseResult:
    lambda_opt: float
    target_t15_s: float
    evaluations: dict[float, float]
    bracket: tuple[float, float]
    root_method: str = "brentq"


def _initial_state(params: ModelParameters) -> np.ndarray:
    return np.array(
        [
            params.initial_core_temperature_C,
            params.initial_skin_temperature_C,
            params.initial_layer1_temperature_C,
            params.initial_pcm_temperature_C,
            params.initial_layer3_temperature_C,
            params.initial_skin_blood_flow_L_m2_h,
        ],
        dtype=float,
    )


def solve_events(
    params: ModelParameters,
    base_pcm: PCMModel,
    lambda_pcm: float,
    settings: SolverSettings,
    *,
    stop_at_15: bool = False,
) -> EventResult:
    """Solve with continuous downward 15/10 degC event detection."""

    pcm = ScaledPCMModel(base_pcm, lambda_pcm)

    def event15(_time_s: float, state: np.ndarray) -> float:
        return float(state[1] - 15.0)

    def event10(_time_s: float, state: np.ndarray) -> float:
        return float(state[1] - 10.0)

    event15.direction = -1.0
    event15.terminal = bool(stop_at_15)
    event10.direction = -1.0
    event10.terminal = True
    events = (event15,) if stop_at_15 else (event15, event10)
    solution = solve_ivp(
        fun=lambda time, state: rhs(time, state, params, pcm, DYNAMIC_SBF),
        t_span=(0.0, settings.horizon_s),
        y0=_initial_state(params),
        method=settings.method,
        rtol=settings.rtol,
        atol=settings.atol,
        max_step=settings.max_step_s,
        events=events,
    )
    if not solution.success:
        raise RuntimeError(f"Problem 4 event solve failed: {solution.message}")
    t15 = float(solution.t_events[0][0]) if len(solution.t_events[0]) else np.nan
    if stop_at_15:
        t10 = np.nan
    else:
        t10 = float(solution.t_events[1][0]) if len(solution.t_events[1]) else np.nan
    return EventResult(
        lambda_pcm=lambda_pcm,
        t15_s=t15,
        t10_s=t10,
        event_15_reached=bool(np.isfinite(t15)),
        event_10_reached=bool(np.isfinite(t10)),
    )


def invert_lambda(
    params: ModelParameters,
    base_pcm: PCMModel,
    target_t15_s: float,
    settings: SolverSettings,
    *,
    initial_upper: float = 1.5,
    known_evaluations: dict[float, float] | None = None,
) -> InverseResult:
    """Find the constrained minimum multiplier that reaches the target."""

    cache = dict(known_evaluations or {})

    def evaluate(lambda_pcm: float) -> float:
        key = float(lambda_pcm)
        if key not in cache:
            event = solve_events(
                params, base_pcm, key, settings, stop_at_15=True
            )
            if not event.event_15_reached:
                raise RuntimeError(f"15 degC event not reached at lambda={key}")
            cache[key] = event.t15_s
        return cache[key]

    t_at_one = evaluate(1.0)
    if t_at_one >= target_t15_s:
        return InverseResult(1.0, target_t15_s, cache, (1.0, 1.0))

    upper = max(float(initial_upper), 1.0 + 1.0e-6)
    while evaluate(upper) < target_t15_s:
        upper *= 1.5
        if upper > settings.maximum_lambda:
            raise RuntimeError(
                f"Could not bracket target below lambda={settings.maximum_lambda}"
            )
    lower_candidates = [
        value
        for value, t15_s in cache.items()
        if value >= 1.0 and value < upper and t15_s < target_t15_s
    ]
    lower = max(lower_candidates, default=1.0)
    root = brentq(
        lambda value: evaluate(float(value)) - target_t15_s,
        lower,
        upper,
        xtol=settings.root_xtol,
        rtol=max(4.0 * np.finfo(float).eps, settings.root_xtol),
    )
    evaluate(float(root))
    return InverseResult(float(root), target_t15_s, cache, (lower, upper))
