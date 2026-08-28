"""Load and validate the reviewed Problem 1/2 CSV interfaces for plotting."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from plot_config import REPOSITORY_ROOT


PROBLEM1_DIR = REPOSITORY_ROOT / "results" / "outputs" / "problem1"
PROBLEM2_DIR = REPOSITORY_ROOT / "results" / "outputs" / "problem2"
SCENARIO_ORDER = ("Q1", "W", "I", "M", "Q2")
TEMPERATURE_COLUMNS = (
    "T_core_C",
    "T_skin_C",
    "T_layer1_C",
    "T_pcm_C",
    "T_layer3_C",
)


@dataclass(frozen=True)
class TemperatureSeries:
    source: Path
    time_s: NDArray[np.float64]
    T_core_C: NDArray[np.float64]
    T_skin_C: NDArray[np.float64]
    T_layer1_C: NDArray[np.float64]
    T_pcm_C: NDArray[np.float64]
    T_layer3_C: NDArray[np.float64]

    @property
    def time_min(self) -> NDArray[np.float64]:
        return self.time_s / 60.0


@dataclass(frozen=True)
class EventSummary:
    source: Path
    case_id: str
    t15_s: float
    t15_min: float
    t10_s: float
    t10_min: float


@dataclass(frozen=True)
class EffectScenario:
    case_id: str
    description: str
    wind_speed_m_s: float
    h_in_W_m2K: float
    metabolic_rate_W_m2: float
    t15_s: float
    t15_min: float


@dataclass(frozen=True)
class VisualizationData:
    q1_series: TemperatureSeries
    q1_summary: EventSummary
    q2_series: TemperatureSeries
    q2_summary: EventSummary
    effects: tuple[EffectScenario, ...]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少正式结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"正式结果文件为空：{path}")
    return rows


def _require_columns(rows: list[dict[str, str]], columns: Iterable[str], path: Path) -> None:
    missing = sorted(set(columns) - set(rows[0]))
    if missing:
        raise ValueError(f"{path} 缺少字段：{missing}")


def _to_float(value: str, column: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} 的 {column} 包含非数值：{value!r}") from exc
    return result


def load_temperature_series(path: Path) -> TemperatureSeries:
    rows = _read_rows(path)
    required = ("time_s", *TEMPERATURE_COLUMNS)
    _require_columns(rows, required, path)
    arrays = {
        column: np.asarray([_to_float(row[column], column, path) for row in rows], dtype=float)
        for column in required
    }
    time_s = arrays["time_s"]
    if not np.isfinite(time_s).all():
        raise ValueError(f"{path} 的 time_s 含 NaN 或无穷值。")
    if np.any(time_s < 0.0):
        raise ValueError(f"{path} 的 time_s 含负值。")
    if time_s.size < 2 or np.any(np.diff(time_s) <= 0.0):
        raise ValueError(f"{path} 的 time_s 必须严格递增且至少包含两个点。")
    for column in TEMPERATURE_COLUMNS:
        if not np.isfinite(arrays[column]).all():
            raise ValueError(f"{path} 的 {column} 含 NaN 或无穷值。")
    return TemperatureSeries(
        source=path,
        time_s=time_s,
        T_core_C=arrays["T_core_C"],
        T_skin_C=arrays["T_skin_C"],
        T_layer1_C=arrays["T_layer1_C"],
        T_pcm_C=arrays["T_pcm_C"],
        T_layer3_C=arrays["T_layer3_C"],
    )


def load_event_summary(path: Path, case_id: str) -> EventSummary:
    rows = _read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"{path} 应恰有一行正式 summary，实际为 {len(rows)} 行。")
    required = ("t15_s", "t15_min", "t10_s", "t10_min")
    _require_columns(rows, required, path)
    row = rows[0]
    values = {column: _to_float(row[column], column, path) for column in required}
    if not np.isfinite(list(values.values())).all():
        raise ValueError(f"{path} 的正式 t15/t10 事件时间不完整。")
    if values["t15_s"] < 0.0 or values["t10_s"] <= values["t15_s"]:
        raise ValueError(f"{path} 的 t15/t10 时间顺序非法。")
    for threshold in (15, 10):
        seconds = values[f"t{threshold}_s"]
        minutes = values[f"t{threshold}_min"]
        if not np.isclose(seconds / 60.0, minutes, atol=1.0e-8, rtol=1.0e-10):
            raise ValueError(f"{path} 的 t{threshold}_s 与 t{threshold}_min 不一致。")
    return EventSummary(source=path, case_id=case_id, **values)


def validate_events(
    series: TemperatureSeries,
    summary: EventSummary,
    *,
    temperature_tolerance_C: float = 0.05,
) -> dict[str, float]:
    """Interpolate only for validation; formal event times remain summary values."""

    checks: dict[str, float] = {}
    for threshold in (15, 10):
        event_s = getattr(summary, f"t{threshold}_s")
        if not (series.time_s[0] <= event_s <= series.time_s[-1]):
            raise ValueError(
                f"{summary.case_id} 的 t{threshold} 不在 {series.source} 时间范围内。"
            )
        interpolated = float(np.interp(event_s, series.time_s, series.T_skin_C))
        error = interpolated - float(threshold)
        checks[f"T_skin_at_t{threshold}_C"] = interpolated
        checks[f"t{threshold}_error_C"] = error
        if abs(error) > temperature_tolerance_C:
            raise ValueError(
                f"{summary.case_id} 的正式 t{threshold} 与皮肤温度曲线不一致："
                f"插值温度为 {interpolated:.6f} °C。"
            )
    return checks


def load_effect_scenarios(path: Path) -> tuple[EffectScenario, ...]:
    rows = _read_rows(path)
    required = (
        "case_id",
        "description",
        "wind_speed_m_s",
        "h_in_W_m2K",
        "metabolic_rate_W_m2",
        "t15_s",
        "t15_min",
    )
    _require_columns(rows, required, path)
    by_id: dict[str, EffectScenario] = {}
    for row in rows:
        case_id = row["case_id"].strip()
        if not case_id:
            raise ValueError(f"{path} 含空 case_id。")
        if case_id in by_id:
            raise ValueError(f"{path} 的 case_id 重复：{case_id}")
        description = row["description"].strip()
        if not description:
            raise ValueError(f"{path} 的 {case_id} 缺少场景 description。")
        scenario = EffectScenario(
            case_id=case_id,
            description=description,
            wind_speed_m_s=_to_float(row["wind_speed_m_s"], "wind_speed_m_s", path),
            h_in_W_m2K=_to_float(row["h_in_W_m2K"], "h_in_W_m2K", path),
            metabolic_rate_W_m2=_to_float(
                row["metabolic_rate_W_m2"], "metabolic_rate_W_m2", path
            ),
            t15_s=_to_float(row["t15_s"], "t15_s", path),
            t15_min=_to_float(row["t15_min"], "t15_min", path),
        )
        numeric = (
            scenario.wind_speed_m_s,
            scenario.h_in_W_m2K,
            scenario.metabolic_rate_W_m2,
            scenario.t15_s,
            scenario.t15_min,
        )
        if not np.isfinite(numeric).all():
            raise ValueError(f"{path} 的 {case_id} 含无效的正式 t15 或参数值。")
        if not np.isclose(scenario.t15_s / 60.0, scenario.t15_min, atol=1.0e-8):
            raise ValueError(f"{path} 的 {case_id} t15 秒/分钟字段不一致。")
        by_id[case_id] = scenario
    missing = [case_id for case_id in SCENARIO_ORDER if case_id not in by_id]
    if missing:
        raise ValueError(f"{path} 缺少图6正式场景：{missing}")
    return tuple(by_id[case_id] for case_id in SCENARIO_ORDER)


def load_visualization_data() -> VisualizationData:
    q1_series = load_temperature_series(PROBLEM1_DIR / "main_timeseries.csv")
    q1_summary = load_event_summary(PROBLEM1_DIR / "main_summary.csv", "Q1")
    q2_series = load_temperature_series(PROBLEM2_DIR / "problem2_timeseries.csv")
    q2_summary = load_event_summary(PROBLEM2_DIR / "problem2_summary.csv", "Q2")
    effects = load_effect_scenarios(PROBLEM2_DIR / "problem2_effect_decomposition.csv")
    validate_events(q1_series, q1_summary)
    validate_events(q2_series, q2_summary)
    return VisualizationData(q1_series, q1_summary, q2_series, q2_summary, effects)
