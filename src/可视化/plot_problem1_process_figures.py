"""Plot the Problem 1 PCM process and stepwise heat-flux pathways.

The figures are generated exclusively from the reviewed main time-series CSV.
No model state is recomputed or modified by this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.可视化.plot_config import (
    PALETTE_A,
    add_minimal_y_grid,
    apply_publication_style,
    finalize_figure,
)


DEFAULT_INPUT = REPOSITORY_ROOT / "results" / "outputs" / "problem1" / "main_timeseries.csv"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "paper" / "figures"

# Exact Group-B colors required by the installed scientific-figure-making skill.
PALETTE_B = {
    "reference": "#ECB66C",
    "baseline": "#DBCB92",
    "comparator": "#EA9E58",
    "emphasis": "#ED8D5A",
}

REQUIRED_COLUMNS = (
    "time_s",
    "c_eff_pcm_J_kgK",
    "pcm_phase_fraction",
    "pcm_latent_released_J",
    "q_cs_W_m2",
    "q_s1_W_m2",
    "q_12_W_m2",
    "q_23_W_m2",
    "q_3inf_W_m2",
)

FLUX_STYLES = {
    "q_cs_W_m2": (r"$q_{cs}$", PALETTE_A["key"], "-"),
    "q_s1_W_m2": (r"$q_{s1}$", PALETTE_A["secondary"], (0, (7, 3))),
    "q_12_W_m2": (r"$q_{12}$", PALETTE_A["positive"], "-"),
    "q_23_W_m2": (r"$q_{23}$", PALETTE_B["reference"], (0, (7, 3))),
    "q_3inf_W_m2": (r"$q_{3\infty}$", PALETTE_B["emphasis"], (0, (2, 2))),
}


def load_timeseries(path: Path) -> pd.DataFrame:
    """Load and validate the reviewed model output used by both figures."""

    if not path.is_file():
        raise FileNotFoundError(f"找不到问题一时间序列：{path}")
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"时间序列缺少字段：{missing}")
    if frame.empty:
        raise ValueError("问题一时间序列为空。")

    values = frame.loc[:, REQUIRED_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("问题一时间序列包含非有限数值。")
    if not frame["time_s"].is_monotonic_increasing or frame["time_s"].duplicated().any():
        raise ValueError("time_s 必须严格单调递增。")
    if frame["time_s"].iloc[0] < 0.0:
        raise ValueError("time_s 不得为负。")

    phase = frame["pcm_phase_fraction"].to_numpy(dtype=float)
    if np.any((phase < -1.0e-9) | (phase > 1.0 + 1.0e-9)):
        raise ValueError("PCM 相变进度必须位于 [0, 1]。")
    latent = frame["pcm_latent_released_J"].to_numpy(dtype=float)
    if np.any(np.diff(latent) < -1.0e-6):
        raise ValueError("冷却过程中累计潜热释放量不应下降。")
    return frame


def phase_interval(frame: pd.DataFrame) -> tuple[float, float]:
    """Return the 1% and 99% latent-release times in seconds."""

    phase = frame["pcm_phase_fraction"].to_numpy(dtype=float)
    time_s = frame["time_s"].to_numpy(dtype=float)
    above_start = np.flatnonzero(phase >= 0.01)
    above_end = np.flatnonzero(phase >= 0.99)
    if above_start.size == 0 or above_end.size == 0:
        raise ValueError("时间序列未覆盖完整的 PCM 相变过程。")
    return float(time_s[above_start[0]]), float(time_s[above_end[0]])


def add_panel_label(ax: Axes, label: str) -> None:
    """Place a publication-size panel identifier outside the data region."""

    ax.text(
        -0.12,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=18.0,
        color="black",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def plot_pcm_time_process(
    frame: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[list[Path], dict[str, float]]:
    """Plot apparent heat capacity and cumulative latent heat against time."""

    start_s, end_s = phase_interval(frame)
    display_end_s = min(float(frame["time_s"].iloc[-1]), max(300.0, end_s * 1.2))
    shown = frame.loc[frame["time_s"] <= display_end_s].copy()
    time_min = shown["time_s"].to_numpy(dtype=float) / 60.0
    c_eff = shown["c_eff_pcm_J_kgK"].to_numpy(dtype=float) / 1000.0
    latent = shown["pcm_latent_released_J"].to_numpy(dtype=float) / 1000.0
    start_min, end_min = start_s / 60.0, end_s / 60.0

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.5), sharex=True)
    top, bottom = axes
    for index, ax in enumerate(axes):
        ax.axvspan(
            start_min,
            end_min,
            color=PALETTE_A["light"],
            alpha=0.50,
            linewidth=0.0,
            label="相变区间" if index == 0 else None,
            zorder=0,
        )
        add_minimal_y_grid(ax)
        ax.margins(x=0.0)

    top.plot(
        time_min,
        c_eff,
        color=PALETTE_A["key"],
        linewidth=2.8,
        label=r"$c_{\mathrm{app}}$",
        zorder=3,
    )
    top.set_ylabel(r"表观比热 / $\mathrm{kJ\,(kg\,K)^{-1}}$")
    top.legend(loc="upper right")
    add_panel_label(top, "(a)")

    bottom.plot(
        time_min,
        latent,
        color=PALETTE_B["emphasis"],
        linewidth=2.8,
        label=r"$Q_{\mathrm{lat}}$",
        zorder=3,
    )
    bottom.set_xlabel("时间 / min")
    bottom.set_ylabel("累计潜热 / kJ")
    bottom.legend(loc="lower right")
    add_panel_label(bottom, "(b)")

    saved = finalize_figure(
        fig,
        output_dir,
        "problem1_pcm_time_process",
        formats,
        dpi=300,
    )
    peak_index = int(frame["c_eff_pcm_J_kgK"].idxmax())
    return saved, {
        "phase_start_s": start_s,
        "phase_end_s": end_s,
        "peak_c_eff_kJ_kgK": float(frame.loc[peak_index, "c_eff_pcm_J_kgK"]) / 1000.0,
        "peak_c_eff_time_s": float(frame.loc[peak_index, "time_s"]),
        "total_latent_heat_kJ": float(frame["pcm_latent_released_J"].max()) / 1000.0,
    }


def _plot_flux_group(
    ax: Axes,
    frame: pd.DataFrame,
    time: np.ndarray,
    columns: tuple[str, ...],
    panel_label: str,
    xlabel: str,
    *,
    show_legend: bool,
) -> None:
    """Render one uncluttered group of heat-flux curves."""

    for column in columns:
        label, color, linestyle = FLUX_STYLES[column]
        ax.plot(
            time,
            frame[column].to_numpy(dtype=float),
            color=color,
            linewidth=2.5,
            linestyle=linestyle,
            label=label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"热流密度 / $\mathrm{W\,m^{-2}}$")
    add_minimal_y_grid(ax)
    add_panel_label(ax, panel_label)
    ax.margins(x=0.0)
    if show_legend:
        ax.legend(loc="best", ncols=len(columns))


def _downsample_full_series(frame: pd.DataFrame, maximum_points: int = 1200) -> pd.DataFrame:
    """Reduce vector-file complexity while retaining both endpoints."""

    if len(frame) <= maximum_points:
        return frame
    stride = int(np.ceil((len(frame) - 1) / (maximum_points - 1)))
    positions = np.unique(np.r_[np.arange(0, len(frame), stride), len(frame) - 1])
    return frame.iloc[positions]


def plot_heat_flux_pathways(
    frame: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
    early_minutes: float,
) -> list[Path]:
    """Plot all five pathway heat fluxes without crowding a single axis."""

    if early_minutes <= 0.0:
        raise ValueError("early_minutes 必须为正数。")
    early = frame.loc[frame["time_s"] <= early_minutes * 60.0]
    if len(early) < 2:
        raise ValueError("早期时间窗内的数据点不足。")
    full = _downsample_full_series(frame)

    human_columns = ("q_cs_W_m2", "q_s1_W_m2")
    clothing_columns = ("q_12_W_m2", "q_23_W_m2", "q_3inf_W_m2")
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0))

    _plot_flux_group(
        axes[0, 0],
        early,
        early["time_s"].to_numpy(dtype=float) / 60.0,
        human_columns,
        "(a)",
        "早期时间 / min",
        show_legend=True,
    )
    _plot_flux_group(
        axes[0, 1],
        early,
        early["time_s"].to_numpy(dtype=float) / 60.0,
        clothing_columns,
        "(b)",
        "早期时间 / min",
        show_legend=True,
    )
    _plot_flux_group(
        axes[1, 0],
        full,
        full["time_s"].to_numpy(dtype=float) / 3600.0,
        human_columns,
        "(c)",
        "暴露时间 / h",
        show_legend=False,
    )
    _plot_flux_group(
        axes[1, 1],
        full,
        full["time_s"].to_numpy(dtype=float) / 3600.0,
        clothing_columns,
        "(d)",
        "暴露时间 / h",
        show_legend=False,
    )

    return finalize_figure(
        fig,
        output_dir,
        "problem1_heat_flux_pathways",
        formats,
        dpi=300,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "pdf", "svg"),
        choices=("png", "pdf", "svg"),
    )
    parser.add_argument("--early-minutes", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    font_path = apply_publication_style()
    frame = load_timeseries(args.input.resolve())
    pcm_paths, metrics = plot_pcm_time_process(frame, args.output_dir, args.formats)
    flux_paths = plot_heat_flux_pathways(
        frame,
        args.output_dir,
        args.formats,
        args.early_minutes,
    )

    print(f"Font: {font_path}")
    for path in (*pcm_paths, *flux_paths):
        print(f"Saved: {path}")
    print(
        "PCM phase interval: "
        f"{metrics['phase_start_s']:.1f}-{metrics['phase_end_s']:.1f} s; "
        f"peak c_app={metrics['peak_c_eff_kJ_kgK']:.3f} kJ/(kg K) "
        f"at {metrics['peak_c_eff_time_s']:.1f} s; "
        f"latent={metrics['total_latent_heat_kJ']:.3f} kJ"
    )


if __name__ == "__main__":
    main()
