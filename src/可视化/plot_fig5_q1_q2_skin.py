"""Plot Figure 5: formal Q1 versus Q2 skin-temperature histories."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from data_loader import EventSummary, TemperatureSeries, load_visualization_data
from plot_config import (
    DEFAULT_FIGURE_DIR,
    DEFAULT_FORMATS,
    PALETTE_A,
    PALETTE_B,
    REFERENCE_COLOR,
    add_minimal_y_grid,
    add_skin_thresholds,
    apply_publication_style,
    finalize_figure,
)


def plot_fig5(
    q1_series: TemperatureSeries,
    q1_summary: EventSummary,
    q2_series: TemperatureSeries,
    q2_summary: EventSummary,
    output_dir: Path = DEFAULT_FIGURE_DIR,
    formats: Iterable[str] = DEFAULT_FORMATS,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    ax.plot(
        q1_series.time_min,
        q1_series.T_skin_C,
        color=PALETTE_B["emphasis"],
        linewidth=3.2,
        label="Q1",
        zorder=5,
    )
    ax.plot(
        q2_series.time_min,
        q2_series.T_skin_C,
        color=PALETTE_A["key"],
        linewidth=3.5,
        label="Q2",
        zorder=6,
    )
    add_skin_thresholds(ax)

    event_specs = (
        (q1_summary.t15_min, "Q1", PALETTE_B["emphasis"], 1.7),
        (q2_summary.t15_min, "Q2", PALETTE_A["key"], -2.9),
    )
    for event_min, label, color, offset in event_specs:
        ax.scatter(
            [event_min],
            [15.0],
            s=80,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=9,
        )
        ax.vlines(
            event_min,
            ymin=9.0,
            ymax=15.0,
            color=REFERENCE_COLOR,
            linewidth=1.2,
            linestyle=(0, (2, 5)),
            alpha=0.72,
            zorder=2,
        )
        ax.text(
            event_min,
            15.0 + offset,
            f"{label}: {event_min:.2f} min",
            ha="center",
            va="bottom" if offset > 0 else "top",
            fontsize=16.0,
            color=color,
            zorder=10,
        )

    x_label = max(10.0, max(q1_summary.t10_min, q2_summary.t10_min) * 0.02)
    ax.text(x_label, 15.65, "15 °C", color=REFERENCE_COLOR, fontsize=16.0, va="bottom")
    ax.text(x_label, 10.65, "10 °C", color=REFERENCE_COLOR, fontsize=16.0, va="bottom")
    ax.set_xlabel(r"时间 $t$ / min")
    ax.set_ylabel(r"皮肤温度 $T_s$ / °C")
    ax.set_xlim(0.0, max(q1_summary.t10_min, q2_summary.t10_min) * 1.015)
    ax.set_ylim(8.5, 38.5)
    ax.legend(loc="upper right", ncol=2, handlelength=2.8)
    add_minimal_y_grid(ax)
    return finalize_figure(
        fig,
        output_dir,
        "fig5_q1_q2_skin_temperature_comparison",
        formats,
        dpi=300,
    )


def main() -> None:
    apply_publication_style()
    data = load_visualization_data()
    for path in plot_fig5(
        data.q1_series,
        data.q1_summary,
        data.q2_series,
        data.q2_summary,
    ):
        print(path)


if __name__ == "__main__":
    main()
