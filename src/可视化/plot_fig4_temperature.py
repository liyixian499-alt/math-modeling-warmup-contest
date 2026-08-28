"""Plot Figure 4: five temperature states of the formal Q2 six-state model."""

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


def plot_fig4(
    series: TemperatureSeries,
    summary: EventSummary,
    output_dir: Path = DEFAULT_FIGURE_DIR,
    formats: Iterable[str] = DEFAULT_FORMATS,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.2, 7.2))
    curves = (
        (series.T_core_C, r"$T_c$", PALETTE_B["baseline"], 2.7, "-", 5),
        (series.T_skin_C, r"$T_s$", PALETTE_A["key"], 3.5, "-", 7),
        (series.T_layer1_C, r"$T_1$", PALETTE_A["positive"], 2.1, "--", 4),
        (series.T_pcm_C, r"$T_2$", PALETTE_A["secondary"], 2.5, "-.", 5),
        (series.T_layer3_C, r"$T_3$", PALETTE_B["reference"], 2.1, ":", 4),
    )
    for values, label, color, linewidth, linestyle, zorder in curves:
        ax.plot(
            series.time_min,
            values,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            zorder=zorder,
        )

    add_skin_thresholds(ax)
    events = (
        (summary.t15_min, 15.0, 1.8, 0.0, "center"),
        (summary.t10_min, 10.0, -1.8, -5.0, "right"),
    )
    for event_min, threshold, text_offset, text_dx, text_ha in events:
        ax.vlines(
            event_min,
            ymin=ax.get_ylim()[0],
            ymax=threshold,
            color=REFERENCE_COLOR,
            linewidth=1.2,
            linestyle=(0, (2, 5)),
            alpha=0.75,
            zorder=2,
        )
        ax.scatter(
            [event_min],
            [threshold],
            s=74,
            color=PALETTE_A["key"],
            edgecolor="white",
            linewidth=1.2,
            zorder=9,
        )
        ax.text(
            event_min + text_dx,
            threshold + text_offset,
            rf"$t_{{{int(threshold)}}}={event_min:.2f}$ min",
            ha=text_ha,
            va="bottom" if text_offset > 0.0 else "top",
            fontsize=16.0,
            color=PALETTE_A["key"],
            zorder=10,
        )

    x_label = max(8.0, summary.t10_min * 0.025)
    ax.text(x_label, 15.65, "15 °C", color=REFERENCE_COLOR, fontsize=16.0, va="bottom")
    ax.text(x_label, 10.65, "10 °C", color=REFERENCE_COLOR, fontsize=16.0, va="bottom")
    ax.set_xlabel(r"时间 $t$ / min")
    ax.set_ylabel(r"温度 $T$ / °C")
    ax.set_xlim(0.0, summary.t10_min * 1.025)
    ax.set_ylim(-35.0, 40.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=5,
        handlelength=2.8,
        columnspacing=1.5,
    )
    add_minimal_y_grid(ax)
    return finalize_figure(
        fig,
        output_dir,
        "fig4_q2_six_state_temperature",
        formats,
        dpi=300,
    )


def main() -> None:
    apply_publication_style()
    data = load_visualization_data()
    for path in plot_fig4(data.q2_series, data.q2_summary):
        print(path)


if __name__ == "__main__":
    main()
