"""Plot Figure 6: Q1/W/I/M/Q2 formal t15 scenario comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from data_loader import EffectScenario, load_visualization_data
from plot_config import (
    AXIS_COLOR,
    DEFAULT_FIGURE_DIR,
    DEFAULT_FORMATS,
    PALETTE_A,
    PALETTE_B,
    add_minimal_y_grid,
    apply_publication_style,
    finalize_figure,
)


def plot_fig6(
    scenarios: tuple[EffectScenario, ...],
    output_dir: Path = DEFAULT_FIGURE_DIR,
    formats: Iterable[str] = DEFAULT_FORMATS,
) -> list[Path]:
    labels = [scenario.case_id for scenario in scenarios]
    values = np.asarray([scenario.t15_min for scenario in scenarios], dtype=float)
    colors = (
        PALETTE_B["reference"],
        PALETTE_A["secondary"],
        PALETTE_A["positive"],
        PALETTE_B["baseline"],
        PALETTE_A["key"],
    )

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    bars = ax.bar(
        labels,
        values,
        width=0.68,
        color=colors,
        edgecolor=AXIS_COLOR,
        linewidth=1.7,
        zorder=3,
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + values.max() * 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=16.0,
            color=AXIS_COLOR,
            zorder=5,
        )

    ax.set_xlabel("场景")
    ax.set_ylabel(r"$t_{15}$ / min")
    ax.set_ylim(0.0, values.max() * 1.13)
    add_minimal_y_grid(ax)
    return finalize_figure(
        fig,
        output_dir,
        "fig6_q2_t15_effect_decomposition",
        formats,
        dpi=600,
    )


def main() -> None:
    apply_publication_style()
    data = load_visualization_data()
    for path in plot_fig6(data.effects):
        print(path)


if __name__ == "__main__":
    main()
