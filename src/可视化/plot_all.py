"""Generate the three reviewed Problem 2 paper figures from existing CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_loader import load_visualization_data, validate_events
from plot_config import DEFAULT_FIGURE_DIR, DEFAULT_FORMATS, apply_publication_style
from plot_fig4_temperature import plot_fig4
from plot_fig5_q1_q2_skin import plot_fig5
from plot_fig6_effect_decomposition import plot_fig6


def generate_all_figures(
    output_dir: Path = DEFAULT_FIGURE_DIR,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> dict[str, list[Path]]:
    apply_publication_style()
    data = load_visualization_data()
    return {
        "fig4": plot_fig4(data.q2_series, data.q2_summary, output_dir, formats),
        "fig5": plot_fig5(
            data.q1_series,
            data.q1_summary,
            data.q2_series,
            data.q2_summary,
            output_dir,
            formats,
        ),
        "fig6": plot_fig6(data.effects, output_dir, formats),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("svg", "pdf", "png"),
        default=list(DEFAULT_FORMATS),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    apply_publication_style()
    data = load_visualization_data()
    q1_checks = validate_events(data.q1_series, data.q1_summary)
    q2_checks = validate_events(data.q2_series, data.q2_summary)

    outputs = {
        "fig4": plot_fig4(data.q2_series, data.q2_summary, args.output_dir, args.formats),
        "fig5": plot_fig5(
            data.q1_series,
            data.q1_summary,
            data.q2_series,
            data.q2_summary,
            args.output_dir,
            args.formats,
        ),
        "fig6": plot_fig6(data.effects, args.output_dir, args.formats),
    }

    delta_min = data.q2_summary.t15_min - data.q1_summary.t15_min
    delta_pct = delta_min / data.q1_summary.t15_min * 100.0
    print("[Fig. 4]")
    print(f"Q2 t15 = {data.q2_summary.t15_min:.2f} min")
    print(f"Q2 t10 = {data.q2_summary.t10_min:.2f} min")
    print(f"event check = {q2_checks['t15_error_C']:+.3e} / {q2_checks['t10_error_C']:+.3e} degC")
    print("\n[Fig. 5]")
    print(f"Q1 t15 = {data.q1_summary.t15_min:.2f} min")
    print(f"Q2 t15 = {data.q2_summary.t15_min:.2f} min")
    print(f"Delta t15 = {delta_min:.2f} min ({delta_pct:.2f}%)")
    print(f"Q1 event check = {q1_checks['t15_error_C']:+.3e} / {q1_checks['t10_error_C']:+.3e} degC")
    print("\n[Fig. 6]")
    q1_t15 = data.effects[0].t15_min
    for scenario in data.effects:
        pct = (scenario.t15_min - q1_t15) / q1_t15 * 100.0
        print(f"{scenario.case_id:<2} = {scenario.t15_min:.2f} min ({pct:+.2f}% vs Q1)")
    print("\n[Saved]")
    for paths in outputs.values():
        for path in paths:
            print(path.resolve())


if __name__ == "__main__":
    main()
