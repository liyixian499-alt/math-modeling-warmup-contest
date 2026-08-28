"""Generate publication-ready figures for Problem 3.

The script reads the reviewed CSV outputs under ``results/outputs/problem3``.
It does not rerun or alter the optimization model.  Final figures are exported
to ``paper/figures`` in both vector PDF and 300 dpi PNG formats.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.text import Text


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = REPOSITORY_ROOT / "results" / "outputs" / "problem3"
DEFAULT_FIGURE_DIR = REPOSITORY_ROOT / "paper" / "figures"
REQUIRED_FONT_FAMILY = "STZhongsong"

PALETTE_A = {
    "light": "#BFDFD2",
    "key": "#51999F",
    "secondary": "#4198AC",
    "positive": "#7BC0CD",
}
PALETTE_B = {
    "reference": "#ECB66C",
    "baseline": "#DBCB92",
    "comparator": "#EA9E58",
    "emphasis": "#ED8D5A",
}

LAYER_COLORS = {
    1: PALETTE_B["reference"],
    2: PALETTE_B["comparator"],
    3: PALETTE_A["secondary"],
    4: PALETTE_A["key"],
}
LAYER_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}


def verify_required_font() -> Path:
    """Return the resolved STZhongsong path or fail without fallback."""

    try:
        resolved = font_manager.findfont(
            REQUIRED_FONT_FAMILY,
            fallback_to_default=False,
        )
    except ValueError as exc:
        raise RuntimeError(
            "Required font STZhongsong (华文中宋) is unavailable. "
            "Install it before generating Problem 3 figures."
        ) from exc
    return Path(resolved)


def apply_publication_style() -> None:
    """Apply the scientific-figure-making skill's required house style."""

    verify_required_font()
    plt.rcParams.update(
        {
            "font.family": REQUIRED_FONT_FAMILY,
            "font.size": 18.0,
            "axes.titlesize": 18.0,
            "axes.labelsize": 18.0,
            "axes.linewidth": 2.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "legend.fontsize": 16.0,
            "legend.frameon": False,
            "lines.linewidth": 2.5,
            "lines.solid_capstyle": "round",
            "mathtext.fontset": "custom",
            "mathtext.rm": REQUIRED_FONT_FAMILY,
            "mathtext.it": REQUIRED_FONT_FAMILY,
            "mathtext.bf": REQUIRED_FONT_FAMILY,
            "text.usetex": False,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _require_columns(frame: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def _read_csv(result_dir: Path, filename: str, required: Iterable[str]) -> pd.DataFrame:
    path = result_dir / filename
    frame = pd.read_csv(path)
    _require_columns(frame, required, path)
    if frame.empty:
        raise ValueError(f"No rows found in {path}")
    return frame


def load_problem3_results(result_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and validate every reviewed dataset used by the figures."""

    result_dir = Path(result_dir)
    candidate = _read_csv(
        result_dir,
        "candidate_summary.csv",
        [
            "outer_layer_count",
            "outer_thickness_mm",
            "garment_mass_kg",
            "maximum_external_load_kg",
            "total_cost_yuan",
            "maximum_total_cost_yuan",
            "standing_time_score_min",
            "safe_score_min",
            "t15_min",
            "t_core_35_min",
        ],
    ).sort_values("outer_layer_count")
    timeseries = _read_csv(
        result_dir,
        "candidate_timeseries.csv",
        [
            "outer_layer_count",
            "time_s",
            "T_core_C",
            "T_skin_C",
            "T_layer1_C",
            "T_pcm_C",
            "T_outer_surface_C",
            "T_outer_1_C",
            "T_outer_2_C",
            "T_outer_3_C",
            "T_outer_4_C",
        ],
    )
    thermal = _read_csv(
        result_dir,
        "sensitivity_thermal.csv",
        ["parameter", "factor", "relative_change_pct"],
    )
    initial_temperature = _read_csv(
        result_dir,
        "sensitivity_initial_temperature.csv",
        [
            "clothing_initial_temperature_C",
            "t_core_35_min",
            "relative_change_pct",
        ],
    ).sort_values("clothing_initial_temperature_C")
    weight_penalty = _read_csv(
        result_dir,
        "sensitivity_weight_penalty.csv",
        [
            "weight_penalty_s_per_kg",
            "outer_layer_count",
            "standing_time_score_min",
            "selected",
            "is_problem_coefficient",
        ],
    )
    manufacturing_cost = _read_csv(
        result_dir,
        "sensitivity_manufacturing_cost.csv",
        [
            "manufacturing_overhead_rate",
            "outer_layer_count",
            "effective_total_cost_yuan",
            "maximum_total_cost_yuan",
            "feasible",
            "selected",
        ],
    )

    expected_layers = np.array([1, 2, 3, 4])
    actual_layers = candidate["outer_layer_count"].to_numpy(dtype=int)
    if not np.array_equal(actual_layers, expected_layers):
        raise ValueError(f"Expected candidate layers 1--4, got {actual_layers.tolist()}")
    for name, frame in {
        "candidate": candidate,
        "timeseries": timeseries,
        "thermal": thermal,
        "initial_temperature": initial_temperature,
        "weight_penalty": weight_penalty,
        "manufacturing_cost": manufacturing_cost,
    }.items():
        if frame.isna().all(axis=1).any():
            raise ValueError(f"Completely empty row found in {name}")

    return {
        "candidate": candidate,
        "timeseries": timeseries,
        "thermal": thermal,
        "initial_temperature": initial_temperature,
        "weight_penalty": weight_penalty,
        "manufacturing_cost": manufacturing_cost,
    }


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18.0,
        color="black",
        fontfamily=REQUIRED_FONT_FAMILY,
    )


def _style_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(
        axis=axis,
        color=PALETTE_A["light"],
        linewidth=0.9,
        alpha=0.58,
        zorder=0,
    )


def _enforce_figure_font(fig: Figure) -> None:
    for artist in fig.findobj(match=Text):
        artist.set_fontfamily(REQUIRED_FONT_FAMILY)


def save_figure(
    fig: Figure,
    output_dir: Path,
    basename: str,
    formats: Iterable[str],
) -> list[Path]:
    """Export one figure with editable text and print-safe padding."""

    _enforce_figure_font(fig)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for suffix in formats:
        normalized = suffix.lower().lstrip(".")
        if normalized not in {"png", "pdf", "svg"}:
            raise ValueError(f"Unsupported figure format: {suffix}")
        path = output_dir / f"{basename}.{normalized}"
        kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "pad_inches": 0.08,
            "facecolor": "white",
        }
        if normalized == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def plot_candidate_overview(
    candidate: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    """Compare objective values and quantify both hard-constraint margins."""

    n = candidate["outer_layer_count"].to_numpy(dtype=int)
    main_score = candidate["standing_time_score_min"].to_numpy(dtype=float)
    safe_score = candidate["safe_score_min"].to_numpy(dtype=float)
    cost_use = (
        candidate["total_cost_yuan"].to_numpy(dtype=float)
        / candidate["maximum_total_cost_yuan"].to_numpy(dtype=float)
        * 100.0
    )
    load_use = (
        candidate["garment_mass_kg"].to_numpy(dtype=float)
        / candidate["maximum_external_load_kg"].to_numpy(dtype=float)
        * 100.0
    )

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.5), layout="constrained")
    panels = [
        (axes[0], main_score, "题意主指标", r"$J$ / min", "(a)"),
        (axes[1], safe_score, "安全修正指标", r"$J_{safe}$ / min", "(b)"),
    ]
    for ax, values, title, ylabel, panel in panels:
        ax.plot(
            n,
            values,
            color=PALETTE_A["key"],
            marker="o",
            markersize=8.5,
            markeredgecolor="black",
            markeredgewidth=1.0,
            zorder=3,
        )
        ax.scatter(
            [n[-1]],
            [values[-1]],
            s=130,
            color=PALETTE_B["emphasis"],
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            f"{values[-1]:.2f}",
            xy=(n[-1], values[-1]),
            xytext=(-12, 12),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=11.5,
            color="black",
        )
        ax.set_title(title)
        _panel_label(ax, panel)
        ax.set_xlabel(r"外层总层数 $N$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(n)
        span = float(values.max() - values.min())
        ax.set_ylim(values.min() - 0.16 * span, values.max() + 0.18 * span)
        _style_grid(ax)

    axes[2].plot(
        n,
        cost_use,
        color=PALETTE_B["emphasis"],
        marker="s",
        markersize=8.5,
        markeredgecolor="black",
        markeredgewidth=1.0,
        label="成本约束",
        zorder=3,
    )
    axes[2].plot(
        n,
        load_use,
        color=PALETTE_A["secondary"],
        marker="D",
        markersize=7.5,
        markeredgecolor="black",
        markeredgewidth=1.0,
        label="承重约束",
        zorder=3,
    )
    axes[2].axhline(100.0, color="black", linewidth=1.7, linestyle="--", zorder=2)
    axes[2].set_title("约束利用率")
    _panel_label(axes[2], "(c)")
    axes[2].set_xlabel(r"外层总层数 $N$")
    axes[2].set_ylabel("约束利用率 / %")
    axes[2].set_xticks(n)
    axes[2].set_ylim(0.0, 108.0)
    axes[2].legend(loc="center right")
    _style_grid(axes[2])
    return save_figure(fig, output_dir, "problem3_candidate_overview", formats)


def plot_temperature_comparison(
    candidate: pd.DataFrame,
    timeseries: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    """Compare core and skin trajectories for all four feasible layer counts."""

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.1), layout="constrained")
    for layer_count in range(1, 5):
        subset = timeseries.loc[
            timeseries["outer_layer_count"].astype(int) == layer_count
        ].sort_values("time_s")
        time_h = subset["time_s"].to_numpy(dtype=float) / 3600.0
        for ax, column in zip(axes, ["T_core_C", "T_skin_C"], strict=True):
            ax.plot(
                time_h,
                subset[column].to_numpy(dtype=float),
                color=LAYER_COLORS[layer_count],
                marker=LAYER_MARKERS[layer_count],
                markevery=max(len(subset) // 10, 1),
                markersize=5.0,
                markeredgecolor="black",
                markeredgewidth=0.55,
                label=rf"$N={layer_count}$",
                zorder=3,
            )

        row = candidate.loc[
            candidate["outer_layer_count"].astype(int) == layer_count
        ].iloc[0]
        axes[0].scatter(
            [float(row["t_core_35_min"]) / 60.0],
            [35.0],
            s=55,
            color=LAYER_COLORS[layer_count],
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
        )
        axes[1].scatter(
            [float(row["t15_min"]) / 60.0],
            [15.0],
            s=55,
            color=LAYER_COLORS[layer_count],
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
        )

    for ax, title, panel, ylabel in [
        (axes[0], "核心温度", "(a)", r"$T_c$ / ℃"),
        (axes[1], "皮肤温度", "(b)", r"$T_s$ / ℃"),
    ]:
        ax.set_title(title)
        _panel_label(ax, panel)
        ax.set_xlabel(r"时间 $t$ / h")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, 12.6)
        _style_grid(ax)

    axes[0].axhline(35.0, color="black", linewidth=1.6, linestyle="--", zorder=1)
    axes[0].text(0.25, 35.18, r"$T_c=35$ ℃", fontsize=11.5)
    axes[0].set_ylim(25.0, 38.3)
    axes[1].axhline(15.0, color="black", linewidth=1.6, linestyle="--", zorder=1)
    axes[1].text(0.25, 15.45, r"$T_s=15$ ℃", fontsize=11.5)
    axes[1].set_ylim(13.0, 39.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        handlelength=2.8,
        columnspacing=1.8,
    )
    return save_figure(fig, output_dir, "problem3_temperature_comparison", formats)


def plot_optimal_temperature_field(
    candidate: pd.DataFrame,
    timeseries: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    """Render the node-wise cooling field of the selected four-layer design."""

    optimal = candidate.loc[candidate["outer_layer_count"].astype(int) == 4].iloc[0]
    safe_limit_s = float(optimal["t_core_35_min"]) * 60.0
    subset = timeseries.loc[
        (timeseries["outer_layer_count"].astype(int) == 4)
        & (timeseries["time_s"].to_numpy(dtype=float) <= safe_limit_s)
    ].sort_values("time_s")
    columns = [
        "T_core_C",
        "T_skin_C",
        "T_layer1_C",
        "T_pcm_C",
        "T_outer_1_C",
        "T_outer_2_C",
        "T_outer_3_C",
        "T_outer_4_C",
    ]
    labels = [
        r"核心 $T_c$",
        r"皮肤 $T_s$",
        r"内层 $T_1$",
        r"PCM $T_2$",
        r"外层 $T_{3,1}$",
        r"外层 $T_{3,2}$",
        r"外层 $T_{3,3}$",
        r"外层 $T_{3,4}$",
    ]
    matrix = subset[columns].to_numpy(dtype=float).T
    time_h = subset["time_s"].to_numpy(dtype=float) / 3600.0
    cmap = LinearSegmentedColormap.from_list(
        "approved_ab",
        [PALETTE_A["key"], "#FFFFFF", PALETTE_B["emphasis"]],
    )
    norm = TwoSlopeNorm(vmin=-40.0, vcenter=0.0, vmax=40.0)

    fig, ax = plt.subplots(figsize=(11.8, 6.5), layout="constrained")
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[float(time_h.min()), float(time_h.max()), -0.5, len(labels) - 0.5],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xlabel(r"时间 $t$ / h")
    ax.set_ylabel("温度节点", labelpad=42)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlim(float(time_h.min()), float(time_h.max()))
    colorbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.045)
    colorbar.set_label(r"温度 $T$ / ℃")
    colorbar.ax.yaxis.label.set_fontfamily(REQUIRED_FONT_FAMILY)
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontfamily(REQUIRED_FONT_FAMILY)
    return save_figure(fig, output_dir, "problem3_optimal_temperature_field", formats)


def plot_sensitivity_and_robustness(
    thermal: pd.DataFrame,
    initial_temperature: pd.DataFrame,
    weight_penalty: pd.DataFrame,
    manufacturing_cost: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    """Combine local sensitivity and design-robustness evidence."""

    parameter_order = [
        "metabolic_rate_W_m2",
        "h_in_W_m2K",
        "lambda_out",
        "outer_conductivity_W_mK",
        "blood_flow_time_constant_s",
    ]
    parameter_labels = {
        "metabolic_rate_W_m2": r"$M_0$",
        "h_in_W_m2K": r"$h_{in}$",
        "lambda_out": r"$\lambda_{out}$",
        "outer_conductivity_W_mK": r"$k_3$",
        "blood_flow_time_constant_s": r"$\tau_{bl}$",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 11.2), layout="constrained")
    ax_thermal, ax_initial, ax_weight, ax_cost = axes.flat

    low_values: list[float] = []
    high_values: list[float] = []
    for parameter in parameter_order:
        rows = thermal.loc[thermal["parameter"] == parameter].sort_values("factor")
        if len(rows) != 2:
            raise ValueError(f"Expected two thermal scenarios for {parameter}")
        low_values.append(float(rows.iloc[0]["relative_change_pct"]))
        high_values.append(float(rows.iloc[1]["relative_change_pct"]))
    y = np.arange(len(parameter_order), dtype=float)
    bar_height = 0.32
    ax_thermal.barh(
        y - bar_height / 2,
        low_values,
        height=bar_height,
        color=PALETTE_A["key"],
        edgecolor="black",
        linewidth=1.0,
        label="低水平",
        zorder=3,
    )
    ax_thermal.barh(
        y + bar_height / 2,
        high_values,
        height=bar_height,
        color=PALETTE_B["emphasis"],
        edgecolor="black",
        linewidth=1.0,
        hatch="//",
        label="高水平",
        zorder=3,
    )
    ax_thermal.axvline(0.0, color="black", linewidth=1.5, zorder=2)
    ax_thermal.set_yticks(y, [parameter_labels[item] for item in parameter_order])
    ax_thermal.invert_yaxis()
    ax_thermal.set_xlim(-18.5, 21.0)
    ax_thermal.set_xlabel(r"$t_{c,35}$ 相对变化 / %")
    ax_thermal.set_title("热学参数敏感性")
    _panel_label(ax_thermal, "(a)")
    ax_thermal.legend(loc="lower right")
    _style_grid(ax_thermal, axis="x")

    initial_x = initial_temperature["clothing_initial_temperature_C"].to_numpy(dtype=float)
    initial_y = initial_temperature["t_core_35_min"].to_numpy(dtype=float)
    ax_initial.plot(
        initial_x,
        initial_y,
        color=PALETTE_A["key"],
        marker="o",
        markersize=8.0,
        markeredgecolor="black",
        markeredgewidth=0.9,
        zorder=3,
    )
    ax_initial.scatter(
        [initial_x[-1]],
        [initial_y[-1]],
        s=120,
        color=PALETTE_B["emphasis"],
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
    )
    ax_initial.set_xlabel(r"服装初始温度 / ℃")
    ax_initial.set_ylabel(r"$t_{c,35}$ / min")
    ax_initial.set_title("服装初始温度")
    _panel_label(ax_initial, "(b)")
    ax_initial.set_xticks(initial_x)
    ax_initial.set_ylim(262.0, 273.5)
    _style_grid(ax_initial)

    penalty_values = np.sort(
        weight_penalty["weight_penalty_s_per_kg"].unique().astype(float)
    )
    for layer_count in range(1, 5):
        rows = weight_penalty.loc[
            weight_penalty["outer_layer_count"].astype(int) == layer_count
        ].sort_values("weight_penalty_s_per_kg")
        ax_weight.plot(
            rows["weight_penalty_s_per_kg"].to_numpy(dtype=float) / 1000.0,
            rows["standing_time_score_min"].to_numpy(dtype=float),
            color=LAYER_COLORS[layer_count],
            marker=LAYER_MARKERS[layer_count],
            markersize=6.0,
            markeredgecolor="black",
            markeredgewidth=0.55,
            label=rf"$N={layer_count}$",
            zorder=3,
        )
    ax_weight.axvline(6.067, color="black", linewidth=1.6, linestyle="--", zorder=2)
    ax_weight.scatter(
        [0.02],
        [
            float(
                weight_penalty.loc[
                    (weight_penalty["outer_layer_count"].astype(int) == 4)
                    & np.isclose(weight_penalty["weight_penalty_s_per_kg"], 20.0),
                    "standing_time_score_min",
                ].iloc[0]
            )
        ],
        s=115,
        color=PALETTE_B["emphasis"],
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
    )
    ax_weight.set_xlim(-0.1, float(penalty_values.max()) / 1000.0 + 0.15)
    ax_weight.set_xlabel(r"重量惩罚系数 $\lambda_m$ / (10³ s/kg)")
    ax_weight.set_ylabel(r"$J$ / min")
    ax_weight.set_title("重量惩罚稳健性")
    _panel_label(ax_weight, "(c)")
    ax_weight.legend(loc="lower left", ncol=2)
    _style_grid(ax_weight)

    maximum_cost = float(manufacturing_cost["maximum_total_cost_yuan"].iloc[0])
    overhead_values = np.sort(
        manufacturing_cost["manufacturing_overhead_rate"].unique().astype(float)
    )
    for layer_count in range(1, 5):
        rows = manufacturing_cost.loc[
            manufacturing_cost["outer_layer_count"].astype(int) == layer_count
        ].sort_values("manufacturing_overhead_rate")
        ax_cost.plot(
            rows["manufacturing_overhead_rate"].to_numpy(dtype=float) * 100.0,
            rows["effective_total_cost_yuan"].to_numpy(dtype=float),
            color=LAYER_COLORS[layer_count],
            marker=LAYER_MARKERS[layer_count],
            markersize=6.0,
            markeredgecolor="black",
            markeredgewidth=0.55,
            label=rf"$N={layer_count}$",
            zorder=3,
        )
    ax_cost.axhline(maximum_cost, color="black", linewidth=1.6, linestyle="--", zorder=2)
    ax_cost.axvline(12.716, color="black", linewidth=1.6, linestyle=":", zorder=2)
    ax_cost.set_xlim(-0.4, float(overhead_values.max()) * 100.0 + 0.4)
    ax_cost.set_xlabel("制造附加率 / %")
    ax_cost.set_ylabel("总成本 / 元")
    ax_cost.set_title("制造附加成本")
    _panel_label(ax_cost, "(d)")
    ax_cost.legend(loc="upper left", ncol=2)
    _style_grid(ax_cost)

    return save_figure(fig, output_dir, "problem3_sensitivity_robustness", formats)


def generate_all_figures(
    result_dir: Path = DEFAULT_RESULT_DIR,
    output_dir: Path = DEFAULT_FIGURE_DIR,
    formats: Iterable[str] = ("png", "pdf"),
) -> dict[str, list[Path]]:
    """Generate the complete Problem 3 figure set."""

    result_dir = Path(result_dir)
    output_dir = Path(output_dir)
    formats = tuple(formats)
    apply_publication_style()
    results = load_problem3_results(result_dir)
    return {
        "candidate_overview": plot_candidate_overview(
            results["candidate"], output_dir, formats
        ),
        "temperature_comparison": plot_temperature_comparison(
            results["candidate"], results["timeseries"], output_dir, formats
        ),
        "optimal_temperature_field": plot_optimal_temperature_field(
            results["candidate"], results["timeseries"], output_dir, formats
        ),
        "sensitivity_robustness": plot_sensitivity_and_robustness(
            results["thermal"],
            results["initial_temperature"],
            results["weight_penalty"],
            results["manufacturing_cost"],
            output_dir,
            formats,
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = generate_all_figures(
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        formats=args.formats,
    )
    for paths in outputs.values():
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
