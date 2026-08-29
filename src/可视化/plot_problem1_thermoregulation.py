"""Plot Problem 1 temperature and skin-blood-flow evolution.

The figure reads the reviewed Problem 1 CSV outputs without rerunning the model.
It follows the local scientific-figure-making skill: STZhongsong typography,
balanced A/B colors, short panel titles, and PNG/PDF publication export.
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
from matplotlib.figure import Figure


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = REPOSITORY_ROOT / "results" / "outputs" / "problem1"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "paper" / "figures"
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


def verify_required_font() -> Path:
    """Return the STZhongsong font path or fail without silent fallback."""

    try:
        resolved = font_manager.findfont(REQUIRED_FONT_FAMILY, fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(
            "未检测到华文中宋（STZhongsong），请安装该字体后再生成论文图。"
        ) from exc
    return Path(resolved)


def apply_publication_style() -> None:
    """Apply the local scientific-figure-making publication style."""

    verify_required_font()
    plt.rcParams.update(
        {
            "font.family": REQUIRED_FONT_FAMILY,
            "font.size": 18.0,
            "mathtext.fontset": "custom",
            "mathtext.rm": REQUIRED_FONT_FAMILY,
            "mathtext.it": REQUIRED_FONT_FAMILY,
            "mathtext.bf": REQUIRED_FONT_FAMILY,
            "text.usetex": False,
            "axes.labelsize": 18.0,
            "axes.titlesize": 18.0,
            "axes.titleweight": "normal",
            "axes.linewidth": 2.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "xtick.major.size": 6.0,
            "ytick.major.size": 6.0,
            "legend.fontsize": 16.0,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
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
        raise ValueError(f"{source} 缺少字段：{missing}")


def load_results(result_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the reviewed Problem 1 time series and summary."""

    timeseries_path = result_dir / "main_timeseries.csv"
    summary_path = result_dir / "main_summary.csv"
    timeseries = pd.read_csv(timeseries_path)
    summary_frame = pd.read_csv(summary_path)
    if len(summary_frame) != 1:
        raise ValueError(f"{summary_path} 应当恰好包含一行")
    _require_columns(
        timeseries,
        [
            "time_s",
            "T_core_C",
            "T_skin_C",
            "skin_blood_flow_actual_L_m2_h",
        ],
        timeseries_path,
    )
    _require_columns(
        summary_frame,
        [
            "t15_s",
            "t10_s",
            "skin_blood_flow_at_t15",
            "skin_blood_flow_at_t10",
        ],
        summary_path,
    )
    return timeseries, summary_frame.iloc[0]


def _save_figure(
    fig: Figure,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    basename = "problem1_temperature_blood_flow_evolution"
    for suffix in formats:
        normalized = suffix.lower().lstrip(".")
        if normalized not in {"png", "pdf", "svg"}:
            raise ValueError(f"不支持的图像格式：{suffix}")
        path = output_dir / f"{basename}.{normalized}"
        kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "pad_inches": 0.06,
            "facecolor": "white",
        }
        if normalized == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def plot_thermoregulation_evolution(
    result_dir: Path = DEFAULT_RESULT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    formats: Iterable[str] = ("png", "pdf"),
) -> tuple[list[Path], dict[str, float]]:
    """Plot core/skin temperatures and actual skin blood flow over time."""

    apply_publication_style()
    timeseries, summary = load_results(Path(result_dir))

    time_h = timeseries["time_s"].to_numpy(dtype=float) / 3600.0
    core = timeseries["T_core_C"].to_numpy(dtype=float)
    skin = timeseries["T_skin_C"].to_numpy(dtype=float)
    blood_flow = timeseries["skin_blood_flow_actual_L_m2_h"].to_numpy(dtype=float)
    t15_h = float(summary["t15_s"]) / 3600.0
    t10_h = float(summary["t10_s"]) / 3600.0
    flow15 = float(summary["skin_blood_flow_at_t15"])
    flow10 = float(summary["skin_blood_flow_at_t10"])

    fig, (ax_temperature, ax_flow) = plt.subplots(
        2,
        1,
        figsize=(10.6, 8.2),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )

    skin_line = ax_temperature.plot(
        time_h,
        skin,
        color=PALETTE_A["key"],
        linewidth=3.2,
        label=r"皮肤温度 $T_s$",
        zorder=5,
    )[0]
    core_line = ax_temperature.plot(
        time_h,
        core,
        color=PALETTE_B["emphasis"],
        linewidth=2.8,
        label=r"核心温度 $T_c$",
        zorder=4,
    )[0]
    for threshold in (15.0, 10.0):
        ax_temperature.axhline(
            threshold,
            color=PALETTE_B["reference"],
            linewidth=1.8,
            linestyle=(0, (6, 4)),
            zorder=1,
        )
    ax_temperature.text(
        0.28,
        15.7,
        r"$T_s=15\,^\circ\mathrm{C}$",
        fontsize=15.0,
        color="black",
        ha="left",
        va="bottom",
    )
    ax_temperature.text(
        0.28,
        10.7,
        r"$T_s=10\,^\circ\mathrm{C}$",
        fontsize=15.0,
        color="black",
        ha="left",
        va="bottom",
    )
    ax_temperature.scatter(
        [t15_h, t10_h],
        [15.0, 10.0],
        s=78,
        color=PALETTE_A["key"],
        edgecolor="white",
        linewidth=1.2,
        zorder=7,
    )
    ax_temperature.annotate(
        rf"$t_{{15}}={t15_h:.2f}\,\mathrm{{h}}$",
        xy=(t15_h, 15.0),
        xytext=(t15_h + 0.55, 18.1),
        fontsize=15.0,
        color=PALETTE_A["key"],
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "color": PALETTE_A["key"], "lw": 1.5},
    )
    ax_temperature.annotate(
        rf"$t_{{10}}={t10_h:.2f}\,\mathrm{{h}}$",
        xy=(t10_h, 10.0),
        xytext=(t10_h - 0.65, 13.0),
        fontsize=15.0,
        color=PALETTE_A["key"],
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "->", "color": PALETTE_A["key"], "lw": 1.5},
    )
    ax_temperature.set_title("(a) 温度响应", loc="left", pad=10)
    ax_temperature.set_ylabel(r"温度 $T$ / $^\circ\mathrm{C}$")
    ax_temperature.set_ylim(8.0, 39.5)
    ax_temperature.set_yticks([10, 15, 20, 25, 30, 35])

    flow_line = ax_flow.plot(
        time_h,
        blood_flow,
        color=PALETTE_A["secondary"],
        linewidth=3.2,
        label=r"实际皮肤血流 $\dot V_{bl}$",
        zorder=5,
    )[0]
    ax_flow.scatter(
        [t15_h, t10_h],
        [flow15, flow10],
        s=78,
        color=PALETTE_A["secondary"],
        edgecolor="white",
        linewidth=1.2,
        zorder=7,
    )
    ax_flow.annotate(
        f"{flow15:.4f}",
        xy=(t15_h, flow15),
        xytext=(t15_h + 0.45, flow15 + 0.62),
        fontsize=15.0,
        color="black",
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.3},
    )
    ax_flow.annotate(
        f"{flow10:.4f}",
        xy=(t10_h, flow10),
        xytext=(t10_h - 0.55, flow10 + 0.62),
        fontsize=15.0,
        color="black",
        ha="right",
        va="center",
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.3},
    )
    ax_flow.set_title("(b) 皮肤血流", loc="left", pad=10)
    ax_flow.set_xlabel(r"时间 $t$ / $\mathrm{h}$")
    ax_flow.set_ylabel(r"$\dot V_{bl}$ / $\mathrm{L}\,(\mathrm{m}^2\,\mathrm{h})^{-1}$")
    ax_flow.set_ylim(0.25, 6.65)
    ax_flow.set_yticks([0.5, 2.0, 3.5, 5.0, 6.3])

    for axis in (ax_temperature, ax_flow):
        axis.set_xlim(0.0, max(float(time_h.max()) * 1.035, t10_h * 1.035))
        axis.axvline(
            t15_h,
            color=PALETTE_B["reference"],
            linewidth=1.6,
            linestyle=(0, (6, 4)),
            zorder=1,
        )
        axis.axvline(
            t10_h,
            color=PALETTE_B["reference"],
            linewidth=1.6,
            linestyle=(0, (6, 4)),
            zorder=1,
        )

    fig.legend(
        [skin_line, core_line, flow_line],
        [skin_line.get_label(), core_line.get_label(), flow_line.get_label()],
        loc="outside upper center",
        ncol=3,
        handlelength=2.8,
        columnspacing=1.6,
    )
    metrics = {
        "t15_h": t15_h,
        "t10_h": t10_h,
        "flow_at_t15": flow15,
        "flow_at_t10": flow10,
    }
    return _save_figure(fig, Path(output_dir), formats), metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png", "pdf"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths, metrics = plot_thermoregulation_evolution(
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        formats=args.formats,
    )
    print(
        f"t15={metrics['t15_h']:.3f} h, t10={metrics['t10_h']:.3f} h, "
        f"Vbl(t15)={metrics['flow_at_t15']:.4f}, "
        f"Vbl(t10)={metrics['flow_at_t10']:.4f}"
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
