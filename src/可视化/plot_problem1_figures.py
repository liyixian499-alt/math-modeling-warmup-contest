"""Generate publication-ready figures for Problem 1.

The script reads the reviewed Problem 1 CSV outputs and the raw DSC workbook.
It does not rerun or alter the thermal model.  By default, figures are exported
to ``paper/figures`` as both 300 dpi PNG files and editable-vector PDF files.
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
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.问题一.config import DEFAULT_DSC_PATH, ModelParameters
from src.问题一.dsc import PCMModel, load_pcm_model


DEFAULT_RESULT_DIR = REPOSITORY_ROOT / "results" / "outputs" / "problem1"
DEFAULT_FIGURE_DIR = REPOSITORY_ROOT / "paper" / "figures"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#6FBF73",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "gray": "#767676",
    "dark": "#272727",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

REQUIRED_PAPER_FONT = "STZhongsong"
TEMPERATURE_RESPONSE_RCPARAMS = {
    "font.family": REQUIRED_PAPER_FONT,
    "font.size": 18.0,
    "mathtext.fontset": "custom",
    "mathtext.rm": REQUIRED_PAPER_FONT,
    "mathtext.it": REQUIRED_PAPER_FONT,
    "mathtext.bf": REQUIRED_PAPER_FONT,
    "text.usetex": False,
    "axes.labelsize": 18.0,
    "axes.linewidth": 2.0,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.labelsize": 16.0,
    "ytick.labelsize": 16.0,
    "legend.fontsize": 15.5,
    "legend.frameon": False,
}
TEMPERATURE_RESPONSE_COLORS = {
    "skin": "#51999F",
    "core": "#ED8D5A",
    "layer1": "#7BC0CD",
    "pcm": "#EA9E58",
    "layer3": "#4198AC",
    "threshold": "#ECB66C",
}


def apply_publication_style() -> None:
    """Apply the figures4papers-inspired house style with Chinese fallbacks."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "axes.linewidth": 1.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "legend.fontsize": 10.5,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "mathtext.fontset": "dejavusans",
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


def _save_figure(
    fig: Figure,
    output_dir: Path,
    basename: str,
    formats: Iterable[str],
) -> list[Path]:
    """Save one figure with stable names and tight, publication-safe padding."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for suffix in formats:
        normalized = suffix.lower().lstrip(".")
        if normalized not in {"png", "pdf", "svg"}:
            raise ValueError(f"Unsupported figure format: {suffix}")
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


def _load_dsc_components(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the raw magnitude, linear baseline and latent DSC peak."""

    workbook = pd.ExcelFile(path)
    frame: pd.DataFrame | None = None
    for sheet in workbook.sheet_names:
        candidate = pd.read_excel(path, sheet_name=sheet)
        if candidate.shape[0] > 0 and candidate.shape[1] >= 2:
            frame = candidate.iloc[:, :2].copy()
            break
    if frame is None:
        raise ValueError(f"No usable two-column DSC sheet found in {path}")

    frame.columns = ["temperature_C", "heat_flow_mW_mg"]
    frame["temperature_C"] = pd.to_numeric(frame["temperature_C"], errors="coerce")
    frame["heat_flow_mW_mg"] = pd.to_numeric(frame["heat_flow_mW_mg"], errors="coerce")
    frame = (
        frame.dropna()
        .groupby("temperature_C", as_index=False)["heat_flow_mW_mg"]
        .mean()
        .sort_values("temperature_C")
    )
    temperature = frame["temperature_C"].to_numpy(dtype=float)
    magnitude = np.abs(frame["heat_flow_mW_mg"].to_numpy(dtype=float))
    baseline = magnitude[0] + (magnitude[-1] - magnitude[0]) * (
        temperature - temperature[0]
    ) / (temperature[-1] - temperature[0])
    latent = np.maximum(magnitude - baseline, 0.0)
    return temperature, magnitude, baseline, latent


def plot_pcm_characteristics(
    dsc_path: Path,
    output_dir: Path,
    formats: Iterable[str],
) -> tuple[list[Path], dict[str, float]]:
    """Plot DSC baseline separation and the resulting apparent heat capacity."""

    pcm: PCMModel = load_pcm_model(dsc_path)
    temperature, magnitude, baseline, latent = _load_dsc_components(dsc_path)
    if not np.allclose(temperature, pcm.temperature_C) or not np.allclose(
        latent, pcm.excess_heat_flow_mW_mg
    ):
        raise RuntimeError("Plot-side DSC reconstruction does not match the model input")

    beta = ModelParameters().dsc_scan_rate_K_min
    latent_heat_kJ_kg = pcm.latent_heat_J_kg(beta) / 1000.0
    total_latent_heat_kJ = ModelParameters().pcm_mass_kg * latent_heat_kJ_kg
    dense_temperature = np.linspace(temperature.min(), temperature.max(), 800)
    dense_baseline = np.interp(dense_temperature, temperature, baseline)
    dense_latent = np.asarray(pcm.q_latent(dense_temperature), dtype=float)
    dense_ceff_kJ_kgK = (
        np.asarray(pcm.effective_heat_capacity_J_kgK(dense_temperature, beta), dtype=float)
        / 1000.0
    )
    positive = dense_latent > max(float(dense_latent.max()) * 1.0e-4, 1.0e-10)
    phase_low = float(dense_temperature[positive][0])
    phase_high = float(dense_temperature[positive][-1])
    peak_index = int(np.argmax(dense_ceff_kJ_kgK))
    peak_temperature = float(dense_temperature[peak_index])
    peak_ceff = float(dense_ceff_kJ_kgK[peak_index])

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), layout="constrained")
    ax_left, ax_right = axes

    ax_left.plot(
        temperature,
        magnitude,
        color=PALETTE["dark"],
        linewidth=1.8,
        marker="o",
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label=r"原始 $|q(T)|$",
        zorder=4,
    )
    ax_left.plot(
        dense_temperature,
        dense_baseline,
        color=PALETTE["red_strong"],
        linewidth=2.0,
        linestyle="--",
        label=r"显热基线 $b(T)$",
        zorder=3,
    )
    ax_left.plot(
        dense_temperature,
        dense_latent,
        color=PALETTE["blue_main"],
        linewidth=2.5,
        label=r"潜热峰 $q_L(T)$",
        zorder=5,
    )
    ax_left.fill_between(
        dense_temperature,
        0.0,
        dense_latent,
        where=positive,
        color=PALETTE["blue_secondary"],
        alpha=0.20,
        linewidth=0,
        zorder=1,
    )
    ax_left.axvspan(phase_low, phase_high, color=PALETTE["green_1"], alpha=0.26, zorder=0)
    ax_left.set_title("(a) DSC 基线分离与潜热峰", loc="left", fontweight="bold")
    ax_left.set_xlabel(r"温度 $T$ / $^\circ$C")
    ax_left.set_ylabel(r"放热能力 / $(\mathrm{mW}\,\mathrm{mg}^{-1})$")
    ax_left.set_xlim(float(temperature.min()), float(temperature.max()))
    ax_left.set_ylim(bottom=0.0)
    ax_left.legend(loc="upper left")
    ax_left.grid(axis="y", color=PALETTE["neutral"], linewidth=0.7, alpha=0.45)

    base_ceff = pcm.base_heat_capacity_J_kgK / 1000.0
    ax_right.plot(
        dense_temperature,
        dense_ceff_kJ_kgK,
        color=PALETTE["blue_main"],
        linewidth=2.8,
        label=r"$c_{\mathrm{eff}}(T)$",
        zorder=4,
    )
    ax_right.fill_between(
        dense_temperature,
        base_ceff,
        dense_ceff_kJ_kgK,
        where=positive,
        color=PALETTE["green_2"],
        alpha=0.48,
        linewidth=0,
        label="相变热容增量",
        zorder=2,
    )
    ax_right.axhline(
        base_ceff,
        color=PALETTE["gray"],
        linewidth=1.5,
        linestyle=":",
        label=r"显热比热 $c_2$",
        zorder=1,
    )
    ax_right.text(
        float(dense_temperature.max()) - 1.05,
        base_ceff + peak_ceff * 0.025,
        rf"$c_2={base_ceff:.2f}\ \mathrm{{kJ/(kg\cdot K)}}$",
        ha="right",
        va="bottom",
        fontsize=10.5,
        color=PALETTE["gray"],
        zorder=7,
    )
    ax_right.scatter(
        [peak_temperature],
        [peak_ceff],
        s=44,
        color=PALETTE["red_strong"],
        edgecolor="white",
        linewidth=0.9,
        zorder=6,
    )
    ax_right.annotate(
        f"峰值 {peak_ceff:.2f} kJ/(kg·K)\n"
        + rf"$T={peak_temperature:.2f}\,^\circ$C",
        xy=(peak_temperature, peak_ceff),
        xytext=(peak_temperature, peak_ceff * 0.61),
        ha="center",
        va="center",
        fontsize=10.5,
        arrowprops={"arrowstyle": "->", "color": PALETTE["gray"], "lw": 1.1},
        zorder=7,
    )
    ax_right.text(
        0.97,
        0.93,
        f"$L={latent_heat_kJ_kg:.1f}$ kJ/kg\n$m_2L={total_latent_heat_kJ:.1f}$ kJ",
        transform=ax_right.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color=PALETTE["dark"],
    )
    ax_right.set_title("(b) PCM 表观比热", loc="left", fontweight="bold")
    ax_right.set_xlabel(r"温度 $T$ / $^\circ$C")
    ax_right.set_ylabel(r"$c_{\mathrm{eff}}$ / $\mathrm{kJ}\,(\mathrm{kg\,K})^{-1}$")
    ax_right.set_xlim(float(temperature.min()), float(temperature.max()))
    ax_right.set_ylim(0.0, peak_ceff * 1.14)
    ax_right.legend(loc="upper left")
    ax_right.grid(axis="y", color=PALETTE["neutral"], linewidth=0.7, alpha=0.45)

    saved = _save_figure(fig, output_dir, "problem1_pcm_characteristics", formats)
    metrics = {
        "latent_heat_kJ_kg": latent_heat_kJ_kg,
        "total_latent_heat_kJ": total_latent_heat_kJ,
        "peak_temperature_C": peak_temperature,
        "peak_ceff_kJ_kgK": peak_ceff,
    }
    return saved, metrics


def _load_problem1_results(result_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    timeseries_path = result_dir / "main_timeseries.csv"
    summary_path = result_dir / "main_summary.csv"
    sensitivity_path = result_dir / "sensitivity_indices.csv"
    timeseries = pd.read_csv(timeseries_path)
    summary_frame = pd.read_csv(summary_path)
    sensitivity = pd.read_csv(sensitivity_path)
    if len(summary_frame) != 1:
        raise ValueError(f"Expected exactly one row in {summary_path}")
    _require_columns(
        timeseries,
        [
            "time_s",
            "T_core_C",
            "T_skin_C",
            "T_layer1_C",
            "T_pcm_C",
            "T_layer3_C",
            "skin_blood_flow_eq_L_m2_h",
            "skin_blood_flow_actual_L_m2_h",
            "alpha_skin",
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
            "alpha_skin_at_t15",
            "alpha_skin_at_t10",
        ],
        summary_path,
    )
    _require_columns(sensitivity, ["parameter", "S_t15", "S_t10"], sensitivity_path)
    return timeseries, summary_frame.iloc[0], sensitivity


def _verify_temperature_response_font() -> None:
    """Fail clearly if the paper's required STZhongsong font is unavailable."""

    try:
        font_manager.findfont(REQUIRED_PAPER_FONT, fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(
            "未检测到华文中宋（STZhongsong），请安装该字体后再生成温度响应图。"
        ) from exc


def plot_temperature_response(
    timeseries: pd.DataFrame,
    summary: pd.Series,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    """Plot the five temperature nodes using the current paper house style."""

    _verify_temperature_response_font()
    with matplotlib.rc_context(TEMPERATURE_RESPONSE_RCPARAMS):
        return _plot_temperature_response_styled(timeseries, summary, output_dir, formats)


def _plot_temperature_response_styled(
    timeseries: pd.DataFrame,
    summary: pd.Series,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    """Plot the five temperature nodes and mark the two skin thresholds."""

    time_h = timeseries["time_s"].to_numpy(dtype=float) / 3600.0
    t15_h = float(summary["t15_s"]) / 3600.0
    t10_h = float(summary["t10_s"]) / 3600.0

    fig, ax = plt.subplots(figsize=(10.8, 6.8), layout="constrained")
    curves = [
        ("T_core_C", r"核心 $T_c$", TEMPERATURE_RESPONSE_COLORS["core"], 2.8, "-", 1.0, 5),
        ("T_skin_C", r"皮肤 $T_s$", TEMPERATURE_RESPONSE_COLORS["skin"], 3.2, "-", 1.0, 6),
        ("T_layer1_C", r"内层 $T_1$", TEMPERATURE_RESPONSE_COLORS["layer1"], 2.0, "-", 0.95, 3),
        ("T_pcm_C", r"PCM 层 $T_2$", TEMPERATURE_RESPONSE_COLORS["pcm"], 2.0, "-.", 0.95, 3),
        ("T_layer3_C", r"外层 $T_3$", TEMPERATURE_RESPONSE_COLORS["layer3"], 2.0, "--", 0.95, 3),
    ]
    for column, label, color, width, linestyle, alpha, zorder in curves:
        ax.plot(
            time_h,
            timeseries[column],
            label=label,
            color=color,
            linewidth=width,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )

    for threshold in (15.0, 10.0):
        ax.axhline(
            threshold,
            color=TEMPERATURE_RESPONSE_COLORS["threshold"],
            linewidth=1.8,
            linestyle=(0, (5, 4)),
            zorder=1,
        )
    ax.text(0.25, 15.8, r"$T_s=15\,^\circ\mathrm{C}$", color="black", fontsize=15.0)
    ax.text(0.25, 10.8, r"$T_s=10\,^\circ\mathrm{C}$", color="black", fontsize=15.0)
    ax.scatter(
        [t15_h, t10_h],
        [15.0, 10.0],
        s=55,
        color=TEMPERATURE_RESPONSE_COLORS["skin"],
        edgecolor="white",
        linewidth=1.0,
        zorder=8,
    )
    ax.annotate(
        rf"$t_{{15}}={t15_h:.2f}\,\mathrm{{h}}$",
        xy=(t15_h, 15.0),
        xytext=(t15_h + 0.55, 20.0),
        ha="left",
        fontsize=15.0,
        color=TEMPERATURE_RESPONSE_COLORS["skin"],
        arrowprops={
            "arrowstyle": "->",
            "color": TEMPERATURE_RESPONSE_COLORS["skin"],
            "lw": 1.5,
        },
    )
    ax.annotate(
        rf"$t_{{10}}={t10_h:.2f}\,\mathrm{{h}}$",
        xy=(t10_h, 10.0),
        xytext=(t10_h - 0.7, 3.5),
        ha="right",
        fontsize=15.0,
        color=TEMPERATURE_RESPONSE_COLORS["skin"],
        arrowprops={
            "arrowstyle": "->",
            "color": TEMPERATURE_RESPONSE_COLORS["skin"],
            "lw": 1.5,
        },
    )
    ax.set_xlabel(r"时间 $t$ / $\mathrm{h}$")
    ax.set_ylabel(r"温度 $T$ / $^\circ\mathrm{C}$")
    ax.set_xlim(0.0, max(float(time_h.max()) * 1.03, t10_h * 1.03))
    ax.set_ylim(-25.0, 41.0)
    ax.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=5,
        columnspacing=1.25,
        handlelength=2.5,
    )
    return _save_figure(fig, output_dir, "problem1_temperature_response", formats)


def _annotate_event_value(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    dx: float,
    dy: float,
    color: str,
) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        ha="right" if dx < 0 else "left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=color,
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.05},
    )


def plot_blood_flow_regulation(
    timeseries: pd.DataFrame,
    summary: pd.Series,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    """Plot delayed skin blood flow and the resulting skin mass fraction."""

    time_h = timeseries["time_s"].to_numpy(dtype=float) / 3600.0
    equilibrium = timeseries["skin_blood_flow_eq_L_m2_h"].to_numpy(dtype=float)
    actual = timeseries["skin_blood_flow_actual_L_m2_h"].to_numpy(dtype=float)
    alpha = timeseries["alpha_skin"].to_numpy(dtype=float)
    t15_h = float(summary["t15_s"]) / 3600.0
    t10_h = float(summary["t10_s"]) / 3600.0
    flow15 = float(summary["skin_blood_flow_at_t15"])
    flow10 = float(summary["skin_blood_flow_at_t10"])
    alpha15 = float(summary["alpha_skin_at_t15"])
    alpha10 = float(summary["alpha_skin_at_t10"])

    fig, (ax_flow, ax_alpha) = plt.subplots(
        2,
        1,
        figsize=(8.8, 7.2),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    ax_flow.plot(
        time_h,
        equilibrium,
        color=PALETTE["red_strong"],
        linestyle="--",
        linewidth=2.2,
        label=r"目标血流 $\dot V_{bl,\mathrm{eq}}$",
        zorder=3,
    )
    ax_flow.plot(
        time_h,
        actual,
        color=PALETTE["blue_main"],
        linewidth=2.8,
        label=r"实际血流 $\dot V_{bl}$",
        zorder=4,
    )
    ax_flow.scatter(
        [t15_h, t10_h],
        [flow15, flow10],
        s=50,
        color=PALETTE["blue_main"],
        edgecolor="white",
        linewidth=0.9,
        zorder=7,
    )
    _annotate_event_value(ax_flow, t15_h, flow15, f"{flow15:.4f}", 0.55, 0.55, PALETTE["blue_main"])
    _annotate_event_value(ax_flow, t10_h, flow10, f"{flow10:.4f}", -0.65, 0.55, PALETTE["blue_main"])
    ax_flow.set_title("(a) 目标与实际皮肤血流", loc="left", fontweight="bold")
    ax_flow.set_ylabel(r"$\dot V_{bl}$ / $\mathrm{L}\,(\mathrm{m}^2\,\mathrm{h})^{-1}$")
    ax_flow.set_ylim(0.25, 6.75)
    ax_flow.legend(loc="upper right")
    ax_flow.grid(axis="y", color=PALETTE["neutral"], linewidth=0.7, alpha=0.42)

    ax_alpha.plot(time_h, alpha, color=PALETTE["green_3"], linewidth=3.0, zorder=4)
    ax_alpha.scatter(
        [t15_h, t10_h],
        [alpha15, alpha10],
        s=52,
        color=PALETTE["green_3"],
        edgecolor="white",
        linewidth=0.9,
        zorder=7,
    )
    _annotate_event_value(ax_alpha, t15_h, alpha15, f"{alpha15:.4f}", 0.55, -0.055, PALETTE["dark"])
    _annotate_event_value(ax_alpha, t10_h, alpha10, f"{alpha10:.4f}", -0.65, -0.055, PALETTE["dark"])
    ax_alpha.set_title("(b) 皮肤等效质量比例", loc="left", fontweight="bold")
    ax_alpha.set_xlabel(r"时间 $t$ / h")
    ax_alpha.set_ylabel(r"$\alpha_s$")
    ax_alpha.set_ylim(0.10, 0.78)
    ax_alpha.grid(axis="y", color=PALETTE["neutral"], linewidth=0.7, alpha=0.42)

    for ax in (ax_flow, ax_alpha):
        ax.set_xlim(0.0, float(time_h.max()) * 1.03)
        ax.axvline(t15_h, color=PALETTE["gray"], linewidth=1.15, linestyle=(0, (5, 4)), zorder=1)
        ax.axvline(t10_h, color=PALETTE["gray"], linewidth=1.15, linestyle=(0, (5, 4)), zorder=1)
    ax_alpha.text(
        t15_h,
        0.125,
        rf"$t_{{15}}={t15_h:.2f}$ h",
        ha="right",
        va="bottom",
        fontsize=10.5,
        color=PALETTE["gray"],
        rotation=90,
    )
    ax_alpha.text(
        t10_h,
        0.125,
        rf"$t_{{10}}={t10_h:.2f}$ h",
        ha="right",
        va="bottom",
        fontsize=10.5,
        color=PALETTE["gray"],
        rotation=90,
    )
    return _save_figure(fig, output_dir, "problem1_blood_flow_regulation", formats)


def plot_sensitivity_indices(
    sensitivity: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    """Plot signed normalized sensitivities for the two threshold times."""

    label_map = {
        "h_in": r"$h_{\mathrm{in}}$",
        "M0": r"$M_0$",
        "lambda_out": r"$\lambda_{\mathrm{out}}$",
        "beta_DSC": r"$\beta_{\mathrm{DSC}}$",
    }
    expected = list(label_map)
    missing = sorted(set(expected) - set(sensitivity["parameter"]))
    if missing:
        raise ValueError(f"Sensitivity rows are missing: {missing}")
    ordered = sensitivity.set_index("parameter").loc[expected].reset_index()

    y = np.arange(len(ordered), dtype=float)
    height = 0.32
    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    bars15 = ax.barh(
        y - height / 2,
        ordered["S_t15"],
        height=height,
        color=PALETTE["blue_secondary"],
        edgecolor=PALETTE["dark"],
        linewidth=0.9,
        label=r"$S_{t_{15}}$",
        zorder=3,
    )
    bars10 = ax.barh(
        y + height / 2,
        ordered["S_t10"],
        height=height,
        color=PALETTE["green_2"],
        edgecolor=PALETTE["dark"],
        linewidth=0.9,
        hatch="///",
        label=r"$S_{t_{10}}$",
        zorder=3,
    )

    for bars in (bars15, bars10):
        for bar in bars:
            value = float(bar.get_width())
            offset = 0.045 if value >= 0.0 else -0.045
            ax.text(
                value + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha="left" if value >= 0.0 else "right",
                va="center",
                fontsize=10.5,
                color=PALETTE["dark"],
            )

    ax.axvline(0.0, color=PALETTE["dark"], linewidth=1.3, zorder=2)
    ax.set_yticks(y, [label_map[value] for value in ordered["parameter"]])
    ax.invert_yaxis()
    ax.set_xlim(-1.72, 1.38)
    ax.set_xlabel(r"归一化局部敏感度 $S_p$")
    ax.set_ylabel("参数")
    ax.set_title("阈值时间的归一化局部敏感度", loc="left", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", color=PALETTE["neutral"], linewidth=0.7, alpha=0.42, zorder=0)
    return _save_figure(fig, output_dir, "problem1_sensitivity_indices", formats)


def generate_all_figures(
    dsc_path: Path = DEFAULT_DSC_PATH,
    result_dir: Path = DEFAULT_RESULT_DIR,
    output_dir: Path = DEFAULT_FIGURE_DIR,
    formats: Iterable[str] = ("png", "pdf"),
) -> dict[str, object]:
    """Generate all four Problem 1 figures and return paths plus key metrics."""

    dsc_path = Path(dsc_path)
    result_dir = Path(result_dir)
    output_dir = Path(output_dir)
    formats = tuple(formats)
    apply_publication_style()
    timeseries, summary, sensitivity = _load_problem1_results(result_dir)

    pcm_paths, pcm_metrics = plot_pcm_characteristics(dsc_path, output_dir, formats)
    temperature_paths = plot_temperature_response(timeseries, summary, output_dir, formats)
    blood_flow_paths = plot_blood_flow_regulation(timeseries, summary, output_dir, formats)
    sensitivity_paths = plot_sensitivity_indices(sensitivity, output_dir, formats)
    return {
        "pcm": pcm_paths,
        "temperature": temperature_paths,
        "blood_flow": blood_flow_paths,
        "sensitivity": sensitivity_paths,
        "pcm_metrics": pcm_metrics,
        "t15_h": float(summary["t15_s"]) / 3600.0,
        "t10_h": float(summary["t10_s"]) / 3600.0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsc-path", type=Path, default=DEFAULT_DSC_PATH)
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
        dsc_path=args.dsc_path,
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        formats=args.formats,
    )
    metrics = outputs["pcm_metrics"]
    print(f"PCM latent heat: {metrics['latent_heat_kJ_kg']:.3f} kJ/kg")
    print(f"PCM total latent heat: {metrics['total_latent_heat_kJ']:.3f} kJ")
    print(f"t15/t10: {outputs['t15_h']:.3f} h / {outputs['t10_h']:.3f} h")
    for key in ("pcm", "temperature", "blood_flow", "sensitivity"):
        for path in outputs[key]:
            print(path)


if __name__ == "__main__":
    main()
