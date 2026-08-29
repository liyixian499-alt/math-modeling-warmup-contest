"""Generate the two publication figures requested for Problem 4.

The script reads the reviewed Problem 4 CSV outputs.  When the dedicated
lambda--t15 plotting interface is absent or incomplete, it supplements the
coarse search with calls to the existing formal Problem 4 event solver.  It
does not alter the thermal model, inverse objective, parameters, or optimum.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.text import Text

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.问题一.dsc import load_pcm_model
from src.问题一.problem4.config import Problem4Config
from src.问题一.problem4.solver import solve_events


DEFAULT_RESULT_DIR = REPOSITORY_ROOT / "results" / "outputs" / "problem4"
DEFAULT_FIGURE_DIR = REPOSITORY_ROOT / "paper" / "figures" / "problem4"
PLOT_DATA_FILENAME = "problem4_plot_lambda_t15.csv"
REQUIRED_FONT_FAMILY = "STZhongsong"
FIGURE_SIZE = (10.0, 6.5)
PNG_DPI = 300

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
AXIS_COLOR = "#202020"

EXPECTED_LAMBDA_OPT = 28.3525284040
EXPECTED_TARGET_MIN = 734.814967549
PROHIBITED_UNPENALIZED_TARGET_MIN = 735.000828799
EXPECTED_SENSITIVITY_PARAMETERS = (
    "h_in",
    "M0",
    "lambda_out",
    "beta_DSC",
)
PARAMETER_LABELS = {
    "h_in": r"$h_{\rm in}$",
    "M0": r"$M_0$",
    "lambda_out": r"$\lambda_{\rm out}$",
    "beta_DSC": r"$\beta$",
}


def verify_required_font() -> Path:
    """Resolve 华文中宋 without allowing Matplotlib font fallback."""

    try:
        resolved = font_manager.findfont(
            REQUIRED_FONT_FAMILY,
            fallback_to_default=False,
        )
    except ValueError as exc:
        raise RuntimeError(
            "未找到绘图必需字体 STZhongsong（华文中宋）；请安装后再生成论文图。"
        ) from exc
    return Path(resolved)


def apply_publication_style() -> Path:
    """Apply the repository's scientific-figure-making house style."""

    font_path = verify_required_font()
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
            "axes.linewidth": 2.0,
            "axes.edgecolor": AXIS_COLOR,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 17.0,
            "ytick.labelsize": 17.0,
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
    return font_path


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    source: Path,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} 缺少字段：{missing}")
    if frame.empty:
        raise ValueError(f"{source} 不包含数据行")


def _read_csv(
    path: Path,
    required: Iterable[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, required, path)
    return frame


def load_problem4_results(result_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and validate the reviewed Problem 4 result tables."""

    result_dir = Path(result_dir)
    search = _read_csv(
        result_dir / "problem4_search.csv",
        ["lambda_pcm", "improvement_pct", "t15_min"],
    )
    optimal = _read_csv(
        result_dir / "problem4_optimal_solution.csv",
        [
            "lambda_opt",
            "improvement_pct",
            "target_t15_min",
            "achieved_t15_min",
        ],
    )
    sensitivity = _read_csv(
        result_dir / "problem4_sensitivity_summary.csv",
        [
            "parameter",
            "normalized_sensitivity_lambda",
            "abs_normalized_sensitivity_lambda",
        ],
    )

    if len(optimal) != 1:
        raise ValueError("problem4_optimal_solution.csv 必须且只能包含一行正式最优解")
    optimal_row = optimal.iloc[0]
    lambda_opt = float(optimal_row["lambda_opt"])
    target_min = float(optimal_row["target_t15_min"])
    achieved_min = float(optimal_row["achieved_t15_min"])
    if not np.isclose(lambda_opt, EXPECTED_LAMBDA_OPT, atol=1.0e-8, rtol=0.0):
        raise ValueError(f"最优倍率不一致：读取到 {lambda_opt:.12f}")
    if not np.isclose(target_min, EXPECTED_TARGET_MIN, atol=1.0e-9, rtol=0.0):
        raise ValueError(f"问题三正式目标时间不一致：读取到 {target_min:.12f} min")
    if np.isclose(
        target_min,
        PROHIBITED_UNPENALIZED_TARGET_MIN,
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("误用了问题三未经重量惩罚的 735.000828799 min 目标")
    if not np.isclose(achieved_min, target_min, atol=1.0e-6, rtol=0.0):
        raise ValueError(
            "正式最优点未达到目标容差："
            f"achieved={achieved_min:.12f}, target={target_min:.12f} min"
        )

    sensitivity_subset = sensitivity.loc[
        sensitivity["parameter"].isin(EXPECTED_SENSITIVITY_PARAMETERS)
    ].copy()
    counts = sensitivity_subset["parameter"].value_counts()
    missing_parameters = [
        name for name in EXPECTED_SENSITIVITY_PARAMETERS if counts.get(name, 0) != 1
    ]
    if missing_parameters:
        raise ValueError(f"敏感度参数缺失或重复：{missing_parameters}")
    values = sensitivity_subset["normalized_sensitivity_lambda"].to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("归一化敏感度包含非有限值")
    recorded_abs = sensitivity_subset[
        "abs_normalized_sensitivity_lambda"
    ].to_numpy(dtype=float)
    if not np.allclose(recorded_abs, np.abs(values), atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("敏感度绝对值字段与有符号字段不一致")

    return {
        "search": search,
        "optimal": optimal,
        "sensitivity": sensitivity_subset,
    }


def _requested_lambda_grid(lambda_opt: float) -> np.ndarray:
    """Return a nonuniform grid with extra resolution around the inverse root."""

    low_region = np.arange(1.0, 24.0, 1.0)
    root_region = np.array(
        [
            24.0,
            25.0,
            26.0,
            26.5,
            27.0,
            27.5,
            28.0,
            28.1,
            28.2,
            28.3,
            lambda_opt,
            28.4,
            28.5,
            28.75,
            29.0,
            29.5,
            30.0,
            31.0,
            32.0,
            33.0,
            34.0,
            35.0,
        ],
        dtype=float,
    )
    coarse_detail = np.array([1.25, 1.5], dtype=float)
    return np.unique(np.concatenate([low_region, coarse_detail, root_region]))


def _plot_data_is_complete(frame: pd.DataFrame, lambda_opt: float) -> bool:
    required = {
        "lambda_pcm",
        "improvement_pct",
        "t15_min",
        "gap_to_target_min",
        "feasible",
        "point_type",
    }
    if not required.issubset(frame.columns) or len(frame) < 40:
        return False
    lambdas = frame["lambda_pcm"].to_numpy(dtype=float)
    values = frame["t15_min"].to_numpy(dtype=float)
    if not np.isfinite(lambdas).all() or not np.isfinite(values).all():
        return False
    requested = _requested_lambda_grid(lambda_opt)
    if not all(
        np.isclose(lambdas, value, atol=1.0e-10).any() for value in requested
    ):
        return False
    optimal = frame.loc[frame["point_type"] == "optimal"]
    if len(optimal) != 1:
        return False
    optimal_row = optimal.iloc[0]
    return bool(
        np.isclose(
            float(optimal_row["lambda_pcm"]),
            EXPECTED_LAMBDA_OPT,
            atol=1.0e-8,
            rtol=0.0,
        )
        and np.isclose(
            float(optimal_row["t15_min"]),
            EXPECTED_TARGET_MIN,
            atol=1.0e-9,
            rtol=0.0,
        )
    )


def prepare_lambda_curve_data(
    results: dict[str, pd.DataFrame],
    result_dir: Path,
    *,
    refresh: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Load or generate the official-model sampling interface for Figure 1."""

    result_dir = Path(result_dir)
    plot_data_path = result_dir / PLOT_DATA_FILENAME
    optimal_row = results["optimal"].iloc[0]
    lambda_opt = float(optimal_row["lambda_opt"])
    target_min = float(optimal_row["target_t15_min"])

    if plot_data_path.is_file() and not refresh:
        cached = pd.read_csv(plot_data_path)
        if _plot_data_is_complete(cached, lambda_opt):
            return cached.sort_values("lambda_pcm").reset_index(drop=True), False

    config = Problem4Config()
    config.validate()
    pcm = load_pcm_model(config.dsc_path)
    requested = _requested_lambda_grid(lambda_opt)
    search = results["search"]
    known = {
        float(row.lambda_pcm): float(row.t15_min)
        for row in search.itertuples(index=False)
    }

    rows: list[dict[str, float | bool | str]] = []
    for lambda_pcm in requested:
        if np.isclose(lambda_pcm, lambda_opt, atol=1.0e-10, rtol=0.0):
            t15_min = target_min
            point_type = "optimal"
        else:
            matched = next(
                (
                    t15
                    for known_lambda, t15 in known.items()
                    if np.isclose(lambda_pcm, known_lambda, atol=1.0e-12, rtol=0.0)
                ),
                None,
            )
            if matched is None:
                event = solve_events(
                    config.thermal,
                    pcm,
                    float(lambda_pcm),
                    config.main_solver,
                    stop_at_15=True,
                )
                if not event.event_15_reached or not np.isfinite(event.t15_s):
                    raise RuntimeError(f"lambda_pcm={lambda_pcm:g} 未触发 15 ℃ 事件")
                t15_min = float(event.t15_s) / 60.0
            else:
                t15_min = matched
            point_type = "sample"
        gap = t15_min - target_min
        rows.append(
            {
                "lambda_pcm": float(lambda_pcm),
                "improvement_pct": (float(lambda_pcm) - 1.0) * 100.0,
                "t15_min": t15_min,
                "gap_to_target_min": gap,
                "feasible": bool(gap >= -1.0e-9),
                "point_type": point_type,
            }
        )

    frame = pd.DataFrame(rows).sort_values("lambda_pcm").reset_index(drop=True)
    sample_values = frame.loc[frame["point_type"] == "sample", "t15_min"].to_numpy(
        dtype=float
    )
    if (np.diff(sample_values) < -1.0e-5).any():
        raise ValueError("正式模型绘图采样出现非单调 t15，停止作图")
    optimal_rows = frame.loc[frame["point_type"] == "optimal"]
    if len(optimal_rows) != 1:
        raise ValueError("绘图数据必须且只能包含一个正式最优点")
    optimal_plot = optimal_rows.iloc[0]
    if not np.isclose(
        float(optimal_plot["lambda_pcm"]),
        EXPECTED_LAMBDA_OPT,
        atol=1.0e-8,
        rtol=0.0,
    ):
        raise ValueError("绘图数据中的最优倍率不一致")
    if not np.isclose(
        float(optimal_plot["t15_min"]),
        EXPECTED_TARGET_MIN,
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise ValueError("绘图数据中的最优阈值时间不一致")

    result_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(plot_data_path, index=False, float_format="%.12g")
    return frame, True


def _style_grid(ax: plt.Axes, axis: str) -> None:
    ax.grid(
        axis=axis,
        color=PALETTE_A["light"],
        linewidth=0.9,
        alpha=0.52,
        zorder=0,
    )
    ax.set_axisbelow(True)


def _enforce_figure_font(fig: Figure) -> None:
    for artist in fig.findobj(match=Text):
        artist.set_fontfamily(REQUIRED_FONT_FAMILY)


def save_figure(
    fig: Figure,
    output_dir: Path,
    basename: str,
    formats: Iterable[str],
    *,
    dpi: int = PNG_DPI,
) -> list[Path]:
    """Validate the no-title policy and export vector PDF plus high-DPI PNG."""

    if fig._suptitle is not None:
        raise ValueError("单幅论文图不得包含 figure-level title")
    for ax in fig.axes:
        if any(ax.get_title(loc=loc) for loc in ("left", "center", "right")):
            raise ValueError("单幅论文图不得包含轴标题")
    _enforce_figure_font(fig)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.5)

    saved: list[Path] = []
    for suffix in formats:
        normalized = suffix.lower().lstrip(".")
        if normalized not in {"png", "pdf"}:
            raise ValueError(f"不支持的图片格式：{suffix}")
        path = output_dir / f"{basename}.{normalized}"
        kwargs: dict[str, object] = {
            "facecolor": "white",
        }
        if normalized == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def plot_lambda_t15_inverse(
    curve: pd.DataFrame,
    optimal: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    """Plot the true-model lambda--t15 curve and its binding intersection."""

    row = optimal.iloc[0]
    lambda_opt = float(row["lambda_opt"])
    target_min = float(row["target_t15_min"])
    x = curve["lambda_pcm"].to_numpy(dtype=float)
    y = curve["t15_min"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    _style_grid(ax, "y")
    ax.plot(
        x,
        y,
        color=PALETTE_A["key"],
        linewidth=2.8,
        marker="o",
        markersize=3.8,
        markerfacecolor="white",
        markeredgecolor=PALETTE_A["key"],
        markeredgewidth=1.1,
        zorder=3,
    )
    ax.axhline(
        target_min,
        color=PALETTE_B["emphasis"],
        linewidth=2.0,
        linestyle=(0, (7, 4)),
        zorder=2,
    )
    ax.axvline(
        lambda_opt,
        color=PALETTE_B["comparator"],
        linewidth=1.8,
        linestyle=(0, (3, 3)),
        zorder=2,
    )
    ax.scatter(
        [lambda_opt],
        [target_min],
        s=105,
        facecolor=PALETTE_B["emphasis"],
        edgecolor=AXIS_COLOR,
        linewidth=1.2,
        zorder=5,
    )

    x_span = float(np.ptp(x))
    y_span = float(np.ptp(y))
    ax.set_xlim(max(0.0, float(x.min()) - 0.04 * x_span), float(x.max()) + 0.03 * x_span)
    ax.set_ylim(float(y.min()) - 0.07 * y_span, float(y.max()) + 0.10 * y_span)
    ax.set_xlabel(r"PCM 放热倍率 $\lambda_{\rm pcm}$")
    ax.set_ylabel(r"皮肤温度降至 15 ℃ 的时间 $t_{15}$ / min")

    ax.text(
        ax.get_xlim()[0] + 0.035 * np.ptp(ax.get_xlim()),
        target_min + 0.018 * np.ptp(ax.get_ylim()),
        rf"$J_3^*={target_min:.3f}\ \mathrm{{min}}$",
        fontsize=16.0,
        color=PALETTE_B["emphasis"],
        ha="left",
        va="bottom",
    )
    ax.annotate(
        rf"$\lambda_{{\rm pcm}}^*={lambda_opt:.3f}$" + "\n" + rf"$t_{{15}}={target_min:.3f}\ \mathrm{{min}}$",
        xy=(lambda_opt, target_min),
        xytext=(0.59, 0.57),
        textcoords="axes fraction",
        fontsize=16.0,
        color=AXIS_COLOR,
        ha="left",
        va="top",
        arrowprops={
            "arrowstyle": "->",
            "color": PALETTE_A["key"],
            "linewidth": 1.7,
            "shrinkA": 4,
            "shrinkB": 7,
        },
    )
    ax.tick_params(direction="out")
    return save_figure(
        fig,
        output_dir,
        "problem4_lambda_t15_inverse",
        formats,
    )


def plot_lambda_sensitivity(
    sensitivity: pd.DataFrame,
    output_dir: Path,
    formats: Iterable[str],
) -> list[Path]:
    """Plot signed inverse-result sensitivities sorted by absolute magnitude."""

    ordered = sensitivity.assign(
        _abs=sensitivity["normalized_sensitivity_lambda"].abs()
    ).sort_values("_abs", ascending=False)
    values = ordered["normalized_sensitivity_lambda"].to_numpy(dtype=float)
    labels = [PARAMETER_LABELS[name] for name in ordered["parameter"]]
    y_positions = np.arange(len(values))
    colors = [
        PALETTE_A["key"] if value >= 0.0 else PALETTE_B["emphasis"]
        for value in values
    ]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    _style_grid(ax, "x")
    bars = ax.barh(
        y_positions,
        values,
        height=0.62,
        color=colors,
        edgecolor=AXIS_COLOR,
        linewidth=1.5,
        zorder=3,
    )
    ax.axvline(0.0, color=AXIS_COLOR, linewidth=1.8, zorder=4)
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"最优放热倍率归一化敏感度 $S_p^{(\lambda)}$")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", direction="out")

    span = float(values.max() - values.min())
    left = float(values.min() - 0.10 * span)
    right = float(values.max() + 0.12 * span)
    ax.set_xlim(left, right)
    label_offset = 0.018 * (right - left)
    for bar, value in zip(bars, values, strict=True):
        if value >= 0.0:
            x_text = value + label_offset
            horizontal_alignment = "left"
        else:
            x_text = value + label_offset
            horizontal_alignment = "left"
        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:+.2f}",
            ha=horizontal_alignment,
            va="center",
            fontsize=16.0,
            color=AXIS_COLOR,
        )
    return save_figure(
        fig,
        output_dir,
        "problem4_lambda_sensitivity",
        formats,
    )


def _parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not formats:
        raise argparse.ArgumentTypeError("至少指定一种输出格式")
    unsupported = sorted(set(formats) - {"png", "pdf"})
    if unsupported:
        raise argparse.ArgumentTypeError(f"不支持的输出格式：{unsupported}")
    return formats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成问题四两张正式论文图")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--formats", type=_parse_formats, default=("pdf", "png"))
    parser.add_argument(
        "--refresh-sampling",
        action="store_true",
        help="忽略已有绘图 CSV，并重新调用正式问题四模型采样",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    font_path = apply_publication_style()
    results = load_problem4_results(args.result_dir)
    curve, generated = prepare_lambda_curve_data(
        results,
        args.result_dir,
        refresh=args.refresh_sampling,
    )
    saved = []
    saved.extend(
        plot_lambda_t15_inverse(
            curve,
            results["optimal"],
            args.output_dir,
            args.formats,
        )
    )
    saved.extend(
        plot_lambda_sensitivity(
            results["sensitivity"],
            args.output_dir,
            args.formats,
        )
    )
    print(f"STZhongsong: {font_path}")
    print(f"plot_data_generated: {generated}")
    print(f"plot_data: {Path(args.result_dir) / PLOT_DATA_FILENAME}")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
