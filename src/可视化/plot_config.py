"""Shared publication style and export helpers for Problem 2 figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.figure import Figure


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURE_DIR = REPOSITORY_ROOT / "results" / "figures"
DEFAULT_FORMATS = ("svg", "pdf", "png")
REQUIRED_FONT_FAMILY = "STZhongsong"

PALETTE_A = {
    "light": "#BFDFD2",
    "key": "#51999F",
    "secondary": "#4198AC",
    "positive": "#7BC0CD",
}
PALETTE_B = {
    "reference": "#DBCB92",
    "baseline": "#ECB66C",
    "comparator": "#EA9E58",
    "emphasis": "#ED8D5A",
}

AXIS_COLOR = "#202020"
REFERENCE_COLOR = PALETTE_B["reference"]


def verify_required_font() -> Path:
    """Resolve the required 华文中宋 font without allowing a fallback."""

    try:
        resolved = font_manager.findfont(
            REQUIRED_FONT_FAMILY,
            fallback_to_default=False,
        )
    except ValueError as exc:
        raise RuntimeError(
            "未找到绘图必需字体 STZhongsong（华文中宋）；请先安装该字体。"
        ) from exc
    return Path(resolved)


def apply_publication_style() -> Path:
    """Apply the scientific-figure-making house style."""

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


def add_minimal_y_grid(ax: plt.Axes) -> None:
    """Add the shared restrained horizontal grid."""

    ax.grid(
        axis="y",
        color=PALETTE_A["light"],
        linewidth=0.9,
        alpha=0.48,
        zorder=0,
    )
    ax.set_axisbelow(True)


def add_skin_thresholds(ax: plt.Axes) -> None:
    """Draw the shared 15 °C and 10 °C threshold language."""

    for threshold in (15.0, 10.0):
        ax.axhline(
            threshold,
            color=REFERENCE_COLOR,
            linewidth=1.5,
            linestyle=(0, (5, 4)),
            alpha=0.88,
            zorder=1,
        )


def finalize_figure(
    fig: Figure,
    output_dir: Path,
    basename: str,
    formats: Iterable[str] = DEFAULT_FORMATS,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Validate the single-panel title policy and export vector plus PNG files."""

    if fig._suptitle is not None:
        raise ValueError("单幅论文图不得包含 figure-level title。")
    for ax in fig.axes:
        if any(ax.get_title(loc=loc) for loc in ("left", "center", "right")):
            raise ValueError("单幅论文图不得包含轴标题。")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.5)
    saved: list[Path] = []
    for suffix in formats:
        normalized = suffix.lower().lstrip(".")
        if normalized not in {"svg", "pdf", "png"}:
            raise ValueError(f"不支持的图片格式：{suffix}")
        path = output_dir / f"{basename}.{normalized}"
        kwargs: dict[str, object] = {
            "bbox_inches": "tight",
            "pad_inches": 0.06,
            "facecolor": "white",
        }
        if normalized == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved
