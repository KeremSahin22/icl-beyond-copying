from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


SMALL_PDF_FIGSIZE: tuple[float, float] = (3.35, 2.25)


MODEL_COLORS: dict[str, str] = {
    "vanilla_1b": "#0072B2",
    "hapax_1b": "#E69F00",
    "hapax_1b_thresh_0.3": "#40B0A6",
    "vanilla_2.8b": "#0072B2",
    "hapax_2.8b": "#E69F00",
}


REPETITION_FIGSIZE: tuple[float, float] = (
    SMALL_PDF_FIGSIZE[0],
    SMALL_PDF_FIGSIZE[1] * 0.93,
)

ICLR_WIDE_FIGSIZE: tuple[float, float] = (7, 4)

TLD_PANEL_FIGSIZE: tuple[float, float] = (
    SMALL_PDF_FIGSIZE[0] * 1.08,
    SMALL_PDF_FIGSIZE[1] * 1.15,
)

TLD_SAMPLE_VARIANTS: tuple[tuple[str, str], ...] = (
    ("", "All Samples"),
    ("_skip_bigrams", "Non-Extractive Samples"),
    ("_skip_bigrams_thresholded", "Thresholded Non-Extractive Samples"),
)


MODEL_TYPE_DISPLAY_NAMES: dict[str, str] = {
    "hapax_1b": "Hapax",
    "hapax_1b_thresh_0.3": "Thresholded-Hapax",
    "vanilla_1b": "Vanilla",
    "hapax_2.8b": "Hapax (2.8B)",
    "vanilla_2.8b": "Vanilla (2.8B)",
}


def display_name_for_model_type(model_type: str) -> str:
    return MODEL_TYPE_DISPLAY_NAMES.get(model_type, model_type)


def apply_iclr_style():
    plt.style.use("default")
    plt.rcParams.update(
        {

            "figure.dpi": 100,
            "figure.figsize": (7, 4),

            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
            "mathtext.fontset": "cm",

            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,

            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,

            "legend.fontsize": 9,
            "legend.frameon": False,
            "legend.columnspacing": 1.0,

            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,

            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def apply_tld_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def apply_induction_scores_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Helvetica",
                "Arial",
                "Liberation Sans",
                "sans-serif",
            ],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "axes.linewidth": 0.85,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure_pdf_png(
    fig,
    base_path: str | Path,
    *,
    pdf_dpi: Optional[float] = None,
    png_dpi: float | int = 300,
    bbox_inches: Optional[str] = "tight",
) -> tuple[Path, Path]:
    stem = Path(base_path)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    pdf_kw: dict = {"format": "pdf", "bbox_inches": bbox_inches}
    if pdf_dpi is not None:
        pdf_kw["dpi"] = pdf_dpi
    fig.savefig(pdf_path, **pdf_kw)
    fig.savefig(png_path, format="png", dpi=png_dpi, bbox_inches=bbox_inches)
    return pdf_path, png_path


def save_pyplot_figure_pdf_png(
    pdf_path: str | Path,
    *,
    pdf_dpi: float | int = 300,
    png_dpi: float | int = 150,
    bbox_inches: Optional[str] = "tight",
) -> tuple[Path, Path]:
    pdf_path = Path(pdf_path)
    png_path = pdf_path.with_suffix(".png")
    plt.savefig(pdf_path, dpi=pdf_dpi, bbox_inches=bbox_inches)
    plt.savefig(png_path, dpi=png_dpi, bbox_inches=bbox_inches)
    return pdf_path, png_path


def micro_avg_abstractive_and_translation_accuracy(
    mean_acc_abstractive: np.ndarray,
    mean_translation_accuracy: np.ndarray,
    n_abstractive_tasks: int,
    n_translation_pairs: int,
) -> np.ndarray:
    a = np.asarray(mean_acc_abstractive, dtype=np.float64)
    t = np.asarray(mean_translation_accuracy, dtype=np.float64)
    n_a = int(n_abstractive_tasks)
    n_t = int(n_translation_pairs)
    if n_a <= 0 and n_t <= 0:
        return np.full_like(a, np.nan, dtype=np.float64)
    if n_t <= 0:
        return a.copy()
    if n_a <= 0:
        return t.copy()
    return (n_a * a + n_t * t) / (n_a + n_t)


def model_line_colors(model_types: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        MODEL_COLORS.get(m, f"C{i % 10}")
        for i, m in enumerate(model_types)
    )


def style_induction_box_axes(
    ax,
    spine_lw: float = 0.85,
    tick_lw: float = 0.75,
    tick_len: float = 3.0,
    labelsize: int = 9,
) -> None:
    ax.grid(False)
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(spine_lw)
    ax.tick_params(direction="out", length=tick_len, width=tick_lw, labelsize=labelsize)
    ax.set_axisbelow(False)


def set_training_step_ticks_induction(
    ax, major: int = 5000, minor: int = 1000
) -> None:
    ax.xaxis.set_major_locator(mticker.MultipleLocator(major))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(minor))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x):}"))
    ax.tick_params(axis="x", which="minor", length=2.0, width=0.6, labelsize=0)


REPETITION_OVER_TIME_CI_ALPHA: float = 0.3
REPETITION_OVER_TIME_XMIN: int = 100
REPETITION_OVER_TIME_XMAX: int = 20100
REPETITION_OVER_TIME_MAJOR_TICKS: tuple[int, ...] = (5000, 10000, 15000, 20000)
REPETITION_OVER_TIME_XLABEL: str = "Training Steps"
def style_repetition_over_time_xaxis(
    ax,
    *,
    step_lo: int | None = None,
    step_hi: int | None = None,
) -> None:
    x_lo = REPETITION_OVER_TIME_XMIN if step_lo is None else int(step_lo)
    if step_hi is None:
        x_hi = REPETITION_OVER_TIME_XMAX
    else:
        x_hi = int(step_hi)
        if x_hi >= 19000:
            x_hi = REPETITION_OVER_TIME_XMAX
    ax.set_xlim(x_lo, max(x_hi, x_lo))
    major = [t for t in REPETITION_OVER_TIME_MAJOR_TICKS if x_lo <= t <= x_hi]
    if major:
        ax.set_xticks(major)
    minor = [t for t in range(1000, 20100, 1000) if x_lo <= t <= x_hi]
    if minor:
        ax.set_xticks(minor, minor=True)
    ax.tick_params(axis="x", which="minor", length=2.0, width=0.6, labelsize=0)
    ax.grid(False)
    ax.set_axisbelow(True)


def set_training_step_ticks_causal(
    ax, major: int = 5000, minor: int = 1000
) -> None:
    ax.xaxis.set_major_locator(mticker.MultipleLocator(major))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(minor))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):}"))
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.grid(False)
    ax.tick_params(direction="out", labelsize=9)

