from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.plotting import (
    TLD_PANEL_FIGSIZE,
    TLD_SAMPLE_VARIANTS,
    model_line_colors,
    apply_tld_plot_style,
    display_name_for_model_type,
    save_figure_pdf_png,
    set_training_step_ticks_causal,
    style_induction_box_axes,
)

_SAMPLE_VARIANT_KEYS: dict[str, str] = {
    "all": "",
    "skip_bigrams": "_skip_bigrams",
    "thresholded": "_skip_bigrams_thresholded",
}

_CI_ALPHA_TLD = 0.2
_CI_ALPHA_POSITION = 0.2
_Y_PAD_FRAC = 0.05

_POSITION_FIELDS: dict[str, tuple[str, str, str]] = {
    "50": ("avg_loss_50", "ci_lower_loss_50", "ci_upper_loss_50"),
    "500": ("avg_loss_500", "ci_lower_loss_500", "ci_upper_loss_500"),
}


def load_jsonl(file_path: Path, max_step: int | None = None) -> list[dict]:
    data: list[dict] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            point = json.loads(line.strip())
            if max_step is not None and point["step"] > max_step:
                continue
            data.append(point)
    return data


def _padded_ylim(y_min: float, y_max: float, *, pad_frac: float = _Y_PAD_FRAC) -> tuple[float, float]:
    span = y_max - y_min
    pad = span * pad_frac if span > 0 else 0.01
    return y_min - pad, y_max + pad


def _panel_tld_ylim(
    tld_dir: Path,
    model_types: tuple[str, ...],
    sample_suffix: str,
    max_step: int | None = None,
) -> tuple[float, float]:
    y_min, y_max = np.inf, -np.inf
    for model_type in model_types:
        path = tld_dir / f"{model_type}_tld{sample_suffix}.jsonl"
        for point in load_jsonl(path, max_step):
            y_min = min(y_min, point["ci_lower_tld"])
            y_max = max(y_max, point["ci_upper_tld"])
    return _padded_ylim(y_min, y_max)


def _panel_position_ylim(
    tld_dir: Path,
    model_types: tuple[str, ...],
    sample_suffix: str,
    position: str,
    max_step: int | None = None,
) -> tuple[float, float]:
    _, ci_lo_key, ci_hi_key = _POSITION_FIELDS[position]
    y_min, y_max = np.inf, -np.inf
    for model_type in model_types:
        path = tld_dir / f"{model_type}_tld{sample_suffix}.jsonl"
        for point in load_jsonl(path, max_step):
            y_min = min(y_min, point[ci_lo_key])
            y_max = max(y_max, point[ci_hi_key])
    return _padded_ylim(y_min, y_max)


def _style_tld_axis(
    ax,
    *,
    ylabel: str,
    title: str | None,
    show_xlabel: bool,
) -> None:
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel("Training steps")
    set_training_step_ticks_causal(ax)
    style_induction_box_axes(ax, spine_lw=0.85, tick_lw=0.75, tick_len=3.0, labelsize=9)


def _plot_tld_series(
    ax, result: list[dict], *, label: str, color: str, linestyle="-"
) -> None:
    step = [point["step"] for point in result]
    avg = [point["avg_tld"] for point in result]
    ci_lower = [point["ci_lower_tld"] for point in result]
    ci_upper = [point["ci_upper_tld"] for point in result]
    ax.plot(
        step, avg, label=label, color=color, alpha=0.9, zorder=3, linestyle=linestyle
    )
    ax.fill_between(
        step, ci_lower, ci_upper, alpha=_CI_ALPHA_TLD, color=color, zorder=2
    )


def _plot_position_series(
    ax,
    result: list[dict],
    *,
    label: str,
    color: str,
    position: str,
) -> None:
    avg_key, ci_lo_key, ci_hi_key = _POSITION_FIELDS[position]
    step = [point["step"] for point in result]
    avg = [point[avg_key] for point in result]
    ci_lower = [point[ci_lo_key] for point in result]
    ci_upper = [point[ci_hi_key] for point in result]
    ax.plot(step, avg, label=label, color=color, alpha=0.9, zorder=3)
    ax.fill_between(
        step, ci_lower, ci_upper, alpha=_CI_ALPHA_POSITION, color=color, zorder=2
    )


def _plot_sample_variant_row(
    axes: np.ndarray,
    *,
    model_types: tuple[str, ...],
    labels: tuple[str, ...],
    colors: tuple[str, ...],
    tld_dir: Path,
    ylabel: str,
    sample_variants: tuple[tuple[str, str], ...],
    max_step: int | None = None,
) -> None:
    for ax, (sample_suffix, panel_title) in zip(axes, sample_variants):
        for model_type, label, color in zip(model_types, labels, colors):
            path = tld_dir / f"{model_type}_tld{sample_suffix}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            _plot_tld_series(
                ax,
                load_jsonl(path, max_step),
                label=label,
                color=color,
                linestyle="-",
            )
        ax.set_ylim(*_panel_tld_ylim(tld_dir, model_types, sample_suffix, max_step))
        _style_tld_axis(
            ax,
            ylabel=ylabel,
            title=panel_title,
            show_xlabel=True,
        )
        ax.legend(loc="best", fontsize=8)


def _plot_position_loss_figure(
    axes: np.ndarray,
    *,
    model_types: tuple[str, ...],
    labels: tuple[str, ...],
    colors: tuple[str, ...],
    tld_dir: Path,
    sample_variants: tuple[tuple[str, str], ...],
    max_step: int | None = None,
) -> None:
    for row, position in enumerate(("50", "500")):
        ylabel = f"Loss (position {position})"
        for col, (sample_suffix, panel_title) in enumerate(sample_variants):
            ax = axes[row, col]
            for model_type, label, color in zip(model_types, labels, colors):
                path = tld_dir / f"{model_type}_tld{sample_suffix}.jsonl"
                if not path.is_file():
                    raise FileNotFoundError(path)
                _plot_position_series(
                    ax,
                    load_jsonl(path, max_step),
                    label=label,
                    color=color,
                    position=position,
                )
            ax.set_ylim(
                *_panel_position_ylim(
                    tld_dir, model_types, sample_suffix, position, max_step
                )
            )
            _style_tld_axis(
                ax,
                ylabel=ylabel,
                title=panel_title if row == 0 else None,
                show_xlabel=(row == 1),
            )
            ax.legend(loc="best", fontsize=8)


def _save_figure(fig: plt.Figure, plots_out: Path, stem: str) -> None:
    fig.tight_layout()

    save_figure_pdf_png(fig, plots_out / stem.replace(".", "_"))
    plt.close(fig)


def _resolve_sample_variants(keys: list[str] | None) -> tuple[tuple[str, str], ...]:
    if keys is None:
        return TLD_SAMPLE_VARIANTS
    title_by_suffix = dict(TLD_SAMPLE_VARIANTS)
    variants: list[tuple[str, str]] = []
    for key in keys:
        if key not in _SAMPLE_VARIANT_KEYS:
            valid = ", ".join(_SAMPLE_VARIANT_KEYS)
            raise ValueError(f"Unknown sample variant {key!r}; expected one of: {valid}")
        suffix = _SAMPLE_VARIANT_KEYS[key]
        variants.append((suffix, title_by_suffix[suffix]))
    return tuple(variants)


def _tld_row_figsize(n_panels: int) -> tuple[float, float]:
    return (TLD_PANEL_FIGSIZE[0] * n_panels, TLD_PANEL_FIGSIZE[1])


def _tld_position_figsize(n_panels: int) -> tuple[float, float]:
    return (TLD_PANEL_FIGSIZE[0] * n_panels, TLD_PANEL_FIGSIZE[1] * 2.05)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_types",
        nargs="+",
        default=["vanilla_1b", "hapax_1b", "hapax_1b_thresh_0.3"],
        help="Model folders under loss_differences/, e.g. vanilla_1b hapax_1b.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels, one per model in --model_types."    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Output filename suffix, e.g. 1b gives tld_1b.pdf.",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--sample_variants",
        nargs="+",
        default=None,
        choices=tuple(_SAMPLE_VARIANT_KEYS),
        help="Sample filters to plot. Defaults to all three.",
    )
    args = parser.parse_args()

    model_types = tuple(args.model_types)
    sample_variants = _resolve_sample_variants(args.sample_variants)
    n_panels = len(sample_variants)
    if args.labels is not None and len(args.labels) != len(model_types):
        raise ValueError(
            f"--labels has {len(args.labels)} entries but --model_types has {len(model_types)}"
        )
    labels = (
        tuple(args.labels)
        if args.labels is not None
        else tuple(display_name_for_model_type(m) for m in model_types)
    )
    colors = model_line_colors(model_types)

    tld_dir = _PROJECT_ROOT / "loss_differences"
    plots_out = tld_dir / "tld_plots"
    plots_out.mkdir(parents=True, exist_ok=True)

    apply_tld_plot_style()

    fig, axes = plt.subplots(1, n_panels, figsize=_tld_row_figsize(n_panels), sharex=True)
    _plot_sample_variant_row(
        axes,
        model_types=model_types,
        labels=labels,
        colors=colors,
        tld_dir=tld_dir,
        ylabel="Token-loss difference",
        sample_variants=sample_variants,
        max_step=args.max_step,
    )
    _save_figure(fig, plots_out, f"tld_{args.name}")

    fig, axes = plt.subplots(
        2, n_panels, figsize=_tld_position_figsize(n_panels), sharex=True
    )
    _plot_position_loss_figure(
        axes,
        model_types=model_types,
        labels=labels,
        colors=colors,
        tld_dir=tld_dir,
        sample_variants=sample_variants,
        max_step=args.max_step,
    )
    _save_figure(fig, plots_out, f"position_losses_{args.name}")
