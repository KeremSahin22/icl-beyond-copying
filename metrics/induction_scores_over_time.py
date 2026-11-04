from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.attn_heads import compute_verbatim_induction_scores
from lib.model_setup import load_model_and_tokenizer
from lib.plotting import (
    MODEL_COLORS,
    MODEL_TYPE_DISPLAY_NAMES,
    SMALL_PDF_FIGSIZE,
    save_figure_pdf_png,
    set_training_step_ticks_induction,
    style_induction_box_axes,
)

MIN_PEAK: float = 0.1


def prefix_scores_npz_path(project_root: Path, model_type: str) -> Path:
    out_dir = project_root / "metrics" / "ind_scores_over_time" / model_type
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{model_type}_prefix_scores.npz"


def resolve_model_types(
    model_size: str, og_pythia: bool
) -> tuple[list[str], list[str], list[str]]:
    if model_size == "1b":
        model_types = ["vanilla_1b", "hapax_1b"]
        all_model_names = ["vanilla_1b", "hapax_1b"]
        comparison_titles = ["Vanilla", "Hapax"]
    elif model_size == "1b-threshold-0.3":
        model_types = ["vanilla_1b", "hapax_1b_thresh_0.3"]
        all_model_names = ["vanilla_1b", "hapax_1b_thresh_0.3"]
        comparison_titles = ["Vanilla", "Thresholded-Hapax"]
    elif model_size == "1b-triple":
        model_types = ["vanilla_1b", "hapax_1b", "hapax_1b_thresh_0.3"]
        all_model_names = model_types
        comparison_titles = [
            MODEL_TYPE_DISPLAY_NAMES.get(m, m) for m in model_types
        ]
    elif model_size == "2.8b":
        if og_pythia:
            model_types = ["EleutherAI/pythia-2.8b", "masked_bigram_2.8b"]
            all_model_names = ["EleutherAI_pythia-2.8b", "masked_bigram_2.8b"]
        else:
            model_types = ["vanilla_2.8b", "hapax_2.8b"]
            all_model_names = ["vanilla_2.8b", "hapax_2.8b"]
        comparison_titles = ["Vanilla", "Hapax"]
    else:
        raise ValueError(f"Unknown model_size: {model_size!r}")

    return model_types, all_model_names, comparison_titles


def load_npz_scores(npz_path: str, max_step: int | None = None):
    data = np.load(npz_path, allow_pickle=True)
    steps = np.array(data["steps"], dtype=int)
    scores = np.array(data["scores"])

    t = min(len(steps), scores.shape[0])
    steps, scores = steps[:t], scores[:t]

    order = np.argsort(steps)
    steps, scores = steps[order], scores[order]
    if max_step is not None:
        keep = steps <= int(max_step)
        steps, scores = steps[keep], scores[keep]
    return steps, scores


def _filter_heads_by_peak(flat, min_peak: float):
    keep = flat.max(axis=0) >= float(min_peak)
    return flat[:, keep]


def _hex_to_rgba(hex_color: str, alpha: float) -> tuple:
    r, g, b = mpl.colors.to_rgb(hex_color)
    return (r, g, b, alpha)


def _style_neutral_axes(ax, *, labelsize: int = 8) -> None:
    style_induction_box_axes(
        ax, spine_lw=0.85, tick_lw=0.75, tick_len=3.0, labelsize=labelsize
    )


def _plot_spaghetti_on_ax(
    ax,
    steps: np.ndarray,
    flat: np.ndarray,
    *,
    color: str,
    alpha_min: float = 0.02,
    alpha_max: float = 0.85,
    gamma_alpha: float = 2.2,
    lw_min: float = 0.35,
    lw_max: float = 1.25,
    gamma_lw: float = 1.4,
) -> None:
    peak_norm = np.clip(flat.max(axis=0), 0.0, 1.0)
    alpha = alpha_min + (alpha_max - alpha_min) * (peak_norm**gamma_alpha)
    lw = lw_min + (lw_max - lw_min) * (peak_norm**gamma_lw)
    for i in np.argsort(alpha):
        ax.plot(
            steps,
            flat[:, i],
            color=color,
            alpha=float(alpha[i]),
            linewidth=float(lw[i]),
            rasterized=True,
            zorder=3,
        )


def plot_models_combined(
    npz_paths: list[str],
    titles: list[str],
    colors: list[str],
    *,
    min_peak: float = MIN_PEAK,
    label_fontsize: int = 9,
    mini_title_fontsize: int = 8,
    save_path: str | None = None,
    max_step: int | None = None,
) -> None:
    n = len(npz_paths)
    if n == 0:
        return

    loaded = []
    for path in npz_paths:
        steps, scores = load_npz_scores(path, max_step)
        flat = _filter_heads_by_peak(scores.reshape(scores.shape[0], -1), min_peak)
        if flat.shape[1] == 0:
            raise ValueError(f"No heads kept for {path} with min_peak={min_peak}")
        loaded.append((steps, flat))

    figsize = (SMALL_PDF_FIGSIZE[0] * (n + 1) * 0.92, SMALL_PDF_FIGSIZE[1] * 1.05)
    fig, axes = plt.subplots(1, n + 1, figsize=figsize, layout="constrained")
    score_axes, ax_summary = axes[:n], axes[n]

    for j, (ax, (steps, flat), title, color) in enumerate(zip(score_axes, loaded, titles, colors)):
        if j > 0:
            ax.sharex(score_axes[0])
        _plot_spaghetti_on_ax(ax, steps, flat, color=color)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title, fontsize=mini_title_fontsize, pad=3)
        set_training_step_ticks_induction(ax, major=5000, minor=1000)
        _style_neutral_axes(ax, labelsize=7)
        if j > 0:
            ax.tick_params(axis="y", labelleft=False)

    ax_summary.sharex(score_axes[0])
    ax_summary.sharey(score_axes[0])
    for (steps, flat), color in zip(loaded, colors):
        mean = flat.mean(axis=1)
        ax_summary.fill_between(
            steps, mean, step="post", color=_hex_to_rgba(color, 0.15), linewidth=0, zorder=1
        )
        ax_summary.plot(steps, mean, color=color, linewidth=1.6, zorder=2, solid_capstyle="round")
    ax_summary.set_ylim(0.0, 1.0)
    ax_summary.set_title("Mean", fontsize=mini_title_fontsize, pad=3)
    ax_summary.legend(
        [mpl.lines.Line2D([0], [0], color=c, lw=1.4) for c in colors],
        titles,
        frameon=False,
        fontsize=7.5,
        loc="upper left",
        handlelength=1.5,
        borderaxespad=0.35,
    )
    set_training_step_ticks_induction(ax_summary, major=5000, minor=1000)
    _style_neutral_axes(ax_summary, labelsize=7)
    ax_summary.tick_params(axis="y", labelleft=False)

    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelbottom=True)
    fig.supxlabel("Training step", fontsize=label_fontsize)
    fig.supylabel("Prefix-matching score", fontsize=label_fontsize)

    if save_path is not None:
        save_figure_pdf_png(fig, save_path, png_dpi=400)
    plt.close(fig)


def checkpoint_steps(first_step: int, last_step: int, step_stride: int) -> np.ndarray:
    if step_stride <= 0:
        raise ValueError("step_stride must be positive")
    if last_step < first_step:
        return np.array([], dtype=np.int64)
    return np.arange(first_step, last_step + 1, step_stride, dtype=np.int64)


def run_checkpoint_sweep(
    project_root: Path,
    model_dir: str,
    save_file: Path,
    first_step: int,
    last_step: int,
    step_stride: int,
    num_of_samples: int = 1000,
) -> None:
    import torch

    save_file = Path(save_file)
    if save_file.exists():
        data = np.load(save_file, allow_pickle=True)
        steps_done = data["steps"].tolist()
        scores = data["scores"]

    else:
        steps_done = []
        scores = None

    for step in checkpoint_steps(first_step, last_step, step_stride):
        if step in steps_done:
            continue

        checkpoint_dir = f"{model_dir}/step={step}"
        model, tokenizer = load_model_and_tokenizer(checkpoint_dir)
        induction_scores = compute_verbatim_induction_scores(
            model, tokenizer, num_of_samples=num_of_samples, batch_size=256
        )
        induction_scores_np = induction_scores.detach().cpu().numpy()
        if scores is None:
            scores = induction_scores_np[None, ...]
        else:
            scores = np.concatenate([scores, induction_scores_np[None, ...]], axis=0)

        steps_done.append(step)
        np.savez(save_file, steps=np.array(steps_done), scores=scores)

        del model, tokenizer, induction_scores, induction_scores_np
        torch.cuda.empty_cache()
        gc.collect()


def run_figure_section(
    project_root: Path,
    model_size: str,
    model_types: list[str],
    comparison_titles: list[str],
    max_step: int | None = None,
) -> None:
    npz_paths = [str(prefix_scores_npz_path(project_root, name)) for name in model_types]
    keep = [i for i, p in enumerate(npz_paths) if os.path.exists(p)]
    if not keep:
        return
    comparison_dir = project_root / "metrics" / "ind_scores_over_time" / "figures"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plot_models_combined(
        [npz_paths[i] for i in keep],
        titles=[comparison_titles[i] for i in keep],
        colors=[MODEL_COLORS.get(model_types[i], "tab:purple") for i in keep],
        save_path=str(comparison_dir / f"induction_scores_comparison_{model_size}.pdf"),
        max_step=max_step,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", type=str, default="2.8b")
    p.add_argument(
        "--max-plot-step",
        type=int,
        default=None,
        help=(
            "Last step to show in the plot."
        ),
    )
    p.add_argument(
        "--model_type_index", type=int, default=1, help="0 for vanilla, 1 for hapax"
    )
    p.add_argument("--og_pythia", action="store_true")
    p.add_argument("--first-step", type=int, default=100)
    p.add_argument(
        "--last-step",
        type=int,
        default=24900,
    )
    p.add_argument("--step-stride", type=int, default=100)
    p.add_argument(
        "--num-of-samples",
        type=int,
        default=1000,
        help="Random sequences per checkpoint for the prefix-matching score.",
    )
    p.add_argument(
        "--plots-only",
        action="store_true",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    model_types, _, comparison_titles = resolve_model_types(
        args.model_size, args.og_pythia
    )

    model_name = model_types[args.model_type_index]
    model_dir = str(project_root / "models" / model_name)
    save_file = prefix_scores_npz_path(project_root, model_name)

    if not args.plots_only:
        run_checkpoint_sweep(
            project_root,
            model_dir,
            save_file,
            args.first_step,
            args.last_step,
            args.step_stride,
            num_of_samples=args.num_of_samples,
        )

    if not args.no_plots:
        run_figure_section(
            project_root,
            args.model_size,
            model_types,
            comparison_titles,
            max_step=args.max_plot_step,
        )

if __name__ == "__main__":
    main()
