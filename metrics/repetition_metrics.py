from __future__ import annotations

import argparse
import gc
import os
import pickle as pkl
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.model_setup import load_model_and_tokenizer
from lib.plotting import (
    REPETITION_FIGSIZE,
    apply_induction_scores_plot_style,
    save_pyplot_figure_pdf_png,
)

from lib.repetition import (
    natural_text_repetition_accuracy_with_ci,
    random_sequence_repetition_accuracy_with_ci,
)


def resolve_model_types(model_size: str) -> list[str]:
    if model_size == "1b":
        return ["vanilla_1b", "hapax_1b"]
    if model_size == "1b-threshold-0.3":
        return ["vanilla_1b", "hapax_1b_thresh_0.3"]
    if model_size == "2.8b":
        return ["vanilla_2.8b", "hapax_2.8b"]
    raise ValueError(f"Unknown model_size: {model_size!r}")


@dataclass(frozen=True)
class _RepetCurve:
    acc: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class _RepetitionPairPanel:
    vanilla: _RepetCurve
    hapax: _RepetCurve


@dataclass(frozen=True)
class _RepetitionPairLoaded:
    steps: list[int]
    pair_stem: str
    minor_ticks: list[int]
    random: _RepetitionPairPanel
    natural: _RepetitionPairPanel


def _extract_ci_series(results_list: list, n_series: int) -> _RepetCurve:
    acc = np.array([r["accuracy"] for r in results_list[:n_series]])
    low = np.array([r["ci_lower"] for r in results_list[:n_series]])
    up = np.array([r["ci_upper"] for r in results_list[:n_series]])
    return _RepetCurve(acc=acc, lower=low, upper=up)


def load_repetition_pair_for_plots(
    model_types: list[str],
    save_dir: Path,
    first_step: int,
    last_step: int,
    step_stride: int = 100,
) -> _RepetitionPairLoaded:
    if len(model_types) < 2:
        raise ValueError("Expected at least two model_types for pair repetition plots.")

    with open(save_dir / f"random_repetition_{model_types[0]}.pkl", "rb") as f:
        random_vanilla = pkl.load(f)
    with open(save_dir / f"random_repetition_{model_types[1]}.pkl", "rb") as f:
        random_hapax = pkl.load(f)
    with open(save_dir / f"natural_text_repetition_{model_types[0]}.pkl", "rb") as f:
        natural_vanilla = pkl.load(f)
    with open(save_dir / f"natural_text_repetition_{model_types[1]}.pkl", "rb") as f:
        natural_hapax = pkl.load(f)

    n_series = min(
        len(random_vanilla),
        len(random_hapax),
        len(natural_vanilla),
        len(natural_hapax),
    )
    steps = [first_step + i * step_stride for i in range(n_series)]
    pair_stem = f"{model_types[0]}_vs_{model_types[1]}"
    x_max = (last_step // 1000 + 1) * 1000
    minor_ticks = list(range(1000, x_max, 1000))

    random_panel = _RepetitionPairPanel(
        vanilla=_extract_ci_series(random_vanilla, n_series),
        hapax=_extract_ci_series(random_hapax, n_series),
    )
    natural_panel = _RepetitionPairPanel(
        vanilla=_extract_ci_series(natural_vanilla, n_series),
        hapax=_extract_ci_series(natural_hapax, n_series),
    )
    return _RepetitionPairLoaded(
        steps=steps,
        pair_stem=pair_stem,
        minor_ticks=minor_ticks,
        random=random_panel,
        natural=natural_panel,
    )


def _plot_repetition_pair_panel(
    ax: plt.Axes,
    steps: list[int],
    panel: _RepetitionPairPanel,
    *,
    alpha: float,
    minor_ticks: list[int],
    y_label: str = "Accuracy",
    show_xlabel: bool = True,
    show_legend: bool = True,
) -> None:
    ax.set_ylim(0, 1.05)
    ax.set_xlim(100, 20100)
    ax.set_xticks([5000, 10000, 15000, 20000])
    ax.set_xticks(minor_ticks, minor=True)

    v, h = panel.vanilla, panel.hapax
    ax.plot(steps, v.acc, label="Vanilla", color="#0072B2")
    ax.fill_between(steps, v.lower, v.upper, color="#0072B2", alpha=alpha)
    ax.plot(steps, h.acc, label="Hapax", color="#E69F00")
    ax.fill_between(steps, h.lower, h.upper, color="#E69F00", alpha=alpha)

    if show_xlabel:
        ax.set_xlabel("Training Steps")
    ax.set_ylabel(y_label)
    if show_legend:
        ax.legend(loc="best", ncol=1, columnspacing=1.5)
    ax.grid(False)
    ax.set_axisbelow(True)


def repetition_tasks_benchmark(
    project_root: Path,
    model_types: list[str],
    random_repetition_seq_len: int = 25,
    natural_text_repetition_seq_len: int = 25,
    max_sample_size: int = 5000,
    batch_size: int = 64,
    first_step: int = 100,
    last_step: int = 19900,
    step_stride: int = 100,
    skip_clean: bool = False,
):
    dataset_name = "wikitext"
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
        cache_dir=str(Path.home() / ".cache"),
    )

    random_repetition_res_all_masked = []
    natural_text_repetition_res_all_masked = []
    natural_text_repetition_res_all_clean = []
    random_repetition_res_all_clean = []

    if not skip_clean:
        with torch.no_grad():
            for step in tqdm(range(first_step, last_step, step_stride)):
                clean_model_path = str(
                    project_root / "models" / model_types[0] / f"step={step}"
                )
                clean_model, tokenizer_clean = load_model_and_tokenizer(
                    clean_model_path
                )
                random_repetition_res = random_sequence_repetition_accuracy_with_ci(
                    clean_model,
                    tokenizer_clean,
                    seq_len=random_repetition_seq_len,
                    num_of_samples=max_sample_size,
                    batch_size=batch_size,
                )

                natural_text_repetition_res = natural_text_repetition_accuracy_with_ci(
                    clean_model,
                    tokenizer_clean,
                    dataset_name,
                    seq_len=natural_text_repetition_seq_len,
                    num_of_samples=max_sample_size,
                    batch_size=batch_size,
                    dataset=dataset,
                )
                random_repetition_res_all_clean.append(random_repetition_res)
                natural_text_repetition_res_all_clean.append(
                    natural_text_repetition_res
                )

                del clean_model, tokenizer_clean
                gc.collect()
                torch.cuda.empty_cache()

    for step in tqdm(range(first_step, last_step, step_stride)):
        with torch.no_grad():
            masked_model_path = str(
                project_root / "models" / model_types[1] / f"step={step}"
            )
            masked_model, tokenizer_masked = load_model_and_tokenizer(masked_model_path)
            random_repetition_res = random_sequence_repetition_accuracy_with_ci(
                masked_model,
                tokenizer_masked,
                seq_len=random_repetition_seq_len,
                num_of_samples=max_sample_size,
                batch_size=batch_size,
            )
            natural_text_repetition_res = natural_text_repetition_accuracy_with_ci(
                masked_model,
                tokenizer_masked,
                dataset_name,
                seq_len=natural_text_repetition_seq_len,
                num_of_samples=max_sample_size,
                batch_size=batch_size,
                dataset=dataset,
            )
            random_repetition_res_all_masked.append(random_repetition_res)
            natural_text_repetition_res_all_masked.append(natural_text_repetition_res)

            del masked_model, tokenizer_masked
            gc.collect()
            torch.cuda.empty_cache()

    return (
        random_repetition_res_all_clean,
        natural_text_repetition_res_all_clean,
        random_repetition_res_all_masked,
        natural_text_repetition_res_all_masked,
    )


def plot_repetition_pair(
    model_size: str,
    model_types: list[str],
    project_root: Path,
    save_dir: Path | None = None,
    output_dir: Path | None = None,
    first_step: int = 100,
    last_step: int = 19900,
    step_stride: int = 100,
    figsize: tuple[float, float] | None = None,
    alpha: float = 0.3,
):
    if save_dir is None:
        save_dir = (
            project_root
            / "metrics"
            / "repetition_metrics"
            / model_size
        )
    if output_dir is None:
        output_dir = save_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if figsize is None:
        figsize = REPETITION_FIGSIZE

    apply_induction_scores_plot_style()

    data = load_repetition_pair_for_plots(
        model_types, save_dir, first_step, last_step, step_stride
    )

    fig, ax = plt.subplots(figsize=figsize)
    _plot_repetition_pair_panel(
        ax,
        data.steps,
        data.random,
        alpha=alpha,
        minor_ticks=data.minor_ticks,
        show_xlabel=True,
        show_legend=True,
    )
    ax.set_title(f"Random Repetition Performance - {model_size.upper()}", pad=8)
    plt.tight_layout()

    save_path = output_dir / f"random_repetition_{data.pair_stem}.pdf"
    pdf_path, png_path = save_pyplot_figure_pdf_png(save_path)

    fig, ax = plt.subplots(figsize=figsize)
    _plot_repetition_pair_panel(
        ax,
        data.steps,
        data.natural,
        alpha=alpha,
        minor_ticks=data.minor_ticks,
        show_xlabel=True,
        show_legend=True,
    )
    ax.set_title(f"Natural Text Repetition Performance - {model_size.upper()}", pad=8)
    plt.tight_layout()

    save_path = output_dir / f"natural_text_repetition_{data.pair_stem}.pdf"
    pdf_path, png_path = save_pyplot_figure_pdf_png(save_path)

    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", type=str, default="2.8b")
    p.add_argument("--first_step", type=int, default=100)
    p.add_argument("--last_step", type=int, default=25000)
    p.add_argument(
        "--step_stride",
        type=int,
        default=100,
    )
    p.add_argument("--random_repetition_seq_len", type=int, default=25)
    p.add_argument("--natural_text_repetition_seq_len", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_sample_size", type=int, default=1000)
    p.add_argument("--skip_clean", action="store_true")
    p.add_argument("--cuda_visible_devices", type=str, default="0")
    p.add_argument(
        "--skip-plots",
        action="store_true",
    )
    p.add_argument(
        "--plots-only",
        action="store_true",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    project_root = PROJECT_ROOT
    model_types = resolve_model_types(args.model_size)

    save_dir = (
        project_root
        / "metrics"
        / "repetition_metrics"
        / args.model_size
    )
    output_dir = save_dir / "figures"

    if not args.plots_only:
        (
            random_repetition_res_all_clean,
            natural_text_repetition_res_all_clean,
            random_repetition_res_all_masked,
            natural_text_repetition_res_all_masked,
        ) = repetition_tasks_benchmark(
            project_root,
            model_types,
            random_repetition_seq_len=args.random_repetition_seq_len,
            natural_text_repetition_seq_len=args.natural_text_repetition_seq_len,
            max_sample_size=args.max_sample_size,
            batch_size=args.batch_size,
            first_step=args.first_step,
            last_step=args.last_step,
            step_stride=args.step_stride,
            skip_clean=args.skip_clean,
        )

        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / f"random_repetition_{model_types[1]}.pkl", "wb") as f:
            pkl.dump(random_repetition_res_all_masked, f)

        with open(
            save_dir / f"natural_text_repetition_{model_types[1]}.pkl", "wb"
        ) as f:
            pkl.dump(natural_text_repetition_res_all_masked, f)

        if not args.skip_clean:
            with open(
                save_dir / f"natural_text_repetition_{model_types[0]}.pkl", "wb"
            ) as f:
                pkl.dump(natural_text_repetition_res_all_clean, f)

            with open(save_dir / f"random_repetition_{model_types[0]}.pkl", "wb") as f:
                pkl.dump(random_repetition_res_all_clean, f)

    if args.skip_plots:
        return

    if args.skip_clean and not args.plots_only:
        return

    plot_repetition_pair(
        model_size=args.model_size,
        model_types=model_types,
        project_root=project_root,
        save_dir=save_dir,
        output_dir=output_dir,
        first_step=args.first_step,
        last_step=args.last_step,
        step_stride=args.step_stride,
        alpha=0.3,
    )

if __name__ == "__main__":
    main()
