from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument("--cuda_visible_devices", type=str, default="0")
_early_args, _ = _early_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _early_args.cuda_visible_devices

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.model_setup import load_model_and_tokenizer
from lib.plotting import ICLR_WIDE_FIGSIZE, apply_iclr_style, save_figure_pdf_png
from lib.translation import (
    BilingualFewShotDataset,
    evaluate_translation_accuracy_with_ci,
)


def _scrub(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_scrub(x) for x in obj)
    if isinstance(obj, np.ndarray):
        return _scrub(obj.tolist())
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def resolve_translation_metrics_dir(project_root: Path, model_size: str) -> Path:
    return project_root / "metrics" / "translation_accuracies" / model_size


def resolve_model_types(model_size: str, og_pythia: bool) -> list[str]:
    if model_size == "1b":
        return ["vanilla_1b", "hapax_1b"]
    if model_size == "1b_thresh0.3":
        return ["vanilla_1b", "hapax_1b_thresh_0.3", "hapax_1b"]
    if model_size == "2.8b":
        if og_pythia:
            return ["EleutherAI/pythia-1b", "masked_bigram_2.8b"]
        return ["vanilla_2.8b", "hapax_2.8b"]
    raise ValueError(f"Unknown model_size: {model_size!r}")


def compute_translation_metrics(
    model,
    tokenizer,
    project_root: Path,
    lang1,
    lang2,
    batch_size: int = 32,
    max_new_tokens: int = 50,
    n_shots: int = 10,
    step=None,
    random_pairs: bool = False,
    confidence_level: float = 0.95,
    lang_labels: bool = False,
):
    if lang1 is None:
        lang1 = ["spa", "jpn", "fra", "por", "ita", "pol", "cmn", "ind"]
    results = {}
    mcnemar_data = {}
    csv_path = project_root / "dataset" / "parallel_concepts.csv"

    for lang in lang1:
        dataset = BilingualFewShotDataset(
            csv_path,
            lang,
            lang2,
            n_shots,
            random_pairs,
            lang_labels,
        )

        device = next(model.parameters()).device
        metrics, predictions = evaluate_translation_accuracy_with_ci(
            model,
            tokenizer,
            dataset,
            device,
            batch_size,
            max_new_tokens,
            random_pairs,
            confidence_level,
        )

        results[f"{lang}_{lang2}"] = metrics
        mcnemar_data[f"{lang}_{lang2}"] = predictions

    return results, mcnemar_data


def _mcnemar_pvalue_masked_vs_clean(masked: list, clean: list) -> float:
    from statsmodels.stats.contingency_tables import mcnemar

    a = masked
    b = clean
    both_correct = sum(p1 and p2 for p1, p2 in zip(a, b))
    model1_only = sum(p1 and not p2 for p1, p2 in zip(a, b))
    model2_only = sum(not p1 and p2 for p1, p2 in zip(a, b))
    both_wrong = sum(not p1 and not p2 for p1, p2 in zip(a, b))

    table = [[both_correct, model1_only], [model2_only, both_wrong]]
    result = mcnemar(table, exact=False, correction=True)
    return float(result.pvalue)


def pack_translation_json(
    metrics_per_pair: dict[str, Any],
    predictions_per_pair: dict[str, list[bool]],
) -> dict[str, Any]:
    out = {}
    for k, m in metrics_per_pair.items():
        if k not in predictions_per_pair:
            raise KeyError(f"missing predictions for {k!r}")
        out[k] = {**m, "predictions": predictions_per_pair[k]}
    return out


def mcnemar_hapax_vs_vanilla_from_json(
    clean_data: dict[str, Any],
    masked_data: dict[str, Any],
    *,
    alpha: float = 0.05,
    fdr_bh: bool = False,
) -> dict[str, bool] | None:
    pred_masked: dict[str, list] = {}
    pred_clean: dict[str, list] = {}
    keys: list[str] = []
    for k in sorted(clean_data.keys()):
        if k not in masked_data:
            return None
        mc = masked_data[k]
        cl = clean_data[k]
        if not isinstance(mc, dict) or not isinstance(cl, dict):
            return None
        if "predictions" not in mc or "predictions" not in cl:
            return None
        pm, pc = mc["predictions"], cl["predictions"]
        if len(pm) != len(pc):
            return None
        keys.append(k)
        pred_masked[k] = [bool(x) for x in pm]
        pred_clean[k] = [bool(x) for x in pc]
    if not pred_masked:
        return None
    pvals = [
        _mcnemar_pvalue_masked_vs_clean(pred_masked[k], pred_clean[k]) for k in keys
    ]
    if fdr_bh:
        from statsmodels.stats.multitest import multipletests

        reject, _, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        return {k: bool(r) for k, r in zip(keys, reject)}
    return {k: bool(p < alpha) for k, p in zip(keys, pvals)}


def prepare_plot_data_two(clean_data, masked_data):
    lang_display = {
        "spa": "Spanish",
        "fra": "French",
        "por": "Portuguese",
        "ita": "Italian",
        "jpn": "Japanese",
        "cmn": "Chinese",
        "pol": "Polish",
        "ind": "Indonesian",
    }

    lang_pairs = sorted(clean_data.keys())

    languages = []
    lang_pair_keys: list[str] = []
    clean_accs = []
    clean_ci_lower = []
    clean_ci_upper = []
    masked_accs = []
    masked_ci_lower = []
    masked_ci_upper = []

    for pair in lang_pairs:
        lang_code = pair.split("_")[0]

        if lang_code == "eng":
            continue
        lang_name = lang_display.get(lang_code, lang_code.upper())
        languages.append(lang_name)
        lang_pair_keys.append(pair)

        clean_accs.append(clean_data[pair]["accuracy"] * 100)
        clean_ci_lower.append(clean_data[pair]["ci_lower"] * 100)
        clean_ci_upper.append(clean_data[pair]["ci_upper"] * 100)

        masked_accs.append(masked_data[pair]["accuracy"] * 100)
        masked_ci_lower.append(masked_data[pair]["ci_lower"] * 100)
        masked_ci_upper.append(masked_data[pair]["ci_upper"] * 100)

    clean_errors = [
        (acc - lower, upper - acc)
        for acc, lower, upper in zip(clean_accs, clean_ci_lower, clean_ci_upper)
    ]
    masked_errors = [
        (acc - lower, upper - acc)
        for acc, lower, upper in zip(masked_accs, masked_ci_lower, masked_ci_upper)
    ]

    clean_errors = list(zip(*clean_errors)) if clean_errors else ([], [])
    masked_errors = list(zip(*masked_errors)) if masked_errors else ([], [])

    return (
        languages,
        clean_accs,
        masked_accs,
        clean_errors,
        masked_errors,
        lang_pair_keys,
    )


def plot_translation_two_models(
    clean_path,
    masked_path,
    save_path=None,
    plot_title: str | None = None,
    n_shots: int = 5,
    *,
    fdr_bh: bool = False,
    joint_bh: bool = False,
    significance_dict: dict[str, bool] | None = None,
    alpha: float = 0.05,
):
    apply_iclr_style()

    with open(clean_path, "r") as f:
        clean_data = json.load(f)
    with open(masked_path, "r") as f:
        masked_data = json.load(f)

    (
        languages,
        clean_accs,
        masked_accs,
        clean_errors,
        masked_errors,
        lang_pair_keys,
    ) = prepare_plot_data_two(clean_data, masked_data)

    if significance_dict is not None:
        significance_hapax = significance_dict
    else:
        significance_hapax = mcnemar_hapax_vs_vanilla_from_json(
            clean_data, masked_data, alpha=alpha, fdr_bh=fdr_bh
        )

    if plot_title is None:
        plot_title = f"{n_shots}-Shot Translation Performance"

    fig, ax = plt.subplots(figsize=ICLR_WIDE_FIGSIZE)

    x = np.arange(len(languages))
    width = 0.36
    color_clean = "#0173B2"
    color_masked = "#DE8F05"

    ax.bar(
        x - width / 2,
        clean_accs,
        width,
        label="Vanilla",
        color=color_clean,
        edgecolor="black",
        linewidth=0.5,
        yerr=clean_errors if clean_errors[0] else None,
        capsize=3,
        error_kw={"linewidth": 0.8},
    )

    ax.bar(
        x + width / 2,
        masked_accs,
        width,
        label="Hapax",
        color=color_masked,
        edgecolor="black",
        linewidth=0.5,
        yerr=masked_errors if masked_errors[0] else None,
        capsize=3,
        error_kw={"linewidth": 0.8},
    )

    if significance_hapax is not None:
        upper_masked = (
            list(masked_errors[1])
            if masked_errors and len(masked_errors) == 2
            else [0.0] * len(lang_pair_keys)
        )
        for i, pair in enumerate(lang_pair_keys):
            if significance_hapax.get(pair):
                y_top = masked_accs[i] + upper_masked[i] + 0.5
                ax.text(
                    x[i] + width / 2,
                    y_top,
                    "†",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_xlabel("Source Language", fontsize=10)
    ax.set_ylabel("Translation Accuracy (%)", fontsize=10)
    ax.set_title(plot_title, fontsize=11, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(languages, rotation=45, ha="right")

    ax.yaxis.grid(True, linestyle="--", alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)

    ax.legend(loc="best", fontsize=9)

    ax.set_ylim(bottom=0)

    max_with_error = 0
    if clean_errors and clean_errors[0]:
        max_clean_with_error = max(
            acc + err for acc, err in zip(clean_accs, clean_errors[1])
        )
        max_with_error = max(max_with_error, max_clean_with_error)
    if masked_errors and masked_errors[0]:
        max_masked_with_error = max(
            acc + err for acc, err in zip(masked_accs, masked_errors[1])
        )
        max_with_error = max(max_with_error, max_masked_with_error)

    extra_top = 0.0
    if significance_hapax and any(significance_hapax.values()):
        extra_top = 0.05
    if max_with_error > 0:
        ax.set_ylim(top=max_with_error * (1.12 + extra_top))

    show_footnote = significance_hapax is None or (
        significance_hapax and any(significance_hapax.values())
    )
    plt.tight_layout(rect=[0, 0.07 if show_footnote else 0, 1, 1])

    if significance_hapax is None:
        fig.text(
            0.5,
            0.02,
            "Add per-example predictions to JSON (re-run evaluation) for McNemar †.",
            ha="center",
            fontsize=7,
            style="italic",
            color="0.35",
        )
    elif any(significance_hapax.values()):
        if joint_bh:
            foot = (
                f"† McNemar paired tests with joint Benjamini--Hochberg FDR across "
                f"abstractive tasks and translation language pairs at α = {alpha:g} "
                "(Hapax vs Vanilla)."
            )
        elif fdr_bh:
            foot = (
                f"† McNemar paired tests with Benjamini--Hochberg FDR control at "
                f"α = {alpha:g} (Hapax vs Vanilla)."
            )
        else:
            foot = f"† McNemar test, p < {alpha:g} uncorrected (Hapax vs Vanilla)."
        fig.text(
            0.5,
            0.02,
            foot,
            ha="center",
            fontsize=7,
            style="italic",
        )

    if save_path:
        save_path = Path(save_path)
        pdf_path, png_path = save_figure_pdf_png(fig, save_path)

        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def release_model(model, tokenizer) -> None:
    import torch

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", type=str, default="2.8b")
    p.add_argument("--step", type=int, default=24900)
    p.add_argument("--lang2", type=str, default="eng")
    p.add_argument("--n_shots", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--skip_clean", action="store_true")
    p.add_argument("--og_pythia", action="store_true")
    p.add_argument("--random_pairs", action="store_true")
    p.add_argument("--cuda_visible_devices", type=str, default="6")
    p.add_argument(
        "--skip-plots",
        action="store_true",
    )
    p.add_argument(
        "--plots-only",
        action="store_true",
    )
    p.add_argument(
        "--translation-dir",
        type=str,
        default=None,
    )
    p.add_argument(
        "--fdr-bh",
        action="store_true",
        help="Apply Benjamini-Hochberg FDR correction across language pairs to the McNemar tests.",
    )
    p.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.05,
        help="Significance level for the McNemar tests.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    model_types = resolve_model_types(args.model_size, args.og_pythia)
    translation_path = (
        Path(args.translation_dir).resolve()
        if args.translation_dir
        else resolve_translation_metrics_dir(project_root, args.model_size)
    )
    translation_path.mkdir(parents=True, exist_ok=True)

    lang1 = None
    lang_labels = False

    clean_path = (
        translation_path
        / f"translation_{model_types[0]}_{args.step}_{args.n_shots}-shot.json"
    )
    masked_path = (
        translation_path
        / f"translation_{model_types[1]}_{args.step}_{args.n_shots}-shot.json"
    )

    if args.plots_only:
        if not clean_path.exists() or not masked_path.exists():
            sys.exit(1)
        if not args.skip_plots:
            save_dir = translation_path / "plots"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / (
                f"{model_types[0]}_vs_{model_types[1]}_{args.step}_{args.n_shots}-shot"
            )
            plot_translation_two_models(
                str(clean_path),
                str(masked_path),
                save_path=str(save_path),
                n_shots=args.n_shots,
                fdr_bh=args.fdr_bh,
                alpha=args.fdr_alpha,
            )

        return

    accuracy_clean: dict[str, Any] | None = None
    mcnemar_data_clean = None

    if not args.skip_clean:
        if not args.og_pythia:
            clean_model_path = str(
                project_root / "models" / model_types[0] / f"step={args.step}"
            )
            clean_model, tokenizer_clean = load_model_and_tokenizer(clean_model_path)
        else:
            clean_model, tokenizer_clean = load_model_and_tokenizer(
                model_types[0], revision=f"step{20000}", use_og=args.og_pythia
            )

        accuracy_clean, mcnemar_data_clean = compute_translation_metrics(
            clean_model,
            tokenizer_clean,
            project_root,
            lang1,
            args.lang2,
            args.batch_size,
            args.max_new_tokens,
            args.n_shots,
            args.step,
            args.random_pairs,
            lang_labels=lang_labels,
        )
        release_model(clean_model, tokenizer_clean)

        with open(
            translation_path
            / f"translation_{model_types[0]}_{args.step}_{args.n_shots}-shot.json",
            "w",
        ) as f:
            json.dump(
                pack_translation_json(accuracy_clean, mcnemar_data_clean),
                f,
            )

    masked_model_path = str(
        project_root / "models" / model_types[1] / f"step={args.step}"
    )
    masked_model, tokenizer_masked = load_model_and_tokenizer(masked_model_path)

    accuracy_masked, mcnemar_data_masked = compute_translation_metrics(
        masked_model,
        tokenizer_masked,
        project_root,
        lang1,
        args.lang2,
        args.batch_size,
        args.max_new_tokens,
        args.n_shots,
        args.step,
        args.random_pairs,
        lang_labels=lang_labels,
    )
    release_model(masked_model, tokenizer_masked)

    with open(
        translation_path
        / f"translation_{model_types[1]}_{args.step}_{args.n_shots}-shot.json",
        "w",
    ) as f:
        json.dump(
            pack_translation_json(accuracy_masked, mcnemar_data_masked),
            f,
        )

    if args.skip_plots:
        return

    if args.skip_clean:
        return


    save_dir = translation_path / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / (
        f"{model_types[0]}_vs_{model_types[1]}_{args.step}_{args.n_shots}-shot"
    )
    plot_translation_two_models(
        str(clean_path),
        str(masked_path),
        save_path=str(save_path),
        n_shots=args.n_shots,
        fdr_bh=args.fdr_bh,
        alpha=args.fdr_alpha,
    )

if __name__ == "__main__":
    main()
