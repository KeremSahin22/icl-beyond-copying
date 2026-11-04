from __future__ import annotations

import argparse
import re
import json
import os
import pickle as pkl
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.model_setup import load_model_and_tokenizer
from lib.plotting import MODEL_COLORS, MODEL_TYPE_DISPLAY_NAMES, REPETITION_FIGSIZE, REPETITION_OVER_TIME_CI_ALPHA, REPETITION_OVER_TIME_XLABEL, apply_induction_scores_plot_style, micro_avg_abstractive_and_translation_accuracy, save_pyplot_figure_pdf_png, set_training_step_ticks_induction, style_induction_box_axes, style_repetition_over_time_xaxis
from lib.translation import BilingualFewShotDataset, evaluate_translation_accuracy_with_ci
from metrics.fv_metrics import (
    format_task_name,
    fv_tasks_benchmark,
    list_icl_tasks,
    release_model,
    resolve_model_types,
)

DEFAULT_LANG1 = ("spa", "jpn", "fra", "por", "ita", "pol", "cmn", "ind")
DEFAULT_MODEL_SIZE = "1b"

DEFAULT_MODEL_TYPES_1B: tuple[str, ...] = (
    "vanilla_1b",
    "hapax_1b",
    "hapax_1b_thresh_0.3",
)

METRIC_NAMES = ("abstractive_icl", "macro_mean_fv", "macro_mean_translation")

DEFAULT_TARGET_LABEL_MODES: tuple[bool, ...] = (False, True)

RANDOM_REPETITION_METRIC_SUBDIRS: dict[str, str] = {
    "vanilla_1b": "1b",
    "hapax_1b": "1b",
    "hapax_1b_thresh_0.3": "1b-threshold-0.3",
}
RANDOM_REPETITION_FIRST_STEP = 100
RANDOM_REPETITION_STEP_STRIDE = 100


@dataclass(frozen=True)
class MetricWithCi:
    mean: float
    ci_lower: float
    ci_upper: float
    std_error: float


def abstractive_save_dir(project_root: Path, model_type: str, *, exclude_target_label: bool) -> Path:
    suffix = "_excl_target" if exclude_target_label else ""
    return project_root / "metrics" / "abstractive_accuracies" / f"{model_type}{suffix}"


def fv_json_path(save_dir: Path, model_type: str, step: int, n_shots: int) -> Path:
    return save_dir / f"abstractive_results_{model_type}_{step}_{n_shots}.json"


def translation_json_path(save_dir: Path, model_type: str, step: int, n_shots: int) -> Path:
    return save_dir / f"translation_{model_type}_{step}_{n_shots}-shot.json"


def summary_npz_path(save_dir: Path, model_type: str) -> Path:
    return save_dir / f"{model_type}_abstractive_accuracies.npz"


def summary_json_path(save_dir: Path, model_type: str) -> Path:
    return save_dir / f"{model_type}_abstractive_over_time.json"


def target_mode_tag(exclude_target_label: bool) -> str:
    return "excl_target" if exclude_target_label else "incl_target"


def series_legend_label(model_type: str, exclude_target_label: bool) -> str:
    display = MODEL_TYPE_DISPLAY_NAMES.get(model_type, model_type)
    if exclude_target_label:
        return f"{display} - Target Excluded"
    return f"{display} - Standard"


def target_mode_plot_kwargs(
    exclude_target_label: bool,
    color: str,
    *,
    repetition_style: bool = False,
    linewidth: float | None = None,
    model_type: str | None = None,
) -> dict:
    line_lw = (
        linewidth
        if linewidth is not None
        else float(plt.rcParams.get("lines.linewidth", 1.0))
    )
    marker_ms = (
        float(plt.rcParams.get("lines.markersize", 3.0)) + 0.3
        if repetition_style
        else 3.5
    )
    common: dict = {
        "color": color,
        "linewidth": line_lw,
        "marker": "o",
        "markersize": marker_ms,
        "zorder": 2,
    }
    if exclude_target_label:
        return {
            **common,
            "linestyle": (0, (5.0, 2.5)),
            "markerfacecolor": color,
            "markeredgecolor": color,
            "markeredgewidth": 0.8,
        }
    return {
        **common,
        "linestyle": "-",
        "markerfacecolor": color,
        "markeredgecolor": color,
        "markeredgewidth": 0.8,
    }


def _add_in_axes_model_legends(
    ax: plt.Axes,
    *,
    fontsize: float,
    vanilla_loc: str = "upper left",
    hapax_loc: str = "lower right",
    handlelength: float = 2.2,
) -> None:
    handles, labels = ax.get_legend_handles_labels()

    if any(" - " in lab for lab in labels) and len(labels) > 4:
        from matplotlib.lines import Line2D

        seen: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            model_lab = label.split(" - ")[0]
            if model_lab in seen:
                continue
            seen[model_lab] = Line2D(
                [], [],
                color=handle.get_color(),
                marker=handle.get_marker(),
                markersize=handle.get_markersize(),
                markerfacecolor=handle.get_color(),
                markeredgecolor=handle.get_color(),
                linestyle="-",
                linewidth=handle.get_linewidth(),
            )
        handles = list(seen.values())
        labels = list(seen.keys())

        mode_key = [
            Line2D([], [], color="0.35", linestyle="-", linewidth=1.4),
            Line2D([], [], color="0.35", linestyle=(0, (5.0, 2.5)), linewidth=1.4),
        ]
        leg_mode = ax.legend(
            mode_key,
            ["Standard", "Target Excluded"],
            loc="lower right",
            frameon=False,
            fontsize=fontsize - 0.5,
            handlelength=handlelength,
            labelspacing=0.3,
            borderaxespad=0.6,
        )
        ax.add_artist(leg_mode)
        ax.legend(
            handles, labels, loc=vanilla_loc, frameon=False, fontsize=fontsize,
            handlelength=handlelength, labelspacing=0.35, borderaxespad=0.6,
        )
        return
    vanilla_h: list = []
    vanilla_l: list[str] = []
    hapax_h: list = []
    hapax_l: list[str] = []
    for handle, label in zip(handles, labels):
        if label.startswith("Vanilla"):
            vanilla_h.append(handle)
            vanilla_l.append(label)
        elif label.startswith("Hapax"):
            hapax_h.append(handle)
            hapax_l.append(label)
    legend_kw = {
        "frameon": False,
        "fontsize": fontsize,
        "handlelength": handlelength,
        "labelspacing": 0.35,
        "borderaxespad": 0.6,
    }
    if vanilla_h:
        leg_vanilla = ax.legend(vanilla_h, vanilla_l, loc=vanilla_loc, **legend_kw)
        ax.add_artist(leg_vanilla)
    if hapax_h:
        ax.legend(hapax_h, hapax_l, loc=hapax_loc, **legend_kw)


def model_types_for_eval(args: argparse.Namespace) -> list[str]:
    if args.model_types:
        return list(args.model_types)
    if args.model_type:
        return [args.model_type]
    if args.model_size in ("1b", "1b-triple"):
        return list(DEFAULT_MODEL_TYPES_1B)
    if args.model_size == "1b-pair":
        return ["vanilla_1b", "hapax_1b"]
    types, _, _ = resolve_model_types(args.model_size)
    return list(types)


def parse_plot_steps(steps_str: str | None) -> tuple[int, ...] | None:
    if steps_str is None:
        return None
    try:
        return tuple(sorted({int(s.strip()) for s in steps_str.split(",")}))
    except ValueError as e:
        raise ValueError(f"Invalid --plot-steps argument: {e}")


def parse_custom_steps(steps_str: str | None) -> np.ndarray | None:
    if steps_str is None:
        return None
    try:
        steps = np.array(sorted(set(int(s.strip()) for s in steps_str.split(","))), dtype=np.int64)
        return steps
    except ValueError as e:
        raise ValueError(f"Invalid --steps argument: {e}")


def resolve_exclude_target_modes(args: argparse.Namespace) -> tuple[bool, ...]:
    if args.only_exclude_target_label and args.only_include_target_label:
        raise ValueError("Cannot set both --only-exclude-target-label and --only-include-target-label.")
    if args.only_exclude_target_label:
        return (True,)
    if args.only_include_target_label:
        return (False,)
    return DEFAULT_TARGET_LABEL_MODES


def checkpoint_steps(first_step: int, last_step: int, step_stride: int) -> np.ndarray:
    if step_stride <= 0:
        raise ValueError("step_stride must be positive")
    if last_step < first_step:
        return np.array([], dtype=np.int64)
    if first_step < step_stride:
        grid = np.arange(step_stride, last_step + 1, step_stride, dtype=np.int64)
        steps = np.unique(np.concatenate(([np.int64(first_step)], grid)))
    else:
        steps = np.arange(first_step, last_step + 1, step_stride, dtype=np.int64)
    if last_step not in steps:
        steps = np.unique(np.concatenate(([np.int64(last_step)], steps)))
    return steps


def unit_std_error(row: dict[str, Any]) -> float:
    if "std_error" in row and row["std_error"] is not None:
        return float(row["std_error"])
    if "ci_lower" in row and "ci_upper" in row:
        return (float(row["ci_upper"]) - float(row["ci_lower"])) / 2.0
    acc = float(row["accuracy"])
    n = int(row.get("n_samples", 1))
    if n <= 0:
        return 0.0
    return float(np.sqrt(max(acc * (1.0 - acc) / n, 0.0)))


def macro_mean_accuracy(aggregate: dict[str, Any]) -> float:
    if not aggregate:
        return float("nan")
    return float(mean(float(v["accuracy"]) for v in aggregate.values()))


def macro_std_error(aggregate: dict[str, Any]) -> float:
    if not aggregate:
        return float("nan")
    n = len(aggregate)
    var_mean = sum(unit_std_error(v) ** 2 for v in aggregate.values()) / (n * n)
    return float(np.sqrt(var_mean))


def ci_from_mean_se(mean: float, se: float, confidence_level: float) -> tuple[float, float]:
    if not np.isfinite(mean) or not np.isfinite(se):
        return float("nan"), float("nan")
    z = float(stats.norm.ppf((1.0 + confidence_level) / 2.0))
    margin = z * se
    return max(0.0, mean - margin), min(1.0, mean + margin)


def macro_metric_with_ci(
    aggregate: dict[str, Any],
    confidence_level: float,
) -> MetricWithCi:
    m = macro_mean_accuracy(aggregate)
    se = macro_std_error(aggregate)
    lo, hi = ci_from_mean_se(m, se, confidence_level)
    return MetricWithCi(mean=m, ci_lower=lo, ci_upper=hi, std_error=se)


def macro_mean_fv_and_translation(
    fv_aggregate: dict[str, Any],
    translation_aggregate: dict[str, Any],
    *,
    confidence_level: float,
) -> tuple[MetricWithCi, MetricWithCi, MetricWithCi, int, int]:
    fv = macro_metric_with_ci(fv_aggregate, confidence_level)
    tr = macro_metric_with_ci(translation_aggregate, confidence_level)
    n_fv = len(fv_aggregate)
    n_tr = len(translation_aggregate)
    combined_mean = float(
        micro_avg_abstractive_and_translation_accuracy(
            np.array([fv.mean]),
            np.array([tr.mean]),
            n_fv,
            n_tr,
        )[0]
    )
    if n_fv + n_tr == 0:
        combined = MetricWithCi(
            mean=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            std_error=float("nan"),
        )
    else:
        w_fv = n_fv / (n_fv + n_tr)
        w_tr = n_tr / (n_fv + n_tr)
        se_combined = float(np.sqrt(w_fv**2 * fv.std_error**2 + w_tr**2 * tr.std_error**2))
        lo, hi = ci_from_mean_se(combined_mean, se_combined, confidence_level)
        combined = MetricWithCi(
            mean=combined_mean,
            ci_lower=lo,
            ci_upper=hi,
            std_error=se_combined,
        )
    return fv, tr, combined, n_fv, n_tr


def _scrub(obj: Any) -> Any:
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
    if isinstance(obj, MetricWithCi):
        return {
            "mean": obj.mean,
            "ci_lower": obj.ci_lower,
            "ci_upper": obj.ci_upper,
            "std_error": obj.std_error,
        }
    return obj


def _npz_scalar_keys() -> list[str]:
    keys: list[str] = []
    for name in METRIC_NAMES:
        keys.extend([name, f"{name}_ci_lower", f"{name}_ci_upper"])
    keys.extend(["macro_mean", "macro_mean_ci_lower", "macro_mean_ci_upper", "task_accuracies", "task_names"])
    return keys


def load_summary_npz(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"steps": []}
    data = np.load(path, allow_pickle=True)
    out: dict[str, Any] = {"steps": data["steps"].tolist()}
    for key in _npz_scalar_keys():
        if key in data:
            val = data[key]
            out[key] = np.array(val) if key != "task_names" else [str(x) for x in val.tolist()]
    return out

_SCALAR_NPZ_KEYS: tuple[str, ...] = (
    "macro_mean_fv",
    "macro_mean_fv_ci_lower",
    "macro_mean_fv_ci_upper",
    "macro_mean_translation",
    "macro_mean_translation_ci_lower",
    "macro_mean_translation_ci_upper",
    "abstractive_icl",
    "abstractive_icl_ci_lower",
    "abstractive_icl_ci_upper",
)


def append_summary_npz(
    path: Path,
    step: int,
    fv_aggregate: dict[str, Any],
    *,
    fv: MetricWithCi,
    tr: MetricWithCi,
    combined: MetricWithCi,
) -> None:
    saved = load_summary_npz(path)
    steps_old = [int(s) for s in saved.get("steps", [])]
    n_old = len(steps_old)

    scalar_rows: dict[int, dict[str, float]] = {}
    for i, s in enumerate(steps_old):
        if s == step:
            continue
        row: dict[str, float] = {}
        for key in _SCALAR_NPZ_KEYS:
            if key in saved:
                arr = np.asarray(saved[key], dtype=np.float64)
                if i < len(arr):
                    row[key] = float(arr[i])
        if row:
            scalar_rows[s] = row

    ordered_tasks = sorted(fv_aggregate.keys())
    task_names_old = [str(x) for x in saved.get("task_names", [])] or ordered_tasks
    task_by_step: dict[int, np.ndarray] = {}
    if "task_accuracies" in saved and n_old:
        ta = np.asarray(saved["task_accuracies"], dtype=np.float64)
        if ta.ndim >= 2:
            for i, s in enumerate(steps_old):
                if s != step and i < ta.shape[0]:
                    task_by_step[s] = ta[i].copy()

    def _acc_row_for_tasks(names: list[str]) -> np.ndarray:
        return np.array(
            [float(fv_aggregate[t]["accuracy"]) if t in fv_aggregate else np.nan for t in names],
            dtype=np.float64,
        )

    scalar_rows[step] = {
        "macro_mean_fv": fv.mean,
        "macro_mean_fv_ci_lower": fv.ci_lower,
        "macro_mean_fv_ci_upper": fv.ci_upper,
        "macro_mean_translation": tr.mean,
        "macro_mean_translation_ci_lower": tr.ci_lower,
        "macro_mean_translation_ci_upper": tr.ci_upper,
        "abstractive_icl": combined.mean,
        "abstractive_icl_ci_lower": combined.ci_lower,
        "abstractive_icl_ci_upper": combined.ci_upper,
    }
    task_by_step[step] = _acc_row_for_tasks(ordered_tasks)
    task_names = ordered_tasks

    if task_names_old != task_names:
        for s, row in list(task_by_step.items()):
            if s == step:
                continue
            realigned = np.full(len(task_names), np.nan, dtype=np.float64)
            for j, name in enumerate(task_names):
                if name in task_names_old:
                    old_j = task_names_old.index(name)
                    if old_j < len(row):
                        realigned[j] = row[old_j]
            task_by_step[s] = realigned

    steps_sorted = np.array(sorted(scalar_rows.keys()), dtype=np.int64)
    arrays = {
        key: np.array([scalar_rows[int(s)].get(key, np.nan) for s in steps_sorted], dtype=np.float64)
        for key in _SCALAR_NPZ_KEYS
    }
    nan_task_row = np.full(len(task_names), np.nan, dtype=np.float64)
    task_arr = np.stack(
        [task_by_step.get(int(s), nan_task_row.copy()) for s in steps_sorted],
        axis=0,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        steps=steps_sorted,
        macro_mean_fv=arrays["macro_mean_fv"],
        macro_mean_fv_ci_lower=arrays["macro_mean_fv_ci_lower"],
        macro_mean_fv_ci_upper=arrays["macro_mean_fv_ci_upper"],
        macro_mean_translation=arrays["macro_mean_translation"],
        macro_mean_translation_ci_lower=arrays["macro_mean_translation_ci_lower"],
        macro_mean_translation_ci_upper=arrays["macro_mean_translation_ci_upper"],
        abstractive_icl=arrays["abstractive_icl"],
        abstractive_icl_ci_lower=arrays["abstractive_icl_ci_lower"],
        abstractive_icl_ci_upper=arrays["abstractive_icl_ci_upper"],
        macro_mean=arrays["macro_mean_fv"],
        macro_mean_ci_lower=arrays["macro_mean_fv_ci_lower"],
        macro_mean_ci_upper=arrays["macro_mean_fv_ci_upper"],
        task_accuracies=task_arr,
        task_names=np.array(task_names, dtype=object),
    )


def load_or_init_summary_json(path: Path, *, model_type: str, eval_kwargs: dict[str, Any]) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "model_type": model_type,
        "eval_kwargs": eval_kwargs,
        "per_step": {},
    }


def per_step_record(
    step: int,
    fv: MetricWithCi,
    tr: MetricWithCi,
    combined: MetricWithCi,
    n_fv: int,
    n_tr: int,
    fv_path: Path,
    tr_path: Path,
) -> dict[str, Any]:
    return {
        "step": step,
        "macro_mean_fv": fv.mean,
        "macro_mean_fv_ci_lower": fv.ci_lower,
        "macro_mean_fv_ci_upper": fv.ci_upper,
        "macro_mean_translation": tr.mean,
        "macro_mean_translation_ci_lower": tr.ci_lower,
        "macro_mean_translation_ci_upper": tr.ci_upper,
        "abstractive_icl": combined.mean,
        "abstractive_icl_ci_lower": combined.ci_lower,
        "abstractive_icl_ci_upper": combined.ci_upper,
        "n_abstractive_tasks": n_fv,
        "n_translation_pairs": n_tr,
        "fv_json_path": fv_path.name,
        "translation_json_path": tr_path.name,
    }


def step_is_complete(
    model_type: str,
    step: int,
    n_shots: int,
    *,
    exclude_target_label: bool,
) -> bool:
    save_dir = abstractive_save_dir(
        PROJECT_ROOT, model_type, exclude_target_label=exclude_target_label
    )
    fv_path = fv_json_path(save_dir, model_type, step, n_shots)
    tr_path = translation_json_path(save_dir, model_type, step, n_shots)
    npz_path = summary_npz_path(save_dir, model_type)
    if not fv_path.is_file() or not tr_path.is_file() or not npz_path.is_file():
        return False
    summary_path = summary_json_path(save_dir, model_type)
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(step) not in summary.get("per_step", {}):
        return False
    npz_data = load_summary_npz(npz_path)
    return step in npz_data.get("steps", []) and "abstractive_icl_ci_lower" in npz_data


def load_translation_json(
    model_type: str,
    step: int,
    n_shots: int,
    *,
    exclude_target_label: bool,
) -> dict[str, Any] | None:
    for excl in (exclude_target_label, not exclude_target_label):
        save_dir = abstractive_save_dir(PROJECT_ROOT, model_type, exclude_target_label=excl)
        tr_path = translation_json_path(save_dir, model_type, step, n_shots)
        if tr_path.is_file():
            return json.loads(tr_path.read_text(encoding="utf-8"))
    return None


def try_process_step_from_json(
    model_type: str,
    step: int,
    n_shots: int,
    *,
    exclude_target_label: bool,
    confidence_level: float,
    summary_json: dict[str, Any],
) -> bool:
    save_dir = abstractive_save_dir(
        PROJECT_ROOT, model_type, exclude_target_label=exclude_target_label
    )
    fv_path = fv_json_path(save_dir, model_type, step, n_shots)
    if not fv_path.is_file():
        return False
    translation_aggregate = load_translation_json(
        model_type, step, n_shots, exclude_target_label=exclude_target_label
    )
    if translation_aggregate is None:
        return False
    fv_aggregate = json.loads(fv_path.read_text(encoding="utf-8"))
    npz_path = summary_npz_path(save_dir, model_type)
    process_step_artifacts(
        save_dir,
        model_type,
        step,
        n_shots,
        fv_aggregate,
        translation_aggregate,
        confidence_level=confidence_level,
        npz_path=npz_path,
        summary_json=summary_json,
        exclude_target_label=exclude_target_label,
    )
    return True


def _evaluate_translation_only(
    model_type: str,
    step: int,
    *,
    lang1: tuple[str, ...],
    lang2: str,
    batch_size: int,
    max_new_tokens: int,
    confidence_level: float,
) -> dict[str, Any]:
    ckpt = str(PROJECT_ROOT / "models" / model_type / f"step={step}")
    model, tokenizer = load_model_and_tokenizer(ckpt)
    try:
        csv_path = PROJECT_ROOT / "dataset" / "parallel_concepts.csv"
        device = next(model.parameters()).device
        translation_aggregate: dict[str, Any] = {}
        for lang in lang1:
            dataset = BilingualFewShotDataset(csv_path, lang, lang2, 5, False, True)
            metrics, _ = evaluate_translation_accuracy_with_ci(
                model,
                tokenizer,
                dataset,
                device,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                random_pairs=False,
                confidence_level=confidence_level,
            )
            translation_aggregate[f"{lang}_{lang2}"] = metrics
    finally:
        release_model(model, tokenizer)
    return translation_aggregate


def evaluate_step_gpu(
    model_type: str,
    step: int,
    *,
    n_shots: int,
    max_sample_size: int,
    batch_size: int,
    max_new_tokens: int,
    translation_max_new_tokens: int,
    exclude_target_modes: tuple[bool, ...],
    abstractive_tasks: list[str],
    lang1: tuple[str, ...],
    lang2: str,
    confidence_level: float,
) -> tuple[dict[bool, dict[str, Any]], dict[str, Any]]:
    ckpt = str(PROJECT_ROOT / "models" / model_type / f"step={step}")
    model, tokenizer = load_model_and_tokenizer(ckpt)
    fv_by_mode: dict[bool, dict[str, Any]] = {}
    try:
        csv_path = PROJECT_ROOT / "dataset" / "parallel_concepts.csv"
        device = next(model.parameters()).device
        translation_aggregate: dict[str, Any] = {}
        for lang in lang1:
            dataset = BilingualFewShotDataset(csv_path, lang, lang2, n_shots, False, False)
            metrics, _ = evaluate_translation_accuracy_with_ci(
                model,
                tokenizer,
                dataset,
                device,
                batch_size=batch_size,
                max_new_tokens=translation_max_new_tokens,
                random_pairs=False,
                confidence_level=confidence_level,
            )
            translation_aggregate[f"{lang}_{lang2}"] = metrics

        for exclude_target_label in exclude_target_modes:
            fv_aggregate, _ = fv_tasks_benchmark(
                model,
                tokenizer,
                abstractive_tasks,
                project_root=PROJECT_ROOT,
                num_of_shots=n_shots,
                max_sample_size=max_sample_size,
                batch_size=batch_size,
                exclude_target_label=exclude_target_label,
                max_new_tokens=max_new_tokens,
                return_per_sample=True,
            )
            fv_by_mode[exclude_target_label] = fv_aggregate
    finally:
        release_model(model, tokenizer)

    return fv_by_mode, translation_aggregate


def process_step_artifacts(
    save_dir: Path,
    model_type: str,
    step: int,
    n_shots: int,
    fv_aggregate: dict[str, Any],
    translation_aggregate: dict[str, Any],
    *,
    confidence_level: float,
    npz_path: Path,
    summary_json: dict[str, Any],
    exclude_target_label: bool,
) -> None:
    fv_m, tr_m, combined, n_fv, n_tr = macro_mean_fv_and_translation(
        fv_aggregate, translation_aggregate, confidence_level=confidence_level
    )
    fv_path = fv_json_path(save_dir, model_type, step, n_shots)
    tr_path = translation_json_path(save_dir, model_type, step, n_shots)

    with open(fv_path, "w", encoding="utf-8") as f:
        json.dump(_scrub(fv_aggregate), f)
    with open(tr_path, "w", encoding="utf-8") as f:
        json.dump(_scrub(translation_aggregate), f)

    append_summary_npz(npz_path, step, fv_aggregate, fv=fv_m, tr=tr_m, combined=combined)

    summary_json.setdefault("per_step", {})[str(step)] = per_step_record(
        step, fv_m, tr_m, combined, n_fv, n_tr, fv_path, tr_path
    )
    summary_json_path(save_dir, model_type).write_text(
        json.dumps(_scrub(summary_json), indent=2),
        encoding="utf-8",
    )


def init_run_state(
    model_type: str,
    args: argparse.Namespace,
    *,
    abstractive_tasks: list[str],
    lang1: tuple[str, ...],
    exclude_target_modes: tuple[bool, ...],
) -> dict[bool, dict[str, Any]]:
    eval_kwargs = {
        "n_shots": args.n_shots,
        "max_sample_size": args.max_sample_size,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "translation_max_new_tokens": args.translation_max_new_tokens,
        "target_label_modes": [target_mode_tag(m) for m in exclude_target_modes],
        "confidence_level": args.confidence_level,
        "n_abstractive_tasks": len(abstractive_tasks),
        "n_translation_pairs": len(lang1),
        "lang1": list(lang1),
        "lang2": args.lang2,
    }
    state: dict[bool, dict[str, Any]] = {}
    for exclude_target_label in exclude_target_modes:
        save_dir = abstractive_save_dir(
            PROJECT_ROOT, model_type, exclude_target_label=exclude_target_label
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        summary_json = load_or_init_summary_json(
            summary_json_path(save_dir, model_type),
            model_type=model_type,
            eval_kwargs=eval_kwargs,
        )
        summary_json["eval_kwargs"] = eval_kwargs
        state[exclude_target_label] = {
            "save_dir": save_dir,
            "summary_json": summary_json,
            "npz_path": summary_npz_path(save_dir, model_type),
        }
    return state


def rebuild_summary_from_json(
    model_type: str,
    args: argparse.Namespace,
    *,
    exclude_target_modes: tuple[bool, ...],
) -> None:
    lang1 = tuple(args.lang1) if args.lang1 else DEFAULT_LANG1
    _, abstractive_tasks = list_icl_tasks(PROJECT_ROOT)
    state = init_run_state(
        model_type,
        args,
        abstractive_tasks=abstractive_tasks,
        lang1=lang1,
        exclude_target_modes=exclude_target_modes,
    )

    custom_steps = parse_custom_steps(args.steps)
    steps_to_eval = custom_steps if custom_steps is not None else checkpoint_steps(args.first_step, args.last_step, args.step_stride)

    for exclude_target_label in exclude_target_modes:
        save_dir = state[exclude_target_label]["save_dir"]
        npz_path = state[exclude_target_label]["npz_path"]
        summary_json = state[exclude_target_label]["summary_json"]
        if not save_dir.is_dir():
            continue
        if npz_path.is_file():
            npz_path.unlink()

        n_rebuilt = 0
        for step in steps_to_eval:
            step = int(step)
            if try_process_step_from_json(
                model_type,
                step,
                args.n_shots,
                exclude_target_label=exclude_target_label,
                confidence_level=args.confidence_level,
                summary_json=summary_json,
            ):
                n_rebuilt += 1


def process_checkpoint_step(
    model_type: str,
    step: int,
    args: argparse.Namespace,
    *,
    abstractive_tasks: list[str],
    lang1: tuple[str, ...],
    exclude_target_modes: tuple[bool, ...],
    state: dict[bool, dict[str, Any]],
) -> None:
    pending: list[bool] = []

    if args.recompute_translation:
        for exclude_target_label in exclude_target_modes:
            save_dir = abstractive_save_dir(
                PROJECT_ROOT, model_type, exclude_target_label=exclude_target_label
            )
            fv_path = fv_json_path(save_dir, model_type, step, args.n_shots)
            npz_path = summary_npz_path(save_dir, model_type)
            summary_path = summary_json_path(save_dir, model_type)

            if fv_path.is_file() and npz_path.is_file() and summary_path.is_file():
                pending.append(exclude_target_label)
    else:
        for exclude_target_label in exclude_target_modes:
            if step_is_complete(
                model_type,
                step,
                args.n_shots,
                exclude_target_label=exclude_target_label,
            ):
                continue
            summary_json = state[exclude_target_label]["summary_json"]
            if try_process_step_from_json(
                model_type,
                step,
                args.n_shots,
                exclude_target_label=exclude_target_label,
                confidence_level=args.confidence_level,
                summary_json=summary_json,
            ):
                continue
            pending.append(exclude_target_label)

    if not pending:
        return

    if args.recompute_translation:
        fv_by_mode: dict[bool, dict[str, Any]] = {}
        modes_need_fv: list[bool] = []

        for exclude_target_label in pending:
            save_dir = abstractive_save_dir(
                PROJECT_ROOT, model_type, exclude_target_label=exclude_target_label
            )
            fv_path = fv_json_path(save_dir, model_type, step, args.n_shots)
            if fv_path.is_file():
                fv_by_mode[exclude_target_label] = json.loads(fv_path.read_text(encoding="utf-8"))
            else:
                modes_need_fv.append(exclude_target_label)

        if modes_need_fv:
            modes_have_fv = [m for m in pending if m not in modes_need_fv]

            fv_computed, translation_aggregate = evaluate_step_gpu(
                model_type,
                step,
                n_shots=args.n_shots,
                max_sample_size=args.max_sample_size,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                translation_max_new_tokens=args.translation_max_new_tokens,
                exclude_target_modes=tuple(modes_need_fv),
                abstractive_tasks=abstractive_tasks,
                lang1=lang1,
                lang2=args.lang2,
                confidence_level=args.confidence_level,
            )
            fv_by_mode.update(fv_computed)

            modes_have_fv = [m for m in pending if m not in modes_need_fv]
            if modes_have_fv:
                translation_fixed = _evaluate_translation_only(
                    model_type,
                    step,
                    lang1=lang1,
                    lang2=args.lang2,
                    batch_size=args.batch_size,
                    max_new_tokens=args.translation_max_new_tokens,
                    confidence_level=args.confidence_level,
                )
                translation_aggregate = translation_fixed
        else:
            translation_aggregate = _evaluate_translation_only(
                model_type,
                step,
                lang1=lang1,
                lang2=args.lang2,
                batch_size=args.batch_size,
                max_new_tokens=args.translation_max_new_tokens,
                confidence_level=args.confidence_level,
            )
    else:
        fv_by_mode, translation_aggregate = evaluate_step_gpu(
            model_type,
            step,
            n_shots=args.n_shots,
            max_sample_size=args.max_sample_size,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            translation_max_new_tokens=args.translation_max_new_tokens,
            exclude_target_modes=tuple(pending),
            abstractive_tasks=abstractive_tasks,
            lang1=lang1,
            lang2=args.lang2,
            confidence_level=args.confidence_level,
        )

    for exclude_target_label in pending:
        fv_aggregate = fv_by_mode[exclude_target_label]
        process_step_artifacts(
            state[exclude_target_label]["save_dir"],
            model_type,
            step,
            args.n_shots,
            fv_aggregate,
            translation_aggregate,
            confidence_level=args.confidence_level,
            npz_path=state[exclude_target_label]["npz_path"],
            summary_json=state[exclude_target_label]["summary_json"],
            exclude_target_label=exclude_target_label,
        )


def run_checkpoint_sweep_for_model(
    model_type: str,
    args: argparse.Namespace,
    *,
    abstractive_tasks: list[str],
    lang1: tuple[str, ...],
    exclude_target_modes: tuple[bool, ...],
) -> None:
    state = init_run_state(
        model_type,
        args,
        abstractive_tasks=abstractive_tasks,
        lang1=lang1,
        exclude_target_modes=exclude_target_modes,
    )
    custom_steps = parse_custom_steps(args.steps)
    steps_to_eval = custom_steps if custom_steps is not None else checkpoint_steps(args.first_step, args.last_step, args.step_stride)

    for step in steps_to_eval:
        process_checkpoint_step(
            model_type,
            int(step),
            args,
            abstractive_tasks=abstractive_tasks,
            lang1=lang1,
            exclude_target_modes=exclude_target_modes,
            state=state,
        )


def run_checkpoint_sweep(args: argparse.Namespace, model_types: list[str]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run on a GPU node.")

    exclude_target_modes = resolve_exclude_target_modes(args)
    lang1 = tuple(args.lang1) if args.lang1 else DEFAULT_LANG1
    _, abstractive_tasks = list_icl_tasks(PROJECT_ROOT)

    for i, model_type in enumerate(model_types):
        run_checkpoint_sweep_for_model(
            model_type,
            args,
            abstractive_tasks=abstractive_tasks,
            lang1=lang1,
            exclude_target_modes=exclude_target_modes,
        )


def load_series(
    model_type: str,
    metric: str,
    *,
    exclude_target_label: bool,
    step_lo: int | None = None,
    step_hi: int | None = None,
    plot_steps: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    save_dir = abstractive_save_dir(
        PROJECT_ROOT, model_type, exclude_target_label=exclude_target_label
    )
    npz_path = summary_npz_path(save_dir, model_type)
    saved = load_summary_npz(npz_path)
    if not saved["steps"]:
        raise FileNotFoundError(
            f"Missing {npz_path}. Run evaluation first:\n"
            f"  python metrics/abstractive_accuracies_over_time.py --model-type {model_type}"
        )
    if metric not in saved:
        if metric == "abstractive_icl" and "macro_mean" in saved and "macro_mean_translation" not in saved:
            raise ValueError(
                f"{npz_path} has FV-only data from an older run. Re-evaluate checkpoints "
                "to populate abstractive_icl (FV + translation)."
            )
        raise ValueError(f"No {metric!r} in {npz_path}")

    steps_arr = np.array(saved["steps"], dtype=int)
    values = np.asarray(saved[metric], dtype=np.float64)
    lo_key, hi_key = f"{metric}_ci_lower", f"{metric}_ci_upper"
    ci_lo = np.asarray(saved[lo_key], dtype=np.float64) if lo_key in saved else None
    ci_hi = np.asarray(saved[hi_key], dtype=np.float64) if hi_key in saved else None

    if step_lo is not None:
        keep = steps_arr >= step_lo
        steps_arr, values = steps_arr[keep], values[keep]
        if ci_lo is not None and ci_hi is not None:
            ci_lo, ci_hi = ci_lo[keep], ci_hi[keep]
    if step_hi is not None:
        keep = steps_arr <= step_hi
        steps_arr, values = steps_arr[keep], values[keep]
        if ci_lo is not None and ci_hi is not None:
            ci_lo, ci_hi = ci_lo[keep], ci_hi[keep]
    if plot_steps is not None:
        keep = np.isin(steps_arr, np.asarray(plot_steps, dtype=int))
        steps_arr, values = steps_arr[keep], values[keep]
        if ci_lo is not None and ci_hi is not None:
            ci_lo, ci_hi = ci_lo[keep], ci_hi[keep]
    return steps_arr, values, ci_lo, ci_hi


def _hex_to_rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(hex_color)
    return (r, g, b, alpha)


def random_repetition_pickle_path(model_type: str) -> Path:
    sub = RANDOM_REPETITION_METRIC_SUBDIRS.get(model_type)
    if sub is None:
        raise ValueError(
            f"No random-repetition pickle mapping for {model_type!r}. "
            f"Known: {sorted(RANDOM_REPETITION_METRIC_SUBDIRS)}"
        )
    return (
        PROJECT_ROOT
        / "metrics"
        / "repetition_metrics"
        / sub
        / f"random_repetition_{model_type}.pkl"
    )


def load_random_repetition_at_steps(
    model_type: str,
    steps: np.ndarray,
    *,
    first_step: int = RANDOM_REPETITION_FIRST_STEP,
    step_stride: int = RANDOM_REPETITION_STEP_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = random_repetition_pickle_path(model_type)
    if not path.is_file():
        raise FileNotFoundError(f"Missing random repetition results: {path}")
    series: list[dict[str, Any]] = pkl.load(path.open("rb"))
    acc = np.full(len(steps), np.nan, dtype=np.float64)
    lo = np.full(len(steps), np.nan, dtype=np.float64)
    hi = np.full(len(steps), np.nan, dtype=np.float64)
    for i, step in enumerate(steps):
        idx = (int(step) - first_step) // step_stride
        if idx < 0 or idx >= len(series):
            continue
        row = series[idx]
        acc[i] = float(row["accuracy"])
        lo[i] = float(row["ci_lower"])
        hi[i] = float(row["ci_upper"])
    return acc, lo, hi


def _abstractive_metric_ylabel(plot_metric: str, *, repetition_style: bool) -> str:
    del repetition_style
    return {
        "abstractive_icl": "Abstractive ICL accuracy",
        "macro_mean_fv": "FV abstractive accuracy",
        "macro_mean_translation": "Translation accuracy",
    }.get(plot_metric, plot_metric)


def _abstractive_metric_title(plot_metric: str, model_types: list[str]) -> str | None:
    if plot_metric != "abstractive_icl":
        return None
    if model_types == ["vanilla_1b", "hapax_1b"]:
        return "Vanilla vs Hapax: Abstractive ICL Over Training"
    names = [MODEL_TYPE_DISPLAY_NAMES.get(m, m) for m in model_types]

    base = []
    for n in names:
        stripped = re.sub(r"\s*\(s\d+[^)]*\)$", "", n)
        if stripped not in base:
            base.append(stripped)
    if len(base) >= 2:
        return f"{' vs '.join(base)}: Abstractive ICL Over Training"
    if len(names) >= 2:
        return f"{' vs '.join(names)}: Abstractive ICL Over Training"
    if len(names) == 1:
        return f"{names[0]}: Abstractive ICL Over Training"
    return "Abstractive ICL Accuracy Over Training"


def _plot_abstractive_evolution_on_ax(
    ax: plt.Axes,
    model_types: list[str],
    *,
    exclude_target_modes: tuple[bool, ...],
    step_lo: int | None,
    step_hi: int | None,
    plot_steps: tuple[int, ...] | None,
    plot_metric: str,
    show_ci: bool,
    repetition_style: bool = False,
) -> None:
    ci_alpha = REPETITION_OVER_TIME_CI_ALPHA if repetition_style else 0.18
    for model_type in model_types:
        base_color = MODEL_COLORS.get(model_type, "tab:purple")
        display = MODEL_TYPE_DISPLAY_NAMES.get(model_type, model_type)
        for exclude_target_label in exclude_target_modes:
            steps, values, ci_lo, ci_hi = load_series(
                model_type,
                plot_metric,
                exclude_target_label=exclude_target_label,
                step_lo=step_lo,
                step_hi=step_hi,
                plot_steps=plot_steps,
            )
            if len(exclude_target_modes) > 1:
                label = series_legend_label(model_type, exclude_target_label)
            else:
                label = display
            if show_ci and ci_lo is not None and ci_hi is not None:
                ax.fill_between(
                    steps,
                    ci_lo,
                    ci_hi,
                    color=_hex_to_rgba(base_color, ci_alpha),
                    linewidth=0,
                    zorder=1,
                )
            plot_kw = target_mode_plot_kwargs(
                exclude_target_label,
                base_color,
                repetition_style=repetition_style,
                model_type=model_type,
            )
            plot_kw["label"] = label
            ax.plot(steps, values, **plot_kw)


def _plot_random_repetition_on_ax(
    ax: plt.Axes,
    model_types: list[str],
    *,
    step_lo: int | None,
    step_hi: int | None,
    plot_steps: tuple[int, ...] | None,
    show_ci: bool,
) -> None:
    ref_steps, _, _, _ = load_series(
        model_types[0],
        "abstractive_icl",
        exclude_target_label=False,
        step_lo=step_lo,
        step_hi=step_hi,
        plot_steps=plot_steps,
    )
    for model_type in model_types:
        color = MODEL_COLORS.get(model_type, "tab:purple")
        display = MODEL_TYPE_DISPLAY_NAMES.get(model_type, model_type)
        acc, ci_lo, ci_hi = load_random_repetition_at_steps(model_type, ref_steps)
        if show_ci:
            ax.fill_between(
                ref_steps,
                ci_lo,
                ci_hi,
                color=_hex_to_rgba(color, 0.18),
                linewidth=0,
                zorder=1,
            )
        ax.plot(
            ref_steps,
            acc,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label=display,
            zorder=2,
        )


def plot_evolution(
    model_types: list[str],
    *,
    exclude_target_modes: tuple[bool, ...],
    step_lo: int | None,
    step_hi: int | None,
    plot_steps: tuple[int, ...] | None,
    save_path: Path | None,
    plot_metric: str,
    show_ci: bool,
    ylim_lo: float = 0.0,
    ylim_hi: float = 1.0,
    with_random_repetition: bool = False,
    repetition_style: bool = True,
) -> None:
    apply_induction_scores_plot_style()
    n_curves = len(model_types) * len(exclude_target_modes)
    if repetition_style and not with_random_repetition:
        figsize = REPETITION_FIGSIZE
    elif with_random_repetition:
        figsize = (REPETITION_FIGSIZE[0] * 2.05, REPETITION_FIGSIZE[1])
    else:
        figsize = (REPETITION_FIGSIZE[0] * 1.35, REPETITION_FIGSIZE[1])

    if with_random_repetition:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax_icl, ax_rep = axes
    else:
        fig, ax_icl = plt.subplots(figsize=figsize)
        ax_rep = None

    _plot_abstractive_evolution_on_ax(
        ax_icl,
        model_types,
        exclude_target_modes=exclude_target_modes,
        step_lo=step_lo,
        step_hi=step_hi,
        plot_steps=plot_steps,
        plot_metric=plot_metric,
        show_ci=show_ci,
        repetition_style=repetition_style,
    )
    ax_icl.set_ylabel(
        _abstractive_metric_ylabel(plot_metric, repetition_style=repetition_style),
        labelpad=6,
    )
    plot_title = _abstractive_metric_title(plot_metric, model_types)
    title_fs = float(plt.rcParams.get("axes.titlesize", 9))
    if plot_title:
        fig.suptitle(plot_title, fontsize=title_fs, y=0.98)
    legend_fs = float(plt.rcParams.get("legend.fontsize", 8))
    if repetition_style and n_curves > 2 and ax_rep is None:
        _add_in_axes_model_legends(
            ax_icl,
            fontsize=legend_fs - 0.5,
            vanilla_loc="upper left",
            hapax_loc="lower right",
            handlelength=2.4,
        )
    elif repetition_style and n_curves > 2:
        ax_icl.legend(
            frameon=False,
            fontsize=legend_fs - 0.5,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            columnspacing=0.8,
            handlelength=1.25,
            labelspacing=0.4,
            borderaxespad=0.0,
        )
    else:
        ax_icl.legend(
            frameon=False,
            fontsize=legend_fs,
            loc="upper left",
            ncol=1,
            columnspacing=1.5,
            handlelength=2.2,
        )

    if ax_rep is not None:
        _plot_random_repetition_on_ax(
            ax_rep,
            model_types,
            step_lo=step_lo,
            step_hi=step_hi,
            plot_steps=plot_steps,
            show_ci=show_ci,
        )
        ax_rep.set_ylabel("Random repetition accuracy")
        ax_rep.legend(frameon=False, fontsize=legend_fs, loc="best")

    xlabel = REPETITION_OVER_TIME_XLABEL if repetition_style else "Training step"
    ax_icl.set_xlabel(xlabel)
    if ax_rep is not None:
        ax_rep.set_xlabel(xlabel)

    for ax in (ax_icl, ax_rep) if ax_rep is not None else (ax_icl,):
        ax.set_ylim(ylim_lo, ylim_hi)
        if repetition_style:
            style_repetition_over_time_xaxis(ax, step_lo=step_lo, step_hi=step_hi)
        else:
            set_training_step_ticks_induction(ax, major=5000, minor=1000)
        style_induction_box_axes(ax)

    if len(exclude_target_modes) == 2:
        target_tag = "incl_and_excl_target"
    elif len(exclude_target_modes) == 1:
        target_tag = target_mode_tag(exclude_target_modes[0])
    else:
        raise ValueError("No target-label modes selected for plotting.")
    ci_tag = "_ci" if show_ci else ""
    rep_tag = "_with_random_rep" if with_random_repetition else ""
    if model_types == ["vanilla_1b", "hapax_1b"]:
        models_tag = "vanilla_vs_hapax"
    else:
        models_tag = "_vs_".join(model_types)
    if save_path is None:
        fig_dir = PROJECT_ROOT / "metrics" / "abstractive_accuracies" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        hi = step_hi if step_hi is not None else "all"
        save_path = (
            fig_dir
            / f"{plot_metric}_{models_tag}_{target_tag}{ci_tag}{rep_tag}_step{hi}.pdf"
        )

    if repetition_style and n_curves > 2 and ax_rep is None:
        fig.subplots_adjust(left=0.15, bottom=0.14, top=0.86, right=0.98)
    elif repetition_style and n_curves > 2:
        fig.subplots_adjust(left=0.10, right=0.52, top=0.88, wspace=0.38)
    elif plot_title:
        fig.tight_layout()
        fig.subplots_adjust(top=0.90)
    else:
        fig.tight_layout()
    save_path = Path(save_path)
    pdf_path, png_path = save_pyplot_figure_pdf_png(save_path, png_dpi=300)

    plt.close(fig)


def _filter_steps(steps: np.ndarray, step_lo: int | None, step_hi: int | None) -> np.ndarray:
    out = steps
    if step_lo is not None:
        out = out[out >= step_lo]
    if step_hi is not None:
        out = out[out <= step_hi]
    return out


def _discover_checkpoint_steps(
    model_types: list[str],
    exclude_target_modes: tuple[bool, ...],
) -> np.ndarray:
    all_steps: set[int] = set()
    for model_type in model_types:
        for exclude_target_label in exclude_target_modes:
            save_dir = abstractive_save_dir(
                PROJECT_ROOT, model_type, exclude_target_label=exclude_target_label
            )
            saved = load_summary_npz(summary_npz_path(save_dir, model_type))
            all_steps.update(int(s) for s in saved.get("steps", []))
    if not all_steps:
        raise FileNotFoundError(
            "No checkpoint steps found in summary NPZ files. Run evaluation first."
        )
    return np.array(sorted(all_steps), dtype=np.int64)


def _ordered_task_names(
    model_types: list[str],
    steps: np.ndarray,
    n_shots: int,
    *,
    include_translation: bool,
) -> tuple[list[str], list[str]]:
    _, abstractive_tasks = list_icl_tasks(PROJECT_ROOT)
    fv_names = sorted(t.rsplit("/", 1)[-1] for t in abstractive_tasks)

    tr_names: list[str] = []
    if include_translation and len(steps) > 0:
        ref_step = int(steps[-1])
        for exclude_target_label in (False, True):
            tr = load_translation_json(
                model_types[0],
                ref_step,
                n_shots,
                exclude_target_label=exclude_target_label,
            )
            if tr:
                tr_names = sorted(tr.keys())
                break
    return fv_names, tr_names


def _translation_display_name(key: str) -> str:
    if "_" in key:
        src, tgt = key.split("_", 1)
        return f"{src.upper()}→{tgt.upper()}"
    return key.replace("_", " ").upper()


def _load_task_accuracy_series(
    model_type: str,
    task: str,
    steps: np.ndarray,
    n_shots: int,
    *,
    exclude_target_label: bool | None,
    is_translation: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step_list: list[int] = []
    acc: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for step in steps:
        step = int(step)
        if is_translation:
            aggregate = load_translation_json(
                model_type,
                step,
                n_shots,
                exclude_target_label=exclude_target_label or False,
            )
        else:
            save_dir = abstractive_save_dir(
                PROJECT_ROOT,
                model_type,
                exclude_target_label=bool(exclude_target_label),
            )
            path = fv_json_path(save_dir, model_type, step, n_shots)
            if not path.is_file():
                continue
            aggregate = json.loads(path.read_text(encoding="utf-8"))
        if aggregate is None or task not in aggregate:
            continue
        row = aggregate[task]
        step_list.append(step)
        acc.append(float(row["accuracy"]))
        lo.append(float(row.get("ci_lower", row["accuracy"])))
        hi.append(float(row.get("ci_upper", row["accuracy"])))
    return (
        np.array(step_list, dtype=np.int64),
        np.array(acc, dtype=np.float64),
        np.array(lo, dtype=np.float64),
        np.array(hi, dtype=np.float64),
    )


def _autoscale_panel_y(ax: plt.Axes, *, pad_frac: float = 0.08) -> None:
    ax.relim()
    ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        ax.set_ylim(0.0, 1.0)
        return
    if ymax <= ymin:
        ymax = ymin + 0.05
    pad = (ymax - ymin) * pad_frac
    lo = max(0.0, ymin - pad)
    hi = min(1.0, ymax + pad)
    if hi <= lo:
        hi = min(1.0, lo + 0.05)
    ax.set_ylim(lo, hi)


def _plot_one_task_on_ax(
    ax: plt.Axes,
    model_types: list[str],
    task: str,
    steps: np.ndarray,
    n_shots: int,
    *,
    exclude_target_modes: tuple[bool, ...],
    is_translation: bool,
    show_ci: bool,
) -> None:
    modes: tuple[bool | None, ...]
    if is_translation:
        modes = (None,)
    else:
        modes = exclude_target_modes

    for model_type in model_types:
        color = MODEL_COLORS.get(model_type, "tab:purple")
        display = MODEL_TYPE_DISPLAY_NAMES.get(model_type, model_type)
        for mode in modes:
            excl = False if mode is None else bool(mode)
            s, y, ci_lo, ci_hi = _load_task_accuracy_series(
                model_type,
                task,
                steps,
                n_shots,
                exclude_target_label=excl,
                is_translation=is_translation,
            )
            if len(s) == 0:
                continue
            if is_translation:
                label = display
            else:
                label = (
                    series_legend_label(model_type, excl)
                    if len(exclude_target_modes) > 1
                    else display
                )
            if show_ci:
                ax.fill_between(
                    s,
                    ci_lo,
                    ci_hi,
                    color=_hex_to_rgba(color, 0.15),
                    linewidth=0,
                    zorder=1,
                )
            if is_translation:
                plot_kw = {
                    "color": color,
                    "linestyle": "-",
                    "linewidth": 1.2,
                    "marker": "o",
                    "markersize": 2.5,
                    "markerfacecolor": color,
                    "markeredgecolor": color,
                    "zorder": 2,
                }
            else:
                plot_kw = target_mode_plot_kwargs(
                    excl,
                    color,
                    repetition_style=False,
                    linewidth=1.2,
                    model_type=model_type,
                )
                plot_kw["markersize"] = 2.5
            plot_kw["label"] = label
            ax.plot(s, y, **plot_kw)


def plot_all_tasks_over_time(
    model_types: list[str],
    *,
    exclude_target_modes: tuple[bool, ...],
    n_shots: int,
    step_lo: int | None,
    step_hi: int | None,
    save_path: Path | None,
    show_ci: bool,
    ylim_lo: float,
    ylim_hi: float,
    ncols: int = 6,
    include_translation: bool = True,
    per_panel_ylim: bool = True,
) -> None:
    apply_induction_scores_plot_style()
    steps = _filter_steps(
        _discover_checkpoint_steps(model_types, exclude_target_modes),
        step_lo,
        step_hi,
    )
    fv_names, tr_names = _ordered_task_names(
        model_types, steps, n_shots, include_translation=include_translation
    )
    panels: list[tuple[str, str, bool]] = [
        (name, format_task_name(name), False) for name in fv_names
    ]
    if include_translation:
        panels.extend(
            (key, _translation_display_name(key), True) for key in tr_names
        )

    n_tasks = len(panels)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_tasks / ncols))
    fig_w = ncols * 2.35
    fig_h = nrows * 1.85
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    legend_handles = None
    legend_labels = None
    for idx, (task_key, title, is_tr) in enumerate(panels):
        ax = axes.flat[idx]
        _plot_one_task_on_ax(
            ax,
            model_types,
            task_key,
            steps,
            n_shots,
            exclude_target_modes=exclude_target_modes,
            is_translation=is_tr,
            show_ci=show_ci,
        )
        ax.set_title(title, fontsize=7, pad=2)
        if per_panel_ylim:
            _autoscale_panel_y(ax)
        else:
            ax.set_ylim(ylim_lo, ylim_hi)
        set_training_step_ticks_induction(ax, major=5000, minor=1000)
        style_induction_box_axes(ax)
        ax.tick_params(labelsize=6)
        if legend_handles is None and ax.get_legend_handles_labels()[0]:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for j in range(n_tasks, nrows * ncols):
        axes.flat[j].axis("off")

    for idx in range(n_tasks):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        if row == nrows - 1:
            ax.set_xlabel("Training step", fontsize=7)
        if col == 0:
            ax.set_ylabel("Accuracy", fontsize=7)

    n_legend = len(legend_labels or [])
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(4, n_legend),
            frameon=False,
            fontsize=6,
            handlelength=1.8,
            columnspacing=1.0,
        )

    if len(exclude_target_modes) == 2:
        target_tag = "incl_and_excl_target"
    elif len(exclude_target_modes) == 1:
        target_tag = target_mode_tag(exclude_target_modes[0])
    else:
        target_tag = "no_target_mode"
    ci_tag = "_ci" if show_ci else ""
    tr_tag = "_with_translation" if include_translation else "_fv_only"
    y_tag = "_autoscale_y" if per_panel_ylim else ""
    if model_types == ["vanilla_1b", "hapax_1b"]:
        models_tag = "vanilla_vs_hapax"
    else:
        models_tag = "_vs_".join(model_types)
    hi = step_hi if step_hi is not None else "all"
    if save_path is None:
        fig_dir = PROJECT_ROOT / "metrics" / "abstractive_accuracies" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        save_path = (
            fig_dir
            / f"per_task_{models_tag}_{target_tag}{ci_tag}{tr_tag}{y_tag}_step{hi}.pdf"
        )

    fig.tight_layout(rect=(0, 0.06, 1, 0.98), h_pad=1.4, w_pad=0.9)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-size",
        type=str,
        default=DEFAULT_MODEL_SIZE,
        choices=[
            "1b",
            "1b-triple",
            "1b-pair",
            "1b-threshold-0.3",
            "2.8b",
        ],
        help=(
            "Models to evaluate when --model-type is not set: 1b and 1b-triple use vanilla, hapax and thresholded hapax; 1b-pair uses vanilla and hapax."       ),
    )
    p.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Evaluate a single model only (overrides --model-size default list).",
    )
    p.add_argument(
        "--model-types",
        nargs="+",
        default=None,
        help="Explicit list of models (overrides --model-size).",
    )
    p.add_argument(
        "--compare-models",
        nargs="+",
        default=None,
        help="Models to plot. Defaults to the models for --model-size.",
    )
    p.add_argument(
        "--first-step",
        type=int,
        default=100,
        help="First checkpoint to evaluate."    )
    p.add_argument(
        "--last-step",
        type=int,
        default=19900,
    )
    p.add_argument(
        "--step-stride",
        type=int,
        default=500,
        help="Checkpoint spacing after --first-step."    )
    p.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated checkpoint steps to evaluate. Overrides --first-step, --last-step and --step-stride.",
    )
    p.add_argument("--n-shots", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--max-sample-size", type=int, default=5000)
    p.add_argument("--max-new-tokens", type=int, default=5)
    p.add_argument(
        "--translation-max-new-tokens",
        type=int,
        default=8,
    )
    p.add_argument("--lang2", type=str, default="eng")
    p.add_argument(
        "--lang1",
        nargs="+",
        default=None,
    )
    p.add_argument(
        "--only-exclude-target-label",
        action="store_true",
        help="Only evaluate with the target label excluded from the demonstrations.",
    )
    p.add_argument(
        "--only-include-target-label",
        action="store_true",
        help="Only evaluate with the target label allowed in the demonstrations.",
    )
    p.add_argument("--confidence-level", type=float, default=0.95)
    p.add_argument("--cuda-visible-devices", type=str, default="0")
    p.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip checkpoint sweep; only plot from saved NPZ.",
    )
    p.add_argument(
        "--rebuild-summary",
        action="store_true",
        help="Recompute the summary from saved JSON results without loading models.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
    )
    p.add_argument(
        "--no-ci",
        action="store_true",
    )
    p.add_argument(
        "--recompute-translation",
        action="store_true",
    )
    p.add_argument(
        "--plot-metric",
        choices=METRIC_NAMES,
        default="abstractive_icl",
    )
    p.add_argument("--step-lo", type=int, default=None)
    p.add_argument("--step-hi", type=int, default=None)
    p.add_argument(
        "--plot-steps",
        type=str,
        default=None,
        help=(
            "Comma-separated steps to plot, so all models are drawn on the same steps."
        ),
    )
    p.add_argument(
        "--ylim-lo",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--ylim-hi",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--with-random-repetition",
        action="store_true",
        help="Add a panel with random repetition accuracy.",
    )
    p.add_argument(
        "--plot-all-tasks",
        action="store_true",
        help="Plot accuracy over training for every task in one PDF.",
    )
    p.add_argument(
        "--per-task-cols",
        type=int,
        default=6,
    )
    p.add_argument(
        "--per-task-fv-only",
        action="store_true",
        help="With --plot-all-tasks, omit translation language-pair subplots.",
    )
    p.add_argument(
        "--per-task-shared-y",
        action="store_true",
        help="With --plot-all-tasks, use --ylim-lo and --ylim-hi for every panel.",
    )
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _maybe_plot_all_tasks(
    args: argparse.Namespace,
    model_types: list[str],
    exclude_target_modes: tuple[bool, ...],
) -> None:
    if not args.plot_all_tasks:
        return
    plot_steps = parse_plot_steps(args.plot_steps)
    out = args.out
    if out is not None and args.plot_metric != "abstractive_icl":
        out = None
    plot_all_tasks_over_time(
        model_types,
        exclude_target_modes=exclude_target_modes,
        n_shots=args.n_shots,
        step_lo=args.step_lo,
        step_hi=args.step_hi,
        plot_steps=plot_steps,
        save_path=out,
        show_ci=not args.no_ci,
        ylim_lo=args.ylim_lo,
        ylim_hi=args.ylim_hi,
        ncols=args.per_task_cols,
        include_translation=not args.per_task_fv_only,
        per_panel_ylim=not args.per_task_shared_y,
    )


def main() -> None:
    args = parse_args()
    plot_steps = parse_plot_steps(args.plot_steps)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    exclude_target_modes = resolve_exclude_target_modes(args)
    model_types = model_types_for_eval(args)

    if args.rebuild_summary:
        for model_type in model_types:
            rebuild_summary_from_json(
                model_type,
                args,
                exclude_target_modes=exclude_target_modes,
            )
        if not args.no_plots:
            _maybe_plot_all_tasks(args, model_types, exclude_target_modes)
            if not args.plot_all_tasks:
                plot_evolution(
                    model_types,
                    exclude_target_modes=exclude_target_modes,
                    step_lo=args.step_lo,
                    step_hi=args.step_hi,
                    plot_steps=plot_steps,
                    save_path=args.out,
                    plot_metric=args.plot_metric,
                    show_ci=not args.no_ci,
                    ylim_lo=args.ylim_lo,
                    ylim_hi=args.ylim_hi,
                    with_random_repetition=args.with_random_repetition,
                )
        return

    if args.plots_only or args.compare_models or args.plot_all_tasks:
        plot_models = args.compare_models or model_types
        _maybe_plot_all_tasks(args, plot_models, exclude_target_modes)
        if args.plots_only or args.compare_models:
            plot_evolution(
                plot_models,
                exclude_target_modes=exclude_target_modes,
                step_lo=args.step_lo,
                step_hi=args.step_hi,
                plot_steps=plot_steps,
                save_path=args.out,
                plot_metric=args.plot_metric,
                show_ci=not args.no_ci,
                ylim_lo=args.ylim_lo,
                ylim_hi=args.ylim_hi,
                with_random_repetition=args.with_random_repetition,
            )
        return

    run_checkpoint_sweep(args, model_types)

    if not args.no_plots:
        _maybe_plot_all_tasks(args, model_types, exclude_target_modes)
        plot_evolution(
            model_types,
            exclude_target_modes=exclude_target_modes,
            step_lo=args.step_lo,
            step_hi=args.step_hi,
            plot_steps=plot_steps,
            save_path=args.out,
            plot_metric=args.plot_metric,
            show_ci=not args.no_ci,
            ylim_lo=args.ylim_lo,
            ylim_hi=args.ylim_hi,
            with_random_repetition=args.with_random_repetition,
        )

if __name__ == "__main__":
    main()
