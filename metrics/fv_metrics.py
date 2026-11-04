from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.fv import fv_icl_tasks_benchmark_with_ci
from lib.model_setup import load_model_and_tokenizer


def list_icl_tasks(project_root: Path) -> tuple[list[str], list[str]]:
    icl_root = project_root / "dataset" / "icl_tasks"
    extractive_dir = icl_root / "extractive"
    abstractive_dir = icl_root / "abstractive"
    extractive_tasks = [
        f"extractive/{f[:-5]}"
        for f in os.listdir(extractive_dir)
        if f.endswith(".json")
    ]
    abstractive_tasks = [
        f"abstractive/{f[:-5]}"
        for f in os.listdir(abstractive_dir)
        if f.endswith(".json")
    ]
    return extractive_tasks, abstractive_tasks


def fv_tasks_benchmark(
    model,
    tokenizer,
    task_names: list[str],
    *,
    project_root: Path,
    num_of_shots: int = 5,
    max_sample_size: int = 5000,
    batch_size: int = 64,
    return_per_sample: bool = True,
    exclude_target_label: bool | None = None,
    max_new_tokens: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_dir = str(project_root / "dataset" / "icl_tasks")
    icl_task_results: dict[str, Any] = {}
    icl_task_results_per_sample: dict[str, Any] = {}
    for task_name in task_names:
        icl_results, per_sample_correct = fv_icl_tasks_benchmark_with_ci(
            model,
            tokenizer,
            task_name=task_name,
            task_dir=task_dir,
            num_of_shots=num_of_shots,
            max_samples=max_sample_size,
            batch_size=batch_size,
            use_wandb=False,
            return_per_sample=return_per_sample,
            exclude_target_label=exclude_target_label,
            max_new_tokens=max_new_tokens,
        )
        short_name = task_name.split("/")[-1]
        icl_task_results[short_name] = icl_results
        icl_task_results_per_sample[short_name] = per_sample_correct
    return icl_task_results, icl_task_results_per_sample


def resolve_model_types(model_size: str) -> tuple[list[str], str, str]:
    if model_size == "1b":
        model_types = ["vanilla_1b", "hapax_1b"]
    elif model_size == "1b-threshold-0.3":
        model_types = ["vanilla_1b", "hapax_1b_thresh_0.3"]
    elif model_size == "2.8b":
        model_types = ["vanilla_2.8b", "hapax_2.8b"]
    else:
        raise ValueError(f"Unknown model_size: {model_size!r}")

    clean_key, masked_key = model_types[0], model_types[1]
    return model_types, clean_key, masked_key


def load_clean_model(
    project_root: Path,
    clean_key: str,
    step: int,
):
    clean_model_path = str(project_root / "models" / clean_key / f"step={step}")
    return load_model_and_tokenizer(clean_model_path)


def load_masked_model(project_root: Path, masked_key: str, step: int):
    masked_model_path = str(project_root / "models" / masked_key / f"step={step}")
    return load_model_and_tokenizer(masked_model_path)


def release_model(model, tokenizer) -> None:
    import torch

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_task_name(task: str) -> str:
    task_names = {
        "capitalize_first_letter": "Capitalize First Letter",
        "capitalize_second_letter": "Capitalize Second Letter",
        "capitalize_last_letter": "Capitalize Last Letter",
        "capitalize": "Capitalize (Full Word)",
        "lowercase_first_letter": "Lowercase First Letter",
        "lowercase_last_letter": "Lowercase Last Letter",
        "singular-plural": "Singular-Plural",
        "plural-singular": "Plural-Singular",
        "present-past": "Present-Past",
        "past-present": "Past-Present",
        "english-spanish": "English-Spanish",
        "english-german": "English-German",
        "english-french": "English-French",
        "synonym": "Synonym",
        "antonym": "Antonym",
        "country-capital": "Country-Capital",
        "capital-country": "Capital-Country",
        "country-currency": "Country-Currency",
        "currency-country": "Currency-Country",
        "landmark-country": "Landmark-Country",
        "country-landmark": "Country-Landmark",
        "product-company": "Product-Company",
        "company-product": "Company-Product",
        "park-country": "Park-Country",
        "country-park": "Country-Park",
        "national_parks": "National Parks",
        "person-occupation": "Person-Occupation",
        "person-instrument": "Person-Instrument",
        "person-sport": "Person-Sport",
        "next_item": "Next Item",
        "prev_item": "Previous Item",
        "next_capital_letter": "Next Capital Letter",
        "word_length": "Word Length",
        "ag_news": "AG News",
        "commonsense_qa": "CommonsenseQA",
        "sentiment": "Sentiment",
        "choose_first_of_3": "Choose First of 3",
        "choose_first_of_5": "Choose First of 5",
        "choose_middle_of_3": "Choose Middle of 3",
        "choose_middle_of_5": "Choose Middle of 5",
        "choose_last_of_3": "Choose Last of 3",
        "choose_last_of_5": "Choose Last of 5",
        "alphabetically_first_3": "Alphabetically First (3)",
        "alphabetically_first_5": "Alphabetically First (5)",
        "alphabetically_last_3": "Alphabetically Last (3)",
        "alphabetically_last_5": "Alphabetically Last (5)",
        "conll2003_person": "CoNLL: Person",
        "conll2003_location": "CoNLL: Location",
        "conll2003_organization": "CoNLL: Organization",
        "animal_v_object_3": "Animal vs Object (3)",
        "animal_v_object_5": "Animal vs Object (5)",
        "fruit_v_animal_3": "Fruit vs Animal (3)",
        "fruit_v_animal_5": "Fruit vs Animal (5)",
        "color_v_animal_3": "Color vs Animal (3)",
        "color_v_animal_5": "Color vs Animal (5)",
        "adjective_v_verb_3": "Adjective vs Verb (3)",
        "adjective_v_verb_5": "Adjective vs Verb (5)",
        "verb_v_adjective_3": "Verb vs Adjective (3)",
        "verb_v_adjective_5": "Verb vs Adjective (5)",
        "concept_v_object_3": "Concept vs Object (3)",
        "concept_v_object_5": "Concept vs Object (5)",
        "object_v_concept_3": "Object vs Concept (3)",
        "object_v_concept_5": "Object vs Concept (5)",
        "squad_val": "SQuAD Validation",
    }
    return task_names.get(task, task.replace("_", " ").title())


def mcnemar_pvalue(predictions1, predictions2) -> float:
    from statsmodels.stats.contingency_tables import mcnemar

    if len(predictions1) != len(predictions2):
        raise ValueError(
            f"McNemar length mismatch: {len(predictions1)} vs {len(predictions2)} "
            "(paired predictions must be aligned)."
        )

    both_correct = sum(p1 and p2 for p1, p2 in zip(predictions1, predictions2))
    model1_only = sum(p1 and not p2 for p1, p2 in zip(predictions1, predictions2))
    model2_only = sum(not p1 and p2 for p1, p2 in zip(predictions1, predictions2))
    both_wrong = sum(not p1 and not p2 for p1, p2 in zip(predictions1, predictions2))

    table = [[both_correct, model1_only], [model2_only, both_wrong]]
    result = mcnemar(table, exact=False, correction=True)
    return float(result.pvalue)


def mcnemar_test(predictions1, predictions2, alpha: float = 0.05) -> bool:
    return bool(mcnemar_pvalue(predictions1, predictions2) < alpha)


def mcnemar_significance_list_bh(
    task_names: list[str],
    per_sample_clean: dict[str, Any],
    per_sample_masked: dict[str, Any],
    *,
    alpha: float = 0.05,
) -> list[bool]:
    from statsmodels.stats.multitest import multipletests

    pvals = [
        mcnemar_pvalue(per_sample_clean[k], per_sample_masked[k]) for k in task_names
    ]
    reject, _, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    return [bool(r) for r in reject]


def generate_latex_table(
    task_names,
    model1_values,
    model1_cis,
    model2_values,
    model2_cis,
    significant=None,
    model1_name: str = "Vanilla",
    model2_name: str = "\\textsc{Hapax Thresh}",
    caption: str = "Performance comparison",
    label: str = "tab:results",
    significance_note: str | None = None,
) -> str:
    if significant is None:
        significant = [False] * len(task_names)

    latex = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        "\\begin{tabular}{lcc@{\\hspace{1.8em}}lcc}",
        "\\toprule",
        f"\\textbf{{Task}} & \\textbf{{{model1_name} (\\%)}} & \\textbf{{{model2_name} (\\%)}} &",
        f"\\textbf{{Task}} & \\textbf{{{model1_name} (\\%)}} & \\textbf{{{model2_name} (\\%)}} \\\\",
        "\\midrule",
    ]

    for i in range(0, len(task_names), 2):
        task1 = task_names[i]
        m1_val1 = model1_values[i] * 100
        m1_ci1 = model1_cis[i] * 100
        m2_val1 = model2_values[i] * 100
        m2_ci1 = model2_cis[i] * 100
        sig1 = significant[i]

        m1_str1 = f"{m1_val1:.1f} $\\pm$ {m1_ci1:.1f}"
        m2_str1 = f"{m2_val1:.1f} $\\pm$ {m2_ci1:.1f}"

        if m2_val1 > m1_val1:
            m2_str1 = f"\\textbf{{{m2_str1}}}"
            if sig1:
                m2_str1 += "$^\\dagger$"
        elif m1_val1 > m2_val1:
            m1_str1 = f"\\textbf{{{m1_str1}}}"
            if sig1:
                m1_str1 += "$^\\dagger$"

        row = f"{task1} & {m1_str1} & {m2_str1}"

        if i + 1 < len(task_names):
            task2 = task_names[i + 1]
            m1_val2 = model1_values[i + 1] * 100
            m1_ci2 = model1_cis[i + 1] * 100
            m2_val2 = model2_values[i + 1] * 100
            m2_ci2 = model2_cis[i + 1] * 100
            sig2 = significant[i + 1]

            m1_str2 = f"{m1_val2:.1f} $\\pm$ {m1_ci2:.1f}"
            m2_str2 = f"{m2_val2:.1f} $\\pm$ {m2_ci2:.1f}"

            if m2_val2 > m1_val2:
                m2_str2 = f"\\textbf{{{m2_str2}}}"
                if sig2:
                    m2_str2 += "$^\\dagger$"
            elif m1_val2 > m2_val2:
                m1_str2 = f"\\textbf{{{m1_str2}}}"
                if sig2:
                    m1_str2 += "$^\\dagger$"

            row += f" & {task2} & {m1_str2} & {m2_str2}"
        else:
            row += " & & &"

        latex.append(row + " \\\\")

    tail = ["\\bottomrule", "\\end{tabular}"]
    if significance_note:
        tail.append(f"\\noindent\\footnotesize {significance_note}\\\\")
    tail.extend(["\\vspace{-0.5em}", "\\end{table}"])
    latex.extend(tail)
    return "\n".join(latex)


def build_latex_for_split(
    desired_task: str,
    save_dir: Path,
    model_types: list[str],
    model_size: str,
    step: int,
    n_shots: int,
    *,
    aggregate_clean: dict[str, Any] | None = None,
    aggregate_masked: dict[str, Any] | None = None,
    per_sample_clean: dict[str, Any] | None = None,
    per_sample_masked: dict[str, Any] | None = None,
    fdr_bh: bool = False,
    fdr_alpha: float = 0.05,
) -> str:
    if aggregate_clean is None:
        with open(
            save_dir / f"{desired_task}_results_{model_types[0]}_{step}_{n_shots}.json"
        ) as f:
            aggregate_clean = json.load(f)
    if aggregate_masked is None:
        with open(
            save_dir / f"{desired_task}_results_{model_types[1]}_{step}_{n_shots}.json"
        ) as f:
            aggregate_masked = json.load(f)
    if per_sample_clean is None:
        with open(
            save_dir
            / f"{desired_task}_results_{model_types[0]}_{model_size}_{step}_{n_shots}_per_sample.json"
        ) as f:
            per_sample_clean = json.load(f)
    if per_sample_masked is None:
        with open(
            save_dir
            / f"{desired_task}_results_{model_types[1]}_{model_size}_{step}_{n_shots}_per_sample.json"
        ) as f:
            per_sample_masked = json.load(f)

    assert aggregate_clean is not None and aggregate_masked is not None
    assert per_sample_clean is not None and per_sample_masked is not None

    task_names = list(aggregate_clean.keys())
    task_names_label = [format_task_name(t) for t in task_names]

    model1_values = [aggregate_clean[t]["accuracy"] for t in task_names]
    model1_cis = [
        (aggregate_clean[t]["ci_upper"] - aggregate_clean[t]["ci_lower"]) / 2
        for t in task_names
    ]
    model2_values = [aggregate_masked[t]["accuracy"] for t in task_names]
    model2_cis = [
        (aggregate_masked[t]["ci_upper"] - aggregate_masked[t]["ci_lower"]) / 2
        for t in task_names
    ]

    if fdr_bh:
        significant_list = mcnemar_significance_list_bh(
            task_names,
            per_sample_clean,
            per_sample_masked,
            alpha=fdr_alpha,
        )
        sig_note = (
            f"\\textit{{Notes:}} $^\\dagger$ McNemar paired tests; "
            f"Benjamini--Hochberg FDR control at $\\alpha = {fdr_alpha:g}$."
        )
    else:
        significant_list = [
            mcnemar_test(per_sample_clean[k], per_sample_masked[k], alpha=fdr_alpha)
            for k in task_names
        ]
        sig_note = (
            f"\\textit{{Notes:}} $^\\dagger$ McNemar paired tests, "
            f"$p < {fdr_alpha:g}$ (uncorrected for multiple comparisons)."
        )

    return generate_latex_table(
        task_names_label,
        model1_values,
        model1_cis,
        model2_values,
        model2_cis,
        significant=significant_list,
        model2_name="\\textsc{Hapax}",
        caption=(
            f"Performance comparison on {desired_task} in-context learning tasks "
            f"({n_shots}-Shot)."
        ),
        label=f"tab:{desired_task}_hapax_results",
        significance_note=sig_note if any(significant_list) else None,
    )


def write_latex_tables(
    save_dir: Path,
    model_types: list[str],
    model_size: str,
    step: int,
    n_shots: int,
    *,
    in_memory: dict[str, dict[str, Any]] | None = None,
    fdr_bh: bool = False,
    fdr_alpha: float = 0.05,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for desired_task in ("abstractive", "extractive"):
        mem = (in_memory or {}).get(desired_task)
        kwargs = {}
        if mem:
            kwargs = {
                "aggregate_clean": mem.get("aggregate_clean"),
                "aggregate_masked": mem.get("aggregate_masked"),
                "per_sample_clean": mem.get("per_sample_clean"),
                "per_sample_masked": mem.get("per_sample_masked"),
            }

        if desired_task == "extractive" and not kwargs:
            json_path = save_dir / f"extractive_results_{model_types[1]}_{step}_{n_shots}.json"
            if not json_path.exists():
                continue
        latex_code = build_latex_for_split(
            desired_task,
            save_dir,
            model_types,
            model_size,
            step,
            n_shots,
            fdr_bh=fdr_bh,
            fdr_alpha=fdr_alpha,
            **kwargs,
        )

        out_path = save_dir / f"{desired_task}_hapax_table.tex"
        out_path.write_text(latex_code, encoding="utf-8")


def save_benchmark_json(
    save_dir: Path,
    model_types: list[str],
    model_size: str,
    step: int,
    n_shots: int,
    *,
    extractive_masked,
    abstractive_masked,
    extractive_masked_per_sample,
    abstractive_masked_per_sample,
    skip_clean: bool,
    extractive_clean=None,
    abstractive_clean=None,
    extractive_clean_per_sample=None,
    abstractive_clean_per_sample=None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    def dump(name: str, obj) -> None:
        path = save_dir / name
        with open(path, "w") as f:
            json.dump(obj, f)

    if not skip_clean:
        if extractive_clean is not None:
            dump(
                f"extractive_results_{model_types[0]}_{step}_{n_shots}.json",
                extractive_clean,
            )
            dump(
                f"extractive_results_{model_types[0]}_{model_size}_{step}_{n_shots}_per_sample.json",
                extractive_clean_per_sample,
            )
        dump(
            f"abstractive_results_{model_types[0]}_{step}_{n_shots}.json",
            abstractive_clean,
        )
        dump(
            f"abstractive_results_{model_types[0]}_{model_size}_{step}_{n_shots}_per_sample.json",
            abstractive_clean_per_sample,
        )

    if extractive_masked is not None:
        dump(
            f"extractive_results_{model_types[1]}_{step}_{n_shots}.json",
            extractive_masked,
        )
        dump(
            f"extractive_results_{model_types[1]}_{model_size}_{step}_{n_shots}_per_sample.json",
            extractive_masked_per_sample,
        )
    dump(
        f"abstractive_results_{model_types[1]}_{step}_{n_shots}.json",
        abstractive_masked,
    )
    dump(
        f"abstractive_results_{model_types[1]}_{model_size}_{step}_{n_shots}_per_sample.json",
        abstractive_masked_per_sample,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_size",
        type=str,
        default="1b",
        choices=[
            "1b",
            "1b-threshold-0.3",
            "2.8b",
        ],
    )
    p.add_argument("--step", type=int, default=24900)
    p.add_argument("--n_shots", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=40)
    p.add_argument("--max_sample_size", type=int, default=5000)
    p.add_argument("--max_new_tokens", type=int, default=5)
    p.add_argument("--skip_clean", action="store_true")
    p.add_argument(
        "--skip_extractive",
        action="store_true",
    )
    p.add_argument("--exclude_target_label", action="store_true")
    p.add_argument("--cuda_visible_devices", type=str, default="0")
    p.add_argument(
        "--tables-only",
        action="store_true",
        help="Write the LaTeX tables from saved JSON results without loading models.",
    )
    p.add_argument(
        "--fdr-bh",
        action="store_true",
        help=(
            "Apply Benjamini-Hochberg FDR correction across tasks to the McNemar tests."
        ),
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    project_root = PROJECT_ROOT
    model_types, clean_key, masked_key = resolve_model_types(args.model_size)

    fv_dir_name = (
        f"{args.model_size}_excl_target" if args.exclude_target_label else args.model_size
    )
    save_dir = (
        project_root
        / "metrics"
        / "fv_accuracies"
        / fv_dir_name
    )

    if args.tables_only:
        write_latex_tables(
            save_dir,
            model_types,
            args.model_size,
            args.step,
            args.n_shots,
            in_memory=None,
            fdr_bh=args.fdr_bh,
            fdr_alpha=args.fdr_alpha,
        )
        return

    extractive_tasks, abstractive_tasks = list_icl_tasks(project_root)
    bench_kw = dict(
        project_root=project_root,
        num_of_shots=args.n_shots,
        max_sample_size=args.max_sample_size,
        batch_size=args.batch_size,
        exclude_target_label=args.exclude_target_label,
        max_new_tokens=args.max_new_tokens,
    )

    in_memory: dict[str, dict[str, Any]] | None = {} if not args.skip_clean else None

    extractive_results_clean = None
    abstractive_results_clean = None
    extractive_results_clean_per_sample = None
    abstractive_results_clean_per_sample = None
    extractive_results_masked = None
    extractive_results_masked_per_sample = None

    if not args.skip_clean:
        clean_model, tokenizer_clean = load_clean_model(
            project_root, clean_key, args.step
        )
        if not args.skip_extractive:
            extractive_results_clean, extractive_results_clean_per_sample = (
                fv_tasks_benchmark(
                    clean_model,
                    tokenizer_clean,
                    extractive_tasks,
                    **bench_kw,
                )
            )
        abstractive_results_clean, abstractive_results_clean_per_sample = (
            fv_tasks_benchmark(
                clean_model,
                tokenizer_clean,
                abstractive_tasks,
                **bench_kw,
            )
        )
        release_model(clean_model, tokenizer_clean)

        if in_memory is not None:
            if not args.skip_extractive:
                in_memory["extractive"] = {
                    "aggregate_clean": extractive_results_clean,
                    "aggregate_masked": None,
                    "per_sample_clean": extractive_results_clean_per_sample,
                    "per_sample_masked": None,
                }
            in_memory["abstractive"] = {
                "aggregate_clean": abstractive_results_clean,
                "aggregate_masked": None,
                "per_sample_clean": abstractive_results_clean_per_sample,
                "per_sample_masked": None,
            }

    masked_model, tokenizer_masked = load_masked_model(
        project_root, masked_key, args.step
    )
    if not args.skip_extractive:
        extractive_results_masked, extractive_results_masked_per_sample = (
            fv_tasks_benchmark(
                masked_model,
                tokenizer_masked,
                extractive_tasks,
                **bench_kw,
            )
        )
    abstractive_results_masked, abstractive_masked_per_sample = fv_tasks_benchmark(
        masked_model,
        tokenizer_masked,
        abstractive_tasks,
        **bench_kw,
    )
    release_model(masked_model, tokenizer_masked)

    if in_memory is not None:
        if not args.skip_extractive:
            in_memory["extractive"]["aggregate_masked"] = extractive_results_masked
            in_memory["extractive"][
                "per_sample_masked"
            ] = extractive_results_masked_per_sample
        in_memory["abstractive"]["aggregate_masked"] = abstractive_results_masked
        in_memory["abstractive"]["per_sample_masked"] = abstractive_masked_per_sample

    save_benchmark_json(
        save_dir,
        model_types,
        args.model_size,
        args.step,
        args.n_shots,
        extractive_masked=extractive_results_masked if not args.skip_extractive else None,
        abstractive_masked=abstractive_results_masked,
        extractive_masked_per_sample=extractive_results_masked_per_sample if not args.skip_extractive else None,
        abstractive_masked_per_sample=abstractive_masked_per_sample,
        skip_clean=args.skip_clean,
        extractive_clean=extractive_results_clean if not args.skip_extractive else None,
        abstractive_clean=abstractive_results_clean,
        extractive_clean_per_sample=extractive_results_clean_per_sample if not args.skip_extractive else None,
        abstractive_clean_per_sample=abstractive_results_clean_per_sample,
    )


    if args.skip_clean:
        return

    write_latex_tables(
        save_dir,
        model_types,
        args.model_size,
        args.step,
        args.n_shots,
        in_memory=in_memory,
        fdr_bh=args.fdr_bh,
        fdr_alpha=args.fdr_alpha,
    )

if __name__ == "__main__":
    main()
