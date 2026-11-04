import os
import json
import random
import torch
import pandas as pd
from tqdm import tqdm
import wandb
import numpy as np
from scipy import stats
from lib.evaluation import matches_target


def fv_icl_tasks_benchmark(
    model,
    tokenizer,
    task_name,
    task_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "icl_tasks"),
    max_samples=5000,
    num_of_shots=5,
    batch_size=64,
    use_wandb=True,
    return_samples_used=False,
    seed=1234,
    return_per_sample=False,
    exclude_target_label=True,
    max_new_tokens=3,
):
    random.seed(seed)

    def generate_few_shot_prompts(
        data,
        num_of_shots=5,
        max_samples=None,
        exclude_target_label=True,
        max_new_tokens=3,
        tokenizer=None,
    ):
        few_shot_prompts = []
        few_shot_targets = []
        skipped_samples = 0

        if max_samples and len(data) > max_samples:
            data = random.sample(data, max_samples)
            samples_used = max_samples
        else:
            samples_used = len(data)

        for point_idx, point in enumerate(data):
            if len(tokenizer.encode(point["output"])) > max_new_tokens:
                samples_used -= 1
                skipped_samples += 1
                continue
            prompt = "Q: " + point["input"] + "\nA:"
            target = point["output"]

            random.seed(seed + point_idx)

            if exclude_target_label:
                available_indices = [
                    i
                    for i in range(len(data))
                    if i != point_idx and data[i]["output"] != target
                ]
            else:
                available_indices = [i for i in range(len(data)) if i != point_idx]

            if len(available_indices) < num_of_shots:
                continue

            sampled_prompts = [
                data[idx] for idx in random.sample(available_indices, num_of_shots)
            ]

            few_shot_prompt = ""
            for sample in sampled_prompts:
                few_shot_prompt += f"Q: {sample['input']}\nA: {sample['output']}\n\n"

            few_shot_prompt += prompt

            few_shot_prompts.append(few_shot_prompt)
            few_shot_targets.append(target)

        if use_wandb:
            wandb.summary[f"samples_for_{task_name}"] = samples_used

        return few_shot_prompts, few_shot_targets, samples_used

    def process_batch(prompts_batch, targets_batch, max_new_tokens=3):
        tokenized_batch = tokenizer(
            prompts_batch, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        batch_predictions = []

        with torch.no_grad():
            outputs = model.generate(
                **tokenized_batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            for i, seq in enumerate(outputs):
                prompt_len = (tokenized_batch.attention_mask[i] == 1).sum().item()
                pad_beginning = (tokenized_batch.attention_mask[i] == 0).sum().item()

                gen_text = tokenizer.decode(
                    seq[prompt_len + pad_beginning :], skip_special_tokens=True
                )

                tgt_text = targets_batch[i].strip()

                if matches_target(gen_text, tgt_text):
                    batch_predictions.append(True)
                else:
                    batch_predictions.append(False)

        return batch_predictions

    if not os.path.exists(task_dir):
        return {}

    json_file = os.path.join(task_dir, f"{task_name}.json")
    if not os.path.exists(json_file):
        return {}
    task_file = json_file

    task_name = os.path.splitext(os.path.basename(task_file))[0]

    try:
        with open(task_file, "r") as f:
            json_data = json.load(f)
        few_shot_prompts, few_shot_targets, samples_used = generate_few_shot_prompts(
            json_data,
            num_of_shots,
            max_samples,
            exclude_target_label=exclude_target_label,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer,
        )
        data = pd.DataFrame({"prompt": few_shot_prompts, "target": few_shot_targets})
    except Exception:
        return None
    correct_predictions = 0
    total_predictions = 0
    per_sample_correct = []

    prompts = data["prompt"].tolist()
    targets = data["target"].tolist()

    for i in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"Processing {task_name}",
        leave=False,
    ):
        batch_prompts = prompts[i : i + batch_size]
        batch_targets = targets[i : i + batch_size]

        try:
            batch_predictions = process_batch(
                batch_prompts, batch_targets, max_new_tokens=max_new_tokens
            )
            correct_predictions += sum(batch_predictions)
            total_predictions += len(batch_predictions)

            per_sample_correct.extend(batch_predictions)
        except Exception:
            continue

    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
    else:
        accuracy = 0.0

    if return_per_sample and return_samples_used:
        return accuracy, samples_used, per_sample_correct
    elif return_per_sample:
        return accuracy, per_sample_correct
    elif return_samples_used:
        return accuracy, samples_used
    else:
        return accuracy


def fv_icl_tasks_benchmark_with_ci(
    model,
    tokenizer,
    task_name,
    task_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "icl_tasks"),
    max_samples=5000,
    num_of_shots=5,
    batch_size=64,
    use_wandb=False,
    confidence_level=0.95,
    return_per_sample=False,
    exclude_target_label=True,
    max_new_tokens=3,
):
    accuracy, n_samples, per_sample_correct = fv_icl_tasks_benchmark(
        model=model,
        tokenizer=tokenizer,
        task_name=task_name,
        task_dir=task_dir,
        max_samples=max_samples,
        num_of_shots=num_of_shots,
        batch_size=batch_size,
        use_wandb=use_wandb,
        return_samples_used=True,
        return_per_sample=return_per_sample,
        exclude_target_label=exclude_target_label,
        max_new_tokens=max_new_tokens,
    )

    z_value = stats.norm.ppf((1 + confidence_level) / 2)

    if accuracy == 0:
        ci_lower = 0
        ci_upper = min(1, 3.0 / n_samples)
        std_error = 0
    elif accuracy == 1:
        ci_lower = max(0, 1 - 3.0 / n_samples)
        ci_upper = 1
        std_error = 0
    else:
        std_error = np.sqrt((accuracy * (1 - accuracy)) / n_samples)
        margin = z_value * std_error
        ci_lower = max(0, accuracy - margin)
        ci_upper = min(1, accuracy + margin)

    return {
        "accuracy": accuracy,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": std_error,
        "n_samples": n_samples,
    }, per_sample_correct
