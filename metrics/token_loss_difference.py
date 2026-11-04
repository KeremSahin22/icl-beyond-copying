import sys
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
from scipy import stats
import torch
import numpy as np
from tqdm import tqdm
import time
import random

sys.path.append(str(Path(__file__).parent.parent))
from mmap_dataset_lightning import setup_pythia_data


def get_model(
    model_type, step, dtype=torch.float16, fs_base=None, model_location=None
):
    if model_type == "EleutherAI/pythia-160m":
        model_name_or_path = model_type
        revision = step
    else:
        model_name_or_path = f"{fs_base}/{model_location}/{model_type}/step={step}"
        revision = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        revision=revision,
        device_map="auto",
        torch_dtype=dtype,
    )

    return model


def index_of_valid_positions(labels):
    return (labels[:, 50] != -100) & (labels[:, 500] != -100)


def compute_losses_at_positions(input_ids, logits, positions=[49, 499]):
    batch_size = input_ids.shape[0]
    device = input_ids.device

    losses = torch.zeros(batch_size, len(positions), device=device)

    for i, pos in enumerate(positions):
        target_tokens = input_ids[:, pos + 1]
        predicted_logits = logits[:, pos, :]

        losses[:, i] = torch.nn.functional.cross_entropy(
            predicted_logits, target_tokens, reduction="none"
        )

    return losses[:, 0], losses[:, 1]


def collect_valid_samples(dataloader, num_of_samples, skip_specific_positions=False):
    collected = []
    num_skipped = 0
    total_collected = 0

    for batch in tqdm(iter(dataloader), desc="Collecting"):
        if total_collected >= num_of_samples:
            break

        batch_input_ids = batch["input_ids"][:, :501]
        current_batch_size = batch_input_ids.shape[0]

        if skip_specific_positions:
            labels = batch["labels"][:, :501]
            keep_mask = index_of_valid_positions(labels)
            batch_input_ids = batch_input_ids[keep_mask]
            num_skipped += current_batch_size - batch_input_ids.shape[0]

            if batch_input_ids.shape[0] == 0:
                continue

        remaining = num_of_samples - total_collected
        batch_input_ids = batch_input_ids[:remaining]
        collected.append(batch_input_ids)
        total_collected += batch_input_ids.shape[0]

    input_ids_tensor = torch.cat(collected, dim=0)

    return input_ids_tensor, num_skipped


def resolve_equiv_lookup_path(model_type, config, explicit_path=None):
    project_root = Path(__file__).parent.parent

    if explicit_path is not None:
        p = Path(explicit_path)
        if p.exists():
            return str(p.resolve())
        raise FileNotFoundError(f"Equivalence lookup not found at {p}")

    candidates = []

    candidates.append(
        project_root / "dataset" / "equiv_lookup" / f"equiv_lookup_{model_type}.pt"
    )

    equiv_from_config = config.get("equivalence_lookup_path")
    if equiv_from_config:
        candidates.extend(
            [
                Path(equiv_from_config),
                project_root / equiv_from_config,
                project_root / "data" / equiv_from_config,
                project_root / "dataset" / "equiv_lookup" / equiv_from_config,
            ]
        )

    equiv_threshold = config.get("equivalence_threshold")
    if equiv_threshold is not None:
        size = "1b" if "1b" in model_type else None
        if size:
            thresh_name = f"equiv_lookup_threshold{equiv_threshold}_{size}.pt"
            candidates.extend(
                [
                    project_root / thresh_name,
                    project_root / "data" / thresh_name,
                    project_root / "dataset" / "equiv_lookup" / thresh_name,
                    project_root
                    / "dataset"
                    / "equiv_lookup"
                    / f"equiv_lookup_hapax_{size}_thresh_{equiv_threshold}.pt",
                ]
            )

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate.resolve())

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find thresholded equivalence lookup. Tried:\n  "
        f"{searched}\n"
        "Pass --equiv_lookup_path to the .pt file used for thresholded training "
        "(e.g. dataset/equiv_lookup/equiv_lookup_hapax_1b_thresh_0.3.pt)."
    )


def process_batches(
    model,
    input_ids_tensor,
    batch_size=256,
    device="cuda",
):
    num_samples = input_ids_tensor.shape[0]
    token_loss_per_sample = np.empty(num_samples, dtype=np.float32)
    loss_50_per_sample = np.empty(num_samples, dtype=np.float32)
    loss_500_per_sample = np.empty(num_samples, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch = input_ids_tensor[start:end].to(device)

            outputs = model(batch)
            logits = outputs.logits

            loss_50, loss_500 = compute_losses_at_positions(batch, logits, [49, 499])
            diff = loss_50 - loss_500

            token_loss_per_sample[start:end] = diff.cpu().numpy()
            loss_50_per_sample[start:end] = loss_50.cpu().numpy()
            loss_500_per_sample[start:end] = loss_500.cpu().numpy()

    return token_loss_per_sample, loss_50_per_sample, loss_500_per_sample


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    fs_base = str(Path(__file__).resolve().parent.parent)
    model_location = "models"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--first_step", type=int, default=100)
    parser.add_argument("--last_step", type=int, default=20000)
    parser.add_argument("--step_stride", type=int, default=100)
    parser.add_argument("--num_of_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--data-base",
        type=str,
        default="data/huggingface_dedupped",
        help="Directory holding document.bin/.idx.",
    )
    parser.add_argument("--use_fp16", action="store_true", default=True)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Overrides num_workers from the config.",
    )
    parser.add_argument("--extra_identifier", type=str, default="")
    parser.add_argument(
        "--samples_cache_path",
        type=str,
        default=None,
        help="Where to cache the collected samples. Inferred from model_type if not set.",
    )
    parser.add_argument(
        "--skip_bigrams",
        action="store_true",
        help="Evaluate only on non-extractive samples.",
    )
    parser.add_argument(
        "--skip_bigrams_thresholded",
        action="store_true",
        help="Like --skip_bigrams, using the thresholded equivalence masking.",
    )
    parser.add_argument(
        "--equiv_lookup_path",
        type=str,
        default=None,
        help="Defaults to equivalence_lookup_path in the config.",
    )

    args = parser.parse_args()

    if args.skip_bigrams and args.skip_bigrams_thresholded:
        raise ValueError(
            "--skip_bigrams and --skip_bigrams_thresholded are mutually exclusive."
        )

    if "2.8b" in args.model_type:
        size = "2.8b"
    elif "1b" in args.model_type:
        size = "1b"
    else:
        raise ValueError(
            f"Cannot infer config size from model_type '{args.model_type}'. "
            "Model type must contain '1b' or '2.8b'."
        )
    config_path = Path(__file__).parent.parent / "configs" / f"pythia-{size}.json"

    dtype = torch.float16 if args.use_fp16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(config_path, "r") as f:
        config = json.load(f)

    config["train-data-paths"] = [f"{args.data_base}/document"]
    config["valid-data-paths"] = [f"{args.data_base}/document"]
    config["test-data-paths"] = [f"{args.data_base}/"]

    config["train_micro_batch_size_per_gpu"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers

    config["mask_bigram_loss"] = args.skip_bigrams
    config["use_equivalence_bigram_masking"] = args.skip_bigrams_thresholded

    equiv_lookup = None
    if args.skip_bigrams_thresholded:
        equiv_path = resolve_equiv_lookup_path(
            args.model_type, config, args.equiv_lookup_path
        )
        equiv_lookup = torch.load(equiv_path, map_location="cpu")

    data_module = setup_pythia_data(config, equiv_lookup=equiv_lookup)
    data_module.setup()

    os.makedirs("results", exist_ok=True)
    os.makedirs("loss_differences", exist_ok=True)
    skip_specific_positions = args.skip_bigrams or args.skip_bigrams_thresholded
    if args.skip_bigrams:
        suffix = "_skip_bigrams"
    elif args.skip_bigrams_thresholded:
        suffix = "_skip_bigrams_thresholded"
    else:
        suffix = ""
    id_part = f"_{args.extra_identifier}" if args.extra_identifier else ""
    filename = f"./loss_differences/{args.model_type}_tld{suffix}{id_part}.jsonl"

    cache_path = args.samples_cache_path
    if cache_path is None:
        os.makedirs("samples_cache", exist_ok=True)
        cache_path = f"./samples_cache/{args.model_type}{suffix}_n{args.num_of_samples}.pt"

    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu")
        valid_input_ids = cache["input_ids"]
        num_skipped = cache["num_skipped"]

    else:
        if skip_specific_positions:
            seed_dataloader = data_module.val_dataloader_with_masking()
        else:
            seed_dataloader = data_module.val_dataloader()

        valid_input_ids, num_skipped = collect_valid_samples(
            seed_dataloader, args.num_of_samples, skip_specific_positions
        )
        del seed_dataloader

        torch.save({"input_ids": valid_input_ids, "num_skipped": num_skipped}, cache_path)

    for step in tqdm(range(args.first_step, args.last_step, args.step_stride), desc="Steps"):
        start_time = time.time()

        model = get_model(
            args.model_type,
            step,
            dtype=dtype,
            fs_base=fs_base,
            model_location=model_location,
        )
        model.eval()
        device = next(model.parameters()).device

        loss_array, loss_50_array, loss_500_array = process_batches(
            model,
            valid_input_ids,
            args.batch_size,
            device,
        )

        confidence_level = 0.95

        avg_score = np.mean(loss_array)
        std_error = stats.sem(loss_array)
        degrees_freedom = len(loss_array) - 1
        confidence_interval = stats.t.interval(
            confidence_level, degrees_freedom, loc=avg_score, scale=std_error
        )

        avg_loss_50 = np.mean(loss_50_array)
        std_error_50 = stats.sem(loss_50_array)
        degrees_freedom_50 = len(loss_50_array) - 1
        confidence_interval_50 = stats.t.interval(
            confidence_level, degrees_freedom_50, loc=avg_loss_50, scale=std_error_50
        )

        avg_loss_500 = np.mean(loss_500_array)
        std_error_500 = stats.sem(loss_500_array)
        degrees_freedom_500 = len(loss_500_array) - 1
        confidence_interval_500 = stats.t.interval(
            confidence_level, degrees_freedom_500, loc=avg_loss_500, scale=std_error_500
        )

        elapsed = time.time() - start_time

        with open(filename, "a") as f:
            record = {
                "step": step,
                "avg_tld": float(avg_score),
                "std_tld_error": float(std_error),
                "ci_lower_tld": float(confidence_interval[0]),
                "ci_upper_tld": float(confidence_interval[1]),
                "avg_loss_50": float(avg_loss_50),
                "std_loss_50_error": float(std_error_50),
                "ci_lower_loss_50": float(confidence_interval_50[0]),
                "ci_upper_loss_50": float(confidence_interval_50[1]),
                "avg_loss_500": float(avg_loss_500),
                "std_loss_500_error": float(std_error_500),
                "ci_lower_loss_500": float(confidence_interval_500[0]),
                "ci_upper_loss_500": float(confidence_interval_500[1]),
                "num_of_skipped_samples": num_skipped,
                "processed_samples": len(loss_array),
                "processing_time": elapsed,
            }
            f.write(json.dumps(record) + "\n")
            if step % 500 == 0:
                f.flush()

        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
