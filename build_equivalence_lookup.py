import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def build_lookup(model, threshold):
    with torch.no_grad():
        embeddings = F.normalize(model.gpt_neox.embed_in.weight.float(), p=2, dim=1)
        return (embeddings @ embeddings.T) >= threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the embedding-similarity lookup used for thresholded bigram masking."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Converted Hugging Face checkpoint of a vanilla model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None, help="Defaults to equivalence_threshold in the config")
    parser.add_argument("--output", type=str, default=None, help="Defaults to equivalence_lookup_path in the config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    threshold = args.threshold if args.threshold is not None else config["equivalence_threshold"]
    output = args.output or config["equivalence_lookup_path"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    lookup = build_lookup(model, threshold)
    torch.save(lookup.cpu(), output)
    print(f"Saved {tuple(lookup.shape)} lookup with threshold {threshold} to {output}")
