from model import PythiaLightningModule
import argparse
import json
import os
import torch
import yaml
from pathlib import Path

def load_local_checkpoint(checkpoint_dir, config_path, attn_implementation=None):
    if attn_implementation is None:
        model = PythiaLightningModule(config_path=config_path)
    else:
        with open(config_path) as f:
            if config_path.endswith(".yaml"):
                config = yaml.safe_load(f)["config"]
            else:
                config = json.load(f)
        config["_attn_implementation"] = attn_implementation
        model = PythiaLightningModule(config=config)
    model_state_file = os.path.join(checkpoint_dir, "checkpoint", "mp_rank_00_model_states.pt")
    checkpoint = torch.load(model_state_file, map_location='cpu')
    state_dict = checkpoint.get('module', checkpoint)

    try:
        model.model.load_state_dict(state_dict)
    except:
        model.load_state_dict(state_dict, strict=False)

    inner = model.model
    inner.to("cuda")
    inner.half()
    inner.eval()
    return inner


def main():
    base = os.environ.get("PYTHIA_BASE") or str(Path(__file__).resolve().parent)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--first_step", type=int, default=100)
    parser.add_argument("--last_step", type=int, default=20000)
    parser.add_argument("--step_interval", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Base directory for hf_output")
    parser.add_argument("--attn_implementation", type=str, default=None,
                         help="Override the attention kernel in hparams.yaml, e.g. sdpa when flash_attn is unavailable")
    args = parser.parse_args()
    model_type = args.model_type
    last_step = args.last_step
    first_step = args.first_step
    output_base = args.output_dir or base
    for step in range(first_step, last_step, args.step_interval):
        checkpoint_dir = (
            f"{base}/trained_models/pythia_output_{model_type}/step={step}.ckpt"
        )
        config_path = f"{base}/trained_models/tensorboard_logs/pythia/{model_type}/hparams.yaml"
        model = load_local_checkpoint(
            checkpoint_dir, config_path, attn_implementation=args.attn_implementation
        )
        model.save_pretrained(
            f"{output_base}/hf_output/{model_type}/step={step}"
        )


if __name__ == "__main__":
    main()
