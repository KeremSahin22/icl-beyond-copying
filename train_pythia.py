import os
import json
import argparse
import signal
import urllib.request
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import TQDMProgressBar
import glob

from model import PythiaModel

from mmap_dataset_lightning import setup_pythia_data


def discord_notify(message: str) -> None:
    webhook = os.environ.get("DISCORD_WEBHOOK")
    if not webhook:
        return
    try:
        payload = json.dumps({"content": message}).encode()
        req = urllib.request.Request(
            webhook,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "curl/7.68.0",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


class CleanupCallback(Callback):
    def __init__(self, model_name, output_dir):
        self.base_dir = os.path.join(output_dir, f"pythia_output_{model_name}")

    def on_validation_end(self, trainer, pl_module):
        base_dir = self.base_dir

        if not os.path.isdir(base_dir):
            return

        step_nums = []
        for sub in os.listdir(base_dir):
            if sub.endswith(".ckpt") and sub != "last.ckpt" and sub.startswith("step="):
                try:
                    step_nums.append(int(sub.replace("step=", "").replace(".ckpt", "")))
                except ValueError:
                    continue
        max_step = max(step_nums) if step_nums else None
        for sub in os.listdir(base_dir):
            if (not sub.endswith(".ckpt")) or sub == "last.ckpt":
                continue

            try:
                step_num = int(sub.replace("step=", "").replace(".ckpt", ""))
                if step_num == max_step:
                    continue
            except ValueError:
                continue

            ckpt_folder = os.path.join(base_dir, sub)

            pattern = os.path.join(ckpt_folder, "**", "*_optim_states.pt")
            for fp in glob.glob(pattern, recursive=True):
                try:
                    os.remove(fp)

                except FileNotFoundError:
                    pass


class DetailedProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        if trainer.optimizers:
            items["lr"] = trainer.optimizers[0].param_groups[0]["lr"]
        return items

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if trainer.max_steps > 0:
            self.train_progress_bar.total = trainer.max_steps
            self.train_progress_bar.n = trainer.global_step
            self.train_progress_bar.set_description("Training")
            self.train_progress_bar.refresh()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % trainer.accumulate_grad_batches == 0:
            self.train_progress_bar.n = trainer.global_step
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
            self.train_progress_bar.refresh()


def train_pythia(args):
    with open(args.config, "r") as f:
        config = json.load(f)

    torch.set_float32_matmul_precision("medium")

    ckpt_base = args.ckpt_dir or args.output_dir

    use_bf16 = bool(config.get("bf16", {}).get("enabled", False))

    deepspeed_config = {
        **(
            {"bf16": {"enabled": True}}
            if use_bf16
            else {
                "fp16": {
                    **config["fp16"],

                    "initial_scale_power": config["fp16"].get("initial_scale_power", 12),
                    "loss_scale_window": config["fp16"].get("loss_scale_window", 1000),
                }
            }
        ),
        "zero_optimization": {
            "stage": (
                config["zero_optimization"]["stage"]
                if not config["reinitialize_optim_for_heads"]
                else 0
            ),
            "allgather_partitions": config["zero_optimization"].get(
                "allgather_partitions", True
            ),
            "allgather_bucket_size": config["zero_optimization"].get(
                "allgather_bucket_size", 200000000
            ),
            "overlap_comm": config["zero_optimization"].get("overlap_comm", True),
            "reduce_scatter": config["zero_optimization"].get("reduce_scatter", True),
            "reduce_bucket_size": config["zero_optimization"].get(
                "reduce_bucket_size", 200000000
            ),
            "contiguous_gradients": config["zero_optimization"].get(
                "contiguous_gradients", True
            ),
        },
        "gradient_clipping": config["gradient_clipping"],
        "train_micro_batch_size_per_gpu": config["train_micro_batch_size_per_gpu"],
        "steps_per_print": config["steps_per_print"],
        "wall_clock_breakdown": config["wall_clock_breakdown"],
    }

    if "data_types" in config:
        deepspeed_config["data_types"] = config["data_types"]

    if config["checkpoint-activations"]:
        deepspeed_config["activation_checkpointing"] = {
            "partition_activations": config["partition-activations"],
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "synchronize_checkpoint_boundary": config["synchronize-each-layer"],
        }

    pl.seed_everything(args.seed)
    config["seed"] = args.seed

    model = PythiaModel.from_config(config=config)

    if args.save_step0_init:
        step0_dir = os.path.join(
            ckpt_base, f"pythia_output_{args.run_name}", "step0_init"
        )
        os.makedirs(step0_dir, exist_ok=True)
        model.model.save_pretrained(step0_dir)
        model.tokenizer.save_pretrained(step0_dir)

    if "train-data-paths" not in config or not config["train-data-paths"]:
        raise ValueError("Config must include 'train-data-paths'")

    if "use_single_file" not in config:
        config["use_single_file"] = True

    equiv_lookup = None
    if config["use_equivalence_bigram_masking"]:
        equiv_path = config["equivalence_lookup_path"]
        if os.path.exists(equiv_path):
            equiv_lookup = torch.load(equiv_path, map_location="cpu")

    resume_sampler_state = None
    if args.resume_from_checkpoint:
        if os.path.isdir(args.resume_from_checkpoint):
            state_path = os.path.join(
                args.resume_from_checkpoint,
                "checkpoint",
                "mp_rank_00_model_states.pt",
            )
        else:
            state_path = args.resume_from_checkpoint
        ckpt_obj = torch.load(state_path, map_location="cpu")
        resume_sampler_state = ckpt_obj.get("data_sampling_state", None)
        if not resume_sampler_state:
            raise ValueError(
                "Expected sampler state in checkpoint but none was found. "
                "Cannot resume training without sampler state."
            )

    data_module = setup_pythia_data(
        config, equiv_lookup=equiv_lookup, resume_sampler_state=resume_sampler_state
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(ckpt_base, f"pythia_output_{args.run_name}"),
        filename="{step}",
        save_top_k=-1,
        every_n_train_steps=config["checkpoint-factor"],

        save_last=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger(
        save_dir=os.path.join(args.output_dir, "tensorboard_logs"),
        name="pythia",
        version=args.run_name,
    )

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if config["zero_optimization"]["stage"] > 0:
            strategy = DeepSpeedStrategy(
                config=deepspeed_config,
            )

        else:
            strategy = "ddp"
    else:
        strategy = "auto"

    gas = config["gas"]

    cbs = [checkpoint_callback, lr_monitor, DetailedProgressBar()]
    if args.save_weights_only:
        cbs.append(CleanupCallback(args.run_name, ckpt_base))

    trainer = pl.Trainer(
        max_steps=config["train-iters"],
        gradient_clip_val=config["gradient_clipping"],
        accumulate_grad_batches=gas,
        limit_val_batches=20,
        val_check_interval=gas * 50,
        precision=(
            "bf16-mixed" if use_bf16 else ("16-mixed" if config["fp16"]["fp16"] else "32")
        ),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        num_nodes=args.num_nodes,
        strategy=strategy,
        callbacks=cbs,
        logger=logger,
        log_every_n_steps=config["log-interval"],
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
    )

    if args.resume_from_checkpoint:
        trainer.fit(
            model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint
        )
    else:
        trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pythia-1b.json",
    )
    parser.add_argument(
        "--save_weights_only",
        action="store_true",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trained_models",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory for checkpoints. Defaults to --output_dir.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--save_step0_init",
        action="store_true",
    )

    signal.signal(signal.SIGTERM, lambda *_: os._exit(1))
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    is_rank0 = os.environ.get("LOCAL_RANK", "0") == "0"

    try:
        if is_rank0:
            discord_notify(f"Training started: `{args.run_name}`")
        train_pythia(args)
        if is_rank0:
            discord_notify(f"Training finished: `{args.run_name}` completed successfully.")
    except Exception:
        import traceback
        tb = traceback.format_exc()

        if is_rank0:
            discord_notify(f"Training CRASHED: `{args.run_name}`\n```\n{tb[-1500:]}\n```")
        os._exit(1)
