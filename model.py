import os
import json
import torch
import math
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from transformers import (
    GPTNeoXForCausalLM,
    GPTNeoXConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from torchmetrics import SumMetric, MeanMetric
import torch.nn as nn
import yaml

try:
    from deepspeed.ops.adam import FusedAdam

    OPTIMIZER_IMPLEMENTATION = FusedAdam

except ImportError:
    OPTIMIZER_IMPLEMENTATION = torch.optim.AdamW


class PythiaLightningModule(pl.LightningModule):
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.token_counter = SumMetric(dist_sync_on_step=True)
        self.effective_token_counter = SumMetric(dist_sync_on_step=True)
        self.loss_meter_train = MeanMetric(dist_sync_on_step=True)
        self.loss_meter_val = MeanMetric(dist_sync_on_step=True)
        self.loss_meter_val_diff = MeanMetric(dist_sync_on_step=True)
        self.loss_meter_ind_loss_func = MeanMetric(dist_sync_on_step=True)
        self.loss_meter_multitask_total = MeanMetric(dist_sync_on_step=True)
        self.loss_meter_neg_bigram_actual_loss = MeanMetric(dist_sync_on_step=True)
        self._samples_seen_in_epoch = 0

        if config_path is not None and os.path.exists(config_path):
            if config_path.endswith(".json"):
                with open(config_path, "r") as f:
                    self.config = json.load(f)
            elif config_path.endswith(".yaml"):
                with open(config_path, "r") as f:
                    self.config = yaml.safe_load(f)["config"]
        elif config is not None:
            self.config = config
        else:
            raise ValueError("Either config_path or config must be provided")

        model_config_obj = self._create_model_config()

        self.model = GPTNeoXForCausalLM(
            config=model_config_obj
        )

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

        self._apply_pythia_inits()

        if self.config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.model.train()

        if model_config_obj.torch_dtype == torch.float16 and (
            self.config["zero_optimization"]["stage"] > 0
        ):
            self.model.half()
        elif model_config_obj.torch_dtype == torch.bfloat16 and (
            self.config["zero_optimization"]["stage"] > 0
        ):
            self.model.bfloat16()

        self.learning_rate = self.config["optimizer"]["params"]["lr"]
        self.min_lr = self.config["min_lr"]
        self.weight_decay = self.config["weight-decay"]
        self.warmup_ratio = self.config["warmup"]
        self.lr_decay_iters = self.config[
            "lr-decay-iters"
        ]
        self.gradient_clip_val = self.config["gradient_clipping"]
        self.betas = self.config["optimizer"]["params"]["betas"]

    def _create_model_config(self) -> GPTNeoXConfig:
        hidden_size = self.config["hidden_size"]

        try:
            torch_dtype_val = getattr(torch, self.config["torch_dtype"])
        except AttributeError:
            torch_dtype_val = torch.float32
        except KeyError:
            torch_dtype_val = torch.float32

        return GPTNeoXConfig(
            vocab_size=self.config[
                "vocab_size"
            ],
            hidden_size=hidden_size,
            num_hidden_layers=self.config["num-layers"],
            num_attention_heads=self.config["num-attention-heads"],
            intermediate_size=hidden_size * 4,
            rotary_pct=self.config["rotary-pct"],
            hidden_dropout=self.config["hidden-dropout"],
            max_position_embeddings=self.config["max-position-embeddings"],
            bos_token_id=0,
            eos_token_id=0,
            attention_dropout=self.config["attention-dropout"],
            tie_word_embeddings=self.config[
                "tie_word_embeddings"
            ],
            attn_implementation=self.config["_attn_implementation"],

            hidden_act=self.config["hidden-act"],
            layer_norm_eps=self.config["layer-norm-eps"],
            use_cache=False,
            initializer_range=self.config["initializer-range"],
            rotary_emb_base=self.config["rotary-emb-base"],
            use_parallel_residual=self.config["use_parallel_residual"],
            torch_dtype=torch_dtype_val,
        )

    def _apply_pythia_inits(self):

        std = self.config["initializer-range"]
        num_layers = self.model.config.num_hidden_layers

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()

                is_attention_output = name.endswith("attention.dense")
                is_mlp_output = name.endswith("mlp.dense_4h_to_h")

                if self.config["use_scaled_init_for_output_weights"] and (
                    is_attention_output or is_mlp_output
                ):
                    module.weight.data.normal_(
                        mean=0.0, std=std / math.sqrt(2.0 * num_layers)
                    )

            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, **inputs):
        return self.model(**inputs)

    def compute_induction_loss_in_batch(self, outputs, batch):
        if self.config["attention_pattern_based_induction_loss"]:
            if outputs.attentions is None:
                raise ValueError(
                    "outputs.attentions is None - model must be called with output_attentions=True"
                )

            attentions = outputs.attentions
            batch_size, seq_len = batch["input_ids"].shape
            input_ids = batch["input_ids"]

            position_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool),
                diagonal=-2,
            )

            similarity = (input_ids.unsqueeze(2) == input_ids.unsqueeze(1)).float()
            matching_positions = similarity * position_mask.unsqueeze(0)

            induction_positions = torch.zeros_like(matching_positions)
            induction_positions[:, :, 1:] = matching_positions[:, :, :-1]

            has_induction_target = (induction_positions.sum(dim=1) > 0).float()
            num_positions_with_targets = has_induction_target.sum(dim=1)

            has_any_induction = (
                num_positions_with_targets > 0
            ).float()

            total_induction = 0
            for layer_attention in attentions:
                induction_per_position = (
                    layer_attention * induction_positions.unsqueeze(1)
                ).sum(dim=3)

                induction_sum = induction_per_position.sum(dim=2)
                normalized_induction = induction_sum / torch.clamp(
                    num_positions_with_targets.unsqueeze(1), min=1
                )

                layer_induction = (
                    normalized_induction.sum(dim=0) / has_any_induction.sum()
                )

                total_induction += layer_induction
            return total_induction.sum()

    def multitask_training(self, batch, batch_idx):
        if self.config["attention_pattern_based_induction_loss"]:
            self.model.config._attn_implementation = (
                "eager"
            )
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_attentions=True,
            )
        else:
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

        lm_loss = outputs.loss

        induction_loss = self.compute_induction_loss_in_batch(
            outputs,
            batch,
        )

        induction_weight = self.config["induction_loss_weight"]

        total_loss = lm_loss + induction_weight * induction_loss

        self.token_counter.update(batch["attention_mask"].sum())
        self.loss_meter_train.update(lm_loss)
        self.loss_meter_multitask_total.update(total_loss)
        self.loss_meter_ind_loss_func.update(induction_loss)

        self.model.config._attn_implementation = "flash_attention_2"

        return total_loss

    def language_modeling_training(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.token_counter.update(batch["attention_mask"].sum())
        loss = outputs.loss
        self.loss_meter_train.update(loss)
        return loss

    def mask_bigram_loss_with_equivalence(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        self.token_counter.update(batch["attention_mask"].sum())
        effective_tokens = (batch["labels"] != -100).sum()
        self.effective_token_counter.update(effective_tokens)
        self.loss_meter_train.update(loss)

        return loss

    def bigram_attention_mask(
        self, batch_tokens: torch.LongTensor, dtype: torch.dtype
    ) -> torch.Tensor:
        B, S = batch_tokens.shape
        device = batch_tokens.device

        causal = torch.tril(torch.zeros(S, S, device=device, dtype=dtype))
        causal = causal + torch.triu(
            torch.full((S, S), float("-inf"), device=device, dtype=dtype), diagonal=1
        )
        masks = causal.unsqueeze(0).expand(B, S, S).clone()

        V = int(batch_tokens.max().item()) + 1
        bigrams = batch_tokens[:, :-1] * V + batch_tokens[:, 1:]

        eq = bigrams.unsqueeze(2) == bigrams.unsqueeze(1)

        idx = torch.arange(S - 1, device=device, dtype=dtype)
        allowed = (idx.unsqueeze(1) - idx.unsqueeze(0)) >= 2
        eq &= allowed.unsqueeze(0)

        b_idx, i_idx, j_idx = eq.nonzero(as_tuple=True)
        masks[b_idx, i_idx, j_idx + 1] = float("-inf")
        if self.config["mask_bigram_position_with_self_included"]:
            masks[b_idx, i_idx, j_idx] = float("-inf")

        return masks.unsqueeze(1)

    def mask_based_training(self, batch, batch_idx):
        dtype = self.model.dtype
        bigram_masks = self.bigram_attention_mask(batch["input_ids"], dtype)

        self.model.config._attn_implementation = "sdpa"
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=bigram_masks,
            labels=batch["labels"],
        )
        self.model.config._attn_implementation = "flash_attention_2"

        self.token_counter.update(batch["attention_mask"].sum())
        loss = outputs.loss
        self.loss_meter_train.update(loss)
        return loss

    def mask_unigram_loss(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        self.token_counter.update(batch["attention_mask"].sum())
        effective_tokens = (batch["labels"] != -100).sum()
        self.effective_token_counter.update(effective_tokens)
        self.loss_meter_train.update(loss)

        return loss

    def mask_bigram_loss(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        self.token_counter.update(batch["attention_mask"].sum())
        effective_tokens = (batch["labels"] != -100).sum()
        self.effective_token_counter.update(effective_tokens)
        self.loss_meter_train.update(loss)

        return loss

    def negate_bigram_loss(self, batch, batch_idx):
        bigram_masks = self.bigram_loss_mask(
            batch["input_ids"], self.tokenizer.vocab_size
        )

        outputs = self.model(
            input_ids=batch["input_ids"],
            labels=None,
        )

        logits = outputs.logits[..., :-1, :].contiguous()
        labels = batch["labels"][..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        ).view(labels.shape)

        bigram_masks_shifted = bigram_masks[:, 1:]
        negative_weight = self.config["negate_bigram_loss_weight"]

        gradient_direction = torch.where(
            bigram_masks_shifted,
            torch.ones_like(per_token_loss),
            -negative_weight * torch.ones_like(per_token_loss),
        )

        valid_mask = (labels != -100).float()
        actual_loss = (per_token_loss * valid_mask).sum() / valid_mask.sum()
        directed_loss = (
            (per_token_loss * gradient_direction) * valid_mask
        ).sum() / valid_mask.sum()

        self.token_counter.update(batch["attention_mask"].sum())
        effective_tokens = bigram_masks.sum()
        self.effective_token_counter.update(effective_tokens)

        self.loss_meter_train.update(actual_loss)
        return directed_loss

    def previous_token_mask(self, batch):
        B, S = batch["input_ids"].shape
        device = batch["input_ids"].device

        causal = torch.tril(torch.zeros(S, S, device=device, dtype=self.model.dtype))
        causal = causal + torch.triu(
            torch.full((S, S), float("-inf"), device=device, dtype=self.model.dtype),
            diagonal=1,
        )
        masks = causal.unsqueeze(0).expand(B, S, S).clone()

        row = torch.arange(1, S, device=device)
        col = row - 1
        masks[:, row, col] = float("-inf")

        return masks.unsqueeze(1)

    def mask_previous_token_position(self, batch, batch_idx):
        previous_token_masks = self.previous_token_mask(batch)

        self.model.config._attn_implementation = "sdpa"
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=previous_token_masks,
            labels=batch["labels"],
        )
        self.model.config._attn_implementation = "flash_attention_2"

        self.token_counter.update(batch["attention_mask"].sum())
        loss = outputs.loss
        self.loss_meter_train.update(loss)
        return loss

    def training_step(self, batch, batch_idx):
        self.model.train()

        if self.config["use_equivalence_bigram_masking"]:
            loss = self.mask_bigram_loss_with_equivalence(batch, batch_idx)
        elif self.config["use_induction_loss"]:
            loss = self.multitask_training(batch, batch_idx)
        elif self.config["mask_bigram_position"]:
            loss = self.mask_based_training(batch, batch_idx)
        elif self.config["mask_bigram_loss"]:
            loss = self.mask_bigram_loss(batch, batch_idx)
        elif self.config.get("mask_unigram_loss", False):
            loss = self.mask_unigram_loss(batch, batch_idx)
        elif self.config["negate_bigram_loss"]:
            loss = self.negate_bigram_loss(batch, batch_idx)
        elif self.config["mask_previous_token_position"]:
            loss = self.mask_previous_token_position(batch, batch_idx)
        else:
            loss = self.language_modeling_training(batch, batch_idx)

        try:
            self._samples_seen_in_epoch += int(batch["input_ids"].shape[0])
        except Exception:
            pass
        return loss

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        data_state = checkpoint["data_sampling_state"]
        self._samples_seen_in_epoch = int(data_state["samples_seen_in_epoch"])
        self._resume_has_samples_seen = self._samples_seen_in_epoch > 0

    def on_train_epoch_start(self) -> None:
        if getattr(self, "_resume_has_samples_seen", False):
            del self._resume_has_samples_seen
        else:
            self._samples_seen_in_epoch = 0

    def on_save_checkpoint(self, checkpoint):
        checkpoint["data_sampling_state"] = {
            "epoch": int(self.current_epoch),
            "samples_seen_in_epoch": int(self._samples_seen_in_epoch),
            "shuffle": bool(self.config["shuffle"]),
            "seed": int(self.config.get("seed", 1234)),
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            self.model.eval()
            loss = self.loss_meter_train.compute()
            if self.config["use_induction_loss"]:
                loss_multitask = self.loss_meter_multitask_total.compute()
                loss_induction = self.loss_meter_ind_loss_func.compute()
            if (
                self.config["mask_bigram_loss"]
                or self.config["negate_bigram_loss"]
                or self.config["use_equivalence_bigram_masking"]
                or self.config.get("mask_unigram_loss", False)
            ):
                effective_tokens = int(self.effective_token_counter.compute())
            total_tokens = int(self.token_counter.compute())
            if self.trainer.is_global_zero:
                self.logger.experiment.add_scalar(
                    "train/loss_vs_tokens",
                    loss.detach().cpu().item(),
                    global_step=total_tokens,
                )
                if (
                    self.config["mask_bigram_loss"]
                    or self.config["negate_bigram_loss"]
                    or self.config["use_equivalence_bigram_masking"]
                ):
                    self.logger.experiment.add_scalar(
                        "train/loss_vs_tokens_effective",
                        loss.detach().cpu().item(),
                        global_step=effective_tokens,
                    )
                self.logger.experiment.add_scalar(
                    "train/loss_vs_steps",
                    loss.detach().cpu().item(),
                    global_step=self.global_step,
                )
                if self.config["use_induction_loss"]:
                    self.logger.experiment.add_scalar(
                        "train/loss_induction",
                        loss_induction.detach().cpu().item(),
                        global_step=self.global_step,
                    )
                    self.logger.experiment.add_scalar(
                        "train/loss_multitask",
                        loss_multitask.detach().cpu().item(),
                        global_step=self.global_step,
                    )

            induction_scores = self.compute_induction_score(
                self.model, self.tokenizer, batch_size=256
            )
            if self.config["reinitialize_heads"]:
                self.reinitialize_high_induction_heads(
                    induction_scores,
                    threshold=self.config["reinitialize_threshold"],
                )
            self.loss_meter_train.reset()
            self.loss_meter_multitask_total.reset()
            self.loss_meter_ind_loss_func.reset()

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = outputs.loss

        self.loss_meter_val.update(loss)

        return loss

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.model.eval()
        loss = self.loss_meter_val.compute()

        if (
            self.config["mask_bigram_loss"]
            or self.config["negate_bigram_loss"]
            or self.config["use_equivalence_bigram_masking"]
            or self.config.get("mask_unigram_loss", False)
        ):
            effective_tokens = int(self.effective_token_counter.compute())
        if self.trainer.is_global_zero:
            self.logger.experiment.add_scalar(
                "val/loss", loss.detach().cpu().item(), global_step=self.global_step
            )
            if (
                self.config["mask_bigram_loss"]
                or self.config["negate_bigram_loss"]
                or self.config["use_equivalence_bigram_masking"]
                or self.config.get("mask_unigram_loss", False)
            ):
                self.logger.experiment.add_scalar(
                    "val/loss_vs_tokens_effective",
                    loss.detach().cpu().item(),
                    global_step=effective_tokens,
                )
                self.logger.experiment.add_scalar(
                    "val/ppl_effective",
                    torch.exp(loss).detach().cpu().item(),
                    global_step=effective_tokens,
                )
            self.logger.experiment.add_scalar(
                "val/ppl",
                torch.exp(loss).detach().cpu().item(),
                global_step=self.global_step,
            )
        self.loss_meter_val.reset()
        self.loss_meter_val_diff.reset()

    def compute_induction_score(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        num_of_samples: int = 2000,
        seq_len: int = 50,
        batch_size: int = 16,
    ) -> torch.Tensor:
        orig_attn_impl = model.config._attn_implementation
        model.config._attn_implementation = "eager"

        induction_scores = torch.zeros(
            model.config.num_hidden_layers, model.config.num_attention_heads
        ).to(model.device)

        vocab_size = tokenizer.vocab_size
        random_sequence = torch.randint(1, vocab_size, (num_of_samples, seq_len))
        random_repetitive_sequence = torch.cat(
            [random_sequence, random_sequence], dim=1
        )
        try:
            with torch.no_grad():
                for i in range(0, num_of_samples, batch_size):
                    begin_index = i
                    end_index = min(i + batch_size, num_of_samples)
                    batch = random_repetitive_sequence[begin_index:end_index, :]
                    input = {"input_ids": batch.to(model.device)}
                    result = model(**input, output_attentions=True)

                    for layer in range(model.config.num_hidden_layers):
                        layer_values = result.attentions[layer]
                        curr_ind_scores = (
                            layer_values.diagonal(offset=-seq_len + 1, dim1=-2, dim2=-1)[
                                ..., 1:
                            ]
                            .mean(dim=-1)
                            .sum(dim=0)
                        )
                        induction_scores[layer] += curr_ind_scores
        finally:
            model.config._attn_implementation = orig_attn_impl

        induction_scores /= num_of_samples

        induction_score_list = []
        for layer_idx in range(model.config.num_hidden_layers):
            layer_scores = induction_scores[layer_idx].cpu().tolist()
            induction_score_list.append(layer_scores)

        global_step = self.global_step
        for layer_idx in range(model.config.num_hidden_layers):
            for head_idx, score in enumerate(induction_score_list[layer_idx]):
                self.logger.experiment.add_scalar(
                    f"induction_scores/layer_{layer_idx}/head_{head_idx}",
                    score,
                    global_step=global_step,
                )

        return induction_score_list

    def reinitialize_high_induction_heads(self, induction_scores, threshold):
        std = self.config["initializer-range"]

        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        hidden_size = self.model.config.hidden_size
        head_size = hidden_size // num_heads
        head_qkv_size = head_size * 3

        reinitialized_heads = []

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                if induction_scores[layer_idx][head_idx] > threshold:
                    layer = self.model.gpt_neox.layers[layer_idx]
                    attn = layer.attention
                    with torch.no_grad():
                        head_start = head_idx * head_qkv_size
                        head_end = (head_idx + 1) * head_qkv_size

                        head_start_dense = head_idx * head_size
                        head_end_dense = (head_idx + 1) * head_size

                        attn.query_key_value.weight[head_start:head_end, :].normal_(
                            mean=0.0, std=std
                        )
                        attn.query_key_value.bias[head_start:head_end].zero_()
                        attn.dense.weight[:, head_start_dense:head_end_dense].normal_(
                            mean=0.0, std=std / math.sqrt(2.0 * num_layers)
                        )
                        attn.dense.bias[head_start_dense:head_end_dense].zero_()

                        if self.config["reinitialize_optim_for_heads"]:
                            optimizer = self.optimizers()
                            if isinstance(optimizer, list):
                                raise ValueError(
                                    "Optimizer is expected to be a single optimizer"
                                )
                            optimizer.state[attn.query_key_value.weight]["exp_avg_sq"][
                                head_start:head_end, :
                            ].zero_()
                            optimizer.state[attn.query_key_value.weight]["exp_avg"][
                                head_start:head_end, :
                            ].zero_()
                            optimizer.state[attn.dense.weight]["exp_avg_sq"][
                                :, head_start_dense:head_end_dense
                            ].zero_()
                            optimizer.state[attn.dense.weight]["exp_avg"][
                                :, head_start_dense:head_end_dense
                            ].zero_()
                            optimizer.state[attn.query_key_value.bias]["exp_avg_sq"][
                                head_start:head_end
                            ].zero_()
                            optimizer.state[attn.query_key_value.bias]["exp_avg"][
                                head_start:head_end
                            ].zero_()
                            optimizer.state[attn.dense.bias]["exp_avg_sq"][
                                head_start_dense:head_end_dense
                            ].zero_()
                            optimizer.state[attn.dense.bias]["exp_avg"][
                                head_start_dense:head_end_dense
                            ].zero_()

                    reinitialized_heads.append(
                        (layer_idx, head_idx, induction_scores[layer_idx][head_idx])
                    )

                    if self.trainer.is_global_zero:
                        self.logger.experiment.add_scalar(
                            f"reinitialized_heads/layer_{layer_idx}_head_{head_idx}",
                            induction_scores[layer_idx][head_idx],
                            global_step=self.global_step,
                        )

        if self.trainer.is_global_zero:
            with open(
                os.path.join(self.trainer.default_root_dir, "reinitialized_heads.txt"),
                "a",
            ) as f:
                f.write(f"{len(reinitialized_heads)}\n")

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.ndim < 2 or any(
                    nd_keyword in name.lower()
                    for nd_keyword in ["bias", "layernorm", "norm", "embedding", "ln_"]
                ):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = OPTIMIZER_IMPLEMENTATION(
            optim_groups,
            lr=self.learning_rate,
            betas=(self.betas[0], self.betas[1]),
            eps=self.config["optimizer"]["params"]["eps"],
        )

        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(self.warmup_ratio * self.lr_decay_iters),
            num_training_steps=self.lr_decay_iters,
            min_lr=self.min_lr,
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler_config]


class PythiaModel:
    @staticmethod
    def from_config(config):
        model = PythiaLightningModule(config=config)
        return model
