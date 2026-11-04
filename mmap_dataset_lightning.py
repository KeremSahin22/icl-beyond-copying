import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from samplers import ResumableDistributedSampler
import numpy as np
from mmap_dataset import MMapIndexedDataset


class MMapPythiaDataset(Dataset):
    def __init__(
        self,
        path,
        start_idx=0,
        end_idx=None,
        equiv_lookup=None,
        mask_bigram_with_self=False,
        mask_bigram_loss=False,
        mask_bigram_loss_with_self_included=False,
        mask_unigram_loss=False,
        seed=1234,
    ):
        self.dataset = MMapIndexedDataset(path, skip_warmup=True)
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(self.dataset) - 1
        self.seq_length = 2049
        self.equiv_lookup = equiv_lookup
        self.mask_bigram_with_self = mask_bigram_with_self
        self.mask_bigram_loss = mask_bigram_loss
        self.mask_bigram_loss_with_self_included = mask_bigram_loss_with_self_included
        self.mask_unigram_loss = mask_unigram_loss
        self.seed = seed

    def __len__(self):
        return self.end_idx - self.start_idx + 1

    def _compute_bigram_mask(self, tokens):
        S = len(tokens)
        loss_mask = torch.ones(S, dtype=torch.bool)

        if S <= 2 or self.equiv_lookup is None:
            return loss_mask

        first_tokens = tokens[:-1]
        second_tokens = tokens[1:]

        equiv_first = self.equiv_lookup[
            first_tokens.unsqueeze(1), first_tokens.unsqueeze(0)
        ]
        equiv_second = self.equiv_lookup[
            second_tokens.unsqueeze(1), second_tokens.unsqueeze(0)
        ]

        bigram_equiv = equiv_first & equiv_second

        valid_mask = torch.tril(torch.ones(S - 1, S - 1, dtype=torch.bool), diagonal=-1)

        bigram_equiv = bigram_equiv & valid_mask

        has_match = bigram_equiv.any(dim=1)

        match_positions = torch.where(has_match)[0]

        if len(match_positions) > 0:
            loss_mask[match_positions + 1] = False
            if self.mask_bigram_with_self:
                raise ValueError("mask_bigram_with_self has not been tested yet")
                loss_mask[torch.clamp(match_positions, min=0)] = False

        return loss_mask

    def bigram_loss_mask_exact(self, tokens):
        S = len(tokens)
        loss_mask = torch.ones(S, dtype=torch.bool)

        if S <= 2:
            return loss_mask

        first_tokens = tokens[:-1]
        second_tokens = tokens[1:]

        bigram_first = first_tokens.unsqueeze(1) == first_tokens.unsqueeze(
            0
        )
        bigram_second = second_tokens.unsqueeze(1) == second_tokens.unsqueeze(
            0
        )

        bigram_equiv = bigram_first & bigram_second

        valid_mask = torch.tril(torch.ones(S - 1, S - 1, dtype=torch.bool), diagonal=-1)

        bigram_equiv = bigram_equiv & valid_mask

        has_match = bigram_equiv.any(dim=1)

        match_positions = torch.where(has_match)[0]

        if len(match_positions) > 0:
            loss_mask[match_positions + 1] = (
                False
            )
            if getattr(self, "mask_bigram_with_self", False):
                raise ValueError("mask_bigram_with_self has not been tested yet")
                loss_mask[torch.clamp(match_positions, min=0)] = False

        return loss_mask

    def unigram_loss_mask(self, tokens):
        S = len(tokens)
        loss_mask = torch.ones(S, dtype=torch.bool)
        seen = set()
        for i, tok in enumerate(tokens.tolist()):
            if tok in seen:
                loss_mask[i] = False
            else:
                seen.add(tok)
        return loss_mask

    def __getitem__(self, idx):
        absolute_idx = idx + self.start_idx
        sequence = self.dataset[absolute_idx]

        if isinstance(sequence, np.ndarray) and len(sequence.shape) == 2:
            sequence = sequence[0]

        tokens = torch.tensor(sequence, dtype=torch.long)
        input_ids = tokens[:-1]
        labels = tokens[:-1]

        if self.equiv_lookup is not None:
            loss_mask = self._compute_bigram_mask(input_ids)
            labels = labels.clone()
            labels[~loss_mask] = -100
        elif self.mask_bigram_loss:
            loss_mask = self.bigram_loss_mask_exact(input_ids)
            labels = labels.clone()
            labels[~loss_mask] = -100
        elif self.mask_unigram_loss:
            loss_mask = self.unigram_loss_mask(input_ids)
            labels = labels.clone()
            labels[~loss_mask] = -100

        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class MMapPythiaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path,
        valid_path=None,
        batch_size=16,
        num_workers=4,
        train_valid_split=0.95,
        single_file=True,
        shuffle=False,
        seed=1234,
        equiv_lookup=None,
        mask_bigram_with_self=False,
        mask_bigram_loss=False,
        mask_bigram_loss_with_self_included=False,
        mask_unigram_loss=False,
        prefetch_factor=2,
        resume_sampler_state=None,
    ):
        super().__init__()
        self.train_path = train_path
        self.valid_path = valid_path if valid_path else train_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.train_valid_split = train_valid_split
        self.single_file = single_file
        self.shuffle = shuffle
        self.seed = seed
        self.equiv_lookup = equiv_lookup
        self.mask_bigram_with_self = mask_bigram_with_self
        self.mask_bigram_loss = mask_bigram_loss
        self.mask_bigram_loss_with_self_included = mask_bigram_loss_with_self_included
        self.mask_unigram_loss = mask_unigram_loss
        self.resume_sampler_state = resume_sampler_state

        self.train_dataset = None
        self.valid_dataset = None

    def prepare_data(self):
        train_idx_path = self.train_path + ".idx"

        if not os.path.exists(train_idx_path):
            raise FileNotFoundError(f"Dataset not found at {self.train_path}")

        if not self.single_file and not os.path.exists(self.valid_path + ".idx"):
            raise FileNotFoundError(
                f"Validation dataset not found at {self.valid_path}"
            )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.single_file:
                temp_dataset = MMapIndexedDataset(self.train_path, skip_warmup=True)
                dataset_size = len(temp_dataset)
                train_size = int(dataset_size * self.train_valid_split)

                self.train_dataset = MMapPythiaDataset(
                    self.train_path,
                    start_idx=0,
                    end_idx=train_size - 1,
                    equiv_lookup=self.equiv_lookup,
                    mask_bigram_with_self=self.mask_bigram_with_self,
                    mask_bigram_loss=self.mask_bigram_loss,
                    mask_bigram_loss_with_self_included=self.mask_bigram_loss_with_self_included,
                    mask_unigram_loss=self.mask_unigram_loss,
                    seed=self.seed,
                )

                self.valid_dataset = MMapPythiaDataset(
                    self.train_path, start_idx=train_size, end_idx=dataset_size - 1
                )

                self.valid_dataset_with_masking = MMapPythiaDataset(
                    self.train_path,
                    start_idx=train_size,
                    end_idx=dataset_size - 1,
                    equiv_lookup=self.equiv_lookup,
                    mask_bigram_with_self=self.mask_bigram_with_self,
                    mask_bigram_loss=self.mask_bigram_loss,
                    mask_bigram_loss_with_self_included=self.mask_bigram_loss_with_self_included,
                    mask_unigram_loss=self.mask_unigram_loss,
                    seed=self.seed,
                )

            else:
                self.train_dataset = MMapPythiaDataset(
                    self.train_path,
                    equiv_lookup=self.equiv_lookup,
                    mask_bigram_with_self=self.mask_bigram_with_self,
                    mask_bigram_loss=self.mask_bigram_loss,
                    mask_bigram_loss_with_self_included=self.mask_bigram_loss_with_self_included,
                    mask_unigram_loss=self.mask_unigram_loss,
                )
                self.valid_dataset = MMapPythiaDataset(self.valid_path)
                self.valid_dataset_with_masking = MMapPythiaDataset(
                    self.valid_path,
                    equiv_lookup=self.equiv_lookup,
                    mask_bigram_with_self=self.mask_bigram_with_self,
                    mask_bigram_loss=self.mask_bigram_loss,
                    mask_bigram_loss_with_self_included=self.mask_bigram_loss_with_self_included,
                    mask_unigram_loss=self.mask_unigram_loss,
                )

    def train_dataloader(self):
        if self.trainer is not None and hasattr(self.trainer, "world_size"):
            num_replicas = int(self.trainer.world_size)

        else:
            num_replicas = 1
        if self.trainer is not None and hasattr(self.trainer, "global_rank"):
            rank = int(self.trainer.global_rank)

        else:
            rank = 0

        sampler = ResumableDistributedSampler(
            dataset_len=len(self.train_dataset),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=False,
        )

        if self.resume_sampler_state:
            sampler.load_state_dict(
                {
                    "epoch": self.resume_sampler_state["epoch"],
                    "start_index": self.resume_sampler_state["samples_seen_in_epoch"],
                    "seed": self.resume_sampler_state["seed"],
                    "shuffle": self.shuffle,
                    "rank": rank,
                    "num_replicas": num_replicas,
                }
            )
            self.resume_sampler_state = None

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader_with_masking(self):
        return DataLoader(
            self.valid_dataset_with_masking,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


def setup_pythia_data(config, equiv_lookup=None, resume_sampler_state=None):
    train_path = config["train-data-paths"][0]
    valid_path = config["valid-data-paths"][0]

    train_path = train_path.strip()
    valid_path = valid_path.strip()

    for ext in [".bin", ".idx"]:
        if train_path.endswith(ext):
            train_path = train_path[: -len(ext)]
        if valid_path.endswith(ext):
            valid_path = valid_path[: -len(ext)]

    single_file = config["use_single_file"]
    shuffle = config["shuffle"]

    data_module = MMapPythiaDataModule(
        train_path=train_path,
        valid_path=valid_path if not single_file else None,
        batch_size=config.get("train_micro_batch_size_per_gpu", 16),
        num_workers=config.get("num_workers", 4),
        prefetch_factor=config.get("prefetch_factor", 2),
        train_valid_split=config.get("train_valid_split", 0.95),
        single_file=single_file,
        shuffle=shuffle,
        seed=config.get("seed", 1234),
        equiv_lookup=equiv_lookup,
        mask_bigram_with_self=config.get("mask_bigram_loss_with_self_included", False),
        mask_bigram_loss=config["mask_bigram_loss"],
        mask_bigram_loss_with_self_included=config[
            "mask_bigram_loss_with_self_included"
        ],
        mask_unigram_loss=config.get("mask_unigram_loss", False),
        resume_sampler_state=resume_sampler_state,
    )

    return data_module
