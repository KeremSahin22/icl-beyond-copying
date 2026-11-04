import math
import torch
from torch.utils.data import Sampler
from typing import Iterator


class ResumableDistributedSampler(Sampler[int]):
    def __init__(
        self,
        dataset_len: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if not isinstance(dataset_len, int) or dataset_len < 0:
            raise ValueError("dataset_len must be non-negative int")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        self.dataset_len = dataset_len
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last

        if self.drop_last and self.dataset_len % self.num_replicas != 0:
            self.num_samples = math.floor(self.dataset_len / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.dataset_len / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "start_index": int(self.start_index),
            "seed": int(self.seed),
            "shuffle": bool(self.shuffle),
            "drop_last": bool(self.drop_last),
            "num_replicas": int(self.num_replicas),
            "rank": int(self.rank),
            "dataset_len": int(self.dataset_len),
        }

    def load_state_dict(self, state: dict) -> None:
        if state is None:
            raise ValueError("State is trying to be loaded but is None")
        if state["rank"] != self.rank or state["num_replicas"] != self.num_replicas:
            raise ValueError(
                "State is trying to be loaded but is for a different rank or num_replicas"
            )

        self.epoch = int(state["epoch"])
        self.start_index = int(state["start_index"])

        self.seed = int(state["seed"])
        self.shuffle = bool(state["shuffle"])

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()

            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_len, generator=g).tolist()
        else:
            indices = list(range(self.dataset_len))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]

        start = min(self.start_index, len(indices))
        indices = indices[start:]

        return iter(indices)

    def __len__(self) -> int:
        remaining = self.num_samples - int(self.start_index)
        if remaining < 0:
            remaining = 0
        return remaining
