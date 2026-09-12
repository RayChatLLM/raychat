# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random
from collections import Counter
from typing import Protocol

from ..core.adapter import DataInst
from ..core.data_loader import DataId, DataLoader
from ..type_support import override


class SamplingState(Protocol):
    """Expose the iteration number used to schedule minibatches."""

    @property
    def i(self) -> int:
        """Current proposal iteration."""


class BatchSampler(Protocol[DataId, DataInst]):
    def next_minibatch_ids(
        self,
        loader: DataLoader[DataId, DataInst],
        state: SamplingState,
    ) -> list[DataId]: ...


class BatchSamplerFactory(Protocol):
    """Create a fresh sampler whose identifier type follows its run's loader."""

    def __call__(
        self,
        *,
        minibatch_size: int,
        rng: random.Random,
    ) -> BatchSampler[DataId, DataInst]:
        """Create a sampler with independent state for one optimization run."""


class EpochShuffledBatchSampler(BatchSampler[DataId, DataInst]):
    """Mirrors the original batching logic:
    - Shuffle ids each epoch
    - Pad to minibatch size with least frequent ids
    - Deterministic via state.rng1
    """

    def __init__(self, minibatch_size: int, rng: random.Random | None = None) -> None:
        self.minibatch_size = minibatch_size
        self.shuffled_ids: list[DataId] = []
        self.epoch = -1
        self.id_freqs: Counter[DataId] = Counter()
        self.last_trainset_size = 0
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    def _update_shuffled(self, loader: DataLoader[DataId, DataInst]) -> None:
        all_ids = list(loader.all_ids())
        trainset_size = len(loader)
        self.last_trainset_size = trainset_size

        if trainset_size == 0:
            self.shuffled_ids = []
            self.id_freqs = Counter()
            return

        self.shuffled_ids = list(all_ids)
        self.rng.shuffle(self.shuffled_ids)
        self.id_freqs = Counter(self.shuffled_ids)

        mod = trainset_size % self.minibatch_size
        num_to_pad = (self.minibatch_size - mod) if mod != 0 else 0
        if num_to_pad > 0:
            for _ in range(num_to_pad):
                selected_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_ids.append(selected_id)
                self.id_freqs[selected_id] += 1

    @override
    def next_minibatch_ids(
        self,
        loader: DataLoader[DataId, DataInst],
        state: SamplingState,
    ) -> list[DataId]:
        trainset_size = len(loader)
        if trainset_size == 0:
            error_message = "Cannot sample a minibatch from an empty loader."
            raise ValueError(error_message)

        base_idx = state.i * self.minibatch_size
        curr_epoch = (
            0 if self.epoch == -1 else base_idx // max(len(self.shuffled_ids), 1)
        )

        needs_refresh = (
            not self.shuffled_ids
            or trainset_size != self.last_trainset_size
            or curr_epoch > self.epoch
        )
        if needs_refresh:
            self.epoch = curr_epoch
            self._update_shuffled(loader)

        assert len(self.shuffled_ids) >= self.minibatch_size
        assert len(self.shuffled_ids) % self.minibatch_size == 0

        base_idx = base_idx % len(self.shuffled_ids)
        end_idx = base_idx + self.minibatch_size
        assert end_idx <= len(self.shuffled_ids)
        return self.shuffled_ids[base_idx:end_idx]
