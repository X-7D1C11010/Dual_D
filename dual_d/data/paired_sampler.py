"""Class-paired source/target batch sampler.

Module purpose:
    Yield source and target batches whose samples are paired by class id. This
    preserves the class-aware assumption used by tensor alignment, contrastive
    loss, and bidirectional feature translation.

Public interface:
    - PairedClassSampler(source_dataset, target_dataset, batch_size)

Usage:
    >>> paired_loader = PairedClassSampler(src_ds, tgt_ds, batch_size=32)
    >>> for src_batch, tgt_batch in paired_loader:
    ...     pass
"""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import DefaultDict, Dict, Iterable, Iterator, List, Tuple

from torch.utils.data import DataLoader, Dataset, Sampler


class _PairedIndexDataset(Dataset):
    """Resolve a sampled source/target index pair inside DataLoader workers."""

    def __init__(self, source_dataset, target_dataset) -> None:
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self) -> int:
        return max(len(self.source_dataset), len(self.target_dataset))

    def __getitem__(self, index_pair: Tuple[int, int]):
        source_index, target_index = index_pair
        return self.source_dataset[source_index], self.target_dataset[target_index]


class _PairedBatchIndexSampler(Sampler[List[Tuple[int, int]]]):
    """Delegate class-aware index generation while DataLoader fetches samples."""

    def __init__(
        self,
        source_indices: Dict[int, List[int]],
        target_indices: Dict[int, List[int]],
        classes: List[int],
        source_size: int,
        target_size: int,
        batch_size: int,
        min_steps_per_epoch: int,
    ) -> None:
        self.source_indices = source_indices
        self.target_indices = target_indices
        self.classes = classes
        self.source_size = int(source_size)
        self.target_size = int(target_size)
        self.batch_size = int(batch_size)
        self.min_steps_per_epoch = int(min_steps_per_epoch)

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        for class_id in self.classes:
            random.shuffle(self.source_indices[class_id])
            random.shuffle(self.target_indices[class_id])
        source_cursors = {class_id: 0 for class_id in self.classes}
        target_cursors = {class_id: 0 for class_id in self.classes}

        def next_index(index_map, cursor_map, class_id: int) -> int:
            pool = index_map[class_id]
            cursor = cursor_map[class_id]
            if cursor >= len(pool):
                random.shuffle(pool)
                cursor = 0
            index = pool[cursor]
            cursor_map[class_id] = cursor + 1
            return index

        for _ in range(len(self)):
            batch_classes = random.choices(self.classes, k=self.batch_size)
            yield [
                (
                    next_index(self.source_indices, source_cursors, class_id),
                    next_index(self.target_indices, target_cursors, class_id),
                )
                for class_id in batch_classes
            ]

    def __len__(self) -> int:
        natural_batches = max(
            1,
            math.ceil(min(self.source_size, self.target_size) / self.batch_size),
        )
        return max(natural_batches, self.min_steps_per_epoch)


class PairedClassSampler:
    """Iterable paired-batch sampler for source and target datasets."""

    def __init__(
        self,
        source_dataset,
        target_dataset,
        batch_size: int,
        min_steps_per_epoch: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.min_steps_per_epoch = max(int(min_steps_per_epoch), 1)
        self.num_workers = max(int(num_workers), 0)

        self.source_indices = self._build_index(source_dataset.labels)
        self.target_indices = self._build_index(target_dataset.labels)
        self.classes = sorted(set(self.source_indices) & set(self.target_indices))
        if not self.classes:
            raise RuntimeError("No common class labels found between source and target.")

        self._batch_sampler = _PairedBatchIndexSampler(
            source_indices=self.source_indices,
            target_indices=self.target_indices,
            classes=self.classes,
            source_size=len(source_dataset),
            target_size=len(target_dataset),
            batch_size=self.batch_size,
            min_steps_per_epoch=self.min_steps_per_epoch,
        )
        loader_options = {
            "dataset": _PairedIndexDataset(source_dataset, target_dataset),
            "batch_sampler": self._batch_sampler,
            "num_workers": self.num_workers,
            "pin_memory": bool(pin_memory),
        }
        if self.num_workers > 0:
            loader_options.update(
                {
                    "persistent_workers": True,
                    "prefetch_factor": 2,
                }
            )
        self._loader = DataLoader(**loader_options)

    @staticmethod
    def _build_index(labels: Iterable[int]) -> Dict[int, List[int]]:
        """Build class id to sample index mapping."""

        indices: DefaultDict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            indices[int(label)].append(idx)
        return dict(indices)

    def __iter__(self):
        """Yield paired source and target batches."""

        return iter(self._loader)

    def __len__(self) -> int:
        """Return number of batches per epoch."""

        return len(self._batch_sampler)
