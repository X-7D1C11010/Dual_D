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
import random
from typing import DefaultDict, Dict, Iterable, List

import torch


class PairedClassSampler:
    """Iterable paired-batch sampler for source and target datasets."""

    def __init__(self, source_dataset, target_dataset, batch_size: int):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.source_indices = self._build_index(source_dataset.labels)
        self.target_indices = self._build_index(target_dataset.labels)
        self.classes = sorted(set(self.source_indices) & set(self.target_indices))
        if not self.classes:
            raise RuntimeError("No common class labels found between source and target.")

    @staticmethod
    def _build_index(labels: Iterable[int]) -> Dict[int, List[int]]:
        """Build class id to sample index mapping."""

        indices: DefaultDict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            indices[int(label)].append(idx)
        return dict(indices)

    def __iter__(self):
        """Yield paired source and target batches."""

        for class_id in self.classes:
            random.shuffle(self.source_indices[class_id])
            random.shuffle(self.target_indices[class_id])
        source_cursors = {class_id: 0 for class_id in self.classes}
        target_cursors = {class_id: 0 for class_id in self.classes}

        def next_index(index_map, cursor_map, class_id: int) -> int:
            """Cycle through a class pool before reusing any sample."""

            pool = index_map[class_id]
            cursor = cursor_map[class_id]
            if cursor >= len(pool):
                random.shuffle(pool)
                cursor = 0
            index = pool[cursor]
            cursor_map[class_id] = cursor + 1
            return index

        num_batches = min(len(self.source_dataset), len(self.target_dataset)) // self.batch_size
        for _ in range(num_batches):
            batch_classes = random.choices(self.classes, k=self.batch_size)
            source_batch_indices = []
            target_batch_indices = []
            for class_id in batch_classes:
                source_batch_indices.append(
                    next_index(self.source_indices, source_cursors, class_id)
                )
                target_batch_indices.append(
                    next_index(self.target_indices, target_cursors, class_id)
                )

            source_batch = torch.utils.data.default_collate(
                [self.source_dataset[idx] for idx in source_batch_indices]
            )
            target_batch = torch.utils.data.default_collate(
                [self.target_dataset[idx] for idx in target_batch_indices]
            )
            yield source_batch, target_batch

    def __len__(self) -> int:
        """Return number of batches per epoch."""

        return min(len(self.source_dataset), len(self.target_dataset)) // self.batch_size
