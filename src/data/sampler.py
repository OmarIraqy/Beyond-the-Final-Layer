"""Identity-balanced and class-aware samplers for ReID training."""

import copy
import math
import random
from collections import defaultdict
from typing import List

import numpy as np
from torch.utils.data import Sampler

from .dataset import Sample


class PKSampler(Sampler):
    """Sample P identities, then K instances for each identity per batch.

    Total batch size = P * K.  The caller should set DataLoader batch_size
    to match (or leave it to the dataloader builder).
    """

    def __init__(self, samples: List[Sample], num_instances: int = 4, batch_size: int = 64):
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.num_pids_per_batch = batch_size // num_instances

        # Build pid -> sample indices mapping
        self.pid_to_indices = defaultdict(list)
        for i, s in enumerate(samples):
            self.pid_to_indices[s.pid].append(i)
        self.pids = list(self.pid_to_indices.keys())

        # Estimate length: each PID contributes ceil(max(num, K) / K) * K indices
        # (iter processes in chunks of K, padding the last chunk if needed)
        self._length = 0
        for pid in self.pids:
            num = len(self.pid_to_indices[pid])
            self._length += math.ceil(max(num, self.num_instances) / self.num_instances) * self.num_instances

    def __iter__(self):
        batch_indices = []
        pid_list = copy.deepcopy(self.pids)
        random.shuffle(pid_list)

        for pid in pid_list:
            indices = copy.deepcopy(self.pid_to_indices[pid])
            if len(indices) < self.num_instances:
                indices = np.random.choice(indices, size=self.num_instances, replace=True).tolist()
            else:
                random.shuffle(indices)

            # Yield in chunks of num_instances
            for start in range(0, len(indices), self.num_instances):
                chunk = indices[start : start + self.num_instances]
                if len(chunk) < self.num_instances:
                    chunk += np.random.choice(
                        self.pid_to_indices[pid],
                        size=self.num_instances - len(chunk),
                        replace=True,
                    ).tolist()
                batch_indices.extend(chunk)

                if len(batch_indices) >= self.batch_size:
                    yield from batch_indices[: self.batch_size]
                    batch_indices = batch_indices[self.batch_size :]

        # Yield remaining
        if batch_indices:
            yield from batch_indices

    def __len__(self):
        return self._length


class ClassBucketPKSampler(Sampler):
    """Emit full PK batches from one class at a time."""

    def __init__(self, samples: List[Sample], num_instances: int = 4, batch_size: int = 64):
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.num_pids_per_batch = max(1, batch_size // num_instances)

        self.class_pid_indices = defaultdict(lambda: defaultdict(list))
        for i, sample in enumerate(samples):
            self.class_pid_indices[sample.class_label][sample.pid].append(i)

        self.classes = list(self.class_pid_indices.keys())
        self._length = self._estimate_length()

    def _sample_pid_chunks(self, indices: List[int]) -> List[List[int]]:
        working = copy.deepcopy(indices)
        if len(working) < self.num_instances:
            working = np.random.choice(working, size=self.num_instances, replace=True).tolist()
        else:
            random.shuffle(working)

        chunks = []
        for start in range(0, len(working), self.num_instances):
            chunk = working[start : start + self.num_instances]
            if len(chunk) < self.num_instances:
                chunk += np.random.choice(
                    indices,
                    size=self.num_instances - len(chunk),
                    replace=True,
                ).tolist()
            chunks.append(chunk)
        return chunks

    def _estimate_length(self) -> int:
        total_batches = 0
        for pid_indices in self.class_pid_indices.values():
            total_chunks = 0
            for indices in pid_indices.values():
                total_chunks += math.ceil(max(len(indices), self.num_instances) / self.num_instances)
            if total_chunks > 0:
                total_batches += math.ceil(total_chunks / self.num_pids_per_batch)
        return total_batches * self.batch_size

    def __iter__(self):
        all_batches = []

        for cls in self.classes:
            pid_chunks = []
            for indices in self.class_pid_indices[cls].values():
                pid_chunks.extend(self._sample_pid_chunks(indices))

            if not pid_chunks:
                continue

            random.shuffle(pid_chunks)
            for start in range(0, len(pid_chunks), self.num_pids_per_batch):
                batch_chunks = pid_chunks[start : start + self.num_pids_per_batch]
                if len(batch_chunks) < self.num_pids_per_batch:
                    batch_chunks += random.choices(
                        pid_chunks,
                        k=self.num_pids_per_batch - len(batch_chunks),
                    )
                batch = []
                for chunk in batch_chunks:
                    batch.extend(chunk)
                all_batches.append(batch)

        random.shuffle(all_batches)
        for batch in all_batches:
            yield from batch

    def __len__(self):
        return self._length


class ClassAwarePKSampler(Sampler):
    """Balance across object classes first, then PK within each class.

    Helps with class imbalance (e.g. trafficsignal >> Container).
    Each batch samples from classes roughly equally, then applies PK
    within each class subset.
    """

    def __init__(self, samples: List[Sample], num_instances: int = 4, batch_size: int = 64):
        self.num_instances = num_instances
        self.batch_size = batch_size

        # Build class -> pid -> indices
        self.class_pid_indices = defaultdict(lambda: defaultdict(list))
        for i, s in enumerate(samples):
            self.class_pid_indices[s.class_label][s.pid].append(i)

        self.classes = list(self.class_pid_indices.keys())
        self.num_classes = len(self.classes)
        # PIDs per class per batch
        pids_per_class = max(1, (batch_size // num_instances) // self.num_classes)
        self.pids_per_class = pids_per_class

        self._length = len(samples)

    def __iter__(self):
        all_indices = []

        # For each class, build PK batches
        class_iters = {}
        for cls in self.classes:
            pid_indices = self.class_pid_indices[cls]
            pids = list(pid_indices.keys())
            random.shuffle(pids)
            class_iters[cls] = (pids, pid_indices)

        # Round-robin across classes
        exhausted = set()
        pid_pointers = {cls: 0 for cls in self.classes}

        while len(exhausted) < self.num_classes:
            batch = []
            for cls in self.classes:
                if cls in exhausted:
                    continue
                pids, pid_indices = class_iters[cls]
                for _ in range(self.pids_per_class):
                    if pid_pointers[cls] >= len(pids):
                        exhausted.add(cls)
                        break
                    pid = pids[pid_pointers[cls]]
                    pid_pointers[cls] += 1
                    indices = list(pid_indices[pid])
                    if len(indices) < self.num_instances:
                        indices = np.random.choice(
                            indices, size=self.num_instances, replace=True
                        ).tolist()
                    else:
                        random.shuffle(indices)
                        indices = indices[: self.num_instances]
                    batch.extend(indices)

            if batch:
                all_indices.extend(batch)

        yield from all_indices

    def __len__(self):
        return self._length
