import torch
import random
from torch.utils.data.sampler import Sampler
from typing import Dict, List, Iterator, Optional
from datasets import CombinedDataset



class PureBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True, drop_last=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Remap original indices to positions in IndexedDataset.indices
        original_to_remapped = {orig_idx: remap_idx for remap_idx, orig_idx in enumerate(data_source.base_dataset.indices)}
        self.class_samples = {}
        for class_label, orig_indices in data_source.base_dataset.class_samples.items():
            remapped_indices = [original_to_remapped[idx] for idx in orig_indices if idx in original_to_remapped]
            self.class_samples[class_label] = remapped_indices

    def __iter__(self):
        all_batches = []
        max_index = len(self.data_source) - 1

        for class_label, sample_indices in self.class_samples.items():
            if self.shuffle:
                random.shuffle(sample_indices)

            for i in range(0, len(sample_indices), self.batch_size):
                batch = sample_indices[i:i + self.batch_size]
                if not self.drop_last or len(batch) == self.batch_size:
                    if any(idx > max_index for idx in batch):
                        raise IndexError(f"Batch contains invalid indices (max={max_index})")
                    all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        total = 0
        for sample_indices in self.class_samples.values():
            num_batches = len(sample_indices) // self.batch_size
            if not self.drop_last and (len(sample_indices) % self.batch_size != 0):
                num_batches += 1
            total += num_batches
        return total
