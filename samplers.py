import torch
import random
from torch.utils.data.sampler import Sampler
from datasets import CombinedDataset

class PureBatchSampler(Sampler):
    def __init__(self, data_source: CombinedDataset, batch_size:int, shuffle=True, drop_last=True, min_batch_size:int=2):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        # divergences need at least this many samples, so a smaller leftover batch is merged into the previous one
        self.min_batch_size = min_batch_size

        # Use target domain for pure batches
        self.class_samples = data_source.target_class_samples

        # Map file path to index in target_samples
        self.path_to_index = {path: idx for idx, (path, _) in enumerate(data_source.target_samples)}

    def _chunk(self, indices):
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

        if self.drop_last:
            return [b for b in batches if len(b) == self.batch_size]

        if len(batches) > 1 and len(batches[-1]) < self.min_batch_size:
            batches[-2].extend(batches.pop())
        elif batches and len(batches[-1]) < self.min_batch_size:
            batches.pop()

        return batches

    def __iter__(self):
        all_batches = []
        max_index = len(self.data_source) - 1

        for class_label, sample_paths in self.class_samples.items():
            indices = [self.path_to_index[path] for path in sample_paths]
            if self.shuffle:
                random.shuffle(indices)

            for batch in self._chunk(indices):
                # Final safety check
                if any(idx > max_index for idx in batch):
                    raise IndexError(f"Batch contains invalid indices (max={max_index})")
                all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        return sum(len(self._chunk(list(range(len(sample_paths))))) for sample_paths in self.class_samples.values())
