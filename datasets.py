from torch.utils.data import Dataset
import numpy as np
import pillow_heif
import glob
from PIL import Image
import os
from typing import List

# Register HEIC opener
pillow_heif.register_heif_opener()

class CombinedDataset(Dataset):
    def __init__(self, data_folder: str, target_split_samples: List[str], source_split_samples: List[str], transform=None):
        self.transform = transform

        self.target_samples = []
        self.source_samples = []

        self.target_class_samples = {}
        self.source_class_samples = {}

        self.class_to_idx = {}
        self.idx_to_class = {}

        self._initialize_mappings(data_folder)
        self._populate_samples(
            split_samples=target_split_samples, domain="target"
        )
        self._populate_samples(
            split_samples=source_split_samples, domain="source"
        )

    def _initialize_mappings(self, data_folder: str):
        # make sure label <-> digit pairing is the same in all cases 
        class_folders = sorted(glob.glob(os.path.join(data_folder, f"*")))
        for idx, folder in enumerate(class_folders):
            class_name = os.path.basename(folder)
            
            self.target_class_samples[class_name] = []
            self.source_class_samples[class_name] = []

            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name

    def _populate_samples(self, split_samples: List[str], domain:str = "target"):
        samples = self.target_samples if domain=="target" else self.source_samples
        class_samples = self.target_class_samples if domain == "target" else self.source_class_samples

        for sample_path in split_samples:
            class_name = os.path.basename(os.path.dirname(sample_path))

            if class_name not in class_samples:
                raise ValueError(f"Class {class_name} not found in initialized class samples!")

            class_samples[class_name].append(sample_path)
            samples.append(
                (sample_path, self.class_to_idx[class_name])
            )

    def get_target_size(self):
        return len(self.target_samples)
    
    def get_source_size(self):
        return len(self.source_samples)

    def __len__(self):
        return len(self.target_samples)
    
    def __getitem__(self, index: int):
        # Get base image and label
        target_image_path, label = self.target_samples[index]

        # Get all aux indices for this label
        class_name = self.idx_to_class[label]
        source_indices = self.source_class_samples[class_name]
        source_sample_path = np.random.choice(source_indices)

        with Image.open(target_image_path) as img:
            target_image = img.copy()

        with Image.open(source_sample_path) as img:
            source_image = img.copy()

        if self.transform is not None:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return target_image, source_image, float(label)
