from torch.utils.data import Dataset
import numpy as np
import pillow_heif
import glob
from PIL import Image
import os
import logging
from typing import List

# Register HEIC opener
pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)

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
        self._validate_and_log_classes()

    def get_class_folders(self, root_dir):
        result = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if filenames:
                result.append(dirpath)

        return sorted(result)

    def _initialize_mappings(self, data_folder: str):
        # make sure label <-> digit pairing is the same in all cases 
        class_folders = self.get_class_folders(data_folder)

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

    def _validate_and_log_classes(self):
        logger.info(
            f"Loaded {len(self.class_to_idx)} classes -- "
            f"target samples: {len(self.target_samples)}, source samples: {len(self.source_samples)}"
        )

        empty_source_classes = []
        for class_name in self.class_to_idx:
            target_count = len(self.target_class_samples[class_name])
            source_count = len(self.source_class_samples[class_name])
            logger.debug(f"\tClass '{class_name}': target={target_count}, source={source_count}")

            if source_count == 0:
                empty_source_classes.append(class_name)

        if empty_source_classes:
            raise ValueError(
                f"The following classes have zero source samples, which would break "
                f"random source sampling in __getitem__: {empty_source_classes}"
            )

    def get_target_size(self):
        return len(self.target_samples)
    
    def get_source_size(self):
        return len(self.source_samples)

    def __len__(self):
        return len(self.target_samples)
    
    def _load_image(self, image_path: str):
        try:
            with Image.open(image_path) as img:
                return img.copy()
        except Exception:
            logger.error(f"Failed to load image at {image_path}")
            raise

    def __getitem__(self, index: int):
        # Get base image and label
        target_image_path, label = self.target_samples[index]

        # Get all aux indices for this label
        class_name = self.idx_to_class[label]
        source_indices = self.source_class_samples[class_name]
        source_sample_path = np.random.choice(source_indices)

        target_image = self._load_image(target_image_path)
        source_image = self._load_image(source_sample_path)

        if self.transform is not None:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        return target_image, source_image, float(label)
