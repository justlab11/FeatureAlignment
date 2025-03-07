import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import cv2
from sklearn.model_selection import train_test_split as tts
import pillow_heif
from torchvision import transforms
import glob
from PIL import Image
import os

# Register HEIC opener
pillow_heif.register_heif_opener()

class HEIFFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.class_samples = {}
        
        self._populate_samples()

    def _populate_samples(self):
        # Find all HEIF files recursively
        for file_path in glob.glob(os.path.join(self.root, "**/*.heif"), recursive=True):
            # Extract class from directory path
            class_name = os.path.basename(os.path.dirname(file_path))
            
            # Add sample to list
            self.samples.append((file_path, class_name))
            
            # Add to class_samples dictionary
            if class_name not in self.class_samples:
                self.class_samples[class_name] = []
            self.class_samples[class_name].append(len(self.samples) - 1)

    def update_samples(self, new_samples):
        self.samples = new_samples
        self.class_samples = {}
        for idx, (_, class_name) in enumerate(self.samples):
            if class_name not in self.class_samples:
                self.class_samples[class_name] = []
            self.class_samples[class_name].append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        
        # Load HEIC image
        image = Image.open(image_path)
        
        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
        
        label = float(label)

        return image, label

    def get_class_samples(self):
        return self.class_samples
    
class CombinedDataset(Dataset):
    def __init__(self, base_dataset: Subset, aux_dataset: Subset):
        self.base_dataset = base_dataset.dataset
        self.aux_dataset = aux_dataset.dataset
        self.unique_sources = True
        
        # Get class samples from both datasets
        self.base_class_samples = base_dataset.class_samples
        self.aux_class_samples = aux_dataset.class_samples
        
        # Store file paths for base and aux datasets
        self.base_file_paths = [self.base_dataset.samples[i][0] for i in range(len(self.base_dataset))]
        self.aux_file_paths = [self.aux_dataset.samples[i][0] for i in range(len(self.aux_dataset))]

    def __len__(self):
        return len(self.base_dataset)
    
    def toggle_unique_sources(self, value:bool):
        self.unique_sources = value

    def __getitem__(self, index):
        base_file_path = self.base_file_paths[index]
        base_image = Image.open(base_file_path)
        
        # Apply transform if provided
        if self.base_dataset.transform is not None:
            base_image = self.base_dataset.transform(base_image)
        
        label = self.base_dataset.samples[index][1]
        
        pair_selection = np.random.uniform()
        
        if self.unique_sources or pair_selection < 0.5:
            # print(True, pair_selection, self.unique_sources)
            # Ensure base and aux are from different sources
            aux_idx = np.random.choice(self.aux_class_samples[label])
            aux_file_path = self.aux_file_paths[aux_idx]
            aux_image = Image.open(aux_file_path)
            
            # Apply transform if provided
            if self.aux_dataset.transform is not None:
                aux_image = self.aux_dataset.transform(aux_image)
        elif pair_selection < 0.75:
            # Base/Base pair
            base_idx = np.random.choice(self.base_class_samples[label])
            aux_file_path = self.base_file_paths[base_idx]
            aux_image = Image.open(aux_file_path)
            
            # Apply transform if provided
            if self.base_dataset.transform is not None:
                aux_image = self.base_dataset.transform(aux_image)
        else:
            # Aux/Aux pair
            aux_idx1 = np.random.choice(self.aux_class_samples[label])
            aux_idx2 = np.random.choice(self.aux_class_samples[label])
            base_file_path = self.aux_file_paths[aux_idx1]
            aux_file_path = self.aux_file_paths[aux_idx2]
            base_image = Image.open(base_file_path)
            
            # Apply transform if provided
            if self.aux_dataset.transform is not None:
                base_image = self.aux_dataset.transform(base_image)
            aux_image = Image.open(aux_file_path)
            
            # Apply transform if provided
            if self.aux_dataset.transform is not None:
                aux_image = self.aux_dataset.transform(aux_image)
        
        label = float(label)

        return base_image, aux_image, label

class PairedMNISTDataset(Dataset):
    """
    Dataset for the paired MNIST digits. The dataset pairs a set of images with the same class from the two datasets.
    NOTE: THERE ARE 2 SAMPLES PER DRAW, MAKE SURE TO HALF THE BATCH SIZE!
    Parameters:
        base_images (np.ndarray) - the base images that are going to be the TRUE training objective.
        base_labels (np.ndarray) - the labels for the base images.
        aux_images (np.ndarray) - the auxiliary images created to improve classification accuracy.
        aux_labels (nd.ndarray) - the labels for the auxiliary images.
    Returns:
        Random pairs of images belonging to the same class in the order of base_sample, aux_sample, label
    """
    def __init__(
            self, 
            base_images: np.ndarray, 
            base_labels: np.ndarray, 
            aux_images: np.ndarray, 
            aux_labels: np.ndarray,
            img_size: int,
            in_memory: bool = False,
        ):
        
        if in_memory:
            resized_base_images = np.zeros((len(base_images), 3, img_size, img_size))
            for i, img in enumerate(base_images):
                new_img = cv2.resize(img.transpose(1,2,0), (img_size,img_size), interpolation=cv2.INTER_LINEAR)
                resized_base_images[i] = new_img.transpose(2,0,1)

            resized_aux_images = np.zeros((len(aux_images), 3, img_size, img_size))
            for i, img in enumerate(aux_images):
                new_img = cv2.resize(img.transpose(1,2,0), (img_size,img_size), interpolation=cv2.INTER_LINEAR)
                resized_aux_images[i] = new_img.transpose(2,0,1)

            self.base_images: torch.tensor = torch.from_numpy(resized_base_images).float() / 255.0
            self.base_labels: torch.tensor = torch.from_numpy(base_labels).long()

            self.aux_images: torch.tensor = torch.from_numpy(resized_aux_images).float() / 255.0
            self.aux_labels: torch.tensor = torch.from_numpy(aux_labels).long()
        
        else:
            self.base_images: torch.tensor = torch.from_numpy(base_images).float() / 255.0
            self.base_labels: torch.tensor = torch.from_numpy(base_labels).long()

            self.aux_images: torch.tensor = torch.from_numpy(aux_images).float() / 255.0
            self.aux_labels: torch.tensor = torch.from_numpy(aux_labels).long()

        self.base_indices_by_class = self._group_indices_by_class(self.base_labels)
        self.aux_indices_by_class = self._group_indices_by_class(self.aux_labels)

        self.unique_sources = False
        self.specific_class = None 

    def __len__(self):
        if self.specific_class is not None:
            return len(self.base_indices_by_class[self.specific_class])
        return len(self.base_images)
    
    def _group_indices_by_class(self, labels):
        indices_by_class = {i: [] for i in range(10)}

        for idx, label in enumerate(labels):
            indices_by_class[label.item()].append(idx)

        return indices_by_class
    
    def __getitem__(self, idx):
        if self.specific_class is not None:
            base_idx = self.base_indices_by_class[self.specific_class][idx]
            label = self.specific_class
        else:
            base_idx = idx
            label = self.base_labels[base_idx].item()

        base_sample = self.base_images[base_idx]

        if self.specific_class is not None:
            aux_idx = np.random.choice(self.aux_indices_by_class[self.specific_class])
        else:
            aux_idx = np.random.choice(self.aux_indices_by_class[label])

        aux_sample = self.aux_images[aux_idx]

        # pair_selection = np.random.uniform()
        # if pair_selection < .7 or self.unique_sources:
        #     base_sample = self.base_images[base_idx]
        #     if self.specific_class is not None:
        #         aux_idx = np.random.choice(self.aux_indices_by_class[self.specific_class])
        #     else:
        #         aux_idx = np.random.choice(self.aux_indices_by_class[label])
        #     aux_sample = self.aux_images[aux_idx]
        # elif pair_selection < .85:
        #     base_sample = self.base_images[base_idx]
        #     if self.specific_class is not None:
        #         aux_idx = np.random.choice(self.base_indices_by_class[self.specific_class])
        #     else:
        #         aux_idx = np.random.choice(self.base_indices_by_class[label])
        #     aux_sample = self.base_images[aux_idx]
        # else:
        #     if self.specific_class is not None:
        #         aux_idx1 = np.random.choice(self.aux_indices_by_class[self.specific_class])
        #         aux_idx2 = np.random.choice(self.aux_indices_by_class[self.specific_class])
        #     else:
        #         aux_idx1 = np.random.choice(self.aux_indices_by_class[label])
        #         aux_idx2 = np.random.choice(self.aux_indices_by_class[label])
        #     base_sample = self.aux_images[aux_idx1]
        #     aux_sample = self.aux_images[aux_idx2]

        return base_sample, aux_sample, label

class DatasetGenerator:
    def __init__(self, images, labels, subset_ratio=.3, base_ds="red", aux_ds="color", train=True):
        self.base_ds = base_ds
        self.aux_ds = aux_ds
        self.train = train

        if train:
            base_subset_size = int(images.shape[0]*subset_ratio)
        else:
            base_subset_size = images.shape[0]

        indices = np.random.choice(images.shape[0], images.shape[0], replace=False)
        self.base_images = images[indices[:base_subset_size]].copy()
        self.base_labels = labels[indices[:base_subset_size]].copy()
        
        self.aux_images = images[indices[base_subset_size:]].copy()
        self.aux_labels = labels[indices[base_subset_size:]].copy()

    def build_base_dataset(self):
        if self.base_ds == "red":
            return self._create_red_mnist(self.base_images), self.base_labels
        elif self.base_ds == "red shade":
            return self._create_red_shade_mnist(self.base_images), self.base_labels
        elif self.base_ds == "color":
            return self._create_colorful_subset(self.base_images, self.base_labels)
        elif self.base_ds == "skip":
            return self._create_skip_subset(self.base_images, self.base_labels)
        else:
            return np.stack([self.base_images] * 3, axis=1), self.base_labels

    def build_aux_dataset(self):
        if self.aux_ds == "red":
            return self._create_red_mnist(self.aux_images), self.aux_labels
        elif self.aux_ds == "red shade":
            return self._create_red_shade_mnist(self.aux_images), self.aux_labels
        if self.aux_ds == "color":
            return self._create_colorful_subset(self.aux_images, self.aux_labels)
        elif self.aux_ds == "skip":
            return self._create_skip_subset(self.aux_images, self.aux_labels)
        else:
            return np.stack([self.aux_images] * 3, axis=1), self.aux_labels

    def _create_red_mnist(self, images):
        red_images = np.zeros((images.shape[0], 3, 28, 28), dtype=np.uint8)
        for i, img in enumerate(images):
            red_images[i] = np.stack([img, np.zeros_like(img), np.zeros_like(img)])
        return red_images

    def _create_red_shade_mnist(self, images):
        red_images = np.zeros((images.shape[0], 3, 28, 28), dtype=np.uint8)
        for i, img in enumerate(images):
            red_shade = np.random.randint(1, 256)
            red_channel = img * red_shade
            red_images[i] = np.stack([red_channel, np.zeros_like(img), np.zeros_like(img)])
        return red_images

    def _create_colorful_subset(self, images, labels):
        color_images = np.zeros((len(images), 3, 28, 28), dtype=np.uint8)
        for i, img in enumerate(images):
            color = np.random.randint(1, 256, size=3) / 255
            color_images[i] = np.stack(
                [color[0] * img, color[1] * img, color[2] * img]
            )  
        return color_images, labels
    
    def _create_skip_subset(self, images, labels):
        lines_on = 4
        lines_off = 4

        if images.ndim == 3:
            images = np.stack([images] * 3, axis=1)  # Add channel dimension

        mask_on = np.ones((images.shape[0], images.shape[1], lines_on, images.shape[3]))
        mask_off = np.zeros((images.shape[0], images.shape[1], lines_off, images.shape[3]))

        mask_cycle = np.concatenate([mask_on, mask_off], axis=2)
        num_cycles = int(np.ceil(images.shape[2] / mask_cycle.shape[2]))
        full_mask = np.tile(mask_cycle, (1, 1, num_cycles, 1))[:, :, :images.shape[2], :]

        skip_images = images * full_mask
        return skip_images, labels
