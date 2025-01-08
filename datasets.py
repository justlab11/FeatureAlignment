import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split as tts

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
            aux_labels: np.ndarray
        ):
        
        self.base_images: torch.tensor = torch.from_numpy(base_images).float() / 255.0
        self.base_labels: torch.tensor = torch.from_numpy(base_labels).long()

        self.aux_images: torch.tensor = torch.from_numpy(aux_images).float() / 255.0
        self.aux_labels: torch.tensor = torch.from_numpy(aux_labels).long()

        self.indices_by_class = self._group_indices_by_class()

    def __len__(self):
        return len(self.base_images)
    
    def _group_indices_by_class(self):
        indices_by_class = {i: [] for i in range(10)}

        for idx, label in enumerate(self.aux_labels):
            indices_by_class[label.item()].append(idx)

        return indices_by_class
    
    def __getitem__(self, idx):
        label = self.base_labels[idx].item()
        base_sample = self.base_images[idx]

        aux_idx = np.random.choice(self.indices_by_class[label])
        aux_sample = self.aux_images[aux_idx]

        return base_sample, aux_sample, label


class DatasetGenerator:
    def __init__(self, images, labels, subset_ratio=.3, base_ds="red", train=True):
        self.base_ds = base_ds
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
        else:
            return self._create_red_shade_mnist(self.base_images), self.base_labels

    def build_aux_dataset(self):
        return self._create_colorful_subset(self.aux_images, self.aux_labels)

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