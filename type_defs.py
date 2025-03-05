from pydantic import BaseModel
from typing import List, Any
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn

class DatasetConfig(BaseModel):
    target_name: str
    target_folder: str
    aux_name: str
    aux_folder: str

    rng_seed: int = 72
    batch_size: int = 16
    base_sample_ratio: float = 0.1

class SaveLocationsConfig(BaseModel):
    model_folder: str = "models"
    file_folder: str = "files"
    image_folder: str = "images"
    logs_folder: str = "logs"

class ClassifierConfig(BaseModel):
    model: str = "resnet18"
    identifier: str = "tiny"
    num_epochs: int = 50

class UNetConfig(BaseModel):
    loss: str = "ebsw"
    num_epochs: int = 50

class Config(BaseModel):
    dataset: DatasetConfig
    save_locations: SaveLocationsConfig
    classifier: ClassifierConfig
    unet: UNetConfig
    verbose: bool = True

class DataLoaderSet(BaseModel):
    train_loader: DataLoader
    test_loader: DataLoader
    val_loader: DataLoader

    class Config:
        arbitrary_types_allowed = True

class EmbeddingSet(BaseModel):
    base_embeds: np.ndarray
    aux_embeds: np.ndarray
    labels: np.ndarray

    class Config:
        arbitrary_types_allowed = True

class ModelSet(BaseModel):
    base: Any
    mixed: Any
    contrast: Any