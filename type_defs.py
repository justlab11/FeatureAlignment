from pydantic import BaseModel
from typing import List, Any, Literal
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn

class DatasetConfig(BaseModel):
    name: str
    folder: str
    train_size: int
    val_size: int
    num_classes: int

class DatasetRootConfig(BaseModel):
    target: DatasetConfig
    source: DatasetConfig
    image_size: Literal["small", "large"]
    rng_seed: int
    batch_size: int

class SaveLocations(BaseModel):
    model_folder: str
    file_folder: str
    image_folder: str
    logs_folder: str

class ClassifierConfig(BaseModel):
    model: str
    identifier: str
    num_epochs: int

class UNetConfig(BaseModel):
    loss: str
    attention: bool
    base_channels: int
    noise_channels: int
    num_warm_start_epochs: int
    num_epochs: int

class Config(BaseModel):
    dataset: DatasetRootConfig
    save_locations: SaveLocations
    classifier: ClassifierConfig
    unet: UNetConfig
    verbose: bool

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

class MetaDatasetConfig(BaseModel):
    name: str
    folder: str
    num_classes: int

class MetaDatasetPair(BaseModel):
    target: str
    source: str

class MetaConfig(BaseModel):
    datasets: List[MetaDatasetConfig]
    dataset_pairs: List[MetaDatasetPair]
    image_sizes: List[str]
    unet_loss: List[str]
    unet_attention: List[bool]