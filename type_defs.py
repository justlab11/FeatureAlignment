from pydantic import BaseModel
from typing import List, Any, Literal
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn

class DatasetConfig(BaseModel):
    name: str
    folder: str
    train_pct: float
    val_pct: float
    num_classes: int
    test_pct: float | None = None  # defaults to whatever is left after train and val

class DatasetRootConfig(BaseModel):
    target: DatasetConfig
    source: DatasetConfig
    image_size: Literal["small", "large"]
    rng_seed: int
    batch_size: int
    # source samples are modified copies of target samples (same class/file name), so source
    # must not contain copies of the target's val/test samples
    paired_source: bool = False

class SaveLocations(BaseModel):
    model_folder: str
    file_folder: str
    image_folder: str
    logs_folder: str

class ClassifierConfig(BaseModel):
    model: str
    identifier: str
    mixed_num_epochs: int
    contrast_num_epochs: int
    train_baseline: bool

class UNetConfig(BaseModel):
    loss: str
    attention: bool
    base_channels: int
    noise_channels: int
    num_warm_start_epochs: int
    num_epochs: int
    patience: int = 2  # non-improving align/classify iterations tolerated before a span stops

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
    paired: bool = False  # source is a modified copy of target (see DatasetRootConfig.paired_source)

class MetaConfig(BaseModel):
    datasets: List[MetaDatasetConfig]
    dataset_pairs: List[MetaDatasetPair]
    image_sizes: List[str]
    unet_loss: List[str]
    unet_attention: List[bool]