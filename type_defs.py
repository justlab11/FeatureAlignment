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
    # random augmentation of the training images only (target and source alike); see transforms.train_augmentations
    augment: Literal["none", "geometric"] = "none"

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
    improvement_se: float = 1.0  # a gain counts only if above this many standard errors of the val accuracy
    alignment_rel_tol: float = 1e-3  # a new best alignment val loss must be this fraction lower than the last best
    body_lr_scale: float = 0.0  # 0 keeps the classifier body frozen in head fine-tuning; >0 trains it at this fraction of the lr
    max_inner_iterations: int | None = None  # cap on align/classify iterations per span; 1 drops the inner loop

class Config(BaseModel):
    dataset: DatasetRootConfig
    save_locations: SaveLocations
    classifier: ClassifierConfig
    unet: UNetConfig
    verbose: bool
    ablation: str = "normal"  # see ablations.py; recorded here so every run says which variant it is

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
    # optional target split sizes (image counts) for this pair; unset ones follow the dataset-family defaults in run_main.py
    target_train_size: int | None = None
    target_val_size: int | None = None
    target_test_size: int | None = None

class MetaConfig(BaseModel):
    datasets: List[MetaDatasetConfig]
    dataset_pairs: List[MetaDatasetPair]
    image_sizes: List[str]
    unet_loss: List[str]
    unet_attention: List[bool]
    ablations: List[str] = ["normal"]  # variants run for every combination (names from ablations.py)
    target_train_sizes: List[int] = [1000]  # labeled target training images (MNIST / SVHN targets only)
    target_val_size: int | None = None  # cap on target validation images; None keeps the default fraction
    source_val_size: int | None = None  # cap on source validation images; None keeps the default fraction