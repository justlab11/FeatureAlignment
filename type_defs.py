from pydantic import BaseModel
from typing import List

class DatasetConfig(BaseModel):
    name: str = "mnist"
    base_ds: str = "normal"
    aux_ds: str = "skip"
    rng_seed: int = 72
    batch_size: int = 64
    base_sample_ratio: float = .2

class SaveLocationsConfig(BaseModel):
    model_folder: str = "models"
    file_folder: str = "files"
    image_folder: str = "images"

class ClassifierConfig(BaseModel):
    identifier: str
    num_filters: List[int] = [8, 16]
    kernel_size: List[int] = [3, 3]
    stride: List[int] = [1, 1]
    padding: List[int] = [1, 1]
    mlp_layer_sizes: List[int] = [32]
    num_classes: int = 10

class UNetConfig(BaseModel):
    loss: str = "ebsw"

class Config(BaseModel):
    dataset: DatasetConfig
    save_locations: SaveLocationsConfig
    classifier: ClassifierConfig
    unet: UNetConfig