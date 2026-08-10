import torch
import torch.nn.functional as F
import yaml
import logging
import csv
from typing import List
from glob import glob
import os
from sklearn.model_selection import train_test_split
from collections import Counter

from models import SmallCustomUNET, LargeCustomUNET, SmallAttentionUNET, LargeAttentionUNET

logger = logging.getLogger(__name__)

def build_splits(folder: str, split_pcts: List[float], seed):
    # collect list of all files in the folder
    files = glob(
        os.path.join(folder, "**", "*"),
        recursive=True
    )
    files = [f for f in files if os.path.isfile(f)]
    logger.info(f"Found {len(files)} files in {folder}")
    labels = []
    samples = []

    # populate samples and labels
    for f in files:
        class_name = os.path.basename(os.path.dirname(f))  # Assumes Unix-style paths
        labels.append(class_name)
        samples.append(f)

    logger.debug(f"Class distribution in {folder}: {dict(Counter(labels))}")

    # collect split information
    train_size, val_size, test_size = split_pcts

    # build test set first
    X_temp, X_test, y_temp, y_test = train_test_split(
        samples, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    # finalize train and val splits
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=seed, stratify=y_temp
    )

    # only return the lists of files (the labels are handled by the dataset class)
    return X_train, X_test, X_val

def append_metrics_row(csv_path: str, row: dict):
    """Append one row of per-epoch metrics to a CSV, writing the header the first time."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def validate_num_layers(model, sample_input, device="cpu"):
    """
    One-time sanity check that model.get_num_layers() matches the number of layer
    outputs a real forward pass produces. Layer-indexed code (cascade/single-layer
    alignment, EBSW plots) silently mis-indexes if these two disagree.
    """
    model.to(device)
    was_training = model.training
    model.eval()

    with torch.no_grad():
        outputs = model(sample_input.to(device))

    if was_training:
        model.train()

    actual = len(outputs)
    reported = model.get_num_layers()

    if actual != reported:
        logger.warning(
            f"{type(model).__name__}.get_num_layers() reports {reported} layers, but a forward "
            f"pass produced {actual} layer outputs. Layer-indexed code may be silently mis-indexing."
        )
    else:
        logger.info(f"{type(model).__name__}.get_num_layers() verified: {reported} layers match forward() output.")

    return actual, reported

def load_yaml(file_path: str):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data

def make_unet(size:str, attention:bool=False, base_channels:int=32, noise_channels:int=8):
    def unet():
        if size == "small" and attention:
            model = SmallAttentionUNET(
                base_channels=base_channels,
                noise_channels=noise_channels
            )
        elif size == "small" and not attention:
            model = SmallCustomUNET(
                base_channels=base_channels,
                noise_channels=noise_channels
            ) 

        elif size == "large" and attention:
            model = LargeAttentionUNET(
                base_channels=base_channels,
                noise_channels=noise_channels
            )

        elif size == "large" and not attention:
            model = LargeCustomUNET(
                base_channels=base_channels,
                noise_channels=noise_channels
            )

        return model
    
    return unet

def compute_layer_loss(base_reps: List[torch.tensor], aux_reps: List[torch.tensor], labels: torch.tensor, layer: int, criterion, device: str):
    # reshape arrays 
    base_reshaped = base_reps[layer].view(base_reps[layer].size(0), -1)
    aux_reshaped = aux_reps[layer].view(aux_reps[layer].size(0), -1)

    if layer == len(base_reps) - 1:  # Output layer
        # compare against the labels instead of the output representation
        base_normed = F.one_hot(labels, base_reshaped.size(1))
        base_normed = base_normed.float()
        aux_normed = F.softmax(aux_reshaped, dim=1)
    else:  # Intermediate layers
        # compare against latent representations
        base_norms = torch.norm(base_reshaped, dim=1, keepdim=True).clamp(min=1e-12)
        aux_norms = torch.norm(aux_reshaped, dim=1, keepdim=True).clamp(min=1e-12)
        base_normed = base_reshaped / base_norms
        aux_normed = aux_reshaped / aux_norms

    return criterion(base_normed, aux_normed, device=device)
