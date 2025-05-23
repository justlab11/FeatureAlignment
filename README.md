# Disjoint Dataset Alignment

This project implements a framework for aligning and training models on disjoint datasets, with a focus on domain adaptation and feature alignment. The system uses a combination of UNet-based alignment models and various classification approaches to improve performance across different domains.

## Project Structure

The codebase is organized into several key components:

- `main.py`: Core training script for single experiment runs
- `run_main.py`: Batch processing script for running multiple experiments sequentially
- `models.py`: Neural network architectures (ResNet, UNet variants, Autoencoder)
- `trainer.py`: Training logic for classifiers and alignment models
- `losses.py`: Custom loss functions for alignment and contrastive learning
- `datasets.py`: Dataset handling and data loading utilities
- `helpers.py`: Utility functions for model building and data processing
- `plotters.py`: Visualization tools for model analysis
- `type_defs.py`: Type definitions and configuration models
- `loss_comparison_analysis.ipynb`: Synthetic 3D experiments comparing divergence measures (CC-EBSW, CC-MMD, InfoNCE, etc.) across different domain alignment scenarios with varying geometric configurations and class distributions

## Installation

1. Create and activate a virtual environment:
```bash
# Clone this repo and cd into it
git clone <repo-url>
cd FeatureAlignment

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Features

- Multiple model architectures:
  - ResNet variants (9, 18, 34, 50, 101, 152)
  - UNet with optional attention mechanisms
  - Custom autoencoder for feature extraction
- Training approaches:
  - Standard supervised learning
  - Contrastive learning with temperature optimization
  - Feature alignment using UNet
- Visualization tools:
  - t-SNE plots for feature visualization
  - Energy-Based Wasserstein distance plots
  - Example image transformations
- Flexible configuration system for experiment management

## Running the Code

There are two ways to run experiments:

### 1. Single Experiment

To run a single experiment, use `main.py` with a configuration file:

```bash
python main.py --config_fname path/to/config.yaml
```

The configuration file should specify:
- Dataset information (source and target)
- Model architecture choices
- Training parameters
- UNet configuration
- Save locations

### 2. Batch Experiments

To run multiple experiments sequentially, use `run_main.py` with a meta-configuration file:

```bash
python run_main.py --config_fname path/to/meta_config.yaml
```

The meta-configuration file allows you to specify:
- Multiple dataset pairs
- Different image sizes
- Various UNet configurations
- Different loss functions

## Configuration Files

### Single Experiment Config

```yaml
dataset:
  target:
    name: "dataset_name"
    folder: "path/to/target"
    train_pct: 0.7
    val_pct: 0.15
    num_classes: 10
  source:
    name: "dataset_name"
    folder: "path/to/source"
    train_pct: 0.7
    val_pct: 0.15
    num_classes: 10
  image_size: "small"  # or "large"
  rng_seed: 42
  batch_size: 32

classifier:
  model: "resnet18"
  identifier: "experiment_name"
  num_epochs: 100

unet:
  loss: "ebsw"  # or "mmdfuse"
  attention: true
  base_channels: 32
  noise_channels: 8
  num_warm_start_epochs: 10
  num_epochs: 100

verbose: true
```

### Meta Config

```yaml
datasets:
  - name: "dataset1"
    folder: "path/to/dataset1"
    num_classes: 10
  - name: "dataset2"
    folder: "path/to/dataset2"
    num_classes: 10

dataset_pairs:
  - target: "dataset1"
    source: "dataset2"

image_sizes: ["small", "large"]
unet_loss: ["ebsw", "mmdfuse"]
unet_attention: [true, false]
```

## Output

The system generates:
- Trained model checkpoints
- Training logs
- Visualization plots:
  - t-SNE embeddings
  - Energy-Based Wasserstein distance plots
  - Example image transformations
- Performance metrics and statistics

## Dependencies

- PyTorch
- torchvision
- numpy
- click
- pydantic
- matplotlib
- scikit-learn

## Notes

- The system supports both small (32x32) and large (128x128) image sizes
- UNet models can be configured with or without attention mechanisms
- Multiple loss functions are available for alignment (EBSW, MMDfuse)
- The framework includes comprehensive logging and visualization tools
