import subprocess
import itertools
import click
import yaml
import glob
import os
from copy import deepcopy

from type_defs import MetaConfig, Config

def load_meta_config(file_path) -> MetaConfig:
    with open(file_path, "r") as f:
        raw_config = yaml.safe_load(f)
    return MetaConfig(**raw_config)

# Define the YAML template as a dictionary
TEMPLATE_YAML = {
    "dataset": {
        "target": {
            "name": "MNIST",
            "folder": "data/mnist_v2",
            "train_size": 1000,
            "val_size": 3000,
        },
        "source": {
            "name": "SVHN",
            "folder": "data/house_mnist_32",
            "train_size": 69000,
            "val_size": 500,
        },
        "image_size": "small",
        "rng_seed": 72,
        "batch_size": 64,
    },
    "save_locations": {
        "model_folder": "models",
        "file_folder": "files",
        "image_folder": "images",
        "logs_folder": "logs",
    },
    "classifier": {
        "model": "resnet9",
        "identifier": "MNIST+SVHN",  # unique identifier for the set of models
        "num_epochs": 100,
    },
    "unet": {
        "loss": "ebsw",
        "attention": False,
        "base_channels": 32,
        "noise_channels": 8,
        "num_warm_start_epochs": 20,
        "num_epochs": 100,
    },
    "verbose": False,
}

@click.command()
@click.option("--config_fname", help="Meta config file used to build ALL other config files and run main.py on all of them")
def main(config_fname):
    meta_config: MetaConfig = load_meta_config(config_fname)

    dataset_map = {dataset.name: dataset for dataset in meta_config.datasets}

    # check if any datasets were not defined by the dataset
    for pair in meta_config.dataset_pairs:
        if pair.target not in dataset_map:
            raise ValueError(f"Invalid dataset: {pair.target}")

        if pair.source not in dataset_map:
            raise ValueError(f"Invalid dataset: {pair.source}")
    
    dataset_options = meta_config.datasets
    #dataset_sizes = [i/10 for i in range(1, 10)]
    dataset_sizes = [.9]
    combinations = list(itertools.product(
        meta_config.dataset_pairs,
        meta_config.image_sizes,
        dataset_sizes
    ))

    # now iterate through and build the yaml files
    for pair, img_size, pct in combinations:
        yaml_data = deepcopy(TEMPLATE_YAML)
        config = Config(**yaml_data)

        target_dataset = next(d for d in dataset_options if d.name == pair.target)
        source_dataset = next(d for d in dataset_options if d.name == pair.source)

        target_ds_files = glob.glob(
            os.path.join(target_dataset.folder, "**", "*"),
            recursive=True
        )
        
        source_ds_files = glob.glob(
            os.path.join(source_dataset.folder, "**", "*"),
            recursive=True
        )

        target_ds_len = len([path for path in target_ds_files if os.path.isfile(path)])
        source_ds_len = len([path for path in source_ds_files if os.path.isfile(path)])

        config.dataset.target.name = target_dataset.name
        config.dataset.source.name = source_dataset.name

        config.dataset.target.folder = target_dataset.folder
        config.dataset.source.folder = source_dataset.folder

        if "mnist" in target_dataset.name.lower():
            config.dataset.target.train_size = 1000
        elif "cifar10" in target_dataset.name.lower():
            config.dataset.target.train_size = 4000
        else:
            config.dataset.target.train_size = int(target_ds_len * .85)
        config.dataset.source.train_size = int((source_ds_len) * pct)

        config.dataset.target.val_size = int(target_ds_len * .1)
        config.dataset.source.val_size = int((source_ds_len) * .05)

        config.dataset.image_size = img_size
        config.dataset.batch_size = 64 if img_size == "small" else 16

        config.classifier.identifier = f"{target_dataset.name}+{source_dataset.name}"

        fname = f"configs/{target_dataset.name}+{source_dataset.name}_{int((source_ds_len) * pct)}_{img_size}.yaml"
        with open(fname, "w") as file:
            yaml.safe_dump(config.model_dump(), file, sort_keys=False)

        subprocess.run([
            "python",
            "main.py",
            "--config_fname",
            fname
        ])


if __name__=="__main__":
    main()
