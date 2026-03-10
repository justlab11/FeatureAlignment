import subprocess
import itertools
import sys
import click
import yaml
import glob
import os
import shutil
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

from copy import deepcopy
from PIL import Image
import datasetops as do
from pathlib import Path
from typing import Union, Tuple

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
            "train_pct": 0.02,
            "val_pct": 0.3,
            "num_classes": 10
        },
        "source": {
            "name": "SVHN",
            "folder": "data/house_mnist_32",
            "train_pct": 0.98,
            "val_pct": 0.01,
            "num_classes": 10
        },
        "image_size": "small",
        "rng_seed": 72,
        "batch_size": 128,
    },
    "save_locations": {
        "model_folder": "models",
        "file_folder": "files",
        "image_folder": "images",
        "logs_folder": "logs",
    },
    "classifier": {
        "model": "vgg",
        "identifier": "MNIST+SVHN",  # unique identifier for the set of models
        "mixed_num_epochs": 40,
        "contrast_num_epochs": 40,
        "train_baseline": False
    },
    "unet": {
        "loss": "ebsw",
        "attention": False,
        "base_channels": 32,
        "noise_channels": 8,
        "num_warm_start_epochs": 20,
        "num_epochs": 30,
    },
    "verbose": False,
}

def build_office31(
    source_name: str,
    target_name: str,
    data_folder: str,
    seed=1,
    image_resize=(240, 240),
    office_path: Union[str, Path, None] = None,
):
    """Get a train-val-test split for Office 31 domain adaptation experiments."""
    test_split_seed = 42  # hard-coded

    num_source_per_class = 20 if source_name == "amazon" else 8
    num_target_per_class = 3

    office_path = Path(office_path)

    # Seed everything for reproducibility
    random.seed(seed)

    # -- You will likely need to import/define `do` to use this code --
    source = do.from_folder_class_data(office_path / source_name / "images").named(
        "s_data", "s_label"
    )
    target = do.from_folder_class_data(office_path / target_name / "images").named(
        "t_data", "t_label"
    )

    source_train = source.shuffle(seed).filter(
        s_label=do.allow_unique(num_source_per_class)
    )

    # Get all source samples and source_train samples as (path, label) tuples
    all_source_samples = [(path, label) for path, label in source]
    all_source_train_samples = [(path, label) for path, label in source_train]
    # Get the remainder (not in source_train)
    remainder_list = [p for p in all_source_samples if p not in all_source_train_samples]

    # 1. Group samples by class
    class_to_samples = defaultdict(list)
    for path, label in remainder_list:
        class_to_samples[label].append(path)

    val_samples = []
    test_samples = []
    other_samples = []

    # 2. For each class, select one for val and one for test (if available)
    for label, samples in class_to_samples.items():
        random.shuffle(samples)
        if samples:
            val_samples.append((samples.pop(), label))
        if samples:
            test_samples.append((samples.pop(), label))
        # The rest go to the pool for further splitting
        other_samples.extend([(path, label) for path in samples])

    # 3. Split the remaining samples by your desired ratio
    random.shuffle(other_samples)
    split_idx = int(len(other_samples) * 0.3)
    rem_test = other_samples[:split_idx]
    rem_val = other_samples[split_idx:]

    source_test = test_samples + rem_test
    source_val = val_samples + rem_val

    # Shuffle final splits for randomness
    random.shuffle(source_test)
    random.shuffle(source_val)

    # Target splits (unchanged)
    target_test, target_trainval = target.split(
        fractions=[0.3, 0.7], seed=test_split_seed
    )
    target_train, target_val = target_trainval.shuffle(seed).split_filter(
        t_label=do.allow_unique(num_target_per_class)
    )
    # Convert to tuples for processing
    target_test = [(data, label) for data, label in target_test]
    target_train = [(data, label) for data, label in target_train]
    target_val = [(data, label) for data, label in target_val]
    source_train = [(data, label) for data, label in source_train]
    
    # --- FILTER OUT CLASSES WITH <3 SAMPLES IN source_val ---
    val_class_counts = Counter([label for _, label in source_val])
    valid_val_classes = {cls for cls, count in val_class_counts.items() if count >= 3}
    # Also match on target_val!
    source_val = [(x, y) for x, y in source_val if y in valid_val_classes]
    target_val = [(x, y) for x, y in target_val if y in valid_val_classes]

    # --- AUGMENT source_test TO MINIMUM 3 SAMPLES PER CLASS ---
    test_class_counts = Counter([label for _, label in source_test])
    for cls, count in test_class_counts.items():
        if count < 3:
            # Find folder for this class
            class_folder = office_path / source_name / "images" / cls
            all_class_images = list(glob.glob(str(class_folder / "*.jpg")))
            already_in_test = {str(path) for path, label in source_test if label == cls}
            remaining_images = [img for img in all_class_images if img not in already_in_test]
            needed = 3 - count
            if needed > 0 and remaining_images:
                additional = random.sample(remaining_images, min(needed, len(remaining_images)))
                source_test.extend([(img, cls) for img in additional])
    # Optionally: re-shuffle
    random.shuffle(source_test)

    # Remove classes from target_test that are missing in source_test
    source_test_classes = set(label for _, label in source_test)
    target_test_classes = set(label for _, label in target_test)

    missing_classes = target_test_classes - source_test_classes

    if missing_classes:
        target_test = [(x,y) for x,y in target_test if y not in missing_classes]

    # Wrap up splits for saving
    target_splits = {
        "train": target_train,
        "test": target_test,
        "val": target_val
    }
    source_splits = {
        "train": source_train,
        "test": source_test,
        "val": source_val
    }

    source_folder = f"{data_folder}/{source_name}-{target_name}-{seed}/{source_name}"
    target_folder = f"{data_folder}/{source_name}-{target_name}-{seed}/{target_name}"

    os.makedirs(source_folder, exist_ok=True)
    os.makedirs(target_folder, exist_ok=True)

    # SAVE SOURCE SPLITS
    for split_name, split in source_splits.items():
        for i, (source_sample, source_label) in enumerate(split):
            source_class_folder = os.path.join(source_folder, split_name, "images", source_label)
            os.makedirs(source_class_folder, exist_ok=True)
            source_file = os.path.join(source_class_folder, f"sample_{i}.jpg")
            with Image.open(source_sample) as img:
                resized_img = img.resize(image_resize)
                resized_img.save(source_file)

    # SAVE TARGET SPLITS
    for split_name, split in target_splits.items():
        for i, (target_sample, target_label) in enumerate(split):
            target_class_folder = os.path.join(target_folder, split_name, "images", target_label)
            os.makedirs(target_class_folder, exist_ok=True)
            target_file = os.path.join(target_class_folder, f"sample_{i}.jpg")
            with Image.open(target_sample) as img:
                resized_img = img.resize(image_resize)
                resized_img.save(target_file)


@click.command()
@click.option("--data_folder")
def main(data_folder="data"):
    words = ["amazon", "webcam", "dslr"]
    pairs = list(itertools.permutations(words, 2))

    for seed, pair in itertools.product(range(1,6), pairs):
        yaml_data = deepcopy(TEMPLATE_YAML)
        config = Config(**yaml_data)
        
        data_path = os.path.abspath(data_folder)

        target = pair[0]
        source = pair[1]
        if target == "amazon" and source == "webcam" and seed ==1:
            continue
            
        build_office31(
            source_name = source,
            target_name = target,
            data_folder = data_folder,
            seed=seed,
            image_resize=(240, 240),
            office_path = data_folder, #automatically downloads to "~/data"
        )

        target_folder = f"{data_folder}/{source}-{target}-{seed}/{target}"
        source_folder = f"{data_folder}/{source}-{target}-{seed}/{source}"

        base_dir = f'{data_folder}/{source}-{target}-{seed}'
        
        domains = [target, source]
        splits = ['train', 'test', 'val']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (domain, split) in enumerate([(d, s) for d in domains for s in splits]):
            root_dir = os.path.join(base_dir, domain, split, 'images')
            class_counts = Counter()

            for dirpath, dirnames, filenames in os.walk(root_dir):
                if dirpath != root_dir:
                    class_name = os.path.basename(dirpath)
                    num_files = len([f for f in filenames if os.path.isfile(os.path.join(dirpath, f))])
                    class_counts[class_name] += num_files


            ax = axes[idx]
            ax.bar(class_counts.keys(), class_counts.values())
            ax.set_title(f'{domain.upper()} - {split.upper()}')
            ax.set_xlabel('Class')
            ax.set_ylabel('Num Samples')
            ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig("test.jpg")

        config.dataset.target.name = target
        config.dataset.source.name = source

        config.dataset.target.folder = target_folder
        config.dataset.source.folder = source_folder

        config.dataset.target.num_classes = 31
        config.dataset.source.num_classes = 31

        config.dataset.image_size = "large" # this isnt necessary
        config.dataset.batch_size = 16

        config.classifier.identifier = f"{target}+{source}+{seed}_v2"
        
        fname = f"configs/{target}+{source}_{seed}.yaml"
        with open(fname, "w") as file:
            yaml.safe_dump(config.model_dump(), file, sort_keys=False)

        print("Starting run for config:", fname)
        print(os.getcwd())

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        try:
            subprocess.run([
                "python",
                "main.py",
                "--config_fname",
                fname
            ], env=env)
        except Exception as e:
            print(f"Error: {e}")

        # delete the folders to save space
        #shutil.rmtree(base_dir)

if __name__=="__main__":
    main()