"""
Builds a synthetic YAML config matching the CURRENT type_defs.Config schema
(not the drifted configs/*.yaml files in the repo -- see checks.py's
schema-drift check for those), sized to run fast: tiny images, tiny UNet
channel counts, tiny batch size. Epoch counts are handled separately via the
FEATUREALIGNMENT_EPOCH_CAP env var (see main.py/trainer.py), not here.
"""
from pathlib import Path

import yaml

CLASS_COUNT = 10  # must match generate_data.CLASS_NAMES length; also required
                   # for main.py's TSNE plotting path, which only runs when
                   # both domains report exactly 10 classes


def build_config(
    target_dir,
    source_dir,
    output_dir,
    samples_per_class=12,
    batch_size=4,
    rng_seed=72,
    num_epochs=1,
    unet_num_epochs=1,
    loss="ebsw",
    attention=False,
    model="resnet9",
    identifier="smoke_test",
    train_pct=0.6,
    val_pct=0.2,
):
    target_name = "smoke_target"
    source_name = "smoke_source"

    config = {
        "dataset": {
            "target": {
                "name": target_name, "folder": str(target_dir),
                "train_pct": train_pct, "val_pct": val_pct, "num_classes": CLASS_COUNT,
            },
            "source": {
                "name": source_name, "folder": str(source_dir),
                "train_pct": train_pct, "val_pct": val_pct, "num_classes": CLASS_COUNT,
            },
            "image_size": "small",  # required: SmallCustomUNET/SmallAttentionUNET assume 32x32
            "rng_seed": rng_seed,
            "batch_size": batch_size,
        },
        "save_locations": {
            "model_folder": "models", "file_folder": "files",
            "image_folder": "images", "logs_folder": "logs",
        },
        "classifier": {
            "model": model,  # resnet9 = cheapest option, no pretrained-weight download
            "identifier": identifier,
            "num_epochs": num_epochs,
        },
        "unet": {
            "loss": loss,
            "attention": attention,
            "base_channels": 4,
            "noise_channels": 2,
            "num_warm_start_epochs": unet_num_epochs,
            "num_epochs": unet_num_epochs,
        },
        "verbose": True,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"smoke_{loss}.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # main.py derives its output folder name and filename suffix from these
    # same values -- compute them here so checks.py knows what to look for
    # without re-deriving main.py's logic. NOTE: main.py counts target/source
    # "size" via glob.glob(folder/**/*, recursive=True), which matches the
    # per-class subdirectories themselves in addition to the image files
    # inside them -- so the true "file" count it uses is (samples_per_class + 1)
    # per class, not samples_per_class.
    total_dirents = CLASS_COUNT * (samples_per_class + 1)
    target_train_size = int(train_pct * total_dirents)
    source_train_size = int(train_pct * total_dirents)
    unet_name = "attention_unet" if attention else "unet"
    classifier_id = f"{identifier}-small-{loss}-{model}-{unet_name}"
    suffix = f"{target_name}={target_train_size}+{source_name}={source_train_size}"

    meta = {
        "classifier_id": classifier_id,
        "suffix": suffix,
        "target_train_size": target_train_size,
        "source_train_size": source_train_size,
    }

    return str(config_path), meta
