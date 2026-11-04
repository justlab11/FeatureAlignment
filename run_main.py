import subprocess
import itertools
import json
import sys
import time
import click
import torch
import yaml
import glob
import os
from copy import deepcopy

import ablations
from type_defs import MetaConfig, Config

# load the config into a format we can work with
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
        "model": "resnet9",
        "identifier": "MNIST+SVHN",  # unique identifier for the set of models
        "mixed_num_epochs": 100,
        "contrast_num_epochs": 100,
        "train_baseline": False
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

def target_split_pcts(pair, dataset_name: str, dataset_len: int, train_size: int, meta_config: MetaConfig):
    """
    (train, val, test) fractions of the target dataset. Defaults follow the dataset family; a pair's own
    target_*_size values (image counts) win. test is None when it should be whatever train and val leave over.
    """
    name = dataset_name.lower()
    test = None
    if "mnist" in name or name == "svhn":
        train = min(1.0, train_size / dataset_len)
        # hold out 10% for test; the rest of the target images stay unused (and available to a paired source)
        test = 0.1
    elif "cifar10" in name:
        train = min(1.0, 4000 / dataset_len)
    else:
        train = .8
    val = min(.1, (1 - train) / 2)

    # smaller validation sets make every epoch's validation pass (the bulk of the run time) much cheaper
    if meta_config.target_val_size is not None:
        val = min(val, meta_config.target_val_size / dataset_len)

    if pair.target_train_size is not None:
        train = pair.target_train_size / dataset_len
    if pair.target_val_size is not None:
        val = pair.target_val_size / dataset_len
    if pair.target_test_size is not None:
        test = pair.target_test_size / dataset_len

    if train + val + (test or 0) > 1 + 1e-9:
        raise ValueError(f"Target splits for {dataset_name} need more than the {dataset_len} available images")
    return train, val, test

def estimate_peak_bytes(fname: str, env: dict) -> int | None:
    """Per-process GPU memory a run needs, from a CPU-only `main.py --estimate_out` pass (None if that failed)."""
    out = fname.replace(".yaml", ".gpu.json")
    result = subprocess.run([sys.executable, "main.py", "--config_fname", fname, "--estimate_out", out], env=env)
    if result.returncode != 0:
        return None
    with open(out) as f:
        return json.load(f)["process_memory_bytes"]

def run_jobs(jobs: list[str], max_parallel: int, memory_margin: float, env: dict, poll_seconds: int = 30, max_oom_retries: int = 2):
    """
    Runs main.py for every config, at most `max_parallel` at a time, as long as the summed peak estimates of the
    running jobs fit in the GPU budget. A job that dies with a CUDA out-of-memory error goes back in the queue.
    """
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        # what is already on the card (display server, other users) is not ours to use; the margin covers estimate error
        budget = total * (1 - memory_margin) - (total - free)
    else:
        budget = float("inf")

    peaks = {}
    if max_parallel > 1:
        for job in jobs:
            peaks[job] = estimate_peak_bytes(job, env) or budget  # unknown -> assume it needs the whole budget
            print(f"{job}: estimated {peaks[job] / 1024**3:.2f} GiB per process (budget {budget / 1024**3:.2f} GiB)")

    queue = sorted(jobs, key=lambda j: -peaks.get(j, 0))  # biggest first so they are not stuck behind small ones
    running: dict = {}  # job -> (process, stdout file)
    retries = {job: 0 for job in jobs}

    while queue or running:
        for job, (proc, log) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[job]
            if log:
                log.close()
            if proc.returncode == 0:
                print(f"Finished {job}")
                continue
            with open(job.replace(".yaml", ".stdout.log"), errors="ignore") as f:
                out_of_memory = "out of memory" in f.read()[-5000:] if max_parallel > 1 else False
            if out_of_memory and retries[job] < max_oom_retries:
                retries[job] += 1
                print(f"{job} ran out of GPU memory, requeueing (attempt {retries[job] + 1})")
                queue.append(job)
            else:
                print(f"Error: {job} exited with code {proc.returncode}")

        reserved = sum(peaks[j] for j in running)
        for job in list(queue):
            if len(running) >= max_parallel:
                break
            if running and reserved + peaks[job] > budget:
                continue  # a job that does not fit has to wait; one that is alone always runs
            queue.remove(job)
            reserved += peaks.get(job, 0)
            print("Starting run for config:", job)
            log = open(job.replace(".yaml", ".stdout.log"), "w") if max_parallel > 1 else None
            proc = subprocess.Popen([sys.executable, "main.py", "--config_fname", job], env=env, stdout=log, stderr=subprocess.STDOUT if log else None)
            running[job] = (proc, log)

        if queue or running:
            time.sleep(poll_seconds)

@click.command()
@click.option("--config_fname", help="Meta config file used to build ALL other config files and run main.py on all of them")
@click.option("--max_parallel", default=1, show_default=True, help="Runs allowed at once; more than one also needs the summed memory estimates to fit the GPU")
@click.option("--memory_margin", default=0.1, show_default=True, help="Fraction of GPU memory kept free for estimate error")
def main(config_fname, max_parallel, memory_margin):
    meta_config: MetaConfig = load_meta_config(config_fname)

    dataset_map = {dataset.name: dataset for dataset in meta_config.datasets}

    # check if any datasets were not defined by the dataset
    for pair in meta_config.dataset_pairs:
        if pair.target not in dataset_map:
            raise ValueError(f"Invalid dataset: {pair.target}")

        if pair.source not in dataset_map:
            raise ValueError(f"Invalid dataset: {pair.source}")
    
    ablations.validate(meta_config.ablations)

    dataset_options = meta_config.datasets
    #dataset_sizes = [i/10 for i in range(1, 10)]
    dataset_sizes = [.7]
    combinations = list(itertools.product(
        meta_config.dataset_pairs,
        meta_config.image_sizes,
        meta_config.unet_loss,
        meta_config.unet_attention,
        dataset_sizes,
        meta_config.target_train_sizes,
        meta_config.ablations
    ))

    # now iterate through and build the yaml files
    jobs = []
    seen = set()
    for pair, img_size, unet_loss, unet_attention, pct, target_train_size, ablation in combinations:
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

        config.dataset.paired_source = pair.paired

        train_pct, val_pct, test_pct = target_split_pcts(pair, target_dataset.name, target_ds_len, target_train_size, meta_config)
        config.dataset.target.train_pct = train_pct
        config.dataset.target.val_pct = val_pct
        config.dataset.target.test_pct = test_pct

        config.dataset.source.train_pct = pct

        config.dataset.source.val_pct = (1-pct)/2

        if meta_config.source_val_size is not None:
            config.dataset.source.val_pct = min(config.dataset.source.val_pct, meta_config.source_val_size / source_ds_len)

        config.dataset.target.num_classes = target_dataset.num_classes
        config.dataset.source.num_classes = source_dataset.num_classes

        config.dataset.image_size = img_size
        config.dataset.batch_size = 32 if img_size == "small" else 16

        config.classifier.identifier = f"{target_dataset.name}+{source_dataset.name}"
        if ablation != "normal":
            config.classifier.identifier += f"-{ablation}"
        config.unet.loss = unet_loss
        config.unet.attention = unet_attention

        suffix = "" if ablation == "normal" else f"_{ablation}"
        fname = f"configs/{target_dataset.name}={round(config.dataset.target.train_pct * target_ds_len)}+{source_dataset.name}={int(source_ds_len * pct)}_{img_size}_{unet_loss}_{'attn' if unet_attention else 'noattn'}{suffix}.yaml"
        if fname in seen:  # e.g. a train-size axis that a pair's own target_train_size overrides
            continue
        seen.add(fname)

        run_config = ablations.apply(config.model_dump(), ablation)
        with open(fname, "w") as file:
            yaml.safe_dump(run_config, file, sort_keys=False)

        jobs.append(fname)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    run_jobs(jobs, max_parallel, memory_margin, env)

if __name__=="__main__":
    main()
