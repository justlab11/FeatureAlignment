import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import os
import pickle
from typing import Literal
import logging

from datasets import DatasetGenerator, PairedMNISTDataset
import helpers
from trainer import Trainer
from plotters import TSNE_Plotter, EBSW_Plotter
from models import TinyCNN, TinyCNN_Headless, TinyCNN_Head, WrapperModelTrainHead, CustomUNET, DynamicCNN
from type_defs import Config, DataLoaderSet, ModelSet

# constants
DEVICE: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"

config_yaml = helpers.load_yaml(r"configs/tiny_config.yaml")
CONFIG: Config = Config(**config_yaml)

MODEL_FOLDER: str = CONFIG.save_locations.model_folder
FILE_FOLDER: str = CONFIG.save_locations.file_folder
IMAGE_FOLDER: str = CONFIG.save_locations.image_folder

BASE: str = CONFIG.dataset.base_ds
AUXILIARY: str = CONFIG.dataset.aux_ds

BATCH_SIZE: int = CONFIG.dataset.batch_size
CLASSIFIER_ID: str = CONFIG.classifier.identifier

log_folder = CONFIG.save_locations.logs_folder
for folder in [MODEL_FOLDER, FILE_FOLDER, IMAGE_FOLDER, log_folder]:
    if not os.path.exists(folder):
        os.mkdir(folder)


# start logging
log_location = os.path.join(log_folder, f"run_{CONFIG.dataset.name}_{CLASSIFIER_ID}.log")
level = logging.DEBUG if CONFIG.verbose else logging.INFO

logging.basicConfig(
    level=level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_location,
    filemode='w'
)

logger = logging.getLogger(__name__)

# dataset creation
logger.info("Creating Datasets")
if CONFIG.dataset.name == "mnist":
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True)

elif CONFIG.dataset.name == "cifar10":
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

else:
    logger.error("Invalid value for dataset name")
    raise ValueError("Invalid value for dataset name")

x_train = train_set.data.numpy()
y_train = train_set.targets.numpy()

x_test = test_set.data.numpy()
y_test = test_set.targets.numpy()

x_test, x_val, y_test, y_val = tts(
    x_test, y_test, test_size=.5, random_state=CONFIG.dataset.rng_seed
)

train_ds_gen: DatasetGenerator = DatasetGenerator(
    images = x_train,
    labels = y_train,
    subset_ratio = CONFIG.dataset.base_sample_ratio,
    base_ds=BASE,
    aux_ds=AUXILIARY
)

base_images_train, base_labels_train = train_ds_gen.build_base_dataset()
aux_images_train, aux_labels_train = train_ds_gen.build_aux_dataset()

logger.info(f"\tTrain Dataset Base Size: {len(base_images_train)}")
logger.info(f"\tTrain Dataset Aux Size: {len(aux_images_train)}")

test_ds_gen: DatasetGenerator = DatasetGenerator(
    images = x_test,
    labels = y_test,
    subset_ratio = .5,
    base_ds=BASE,
    aux_ds=AUXILIARY
)

base_images_test, base_labels_test = test_ds_gen.build_base_dataset()
aux_images_test, aux_labels_test = test_ds_gen.build_aux_dataset()

logger.info(f"\tTest Dataset Base Size: {len(base_images_test)}")
logger.info(f"\tTest Dataset Aux Size: {len(aux_images_test)}")

val_ds_gen: DatasetGenerator = DatasetGenerator(
    images = x_val,
    labels = y_val,
    subset_ratio = .5,
    base_ds=BASE,
    aux_ds=AUXILIARY
)

base_images_val, base_labels_val = val_ds_gen.build_base_dataset()
aux_images_val, aux_labels_val = val_ds_gen.build_aux_dataset()

logger.info(f"\tValidation Dataset Base Size: {len(base_images_val)}")
logger.info(f"\tValidation Dataset Aux Size: {len(aux_images_val)}")

train_dataset: PairedMNISTDataset = PairedMNISTDataset(
    base_images=base_images_train,
    base_labels=base_labels_train,
    aux_images=aux_images_train,
    aux_labels=aux_labels_train
)

test_dataset: PairedMNISTDataset = PairedMNISTDataset(
    base_images=base_images_test,
    base_labels=base_labels_test,
    aux_images=aux_images_test,
    aux_labels=aux_labels_test
)

val_dataset: PairedMNISTDataset = PairedMNISTDataset(
    base_images=base_images_val,
    base_labels=base_labels_val,
    aux_images=aux_images_val,
    aux_labels=aux_labels_val
)

train_loader: DataLoader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

test_loader: DataLoader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    drop_last=True
)

val_loader: DataLoader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    drop_last=True
)

dl_set = DataLoaderSet(
    train_loader=train_loader,
    test_loader=test_loader,
    val_loader=val_loader
)

for x, y, z in val_loader:
    INPUT_SHAPE = x.shape
    break

logger.info(f"\tShape of Inputs: {INPUT_SHAPE}")


logger.info("Training Base Model")
base_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_base_classifier_{BASE}+{AUXILIARY}.pt"

model = DynamicCNN(
    input_shape=INPUT_SHAPE,
    num_filters=CONFIG.classifier.num_filters,
    kernel_size=CONFIG.classifier.kernel_size,
    stride=CONFIG.classifier.stride,
    padding=CONFIG.classifier.padding,
    mlp_layer_sizes=CONFIG.classifier.mlp_layer_sizes,
    num_classes=CONFIG.classifier.num_classes
)

base_model_trainer = Trainer(
    classifier = model,
    dataloaders=dl_set
)

base_model_trainer.classification_train_loop(
    filename = base_model_file,
    device=DEVICE,
    mode="base_only",
    num_epochs=10
)


logger.info("Training Mixed Model")
mixed_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_classifier_{BASE}+{AUXILIARY}.pt"

model = DynamicCNN(
    input_shape=INPUT_SHAPE,
    num_filters=CONFIG.classifier.num_filters,
    kernel_size=CONFIG.classifier.kernel_size,
    stride=CONFIG.classifier.stride,
    padding=CONFIG.classifier.padding,
    mlp_layer_sizes=CONFIG.classifier.mlp_layer_sizes,
    num_classes=CONFIG.classifier.num_classes
)

mixed_model_trainer = Trainer(
    classifier = model,
    dataloaders=dl_set
)

mixed_model_trainer.classification_train_loop(
    filename = mixed_model_file,
    device=DEVICE,
    mode="mixed",
    num_epochs=10
)


logger.info("Training Contrastive Model")
contrast_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_body_{BASE}+{AUXILIARY}.pt"

model = DynamicCNN(
    input_shape=INPUT_SHAPE,
    num_filters=CONFIG.classifier.num_filters,
    kernel_size=CONFIG.classifier.kernel_size,
    stride=CONFIG.classifier.stride,
    padding=CONFIG.classifier.padding,
    mlp_layer_sizes=CONFIG.classifier.mlp_layer_sizes,
    num_classes=CONFIG.classifier.num_classes
)

contrast_model_trainer = Trainer(
    classifier = model,
    dataloaders=dl_set,
    contrastive=True
)

contrast_model_trainer.contrastive_train_loop(
    filename = contrast_model_file,
    device=DEVICE,
    num_epochs=10
)


logger.info("Getting Model Accuracy")
base_acc = base_model_trainer.evaluate_model(DEVICE)
logger.info(f"Contrastive Model Accuracy: {round(base_acc*100, 2)}%")

mixed_acc = mixed_model_trainer.evaluate_model(DEVICE)
logger.info(f"Contrastive Model Accuracy: {round(mixed_acc*100, 2)}%")

contrast_acc = contrast_model_trainer.evaluate_model(DEVICE)
logger.info(f"Contrastive Model Accuracy: {round(contrast_acc*100, 2)}%")


logger.info("Generating TSNE Plot")
tsne_plot_file = f'{IMAGE_FOLDER}/{CLASSIFIER_ID}_TSNE_{BASE}+{AUXILIARY}.pdf'

models = ModelSet(
    base=base_model_trainer.classifier,
    mixed=mixed_model_trainer.classifier,
    contrast=contrast_model_trainer.classifier
)

accuracies = ModelSet(
    base=base_acc,
    mixed=mixed_acc,
    contrast=contrast_acc
)

tsne_plotter = TSNE_Plotter(
    dataloaders=dl_set,
    embed_size=CONFIG.classifier.mlp_layer_sizes[-1]
)

tsne_plotter.plot_tsne(
    classifiers=models,
    accuracies=accuracies,
    device=DEVICE,
    filename=tsne_plot_file,
    base=BASE,
    aux=AUXILIARY
)


logger.info("Generating Energy-Based Wasserstein Distance Plot")
ebsw_plot_file = f'{IMAGE_FOLDER}/{CLASSIFIER_ID}_EBSW_{BASE}+{AUXILIARY}.pdf'

ebsw_plotter = EBSW_Plotter(dataloaders=dl_set)
num_layers = base_model_trainer.classifier.get_num_layers()

ebsw_plotter.plot_ebsw(
    models=models,
    layers=[i for i in range(num_layers)],
    device=DEVICE,
    filename=ebsw_plot_file,
    num_projections=256
)


logger.info("Training UNET for Base Model")
base_unet = CustomUNET()
base_unet_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_base_unet_{BASE}+{AUXILIARY}.pt"

base_model_trainer.unet = base_unet

base_model_trainer.unet_train_loop(
    filename = base_unet_fname,
    device=DEVICE
)