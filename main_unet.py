import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from typing import Literal
import logging

from datasets import DatasetGenerator, PairedMNISTDataset, CombinedDataset, HEIFFolder
import helpers
from trainer import Trainer
from transforms import EnsureThreeChannelsPIL
from plotters import TSNE_Plotter, EBSW_Plotter
from models import CustomUNET, DynamicCNN, DynamicResNet
from type_defs import Config, DataLoaderSet, ModelSet

# constants
DEVICE: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"

config_yaml = helpers.load_yaml(r"configs/tiny_config.yaml")
CONFIG: Config = Config(**config_yaml)

MODEL_FOLDER: str = CONFIG.save_locations.model_folder
FILE_FOLDER: str = CONFIG.save_locations.file_folder
IMAGE_FOLDER: str = CONFIG.save_locations.image_folder

BASE: str = CONFIG.dataset.base_name
AUXILIARY: str = CONFIG.dataset.aux_name

BATCH_SIZE: int = CONFIG.dataset.batch_size
CLASSIFIER_ID: str = CONFIG.classifier.identifier

log_folder = CONFIG.save_locations.logs_folder
for folder in [MODEL_FOLDER, FILE_FOLDER, IMAGE_FOLDER, log_folder]:
    if not os.path.exists(folder):
        os.mkdir(folder)


# start logging
log_location = os.path.join(log_folder, f"run_{CLASSIFIER_ID}.log")
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
base_dir = CONFIG.dataset.base_folder
aux_dir = CONFIG.dataset.aux_folder

transform = transforms.Compose([
    EnsureThreeChannelsPIL(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

base_full_dataset = HEIFFolder(base_dir, transform=transform)
aux_full_dataset = HEIFFolder(aux_dir, transform=transform)

combined_dataset = CombinedDataset(
    base_dataset=base_full_dataset,
    aux_dataset=aux_full_dataset
)

train_size = int(.1 * len(combined_dataset))
val_size = int(.1 * len(combined_dataset))
test_size = len(combined_dataset) - train_size - val_size

train_ds, test_ds, val_ds = torch.utils.data.random_split(
    combined_dataset,
    [train_size, test_size, val_size]
)

logger.info(f"\tTrain Dataset Size: {len(train_ds)}")
logger.info(f"\tTest Dataset Size: {len(test_ds)}")
logger.info(f"\tValidation Dataset Aux Size: {len(val_ds)}")

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False
)

val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False
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

logger.info("TRAINING CLASSIFIERS\n--------------------")

logger.info("Training Base Model")
base_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_base_classifier_{BASE}+{AUXILIARY}.pt"

model = DynamicResNet(
    resnet_type='resnet9',
    num_classes=10
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

model = DynamicResNet(
    resnet_type='resnet9',
    num_classes=10
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

model = DynamicResNet(
    resnet_type='resnet9',
    num_classes=10
)


contrast_model_trainer = Trainer(
    classifier = model,
    dataloaders=dl_set,
    contrastive=True
)

best_temp = contrast_model_trainer.contrastive_train_loop(
    filename = contrast_model_file,
    device=DEVICE,
    num_epochs=10
)


logger.info("Getting Model Accuracy")
base_acc = base_model_trainer.evaluate_model(DEVICE)
logger.info(f"Base Model Accuracy: {round(base_acc*100, 2)}%")

mixed_acc = mixed_model_trainer.evaluate_model(DEVICE)
logger.info(f"Mixed Model Accuracy: {round(mixed_acc*100, 2)}%")

contrast_acc = contrast_model_trainer.evaluate_model(DEVICE)
logger.info(f"Contrastive Model Accuracy: {round(contrast_acc*100, 2)}%")


logger.info("GENERATING INITIAL PLOTS\n------------------------")

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
    models=models,
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

logger.info("STARTING UNET WARM STARTS\n-------------------------")


logger.info("Training UNET for Base Model")
base_unet = CustomUNET()
base_unet_ws_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_base_unet_WS_{BASE}+{AUXILIARY}.pt"

base_model_trainer.unet = base_unet

base_model_trainer.unet_train_loop(
    filename = base_unet_ws_fname,
    device=DEVICE
)


logger.info("Training UNET for Mixed Model")
mixed_unet = CustomUNET()
mixed_unet_ws_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_unet_WS_{BASE}+{AUXILIARY}.pt"

mixed_model_trainer.unet = mixed_unet

mixed_model_trainer.unet_train_loop(
    filename = mixed_unet_ws_fname,
    device=DEVICE
)


logger.info("Training UNET for Contrast Model")
contrast_unet = CustomUNET()
contrast_unet_ws_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_unet_WS_{BASE}+{AUXILIARY}.pt"

contrast_model_trainer.unet = contrast_unet

contrast_model_trainer.unet_train_loop(
    filename = contrast_unet_ws_fname,
    device=DEVICE
)


logger.info("Getting Model + UNET (WS) Accuracy")

base_acc = base_model_trainer.evaluate_model(DEVICE)
logger.info(f"Base Model + UNET (WS) Accuracy: {round(base_acc*100, 2)}%")

mixed_acc = mixed_model_trainer.evaluate_model(DEVICE)
logger.info(f"Mixed Model + UNET (WS) Accuracy: {round(mixed_acc*100, 2)}%")

contrast_acc = contrast_model_trainer.evaluate_model(DEVICE)
logger.info(f"Contrastive Model + UNET (WS) Accuracy: {round(contrast_acc*100, 2)}%")


logger.info("GENERATING WARM START UNET PLOTS\n--------------------------------")

logger.info("Generating TSNE Plot")
tsne_plot_file = f'{IMAGE_FOLDER}/{CLASSIFIER_ID}_TSNE_UNET_WS_{BASE}+{AUXILIARY}.pdf'

models = ModelSet(
    base=base_model_trainer.classifier,
    mixed=mixed_model_trainer.classifier,
    contrast=contrast_model_trainer.classifier
)

unet_models = ModelSet(
    base=base_unet,
    mixed=mixed_unet,
    contrast=contrast_unet
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
    models=models,
    unet_models=unet_models,
    accuracies=accuracies,
    device=DEVICE,
    filename=tsne_plot_file,
    base=BASE,
    aux=AUXILIARY
)


logger.info("Generating Energy-Based Wasserstein Distance Plot")
ebsw_plot_file = f'{IMAGE_FOLDER}/{CLASSIFIER_ID}_EBSW_UNET_WS_{BASE}+{AUXILIARY}.pdf'

ebsw_plotter = EBSW_Plotter(dataloaders=dl_set)
num_layers = base_model_trainer.classifier.get_num_layers()

ebsw_plotter.plot_ebsw(
    models=models,
    unet_models=unet_models,
    layers=[i for i in range(num_layers)],
    device=DEVICE,
    filename=ebsw_plot_file,
    num_projections=256
)


logger.info("STARTING UNET/CLASSIFER TRAIN CYCLES\n------------------------------------")

logger.info("Training UNET/Classifier for Base Model")
base_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_base_unet_FINAL_{BASE}+{AUXILIARY}.pt"
base_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_base_classifier_FINAL_{BASE}+{AUXILIARY}.pt"


base_model_trainer.unet_classifier_train_loop(
    unet_filename=base_unet_final_fname,
    classifier_filename=base_classifier_final_fname,
    device=DEVICE
)


logger.info("Training UNET/Classifier for Mixed Model")
mixed_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_unet_FINAL_{BASE}+{AUXILIARY}.pt"
mixed_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_classifier_FINAL_{BASE}+{AUXILIARY}.pt"


mixed_model_trainer.unet_classifier_train_loop(
    unet_filename=mixed_unet_final_fname,
    classifier_filename=mixed_classifier_final_fname,
    device=DEVICE
)


logger.info("Training UNET/Classifier for Contrast Model")
contrast_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_unet_FINAL_{BASE}+{AUXILIARY}.pt"
contrast_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_classifier_FINAL_{BASE}+{AUXILIARY}.pt"


contrast_model_trainer.unet_contrastive_train_loop(
    unet_filename=contrast_unet_final_fname,
    classifier_filename=contrast_classifier_final_fname,
    best_temp=best_temp,
    device=DEVICE
)


logger.info("Getting Model + UNET (FINAL) Accuracy")

base_acc = base_model_trainer.evaluate_model(DEVICE)
logger.info(f"Base Model + UNET (FINAL) Accuracy: {round(base_acc*100, 2)}%")

mixed_acc = mixed_model_trainer.evaluate_model(DEVICE)
logger.info(f"Mixed Model + UNET (FINAL) Accuracy: {round(mixed_acc*100, 2)}%")

contrast_acc = contrast_model_trainer.evaluate_model(DEVICE)
logger.info(f"Contrastive Model + UNET (FINAL) Accuracy: {round(contrast_acc*100, 2)}%")


logger.info("GENERATING FINAL UNET PLOTS\n---------------------------")

logger.info("Generating TSNE Plot")
tsne_plot_file = f'{IMAGE_FOLDER}/{CLASSIFIER_ID}_TSNE_UNET_FINAL_{BASE}+{AUXILIARY}.pdf'

models = ModelSet(
    base=base_model_trainer.classifier,
    mixed=mixed_model_trainer.classifier,
    contrast=contrast_model_trainer.classifier
)

unet_models = ModelSet(
    base=base_unet,
    mixed=mixed_unet,
    contrast=contrast_unet
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
    models=models,
    unet_models=unet_models,
    accuracies=accuracies,
    device=DEVICE,
    filename=tsne_plot_file,
    base=BASE,
    aux=AUXILIARY
)


logger.info("Generating Energy-Based Wasserstein Distance Plot")
ebsw_plot_file = f'{IMAGE_FOLDER}/{CLASSIFIER_ID}_EBSW_UNET_FINAL_{BASE}+{AUXILIARY}.pdf'

ebsw_plotter = EBSW_Plotter(dataloaders=dl_set)
num_layers = base_model_trainer.classifier.get_num_layers()

ebsw_plotter.plot_ebsw(
    models=models,
    unet_models=unet_models,
    layers=[i for i in range(num_layers)],
    device=DEVICE,
    filename=ebsw_plot_file,
    num_projections=256
)