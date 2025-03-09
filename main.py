import torch
from torchvision import transforms
import numpy as np
import os
from typing import Literal, Callable
import logging
import click

import datasets
import helpers
import trainer
import transforms as tr
import plotters
import models
import type_defs

@click.command()
@click.option("--config_fname", help="Config file path for the script (likely in configs folder)")
def main(config_fname):
    DEVICE: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"

    config_yaml = helpers.load_yaml(config_fname)
    CONFIG: type_defs.Config = type_defs.Config(**config_yaml)

    # define constants

    # create a sub directory to go in the folders (stores the data in a folder related to the model information)
    unet_name = "attention_unet" if CONFIG.unet.attention else "unet"
    SUB_FOLDER = f"{CONFIG.classifier.model}-{unet_name}" 

    # folder locations
    MODEL_FOLDER: str = os.path.join(
        CONFIG.save_locations.model_folder,
        SUB_FOLDER
    )
    FILE_FOLDER: str = os.path.join(
        CONFIG.save_locations.file_folder,
        SUB_FOLDER
    )
    IMAGE_FOLDER: str = os.path.join(
        CONFIG.save_locations.image_folder,
        SUB_FOLDER
    )

    # domain names
    TARGET: str = CONFIG.dataset.target.name
    SOURCE: str = CONFIG.dataset.source.name

    # data stuff
    BATCH_SIZE: int = CONFIG.dataset.batch_size
    CLASSIFIER_ID: str = CONFIG.classifier.identifier

    # data size info
    TARGET_SIZE: int = CONFIG.dataset.source.train_size
    SOURCE_SIZE: int = CONFIG.dataset.source.train_size

    build_unet: Callable = helpers.make_unet(
        size=CONFIG.dataset.image_size,
        attention=CONFIG.unet.attention,
        base_channels=CONFIG.unet.base_channels,
        noise_channels=CONFIG.unet.noise_channels
    )

    log_folder: str = os.path.join(
        CONFIG.save_locations.logs_folder,
        SUB_FOLDER
    )
    for folder in [MODEL_FOLDER, FILE_FOLDER, IMAGE_FOLDER, log_folder]:
        os.makedirs(folder, exist_ok=True)

    # enable reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(CONFIG.dataset.rng_seed)
    torch.cuda.manual_seed(CONFIG.dataset.rng_seed)
    np.random.seed(CONFIG.dataset.rng_seed)

    # start logging
    log_location: str = os.path.join(log_folder, f"run_{CLASSIFIER_ID}.log")
    level: int = logging.DEBUG if CONFIG.verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_location,
        filemode='w'
    )

    logger = logging.getLogger(__name__)
    logger.info(f"DEVICE: {DEVICE}")
    logger.info(f"Classifier ID: {CLASSIFIER_ID}")
    logger.info(f"Target Dataset: {TARGET}")
    logger.info(f"Source Dataset: {SOURCE}\n")

    # dataset creation
    logger.info("Creating Datasets")
    target_dir: str = CONFIG.dataset.target.folder
    source_dir: str = CONFIG.dataset.source.folder

    if CONFIG.dataset.image_size == "small":
        transform = transforms.Compose([
            transforms.Resize(32),  # Ensure consistency if needed
            tr.EnsureThreeChannelsPIL(),  # Convert to 3 channels
            transforms.ToTensor(),
        ])
    elif CONFIG.dataset.image_size == "large":
        transform = transforms.Compose([
            transforms.Resize(224),  # Ensure consistency if needed
            tr.EnsureThreeChannelsPIL(),  # Convert to 3 channels
            transforms.ToTensor(),
        ])

    target_full_dataset: datasets.HEIFFolder = datasets.HEIFFolder(
        target_dir, transform=transform
    )
    target_inds: np.ndarray = np.arange(len(target_full_dataset))
    np.random.shuffle(target_inds)

    target_train_size: int = CONFIG.dataset.target.train_size
    target_val_size: int = CONFIG.dataset.target.val_size

    target_train_inds: np.ndarray = target_inds[:target_train_size]
    target_val_inds: np.ndarray = target_inds[target_train_size:target_train_size+target_val_size]
    target_test_inds: np.ndarray = target_inds[target_train_size+target_val_size:]

    source_full_dataset: datasets.HEIFFolder = datasets.HEIFFolder(
        source_dir, transform=transform
    )
    source_inds: np.ndarray = np.arange(len(source_full_dataset))
    np.random.shuffle(source_inds)

    source_train_size: int = CONFIG.dataset.source.train_size
    source_val_size: int = CONFIG.dataset.source.val_size

    source_train_inds: np.ndarray = source_inds[:source_train_size]
    source_val_inds: np.ndarray = source_inds[source_train_size:source_train_size+source_val_size]
    source_test_inds: np.ndarray = source_inds[source_train_size+source_val_size:]

    target_train_ds: datasets.IndexedDataset = datasets.IndexedDataset(
        base_dataset=target_full_dataset,
        indices=target_train_inds
    )
    target_test_ds: datasets.IndexedDataset = datasets.IndexedDataset(
        base_dataset=target_full_dataset,
        indices=target_test_inds
    )
    target_val_ds: datasets.IndexedDataset = datasets.IndexedDataset(
        base_dataset=target_full_dataset,
        indices=target_val_inds
    )

    source_train_ds: datasets.IndexedDataset = datasets.IndexedDataset(
        base_dataset=source_full_dataset,
        indices=source_train_inds
    )
    source_test_ds: datasets.IndexedDataset = datasets.IndexedDataset(
        base_dataset=source_full_dataset,
        indices=source_test_inds
    )
    source_val_ds: datasets.IndexedDataset = datasets.IndexedDataset(
        base_dataset=source_full_dataset,
        indices=source_val_inds
    )

    train_ds: datasets.CombinedDataset = datasets.CombinedDataset(
        base_dataset=target_train_ds,
        aux_dataset=source_train_ds
    )

    test_ds: datasets.CombinedDataset = datasets.CombinedDataset(
        base_dataset=target_test_ds,
        aux_dataset=source_train_ds
    )

    val_ds: datasets.CombinedDataset = datasets.CombinedDataset(
        base_dataset=target_val_ds,
        aux_dataset=source_train_ds
    )

    logger.info(f"\tTrain Dataset Size: {len(train_ds)} (Target: {len(target_train_ds)}, Source: {len(source_train_ds)})")
    logger.info(f"\tTest Dataset Size: {len(test_ds)} (Target: {len(target_test_ds)}, Source: {len(source_test_ds)})")
    logger.info(f"\tValidation Dataset Aux Size: {len(val_ds)} (Target: {len(target_val_ds)}, Source: {len(source_val_ds)})")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    dl_set: type_defs.DataLoaderSet = type_defs.DataLoaderSet(
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader
    )

    for x, _, _ in val_loader:
        INPUT_SHAPE = x.shape
        break

    logger.info("TRAINING CLASSIFIERS\n--------------------")

    logger.info("\nTraining Baseline Model")
    baseline_ae_filename = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_baseline_autoencoder_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    baseline_unet_filename = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_baseline_unet_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    baseline_classifier_filename = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_baseline_classifier_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    baseline_autoencoder = models.CustomAutoencoder()
    baseline_unet = build_unet()
    baseline_classifier = models.DynamicResNet(
        resnet_type=CONFIG.classifier.model,
        num_classes=10,
    )

    baseline_model_trainer = trainer.PreloaderTrainer(
        autoencoder=baseline_autoencoder,
        unet = baseline_unet,
        classifier = baseline_classifier,
        dataloaders=dl_set
    )

    baseline_model_trainer.unet_preloader_train_loop(
        ae_filename=baseline_ae_filename,
        unet_filename=baseline_unet_filename,
        device=DEVICE,
        train_both=True
    )

    baseline_model_trainer.classification_train_loop(
        classifier_filename=baseline_classifier_filename,
        device=DEVICE,
        num_epochs=100
    )

    baseline_val_acc = baseline_model_trainer.evaluate_model(device=DEVICE)

    logger.info("\nTraining Base Model")
    base_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_base_classifier_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    if not os.path.exists(base_model_file):
        model: models.DynamicResNet = models.DynamicResNet(
            resnet_type=CONFIG.classifier.model,
            num_classes=10,
        )

        base_model_trainer: trainer.Trainer = trainer.Trainer(
            classifier = model,
            dataloaders=dl_set
        )

        base_model_trainer.classification_train_loop(
            filename = base_model_file,
            device=DEVICE,
            num_epochs=CONFIG.classifier.num_epochs,
            mode="base_only"
        )

    logger.info("\nTraining Mixed Model")
    mixed_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_classifier_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    if not os.path.exists(mixed_model_file):
        model: models.DynamicResNet = models.DynamicResNet(
            resnet_type=CONFIG.classifier.model,
            num_classes=10
        )

        mixed_model_trainer: trainer.Trainer = trainer.Trainer(
            classifier = model,
            dataloaders=dl_set
        )

        mixed_model_trainer.classification_train_loop(
            filename = mixed_model_file,
            device=DEVICE,
            num_epochs=CONFIG.classifier.num_epochs,
            mode="mixed",
        )

    logger.info("\nTraining Contrastive Model")
    contrast_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_body_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    contrast_full_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_full_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    if not os.path.exists(contrast_full_model_file): 
        model: models.DynamicResNet = models.DynamicResNet(
            resnet_type=CONFIG.classifier.model,
            num_classes=10
        )

        contrast_model_trainer: trainer.Trainer = trainer.Trainer(
            classifier = model,
            dataloaders=dl_set,
            contrastive=True
        )

        best_temp = contrast_model_trainer.contrastive_train_loop(
            filename = contrast_model_file,
            device=DEVICE,
            num_epochs=CONFIG.classifier.num_epochs,
            temp_range=[0.05, 0.1, 0.15],
        )


    logger.info("\nGetting Model Accuracy")
    base_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_base_classifier_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    mixed_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_classifier_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    contrast_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_full_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    base_model = models.DynamicResNet(
        resnet_type=CONFIG.classifier.model,
        num_classes=10
    )
    base_model.load_state_dict(torch.load(base_model_file, weights_only=True))
    base_model_trainer = trainer.Trainer(
        classifier = base_model,
        dataloaders=dl_set
    )
    base_model_trainer.classifier = base_model

    mixed_model = models.DynamicResNet(
        resnet_type=CONFIG.classifier.model,
        num_classes=10
    )
    mixed_model.load_state_dict(torch.load(mixed_model_file, weights_only=True))
    mixed_model_trainer = trainer.Trainer(
        classifier = mixed_model,
        dataloaders=dl_set
    )
    mixed_model_trainer.classifier = mixed_model

    contrast_model = models.DynamicResNet(
        resnet_type=CONFIG.classifier.model,
        num_classes=10
    )
    contrast_model.load_state_dict(torch.load(contrast_model_file, weights_only=True))
    contrast_model_trainer = trainer.Trainer(
        classifier = contrast_model,
        dataloaders=dl_set,
        contrastive=True
    )
    contrast_model_trainer.classifier = contrast_model

    # logger.info(f"Baseline Model Accuracy: {round(baseline_val_acc*100, 2)}%")

    base_acc: float = base_model_trainer.evaluate_model(DEVICE)
    logger.info(f"Base Model Accuracy: {round(base_acc*100, 2)}%")

    mixed_acc: float = mixed_model_trainer.evaluate_model(DEVICE)
    logger.info(f"Mixed Model Accuracy: {round(mixed_acc*100, 2)}%")

    contrast_acc: float = contrast_model_trainer.evaluate_model(DEVICE)
    logger.info(f"Contrastive Model Accuracy: {round(contrast_acc*100, 2)}%")

    logger.info("\nGenerating TSNE Plot")
    tsne_plot_file: str = f"{IMAGE_FOLDER}/{CLASSIFIER_ID}_TSNE_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pdf"

    model_set = type_defs.ModelSet(
        base=base_model_trainer.classifier,
        mixed=mixed_model_trainer.classifier,
        contrast=contrast_model_trainer.classifier
    )

    accuracies = type_defs.ModelSet(
        base=base_acc,
        mixed=mixed_acc,
        contrast=contrast_acc
    )

    tsne_plotter = plotters.TSNE_Plotter(
        dataloaders=dl_set,
        embed_size=mixed_model.get_body_output_size(),
        bs=BATCH_SIZE
    )

    tsne_plotter.plot_tsne(
        models=model_set,
        accuracies=accuracies,
        device=DEVICE,
        filename=tsne_plot_file,
        base=TARGET,
        aux=SOURCE
    )

    logger.info("\nGenerating Energy-Based Wasserstein Distance Plot")
    ebsw_plot_file = f"{IMAGE_FOLDER}/{CLASSIFIER_ID}_EBSW_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pdf"

    ebsw_plotter = plotters.EBSW_Plotter(
        dataloaders=dl_set,
        batch_size=BATCH_SIZE
    )
    num_layers = base_model_trainer.classifier.get_num_layers()

    ebsw_plotter.plot_ebsw(
        models=model_set,
        layers=[i for i in range(num_layers-1)],
        device=DEVICE,
        filename=ebsw_plot_file,
        num_projections=256
    )

    logger.info("\nSTARTING UNET/CLASSIFER TRAIN CYCLES\n------------------------------------")

    logger.info("Training UNET/Classifier for Mixed Model")
    mixed_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_unet_FINAL_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    mixed_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_classifier_FINAL_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    mixed_model_trainer.unet = build_unet()

    mixed_model_trainer.unet_classifier_train_loop(
        unet_filename=mixed_unet_final_fname,
        classifier_filename=mixed_classifier_final_fname,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    logger.info("Training UNET/Classifier for Contrast Model")
    contrast_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_unet_FINAL_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    contrast_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_classifier_FINAL_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    contrast_model_trainer.unet = build_unet()

    contrast_model_trainer.unet_classifier_train_loop(
        unet_filename=contrast_unet_final_fname,
        classifier_filename=contrast_classifier_final_fname,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    logger.info("\nGetting Model Accuracy With UNET Models")

    mixed_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_unet_FINAL_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    mixed_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_classifier_FINAL_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    contrast_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_unet_FINAL_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    contrast_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_classifier_FINAL_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    mixed_model = models.DynamicResNet(
        resnet_type=CONFIG.classifier.model,
        num_classes=10
    )
    mixed_model.load_state_dict(torch.load(mixed_classifier_final_fname, weights_only=True))
    mixed_unet = build_unet()
    mixed_unet.load_state_dict(torch.load(mixed_unet_final_fname, weights_only=True))

    mixed_model_trainer.classifier = mixed_model
    mixed_model_trainer.unet = mixed_unet

    contrast_model = models.DynamicResNet(
        resnet_type=CONFIG.classifier.model,
        num_classes=10
    )
    contrast_model.load_state_dict(torch.load(contrast_classifier_final_fname, weights_only=True))
    contrast_unet = build_unet()
    contrast_unet.load_state_dict(torch.load(contrast_unet_final_fname, weights_only=True))

    contrast_model_trainer.classifier = contrast_model
    contrast_model_trainer.unet = contrast_unet

    # logger.info(f"Baseline Model Accuracy: {round(baseline_val_acc*100, 2)}%")

    base_acc = base_model_trainer.evaluate_model(DEVICE)
    logger.info(f"Base Model Accuracy w/ UNET: {round(base_acc*100, 2)}%")

    mixed_acc = mixed_model_trainer.evaluate_model(DEVICE)
    logger.info(f"Mixed Model Accuracy w/ UNET: {round(mixed_acc*100, 2)}%")

    contrast_acc = contrast_model_trainer.evaluate_model(DEVICE)
    logger.info(f"Contrastive Model Accuracy w/ UNET: {round(contrast_acc*100, 2)}%")

    logger.info("Generating Image Example Plots")
    base_example_file = f"{IMAGE_FOLDER}/{CLASSIFIER_ID}_base_examples_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pdf"
    mixed_example_file = f"{IMAGE_FOLDER}/{CLASSIFIER_ID}_mixed_examples_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pdf"
    contrast_example_file = f"{IMAGE_FOLDER}/{CLASSIFIER_ID}_contrast_examples_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pdf"

    plotters.plot_examples(
        dataset=val_ds,
        unet_model=None,
        filename=base_example_file,
        device=DEVICE
    )

    plotters.plot_examples(
        dataset=train_ds,
        unet_model=mixed_model_trainer.unet,
        filename=mixed_example_file,
        device=DEVICE
    )

    plotters.plot_examples(
        dataset=train_ds,
        unet_model=contrast_model_trainer.unet,
        filename=contrast_example_file,
        device=DEVICE
    )


    logger.info("\nGenerating TSNE w/ UNET Plot")
    tsne_plot_file = f"{IMAGE_FOLDER}/{CLASSIFIER_ID}_TSNE_UNET_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pdf"

    model_set = type_defs.ModelSet(
        base=base_model_trainer.classifier,
        mixed=mixed_model_trainer.classifier,
        contrast=contrast_model_trainer.classifier
    )

    accuracies = type_defs.ModelSet(
        base=base_acc,
        mixed=mixed_acc,
        contrast=contrast_acc
    )

    unets = type_defs.ModelSet(
        base=None,
        mixed=mixed_model_trainer.unet,
        contrast=contrast_model_trainer.unet
    )

    tsne_plotter = plotters.TSNE_Plotter(
        dataloaders=dl_set,
        embed_size=mixed_model.get_body_output_size(),
        bs=BATCH_SIZE
    )

    tsne_plotter.plot_tsne(
        models=model_set,
        unet_models=unets,
        accuracies=accuracies,
        device=DEVICE,
        filename=tsne_plot_file,
        base=TARGET,
        aux=SOURCE
    )

    logger.info("Generating Energy-Based Wasserstein Distance Plot")
    ebsw_plot_file = f"{IMAGE_FOLDER}/{CLASSIFIER_ID}{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pdf"

    ebsw_plotter = plotters.EBSW_Plotter(
        dataloaders=dl_set,
        batch_size=BATCH_SIZE
    )
    num_layers = base_model_trainer.classifier.get_num_layers()

    ebsw_plotter.plot_ebsw(
        models=model_set,
        unet_models=unets,
        layers=[i for i in range(num_layers-1)],
        device=DEVICE,
        filename=ebsw_plot_file,
        num_projections=256
    )

    logger.info(f"Finished {CLASSIFIER_ID} Dataset")

if __name__=="__main__":
    main()