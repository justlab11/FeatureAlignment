import torch
from torchvision import transforms
import numpy as np
import os
from typing import Literal, Callable
import logging
import click
import glob
import random

import datasets
import helpers
import trainer
import transforms as tr
import plotters
import models
import type_defs
import samplers

@click.command()
@click.option("--config_fname", help="Config file path for the script (likely in configs folder)")
def main(config_fname):
    DEVICE: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"

    config_yaml = helpers.load_yaml(config_fname)
    CONFIG: type_defs.Config = type_defs.Config(**config_yaml)

    # define constants
    target_dir: str = CONFIG.dataset.target.folder
    source_dir: str = CONFIG.dataset.source.folder

    # domain names
    TARGET: str = CONFIG.dataset.target.name
    SOURCE: str = CONFIG.dataset.source.name

    # data stuff
    BATCH_SIZE: int = CONFIG.dataset.batch_size
    CLASSIFIER_ID: str = CONFIG.classifier.identifier
    
    unet_name = "attention_unet" if CONFIG.unet.attention else "unet"
    CLASSIFIER_ID += f"-{CONFIG.dataset.image_size}-{CONFIG.unet.loss}-{CONFIG.classifier.model}-{unet_name}"

    # data size info
    target_size = len(glob.glob(
        os.path.join(target_dir, "**", "*"),
        recursive=True
    ))

    source_size = len(glob.glob(
        os.path.join(source_dir, "**", "*"),
        recursive=True
    ))

    TARGET_TRAIN_SIZE: int = int(CONFIG.dataset.target.train_pct * target_size)
    SOURCE_TRAIN_SIZE: int = int(CONFIG.dataset.source.train_pct * source_size)

    LOSS: str = CONFIG.unet.loss
    RNG: int = CONFIG.dataset.rng_seed
    TARGET_NUM_CLASSES: int = CONFIG.dataset.target.num_classes
    SOURCE_NUM_CLASSES: int = CONFIG.dataset.source.num_classes

    # folder locations
    MODEL_FOLDER: str = os.path.join(
        CLASSIFIER_ID,
        "models",
    )
    FILE_FOLDER: str = os.path.join(
        CLASSIFIER_ID,
        "files",
    )
    IMAGE_FOLDER: str = os.path.join(
        CLASSIFIER_ID,
        "images",
    )

    log_folder: str = os.path.join(
        CLASSIFIER_ID,
        "logs",
    )

    print(f"Run location: {CLASSIFIER_ID}")
    for folder in [MODEL_FOLDER, FILE_FOLDER, IMAGE_FOLDER, log_folder]:
        os.makedirs(folder, exist_ok=True)

    # function to build unet with given parameters
    build_unet: Callable = helpers.make_unet(
        size=CONFIG.dataset.image_size,
        attention=CONFIG.unet.attention,
        base_channels=CONFIG.unet.base_channels,
        noise_channels=CONFIG.unet.noise_channels
    )

    # enable reproducibility
    torch.manual_seed(RNG)
    torch.cuda.manual_seed(RNG)
    np.random.seed(RNG)
    torch.backends.cudnn.deterministic = True
    random.seed(RNG)

    # start logging
    log_location: str = os.path.join(log_folder, f"run_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.log")
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
    logger.info("CREATING DATASETS")
    logger.info("-----------------")

    if CONFIG.dataset.image_size == "small":
        transform = transforms.Compose([
            transforms.Resize(32),  # Ensure consistency if needed
            tr.EnsureThreeChannelsPIL(),  # Convert to 3 channels
            transforms.ToTensor(),
        ])
    elif CONFIG.dataset.image_size == "large":
        transform = transforms.Compose([
            transforms.Resize(128),  # Ensure consistency if needed
            tr.EnsureThreeChannelsPIL(),  # Convert to 3 channels
            transforms.ToTensor(),
        ])

    target_train_split, target_test_split, target_val_split = helpers.build_splits(
        folder=target_dir,
        split_pcts=[
            CONFIG.dataset.target.train_pct,
            CONFIG.dataset.target.val_pct,
            1 - CONFIG.dataset.target.train_pct - CONFIG.dataset.target.val_pct
        ],
        seed=RNG
    )
    
    source_train_split, source_test_split, source_val_split = helpers.build_splits(
        folder=source_dir,
        split_pcts=[
            CONFIG.dataset.source.train_pct,
            CONFIG.dataset.source.val_pct,
            1 - CONFIG.dataset.source.train_pct - CONFIG.dataset.source.val_pct
        ],
        seed=RNG
    )

    train_ds: datasets.CombinedDataset = datasets.CombinedDataset(
        data_folder=target_dir,
        target_split_samples=target_train_split,
        source_split_samples=source_train_split,
        transform=transform
    )

    test_ds: datasets.CombinedDataset = datasets.CombinedDataset(
        data_folder=target_dir,
        target_split_samples=target_test_split,
        source_split_samples=source_test_split,
        transform=transform
    )

    val_ds: datasets.CombinedDataset = datasets.CombinedDataset(
        data_folder=target_dir,
        target_split_samples=target_val_split,
        source_split_samples=source_val_split,
        transform=transform
    )

    logger.info(f"\tTrain Dataset - Target: {train_ds.get_target_size()}, Source: {train_ds.get_source_size()}")
    logger.info(f"\tTest Dataset - Target: {test_ds.get_target_size()}, Source: {test_ds.get_source_size()}")
    logger.info(f"\tValidation Dataset - Target: {val_ds.get_target_size()}, Source: {val_ds.get_source_size()}")

    NUM_WORKERS = 7 # for runpod

    cls_train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=False, pin_memory=True
    )

    cls_test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    cls_val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    cls_dl_set: type_defs.DataLoaderSet = type_defs.DataLoaderSet(
        train_loader=cls_train_loader,
        test_loader=cls_test_loader,
        val_loader=cls_val_loader
    )

    train_sampler = samplers.PureBatchSampler(
        data_source=train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    test_sampler = samplers.PureBatchSampler(
        data_source=test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    val_sampler = samplers.PureBatchSampler(
        data_source=val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    align_train_loader = torch.utils.data.DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True
    )

    align_test_loader = torch.utils.data.DataLoader(
        test_ds, batch_sampler=test_sampler, num_workers=NUM_WORKERS, pin_memory=True
    )

    align_val_loader = torch.utils.data.DataLoader(
        val_ds, batch_sampler=val_sampler, num_workers=NUM_WORKERS, pin_memory=True
    )

    align_dl_set: type_defs.DataLoaderSet = type_defs.DataLoaderSet(
        train_loader=align_train_loader,
        test_loader=align_test_loader,
        val_loader=align_val_loader
    )

    for x, _, _ in cls_val_loader:
        INPUT_SHAPE = x.shape
        logger.info(f"Input Shape: {INPUT_SHAPE}")
        break

    logger.info("TRAINING CLASSIFIERS")
    logger.info("--------------------")

    logger.info("\nTraining Baseline Model")
    baseline_ae_filename = f"{MODEL_FOLDER}/baseline_autoencoder_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"
    baseline_unet_filename = f"{MODEL_FOLDER}/baseline_unet_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"
    baseline_classifier_filename = f"{MODEL_FOLDER}/baseline_classifier_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"

    baseline_autoencoder = models.CustomAutoencoder()
    baseline_unet = build_unet()
    baseline_classifier = models.DynamicResNet(
        resnet_type=CONFIG.classifier.model,
        num_classes=TARGET_NUM_CLASSES,
    )

    baseline_model_trainer = trainer.PreloaderTrainer(
        autoencoder = baseline_autoencoder,
        alignment_model = baseline_unet,
        classifier = baseline_classifier,
        cls_dataloaders = cls_dl_set,
        align_dataloaders=align_dl_set
    )

    baseline_model_trainer.alignment_preloader_train_loop(
        ae_filename=baseline_ae_filename,
        alignment_filename=baseline_unet_filename,
        device=DEVICE,
        train_both=True
    )

    baseline_model_trainer.classification_train_loop(
       classifier_filename=baseline_classifier_filename,
       device=DEVICE,
       num_epochs=100,
       use_alignment=True
    )

    _, baseline_acc = baseline_model_trainer.evaluate_model(device=DEVICE, test=True)
    logger.info(baseline_acc)

    logger.info("\nTraining Base Model")
    base_model_file = f"{MODEL_FOLDER}/base_classifier_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"

    model: models.DynamicResNet = models.DynamicResNet(
        resnet_type=CONFIG.classifier.model,
        num_classes=TARGET_NUM_CLASSES,
    )

    unet = build_unet()

    base_model_trainer = trainer.FullTrainer(
        classifier=model,
        alignment_model=unet,
        alignment_loss=LOSS,
        classifier_dataloaders=cls_dl_set,
        alignment_dataloaders=align_dl_set,
        file_folder = FILE_FOLDER,
        classifier_name="base"
    )

    if not os.path.exists(base_model_file):
        base_model_trainer.classifier_trainer.classification_train_loop(
            classifier_filename = base_model_file,
            device=DEVICE,
            num_epochs=CONFIG.classifier.num_epochs,
            target_only=True,
            use_alignment=False
        )
    else:
        base_model_trainer.classifier.load_state_dict(torch.load(base_model_file))

    logger.info("\nTraining Mixed Model")
    mixed_model_file = f"{MODEL_FOLDER}/mixed_classifier_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"

    model: models.DynamicResNet = models.DynamicResNet(
        resnet_type=CONFIG.classifier.model,
        num_classes=TARGET_NUM_CLASSES,
    )

    unet = build_unet()

    mixed_model_trainer = trainer.FullTrainer(
        classifier=model,
        alignment_model=unet,
        alignment_loss=LOSS,
        classifier_dataloaders=cls_dl_set,
        alignment_dataloaders=align_dl_set,
        file_folder = FILE_FOLDER,
        classifier_name="mixed"
    )

    if not os.path.exists(mixed_model_file):
        mixed_model_trainer.classifier_trainer.classification_train_loop(
            classifier_filename = mixed_model_file,
            device=DEVICE,
            num_epochs=CONFIG.classifier.num_epochs,
            target_only=False,
            use_alignment=False
        )
    else:
        mixed_model_trainer.classifier.load_state_dict(torch.load(mixed_model_file))

    logger.info("\nTraining Contrastive Model")
    contrast_model_file = f"{MODEL_FOLDER}/contrast_body_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"
    contrast_full_model_file = f"{MODEL_FOLDER}/contrast_full_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"

    model: models.DynamicResNet = models.DynamicResNet(
        resnet_type=CONFIG.classifier.model,
        num_classes=TARGET_NUM_CLASSES
    )

    unet = build_unet()

    contrast_model_trainer = trainer.FullTrainer(
        classifier=model,
        alignment_model=unet,
        alignment_loss=LOSS,
        classifier_dataloaders=cls_dl_set,
        alignment_dataloaders=align_dl_set,
        file_folder = FILE_FOLDER,
        classifier_name="contrast"
    )

    if not os.path.exists(contrast_full_model_file):
        best_temp = contrast_model_trainer.classifier_trainer.contrastive_train_loop(
            filename = contrast_model_file,
            device=DEVICE,
            num_epochs=CONFIG.classifier.num_epochs,
            temp_range=[0.05],
        )
    else:
        contrast_model_trainer.classifier.load_state_dict(torch.load(contrast_full_model_file))

    logger.info("\nGetting Model Accuracy")
    base_model_file = f"{MODEL_FOLDER}/base_classifier_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"
    mixed_model_file = f"{MODEL_FOLDER}/mixed_classifier_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"
    contrast_model_file = f"{MODEL_FOLDER}/contrast_full_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"

    logger.info(f"Baseline Model Accuracy: {round(baseline_acc*100, 2)}%")

    _, base_acc = base_model_trainer.classifier_trainer.evaluate_model(DEVICE, use_alignment=False, test=True)
    logger.info(f"Base Model Accuracy: {round(base_acc*100, 2)}%")

    _, mixed_acc = mixed_model_trainer.classifier_trainer.evaluate_model(DEVICE, use_alignment=False, test=True)
    logger.info(f"Mixed Model Accuracy: {round(mixed_acc*100, 2)}%")

    _, contrast_acc = contrast_model_trainer.classifier_trainer.evaluate_model(DEVICE, use_alignment=False, test=True)
    logger.info(f"Contrastive Model Accuracy: {round(contrast_acc*100, 2)}%")

    logger.info("\nGenerating TSNE Plot")
    tsne_plot_file: str = f"{IMAGE_FOLDER}/TSNE_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pdf"

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
    if TARGET_NUM_CLASSES == 10 and SOURCE_NUM_CLASSES == 10: 
        tsne_plotter = plotters.TSNE_Plotter(
            dataloaders=cls_dl_set,
            embed_size=model.get_body_output_size(),
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
    ebsw_plot_file = f"{IMAGE_FOLDER}/DIV_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pdf"

    ebsw_plotter = plotters.EBSW_Plotter(
        dataloaders=cls_dl_set,
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
    mixed_unet_final_fname = f"{MODEL_FOLDER}/mixed_unet_FINAL_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"
    mixed_classifier_final_fname = f"{MODEL_FOLDER}/mixed_classifier_FINAL_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"
    mixed_examples_final_fname = f"{IMAGE_FOLDER}/mixed_examples_FINAL_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"

    mixed_model_trainer.cascading_train_loop(
        alignment_fname=mixed_unet_final_fname,
        classifier_fname=mixed_classifier_final_fname,
        examples_fname=mixed_examples_final_fname,
        device=DEVICE,
    )

    logger.info("Training UNET/Classifier for Contrast Model")
    contrast_unet_final_fname = f"{MODEL_FOLDER}/contrast_unet_FINAL_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"
    contrast_classifier_final_fname = f"{MODEL_FOLDER}/contrast_classifier_FINAL_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"
    contrast_examples_final_fname = f"{IMAGE_FOLDER}/contrast_examples_FINAL_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pt"

    contrast_model_trainer.cascading_train_loop(
        alignment_fname=contrast_unet_final_fname,
        classifier_fname=contrast_classifier_final_fname,
        examples_fname=contrast_examples_final_fname,
        device=DEVICE,
    )

    logger.info("\nGetting Model Accuracy With UNET Models")

    logger.info(f"Baseline Model Accuracy: {round(baseline_acc*100, 2)}%")

    _, base_acc = base_model_trainer.classifier_trainer.evaluate_model(DEVICE, use_alignment=False, test=True)
    logger.info(f"Base Model Accuracy w/ UNET: {round(base_acc*100, 2)}%")

    _, mixed_acc = mixed_model_trainer.classifier_trainer.evaluate_model(DEVICE, use_alignment=False, test=True)
    logger.info(f"Mixed Model Accuracy w/ UNET: {round(mixed_acc*100, 2)}%")

    _, contrast_acc = contrast_model_trainer.classifier_trainer.evaluate_model(DEVICE, use_alignment=False, test=True)
    logger.info(f"Contrastive Model Accuracy w/ UNET: {round(contrast_acc*100, 2)}%")

    logger.info("Generating Image Example Plots")
    mixed_example_file = f"{IMAGE_FOLDER}/mixed_examples_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pdf"
    contrast_example_file = f"{IMAGE_FOLDER}/contrast_examples_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pdf"

    plotters.plot_examples(
        dataset=test_ds,
        alignment_model=mixed_model_trainer.alignment_model,
        filename=mixed_example_file,
        device=DEVICE
    )

    plotters.plot_examples(
        dataset=test_ds,
        alignment_model=contrast_model_trainer.alignment_model,
        filename=contrast_example_file,
        device=DEVICE
    )


    logger.info("\nGenerating TSNE w/ UNET Plot")
    tsne_plot_file = f"{IMAGE_FOLDER}/TSNE_UNET_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pdf"

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

    alignment_models = type_defs.ModelSet(
        base=None,
        mixed=mixed_model_trainer.alignment_model,
        contrast=contrast_model_trainer.alignment_model
    )
    if TARGET_NUM_CLASSES == 10 and SOURCE_NUM_CLASSES == 10:
        tsne_plotter = plotters.TSNE_Plotter(
            dataloaders=cls_dl_set,
            embed_size=model.get_body_output_size(),
            bs=BATCH_SIZE
        )

        tsne_plotter.plot_tsne(
            models=model_set,
            unet_models=alignment_models,
            accuracies=accuracies,
            device=DEVICE,
            filename=tsne_plot_file,
            base=TARGET,
            aux=SOURCE
        )

    logger.info("Generating Energy-Based Wasserstein Distance Plot")
    ebsw_plot_file = f"{IMAGE_FOLDER}/DIV_UNET_{TARGET}={TARGET_TRAIN_SIZE}+{SOURCE}={SOURCE_TRAIN_SIZE}.pdf"

    ebsw_plotter = plotters.EBSW_Plotter(
        dataloaders=cls_dl_set,
        batch_size=BATCH_SIZE
    )
    num_layers = base_model_trainer.classifier.get_num_layers()

    ebsw_plotter.plot_ebsw(
        models=model_set,
        alignment_models=alignment_models,
        layers=[i for i in range(num_layers-1)],
        device=DEVICE,
        filename=ebsw_plot_file,
        num_projections=256
    )

    logger.info(f"Finished {CLASSIFIER_ID} Dataset")

if __name__=="__main__":
    main()
