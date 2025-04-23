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
import samplers

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
    CLASSIFIER_ID += f"-{CONFIG.dataset.image_size}-{CONFIG.unet.loss}"

    # data size info
    TARGET_SIZE: int = CONFIG.dataset.target.train_size
    SOURCE_SIZE: int = CONFIG.dataset.source.train_size

    LOSS: str = CONFIG.unet.loss
    TARGET_NUM_CLASSES: int = CONFIG.dataset.target.num_classes
    SOURCE_NUM_CLASSES: int = CONFIG.dataset.source.num_classes

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
    for folder in [MODEL_FOLDER, FILE_FOLDER, IMAGE_FOLDER]:
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, CLASSIFIER_ID), exist_ok=True)

    os.makedirs(log_folder, exist_ok=True)

    # enable reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(CONFIG.dataset.rng_seed)
    torch.cuda.manual_seed(CONFIG.dataset.rng_seed)
    np.random.seed(CONFIG.dataset.rng_seed)

    # start logging
    log_location: str = os.path.join(log_folder, f"run_{CLASSIFIER_ID}_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}_{LOSS}.log")
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
        dataset=target_full_dataset,
        indices=target_train_inds
    )
    target_test_ds: datasets.IndexedDataset = datasets.IndexedDataset(
        dataset=target_full_dataset,
        indices=target_test_inds
    )
    target_val_ds: datasets.IndexedDataset = datasets.IndexedDataset(
        dataset=target_full_dataset,
        indices=target_val_inds
    )

    source_train_ds: datasets.IndexedDataset = datasets.IndexedDataset(
        dataset=source_full_dataset,
        indices=source_train_inds
    )
    source_test_ds: datasets.IndexedDataset = datasets.IndexedDataset(
        dataset=source_full_dataset,
        indices=source_test_inds
    )
    source_val_ds: datasets.IndexedDataset = datasets.IndexedDataset(
        dataset=source_full_dataset,
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

    logger.info(f"\t{sorted(list(train_ds.base_dataset.class_samples.keys()))}")
    logger.info(f"\t{sorted(list(train_ds.aux_dataset.class_samples.keys()))}")

    logger.info(f"\t{sorted(list(test_ds.base_dataset.class_samples.keys()))}")
    logger.info(f"\t{sorted(list(test_ds.aux_dataset.class_samples.keys()))}")

    logger.info(f"\t{sorted(list(val_ds.base_dataset.class_samples.keys()))}")
    logger.info(f"\t{sorted(list(val_ds.aux_dataset.class_samples.keys()))}")

    drop_last = True if BATCH_SIZE < 64 else False

    cls_train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=drop_last
    )

    cls_test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop_last
    )

    cls_val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop_last
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
        drop_last=drop_last
    )

    test_sampler = samplers.PureBatchSampler(
        data_source=test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=drop_last
    )

    val_sampler = samplers.PureBatchSampler(
        data_source=train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=drop_last
    )

    align_train_loader = torch.utils.data.DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=2
    )

    align_test_loader = torch.utils.data.DataLoader(
        test_ds, batch_sampler=test_sampler
    )

    align_val_loader = torch.utils.data.DataLoader(
        val_ds, batch_sampler=val_sampler
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

    logger.info("TRAINING CLASSIFIERS\n--------------------")

    logger.info("\nTraining Baseline Model")
    baseline_ae_filename = f"{MODEL_FOLDER}/{CLASSIFIER_ID}/baseline_autoencoder_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    baseline_unet_filename = f"{MODEL_FOLDER}/{CLASSIFIER_ID}/baseline_unet_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    baseline_classifier_filename = f"{MODEL_FOLDER}/{CLASSIFIER_ID}/baseline_classifier_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    # baseline_autoencoder = models.CustomAutoencoder()
    # baseline_unet = build_unet()
    # baseline_classifier = models.DynamicResNet(
    #     resnet_type=CONFIG.classifier.model,
    #     num_classes=10,
    # )

    # baseline_model_trainer = trainer.PreloaderTrainer(
    #     autoencoder=baseline_autoencoder,
    #     unet = baseline_unet,
    #     classifier = baseline_classifier,
    #     dataloaders=cls_dl_set
    # )

    #baseline_model_trainer.unet_preloader_train_loop(
    #    ae_filename=baseline_ae_filename,
    #    unet_filename=baseline_unet_filename,
    #    device=DEVICE,
    #    train_both=True
    #)

    #baseline_model_trainer.classification_train_loop(
    #    classifier_filename=baseline_classifier_filename,
    #    device=DEVICE,
    #    num_epochs=100
    #)

    #baseline_val_acc = baseline_model_trainer.evaluate_model(device=DEVICE)

    logger.info("\nTraining Base Model")
    base_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}/base_classifier_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    model: models.DynamicResNet = models.DynamicResNet(
        resnet_type=CONFIG.classifier.model,
        num_classes=TARGET_NUM_CLASSES,
    )

    unet = build_unet()

    base_model_trainer = trainer.FullTrainer(
        classifier=model,
        unet=unet,
        unet_loss=LOSS,
        classifier_dataloaders=cls_dl_set,
        unet_dataloaders=align_dl_set,
        file_folder = os.path.join(FILE_FOLDER, CLASSIFIER_ID)
    )

    if not os.path.exists(base_model_file):
        base_model_trainer.classifier_trainer.classification_train_loop(
            classifier_filename = base_model_file,
            device=DEVICE,
            num_epochs=CONFIG.classifier.num_epochs,
            target_only=True,
            use_unet=False
        )
    else:
        base_model_trainer.classifier.load_state_dict(torch.load(base_model_file))

    logger.info("\nTraining Mixed Model")
    mixed_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}/mixed_classifier_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    model: models.DynamicResNet = models.DynamicResNet(
        resnet_type=CONFIG.classifier.model,
        num_classes=TARGET_NUM_CLASSES,
    )

    unet = build_unet()

    mixed_model_trainer = trainer.FullTrainer(
        classifier=model,
        unet=unet,
        unet_loss=LOSS,
        classifier_dataloaders=cls_dl_set,
        unet_dataloaders=align_dl_set,
        file_folder = os.path.join(FILE_FOLDER, CLASSIFIER_ID)
    )

    if not os.path.exists(mixed_model_file):
        mixed_model_trainer.classifier_trainer.classification_train_loop(
            classifier_filename = mixed_model_file,
            device=DEVICE,
            num_epochs=CONFIG.classifier.num_epochs,
            target_only=False,
            use_unet=False
        )
    else:
        mixed_model_trainer.classifier.load_state_dict(torch.load(mixed_model_file))

    logger.info("\nTraining Contrastive Model")
    contrast_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}/contrast_body_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    contrast_full_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}/contrast_full_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    model: models.DynamicResNet = models.DynamicResNet(
        resnet_type=CONFIG.classifier.model,
        num_classes=TARGET_NUM_CLASSES
    )

    unet = build_unet()

    contrast_model_trainer = trainer.FullTrainer(
        classifier=model,
        unet=unet,
        unet_loss=LOSS,
        classifier_dataloaders=cls_dl_set,
        unet_dataloaders=align_dl_set,
        file_folder = os.path.join(FILE_FOLDER, CLASSIFIER_ID)
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
    base_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}/base_classifier_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    mixed_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}/mixed_classifier_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    contrast_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}/contrast_full_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    #logger.info(f"Baseline Model Accuracy: {round(baseline_val_acc*100, 2)}%")

    _, base_acc = base_model_trainer.classifier_trainer.evaluate_model(DEVICE, use_unet=False)
    logger.info(f"Base Model Accuracy: {round(base_acc*100, 2)}%")

    _, mixed_acc = mixed_model_trainer.classifier_trainer.evaluate_model(DEVICE, use_unet=False)
    logger.info(f"Mixed Model Accuracy: {round(mixed_acc*100, 2)}%")

    _, contrast_acc = contrast_model_trainer.classifier_trainer.evaluate_model(DEVICE, use_unet=False)
    logger.info(f"Contrastive Model Accuracy: {round(contrast_acc*100, 2)}%")

    logger.info("\nGenerating TSNE Plot")
    tsne_plot_file: str = f"{IMAGE_FOLDER}/{CLASSIFIER_ID}/TSNE_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pdf"

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
    ebsw_plot_file = f"{IMAGE_FOLDER}/{CLASSIFIER_ID}/EBSW_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pdf"

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
    mixed_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}/mixed_unet_FINAL_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    mixed_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}/mixed_classifier_FINAL_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    mixed_examples_final_fname = f"{IMAGE_FOLDER}/{CLASSIFIER_ID}/mixed_examples_FINAL_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    mixed_model_trainer.cascading_train_loop(
        unet_fname=mixed_unet_final_fname,
        classifier_fname=mixed_classifier_final_fname,
        examples_fname=mixed_examples_final_fname,
        device=DEVICE,
    )

    logger.info("Training UNET/Classifier for Contrast Model")
    contrast_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}/contrast_unet_FINAL_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    contrast_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}/contrast_classifier_FINAL_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"
    contrast_examples_final_fname = f"{IMAGE_FOLDER}/{CLASSIFIER_ID}/contrast_examples_FINAL_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pt"

    contrast_model_trainer.cascading_train_loop(
        unet_fname=contrast_unet_final_fname,
        classifier_fname=contrast_classifier_final_fname,
        examples_fname=contrast_examples_final_fname,
        device=DEVICE,
    )

    logger.info("\nGetting Model Accuracy With UNET Models")

    # logger.info(f"Baseline Model Accuracy: {round(baseline_val_acc*100, 2)}%")

    _, base_acc = base_model_trainer.classifier_trainer.evaluate_model(DEVICE)
    logger.info(f"Base Model Accuracy w/ UNET: {round(base_acc*100, 2)}%")

    _, mixed_acc = mixed_model_trainer.classifier_trainer.evaluate_model(DEVICE, use_unet=True)
    logger.info(f"Mixed Model Accuracy w/ UNET: {round(mixed_acc*100, 2)}%")

    _, contrast_acc = contrast_model_trainer.classifier_trainer.evaluate_model(DEVICE, use_unet=True)
    logger.info(f"Contrastive Model Accuracy w/ UNET: {round(contrast_acc*100, 2)}%")

    logger.info("Generating Image Example Plots")
    mixed_example_file = f"{IMAGE_FOLDER}/{CLASSIFIER_ID}/mixed_examples_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pdf"
    contrast_example_file = f"{IMAGE_FOLDER}/{CLASSIFIER_ID}/contrast_examples_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pdf"

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
    tsne_plot_file = f"{IMAGE_FOLDER}/{CLASSIFIER_ID}/TSNE_UNET_{TARGET}={TARGET_SIZE}+{SOURCE}={SOURCE_SIZE}.pdf"

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
    if TARGET_NUM_CLASSES != 10 or SOURCE_NUM_CLASSES != 10:
        tsne_plotter = plotters.TSNE_Plotter(
            dataloaders=cls_dl_set,
            embed_size=model.get_body_output_size(),
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
        dataloaders=cls_dl_set,
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
