import torch
from torchvision import transforms
import numpy as np
import os
from typing import Literal
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

    MODEL_FOLDER: str = CONFIG.save_locations.model_folder
    FILE_FOLDER: str = CONFIG.save_locations.file_folder
    IMAGE_FOLDER: str = CONFIG.save_locations.image_folder

    TARGET: str = CONFIG.dataset.target_name
    AUXILIARY: str = CONFIG.dataset.aux_name

    BATCH_SIZE: int = CONFIG.dataset.batch_size
    CLASSIFIER_ID: str = CONFIG.classifier.identifier

    log_folder: str = CONFIG.save_locations.logs_folder
    for folder in [MODEL_FOLDER, FILE_FOLDER, IMAGE_FOLDER, log_folder]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    # enable reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(CONFIG.dataset.rng_seed)
    torch.cuda.manual_seed(CONFIG.dataset.rng_seed)
    np.random.seed(CONFIG.dataset.rng_seed)

    # start logging
    log_location: str = os.path.join(log_folder, f"run_{CLASSIFIER_ID}.log")
    level = logging.DEBUG if CONFIG.verbose else logging.INFO

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
    logger.info(f"Auxiliary Dataset: {AUXILIARY}\n")

    # dataset creation
    logger.info("Creating Datasets")
    target_dir = CONFIG.dataset.target_folder
    aux_dir = CONFIG.dataset.aux_folder

    transform = transforms.Compose([
        transforms.Resize(32),  # Ensure consistency if needed
        # transforms.Pad(2),  # Pad to 32x32
        # transforms.RandomCrop(32, padding=4),  # Random crop with padding as specified
        tr.EnsureThreeChannelsPIL(),  # Convert to 3 channels
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))  # MNIST standard normalization
    ])


    split_gen = torch.Generator()
    split_gen.manual_seed(CONFIG.dataset.rng_seed)

    target_full_dataset = datasets.HEIFFolder(target_dir, transform=transform)

    train_size = 1000
    val_size = train_size * 2
    test_size = len(target_full_dataset) - train_size - val_size

    target_train_ds, target_test_ds, target_val_ds = torch.utils.data.random_split(
        target_full_dataset,
        [train_size, test_size, val_size],
        generator=split_gen
    )

    aux_full_dataset = datasets.HEIFFolder(aux_dir, transform=transform)

    train_size = 2000
    val_size = train_size * 2
    test_size = len(aux_full_dataset) - train_size - val_size

    aux_train_ds, aux_test_ds, aux_val_ds = torch.utils.data.random_split(
        aux_full_dataset,
        [train_size, test_size, val_size],
        generator=split_gen
    )

    train_ds = datasets.CombinedDataset(
        base_dataset=target_train_ds,
        aux_dataset=aux_train_ds
    )

    test_ds = datasets.CombinedDataset(
        base_dataset=target_test_ds,
        aux_dataset=aux_test_ds
    )

    val_ds = datasets.CombinedDataset(
        base_dataset=target_val_ds,
        aux_dataset=aux_val_ds
    )

    # train_ds, test_ds, val_ds = torch.utils.data.random_split(
    #     combined_dataset,
    #     [train_size, test_size, val_size],
    #     generator=split_gen
    # )

    logger.info(f"\tTrain Dataset Size: {len(train_ds)} (Target: {len(target_train_ds)}, Aux: {len(aux_train_ds)})")
    logger.info(f"\tTest Dataset Size: {len(test_ds)} (Target: {len(target_test_ds)}, Aux: {len(aux_test_ds)})")
    logger.info(f"\tValidation Dataset Aux Size: {len(val_ds)} (Target: {len(target_val_ds)}, Aux: {len(aux_val_ds)})")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    dl_set = type_defs.DataLoaderSet(
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader
    )

    for x, _, _ in val_loader:
        INPUT_SHAPE = x.shape
        break

    logger.info("TRAINING CLASSIFIERS\n--------------------")

    # logger.info("\nTraining Baseline Model")
    # baseline_ae_filename = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_baseline_autoencoder_{TARGET}+{AUXILIARY}.pt"
    # baseline_unet_filename = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_baseline_unet_{TARGET}+{AUXILIARY}.pt"
    # baseline_classifier_filename = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_baseline_classifier_{TARGET}+{AUXILIARY}.pt"

    # baseline_autoencoder = models.CustomUNET()
    # baseline_unet = models.CustomUNET()
    # baseline_classifier = models.DynamicResNet(
    #     resnet_type='resnet9',
    #     num_classes=10,
    # )

    # baseline_model_trainer = trainer.PreloaderTrainer(
    #     autoencoder=baseline_autoencoder,
    #     unet = baseline_unet,
    #     classifier = baseline_classifier,
    #     dataloaders=dl_set
    # )

    # baseline_model_trainer.unet_preloader_train_loop(
    #     ae_filename=baseline_ae_filename,
    #     unet_filename=baseline_unet_filename,
    #     device=DEVICE,
    #     train_both=True
    # )

    # baseline_model_trainer.classification_train_loop(
    #     classifier_filename=baseline_classifier_filename,
    #     device=DEVICE,
    #     num_epochs=100
    # )

    # baseline_val_acc = baseline_model_trainer.evaluate_model(device=DEVICE)

    logger.info("\nTraining Base Model")
    base_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_base_classifier_{TARGET}+{AUXILIARY}.pt"

    if not os.path.exists(base_model_file):
        model = models.DynamicResNet(
            resnet_type='resnet9',
            num_classes=10,
        )

        base_model_trainer = trainer.Trainer(
            classifier = model,
            dataloaders=dl_set
        )

        base_model_trainer.classification_train_loop(
            filename = base_model_file,
            device=DEVICE,
            mode="base_only"
        )

    logger.info("\nTraining Mixed Model")
    mixed_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_classifier_{TARGET}+{AUXILIARY}.pt"

    if not os.path.exists(mixed_model_file):
        model = models.DynamicResNet(
            resnet_type='resnet9',
            num_classes=10
        )

        mixed_model_trainer = trainer.Trainer(
            classifier = model,
            dataloaders=dl_set
        )

        mixed_model_trainer.classification_train_loop(
            filename = mixed_model_file,
            device=DEVICE,
            mode="mixed",
        )

    logger.info("\nTraining Contrastive Model")
    contrast_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_LF_body_{TARGET}+{AUXILIARY}.pt"
    contrast_full_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_LF_full_{TARGET}+{AUXILIARY}.pt"

    if not os.path.exists(contrast_full_model_file): 
        model = models.DynamicResNet(
            resnet_type='resnet9',
            num_classes=10
        )

        contrast_model_trainer = trainer.Trainer(
            classifier = model,
            dataloaders=dl_set,
            contrastive=True
        )

        best_temp = contrast_model_trainer.contrastive_train_loop(
            filename = contrast_model_file,
            device=DEVICE,
            temp_range=[0.05],
        )


    logger.info("\nGetting Model Accuracy")
    base_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_base_classifier_{TARGET}+{AUXILIARY}.pt"
    mixed_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_classifier_{TARGET}+{AUXILIARY}.pt"
    contrast_model_file = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_LF_full_{TARGET}+{AUXILIARY}.pt"

    base_model = models.DynamicResNet(
        resnet_type='resnet9',
        num_classes=10
    )
    base_model.load_state_dict(torch.load(base_model_file, weights_only=True))
    base_model_trainer = trainer.Trainer(
        classifier = base_model,
        dataloaders=dl_set
    )
    base_model_trainer.classifier = base_model

    mixed_model = models.DynamicResNet(
        resnet_type='resnet9',
        num_classes=10
    )
    mixed_model.load_state_dict(torch.load(mixed_model_file, weights_only=True))
    mixed_model_trainer = trainer.Trainer(
        classifier = mixed_model,
        dataloaders=dl_set
    )
    mixed_model_trainer.classifier = mixed_model

    contrast_model = models.DynamicResNet(
        resnet_type='resnet9',
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

    base_acc = base_model_trainer.evaluate_model(DEVICE)
    logger.info(f"Base Model Accuracy: {round(base_acc*100, 2)}%")

    mixed_acc = mixed_model_trainer.evaluate_model(DEVICE)
    logger.info(f"Mixed Model Accuracy: {round(mixed_acc*100, 2)}%")

    contrast_acc = contrast_model_trainer.evaluate_model(DEVICE)
    logger.info(f"Contrastive Model Accuracy: {round(contrast_acc*100, 2)}%")

    logger.info("\nGenerating TSNE Plot")
    tsne_plot_file = f'{IMAGE_FOLDER}/{CLASSIFIER_ID}_TSNE_{TARGET}+{AUXILIARY}.pdf'

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
        aux=AUXILIARY
    )

    logger.info("\nGenerating Energy-Based Wasserstein Distance Plot")
    ebsw_plot_file = f'{IMAGE_FOLDER}/{CLASSIFIER_ID}_EBSW_{TARGET}+{AUXILIARY}.pdf'

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

    logger.info("Training UNET/Classifier for Base Model")
    base_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_base_unet_FINAL_{TARGET}+{AUXILIARY}.pt"
    base_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_base_classifier_FINAL_{TARGET}+{AUXILIARY}.pt"

    base_model_trainer.unet = models.CustomUNET()

    base_model_trainer.unet_classifier_train_loop(
        unet_filename=base_unet_final_fname,
        classifier_filename=base_classifier_final_fname,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )


    logger.info("Training UNET/Classifier for Mixed Model")
    mixed_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_unet_FINAL_{TARGET}+{AUXILIARY}.pt"
    mixed_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_classifier_FINAL_{TARGET}+{AUXILIARY}.pt"

    mixed_model_trainer.unet = models.CustomUNET()

    mixed_model_trainer.unet_classifier_train_loop(
        unet_filename=mixed_unet_final_fname,
        classifier_filename=mixed_classifier_final_fname,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )


    logger.info("Training UNET/Classifier for Contrast Model")
    contrast_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_unet_FINAL_{TARGET}+{AUXILIARY}.pt"
    contrast_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_classifier_FINAL_{TARGET}+{AUXILIARY}.pt"

    contrast_model_trainer.unet = models.CustomUNET()

    contrast_model_trainer.unet_classifier_train_loop(
        unet_filename=contrast_unet_final_fname,
        classifier_filename=contrast_classifier_final_fname,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    logger.info("\nGetting Model Accuracy With UNET Models")

    base_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_base_unet_FINAL_{TARGET}+{AUXILIARY}.pt"
    base_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_base_classifier_FINAL_{TARGET}+{AUXILIARY}.pt"
    mixed_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_unet_FINAL_{TARGET}+{AUXILIARY}.pt"
    mixed_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_mixed_classifier_FINAL_{TARGET}+{AUXILIARY}.pt"
    contrast_unet_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_unet_FINAL_{TARGET}+{AUXILIARY}.pt"
    contrast_classifier_final_fname = f"{MODEL_FOLDER}/{CLASSIFIER_ID}_contrast_classifier_FINAL_{TARGET}+{AUXILIARY}.pt"

    base_model = models.DynamicResNet(
        resnet_type='resnet9',
        num_classes=10
    )
    base_model.load_state_dict(torch.load(base_classifier_final_fname, weights_only=True))
    base_unet = models.CustomUNET()
    base_unet.load_state_dict(torch.load(base_unet_final_fname, weights_only=True))
    base_model_trainer = trainer.Trainer(
        classifier = base_model,
        dataloaders=dl_set
    )
    base_model_trainer.classifier = base_model
    base_model_trainer.unet = base_unet

    mixed_model = models.DynamicResNet(
        resnet_type='resnet9',
        num_classes=10
    )
    mixed_model.load_state_dict(torch.load(mixed_classifier_final_fname, weights_only=True))
    mixed_unet = models.CustomUNET()
    mixed_unet.load_state_dict(torch.load(mixed_unet_final_fname, weights_only=True))
    mixed_model_trainer = trainer.Trainer(
        classifier = mixed_model,
        dataloaders=dl_set
    )
    mixed_model_trainer.classifier = mixed_model
    mixed_model_trainer.unet = mixed_unet

    contrast_model = models.DynamicResNet(
        resnet_type='resnet9',
        num_classes=10
    )
    contrast_model.load_state_dict(torch.load(contrast_classifier_final_fname, weights_only=True))
    contrast_unet = models.CustomUNET()
    contrast_unet.load_state_dict(torch.load(contrast_unet_final_fname, weights_only=True))
    contrast_model_trainer = trainer.Trainer(
        classifier = contrast_model,
        dataloaders=dl_set,
        contrastive=True
    )
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
    base_example_file = f'{IMAGE_FOLDER}/{CLASSIFIER_ID}_base_examples_{TARGET}+{AUXILIARY}.pdf'
    mixed_example_file = f'{IMAGE_FOLDER}/{CLASSIFIER_ID}_mixed_examples_{TARGET}+{AUXILIARY}.pdf'
    contrast_example_file = f'{IMAGE_FOLDER}/{CLASSIFIER_ID}_contrast_examples_{TARGET}+{AUXILIARY}.pdf'

    plotters.plot_examples(
        dataset=val_ds,
        unet_model=base_model_trainer.unet,
        filename=base_example_file,
        device=DEVICE
    )

    plotters.plot_examples(
        dataset=val_ds,
        unet_model=mixed_model_trainer.unet,
        filename=mixed_example_file,
        device=DEVICE
    )

    plotters.plot_examples(
        dataset=val_ds,
        unet_model=contrast_model_trainer.unet,
        filename=contrast_example_file,
        device=DEVICE
    )


    logger.info("\nGenerating TSNE w/ UNET Plot")
    tsne_plot_file = f'{IMAGE_FOLDER}/{CLASSIFIER_ID}_TSNE_UNET_{TARGET}+{AUXILIARY}.pdf'

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
        base=base_model_trainer.unet,
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
        aux=AUXILIARY
    )

if __name__=="__main__":
    main()