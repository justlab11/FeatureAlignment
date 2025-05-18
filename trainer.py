import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np 
from os import path
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import gc
from typing import List

from models import DynamicCNN
from losses import supervised_contrastive_loss, ISEBSW, mmdfuse
from datasets import CombinedDataset
from type_defs import DataLoaderSet
from helpers import compute_layer_loss
from plotters import plot_examples, EBSW_Plotter, divergence_plots

logger = logging.getLogger(__name__)

class ClassifierTrainer:
    """
    ClassifierTrainer handles all training, validation, contrastive learning, and evaluation routines for a classification model.

    This class supports:
        - Standard supervised training with flexible data loaders
        - Supervised contrastive learning with temperature optimization
        - Fine-tuning and evaluation workflows
        - Optional integration of a alignment model for feature preprocessing
        - Management of model checkpointing, parameter resets, and optimizer scheduling

    Args:
        classifier (nn.Module): The main classification model.
        alignment_model (nn.Module): Optional alignment model for feature preprocessing or augmentation.
        dataloaders (DataLoaderSet): Named tuple or object containing train, validation, and test DataLoader instances.

    Attributes:
        classifier (nn.Module): The classification model being trained.
        alignment_model (nn.Module): The alignment model for optional preprocessing.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        proj_head (nn.Module): Projection head for contrastive learning tasks.
    """

    def __init__(self, classifier, alignment_model, dataloaders: DataLoaderSet):
        # initialize the models used for the trainers
        self.classifier: nn.Module = classifier
        self.alignment_model: nn.Module = alignment_model

        # 
        self.train_loader: DataLoader = dataloaders.train_loader
        self.test_loader: DataLoader= dataloaders.test_loader
        self.val_loader: DataLoader = dataloaders.val_loader

        # in case we need to do contrastive learning, we build out a projection head
        body_output_size: int = self.classifier.get_body_output_size()
        self.proj_head: nn.Module = nn.Sequential(
            nn.Linear(body_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def _classification_run(self, optimizer, dataloader, device, target_only=False, train=True, use_alignment=False):
        """
        Run one epoch of training or evaluation on the given dataset.
        
        Args:
        - model: The neural network model
        - optimizer: The optimizer for the model (used only if train=True)
        - dataloader: DataLoader instance for the dataset
        - target_only: Boolean, if True, only use the samples from the target dataset
        - train: Boolean, if True, run in training mode; if False, run in evaluation mode
        - use_alignment: Boolean, if True, pass source samples through alignment model 
        
        Returns:
        - Average loss for the epoch
        - Accuracy for the epoch (for classification) or None (for contrastive)
        """

        criterion = nn.CrossEntropyLoss()
        self.classifier.to(device)

        if train:
            self.classifier.train()
        else:
            self.classifier.eval()

        self.alignment_model.to(device)
        self.alignment_model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for target_samples, source_samples, labels in dataloader:
            labels = labels.long()
            target_samples, source_samples, labels = (
                target_samples.to(device), 
                source_samples.to(device), 
                labels.to(device)
            )
            
            if train:
                optimizer.zero_grad()
            
            with torch.set_grad_enabled(train):
                if target_only:
                    outputs = self.classifier(target_samples)[-1]
                    loss = criterion(outputs, labels)
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                
                else:
                    if use_alignment:
                        source_samples = self.alignment_model(source_samples)[-1]
                        source_samples = source_samples.detach()

                    inputs = torch.cat((target_samples, source_samples), 0)

                    outputs = self.classifier(inputs)[-1]
                    loss = criterion(outputs, labels.repeat(2))
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0) * 2
                    correct += predicted.eq(labels.repeat(2)).sum().item()
            
            if train:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct / total
        
        return epoch_loss, epoch_accuracy
    
    def _contrastive_run(self, dataloader, device, optimizer=None, train=True, temperature=0.1, use_alignment=False):
        """
        Runs a single epoch of supervised contrastive learning, either in training or evaluation mode.

        Args:
            dataloader (DataLoader): DataLoader providing batches of (target_samples, source_samples, labels).
            device (torch.device): Device to perform computation on.
            optimizer (torch.optim.Optimizer, optional): Optimizer for parameter updates (used if train=True).
            train (bool, optional): If True, runs in training mode; otherwise, evaluation mode. Default is True.
            temperature (float, optional): Temperature parameter for the contrastive loss. Default is 0.1.
            use_alignment (bool, optional): If True, processes source samples with the alignment model before contrastive learning. Default is False.

        Returns:
            float: Average contrastive loss over the epoch.
        """
        criterion = supervised_contrastive_loss
        self.classifier.set_freeze_head(True)
        self.classifier.set_freeze_body(False)
        self.classifier.to(device)
        self.proj_head.to(device)

        if train:
            self.classifier.train()
            self.proj_head.train()
        else:
            self.classifier.eval()
            self.proj_head.eval()
        
        running_loss = 0.0

        self.alignment_model.to(device)
        self.alignment_model.eval()
        
        for target_samples, source_samples, labels in dataloader:
            labels = labels.long()
            group_labels = torch.cat([
                torch.zeros(len(target_samples)),
                torch.ones(len(source_samples)),
            ]).long()
            group_labels = group_labels.reshape(1, -1)
            target_samples, source_samples, labels = (
                target_samples.to(device),
                source_samples.to(device), 
                labels.to(device)
            )
            
            if use_alignment:
                source_samples = self.alignment_model(source_samples)[-1]
                
            if train:
                optimizer.zero_grad()
            
            with torch.set_grad_enabled(train):
                inputs = torch.cat((target_samples, source_samples), 0)
                features = self.classifier(inputs)[-1]
                features = features.reshape(inputs.shape[0], -1)
                projected = self.proj_head(features)
                loss = criterion(projected, labels.repeat(2), temperature=temperature)
            
            if train:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)

        return epoch_loss
    
    def _optimize_temperature(self, temp_range, device, classifier_filename, num_epochs=100):
        """
        Performs a grid search over the provided temperature range to find the optimal temperature for contrastive learning.

        For each temperature, the classifier and projection head parameters are reset, the model is trained, and validation loss is tracked.
        The best model (with lowest validation loss) and corresponding temperature are saved and returned.

        Args:
            temp_range (iterable): Range of temperature values to evaluate.
            device (torch.device): Device to perform training on.
            classifier_filename (str): Path to save the best model weights.
            num_epochs (int, optional): Number of epochs to train for each temperature. Default is 100.

        Returns:
            tuple: (best_temp, best_val_loss) where best_temp is the optimal temperature and best_val_loss is the corresponding validation loss.
        """

        best_temp = None
        best_val_loss = float('inf')
        
        for temp in temp_range:
            logger.info(f"\tStarting temp: {temp}")
            self.classifier.reset_parameters()
            self.reset_parameters(self.proj_head)
            
            optimizer = optim.Adam(
                list(self.classifier.parameters()) + list(self.proj_head.parameters()),
                lr=0.001, 
                weight_decay=1e-5
            )
                        
            for epoch in range(num_epochs):
                _ = self._contrastive_run(
                    optimizer=optimizer,
                    dataloader=self.train_loader,
                    device=device,
                    temperature=temp
                )
                
                val_loss = self._contrastive_run(
                    dataloader=self.val_loader,
                    device=device,
                    train=False,
                    temperature=temp
                )
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_temp = temp
                    torch.save(self.classifier.state_dict(), classifier_filename)
        
        return best_temp, best_val_loss

    def reset_parameters(self, module):
        """
        Recursively resets the parameters of all submodules within the given module that implement a 'reset_parameters' method.

        Args:
            module (nn.Module): The PyTorch module whose parameters should be reset.
        """

        for m in module.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def contrastive_train_loop(self, device, filename, temp_range=[0.05, 0.1, 0.15], num_epochs=100, best_temp=None):
        """
        Performs contrastive pre-training with temperature optimization, then fine-tunes the classifier head for downstream classification.

        Args:
            device (torch.device): Device to perform training on.
            filename (str): Path to save and load the best model weights.
            temp_range (list, optional): List of temperature values to search for contrastive loss. Default is [0.05, 0.1, 0.15].
            num_epochs (int, optional): Number of epochs for each training phase. Default is 100.
            best_temp (float, optional): If provided, skips temperature search and uses this value.

        Returns:
            float: The best temperature value found (or provided) for contrastive training.
        """
        
        # Step 1: Optimize temperature and train body
        if best_temp == None:
            best_temp, best_val_loss = self._optimize_temperature(temp_range, device, filename, num_epochs)
        
            logger.info(f"\tBest temperature: {best_temp:.4f}, Best validation loss: {best_val_loss:.4f}")
        
        # Load the best model
        self.classifier.load_state_dict(torch.load(filename, weights_only=True))
        
        # Step 2: Freeze body and train head
        self.classifier.set_freeze_head(False)
        self.classifier.set_freeze_body(True)
        
        # Train the head
        head_filename = filename.replace('body', 'full')
        self.classification_train_loop(head_filename, device, num_epochs=100)
        
        # Optionally, unfreeze everything for potential fine-tuning
        self.classifier.set_freeze_head(False)
        self.classifier.set_freeze_body(False)

        logger.info("Contrastive training and head training completed.")
        return best_temp

    def classification_train_loop(self, classifier_filename, device, num_epochs=100, target_only=False, use_alignment=False):
        """
        Trains the classifier using supervised learning, tracks best validation performance, and saves the best model checkpoint.

        Args:
            classifier_filename (str): Path to save the best model weights.
            device (torch.device): Device to perform training on.
            num_epochs (int, optional): Number of training epochs. Default is 100.
            target_only (bool, optional): If True, only target samples are used for training. Default is False.
            use_alignment (bool, optional): If True, processes inputs with the alignment model before classification. Default is False.

        Returns:
            float: Best validation loss achieved during training.
        """

        optimizer = optim.SGD(self.classifier.parameters(), lr=1e-2, momentum=.9, 
                              nesterov=True, weight_decay=1e-2)

        scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=20,
            T_mult=1,
            eta_min=1e-8
        )

        best_val_loss = 1e7

        self.classifier.to(device)
        self.alignment_model.to(device)
        self.alignment_model.eval()

        for epoch in range(num_epochs):
            train_loss, train_acc = self._classification_run(
                optimizer=optimizer,
                dataloader=self.train_loader,
                device=device,
                target_only=target_only,
                train=True,
                use_alignment=use_alignment
            )

            current_lr = optimizer.param_groups[0]['lr']
            logger.debug(f"Epoch {epoch+1}: Current LR: {current_lr:.2e}")

            scheduler.step() 

            val_loss, val_acc = self._classification_run(
                optimizer=None,
                dataloader=self.val_loader,
                device=device,
                target_only=True,
                train=False,
                use_alignment=use_alignment
            )

            test_loss, test_acc = self._classification_run(
                optimizer=None,
                dataloader=self.test_loader,
                device=device,
                target_only=True,
                train=False,
                use_alignment=use_alignment
            )

            log_message = f"\tEpoch {epoch+1}: Train- {train_loss:.4f} {train_acc*100:.2f}, Val- {val_loss:.4f} {val_acc*100:.2f}, Test - {test_loss:.4f} {test_acc*100:.2f}"

            if val_loss < best_val_loss:
                log_message += " <- New Best"
                best_val_loss = val_loss
                torch.save(self.classifier.state_dict(), classifier_filename)

            logger.info(log_message)
        
        self.classifier.load_state_dict(torch.load(classifier_filename, weights_only=True))

        return best_val_loss    

    def evaluate_model(self, device, use_alignment=False, test=False):
        """
        Evaluates the model on the validation or test set.

        Args:
            device: torch.device to evaluate on.
            use_alignment: Whether to use the alignment model for preprocessing.
            test: If True, evaluates on test set; else on validation set.

        Returns:
            loss (float), acc (float): Average loss and accuracy.
        """
        if test:
            loss, acc = self._classification_run(
                optimizer=None,
                dataloader=self.test_loader,
                device=device,
                target_only=True,
                train=False,
                use_alignment=use_alignment
            )
        else:
            loss, acc = self._classification_run(
                optimizer=None,
                dataloader=self.val_loader,
                device=device,
                target_only=True,
                train=False,
                use_alignment=use_alignment
            )
        return loss, acc


class AlignmentTrainer:
    """
    AlignmentTrainer manages the training and evaluation routines for aligning an alignment models's feature representations with those of a classifier model.

    This class is designed for advanced domain adaptation or feature alignment tasks, where the alignment model is trained to transform source samples such that their representations match those of the target samples at specified layers of a fixed classifier. It supports flexible loss selection for the alignment objective and robust training practices, including gradient clipping and NaN/Inf gradient checks.

    Args:
        classifier (nn.Module): The fixed classifier model whose feature representations serve as alignment targets.
        alignment_model (nn.Module): The alignment model to be trained for feature alignment.
        dataloaders (DataLoaderSet): Object containing train, validation, and test DataLoader instances.
        alignment_loss (str): Specifies which loss function to use for alignment ('ebsw' or 'mmdfuse').

    Attributes:
        classifier (nn.Module): The classifier model used for feature extraction and alignment targets.
        alignment_model (nn.Module): The alignment model being trained for alignment.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        criterion (callable): The loss function used for alignment.
    """
    def __init__(self, classifier, alignment_model, dataloaders: DataLoaderSet, alignment_loss):
        self.classifier = classifier
        self.alignment_model = alignment_model

        self.train_loader: DataLoader = dataloaders.train_loader
        self.test_loader: DataLoader= dataloaders.test_loader
        self.val_loader: DataLoader = dataloaders.val_loader

        if alignment_loss == "ebsw":
            self.criterion = ISEBSW
        elif alignment_loss == "mmdfuse":
            self.criterion = mmdfuse
        else:
            raise ValueError(f"Unknown alignment_loss: {alignment_loss}")

    def cascade_alignment_train_loop(self, layers, device, alignment_fname, epochs=100):
        """
        Trains the alignment model to align its feature representations with those of the classifier across specified layers using a cascade alignment strategy.

        For each epoch, the alignment model is optimized to minimize the alignment loss between its outputs (after transforming source samples) and the classifier's outputs on target samples, across the given layers. The best UNet weights (based on validation loss) are saved and restored at the end. Includes gradient clipping and checks for NaN/Inf gradients for stability.

        Args:
            layers (list[int]): Indices of classifier layers to use for alignment loss computation.
            device (torch.device): Device to perform training on.
            alignment_fname (str): Path to save the best alignment model weights.
            epochs (int, optional): Number of training epochs. Default is 100.

        Returns:
            float: The best validation loss achieved during training.
        """
        torch.save(self.alignment_model.state_dict(), alignment_fname)
        
        criterion = self.criterion
        self.alignment_model.to(device)
        self.classifier.to(device)

        self.classifier.set_freeze_head(False)
        self.classifier.set_freeze_body(False)
        self.classifier.eval()

        alignment_optimizer = optim.Adam(self.alignment_model.parameters(), lr=3e-4, weight_decay=1e-5)
        alignment_scheduler = CosineAnnealingLR(alignment_optimizer, T_max=epochs, eta_min=1e-6)

        best_val = float('inf')
        max_norm = 1
        alignment_nan = False

        for epoch in range(epochs):
            self.alignment_model.train()
            train_loss = 0

            for target_samples, source_samples, labels in self.train_loader:
                labels = labels.long()

                target_samples, source_samples, labels = (
                    target_samples.to(device),
                    source_samples.to(device),
                    labels.to(device),
                )

                alignment_optimizer.zero_grad()

                alignment_output = self.alignment_model(source_samples)[-1]
                target_reps = self.classifier(target_samples)
                source_reps = self.classifier(alignment_output)

                loss = sum(
                    compute_layer_loss(target_reps, source_reps, labels, layer, criterion, device)
                    for layer in layers
                )

                # loss += 0.05 * torch.nn.MSELoss()(target_samples, target_reps[-1])

                loss.backward()
                for name, param in self.alignment_model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"NaN detected in gradient of {name}")
                            alignment_nan = True
                            break
                        if torch.isinf(param.grad).any():
                            print(f"Inf detected in gradient of {name}")
                            alignment_nan = True
                            break

                if alignment_nan:
                    break

                torch.nn.utils.clip_grad_norm_(self.alignment_model.parameters(), max_norm)

                alignment_optimizer.step()

                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)

            alignment_scheduler.step()
            if alignment_nan:
                alignment_nan = False
                break

            self.alignment_model.eval()
            val_loss = 0
            for target_samples, source_samples, labels in self.val_loader:
                labels = labels.long()

                target_samples, source_samples, labels = (
                    target_samples.to(device),
                    source_samples.to(device),
                    labels.to(device),
                )

                with torch.no_grad():
                    alignment_output = self.alignment_model(source_samples)[-1]
                    target_reps = self.classifier(target_samples)
                    source_reps = self.classifier(alignment_output)

                    loss = sum(
                        compute_layer_loss(target_reps, source_reps, labels, layer, criterion, device)
                        for layer in layers
                    )

                val_loss += loss.item()
            
            val_loss /= len(self.val_loader)

            gc.collect()
            torch.cuda.empty_cache()
            
            log_message = f"\tAlignment Epoch {epoch+1}: Train- {train_loss}, Val- {val_loss}"
            if val_loss < best_val:
                log_message += " <- New Best"
                best_val = val_loss
                torch.save(self.alignment_model.state_dict(), alignment_fname)

            logger.info(log_message)
        
        self.alignment_model.load_state_dict(torch.load(alignment_fname, weights_only=True)) 

        return best_val 

class FullTrainer:
    def __init__(
        self,
        classifier: nn.Module,
        alignment_model: nn.Module,
        classifier_dataloaders: DataLoaderSet,
        alignment_dataloaders: DataLoaderSet,
        file_folder: str,
        alignment_loss: str,
        classifier_name: str
    ):
        """
        Initializes the FullTrainer, which orchestrates classifier and alignment model training.

        Args:
            classifier (nn.Module): The classification model.
            alignment_model (nn.Module): The model to be aligned (e.g., UNet or other).
            classifier_dataloaders (DataLoaderSet): Dataloaders for classifier training/validation/testing.
            alignment_dataloaders (DataLoaderSet): Dataloaders for alignment training/validation/testing.
            file_folder (str): Path for saving models and logs.
            alignment_loss (str): Loss function for alignment model ('ebsw' or 'mmdfuse').
            classifier_name (str): Name identifier for the classifier.
        """
        self.classifier: nn.Module = classifier
        self.alignment_model: nn.Module = alignment_model
        self.alignment_loss: str = alignment_loss
        self.classifier_dataloaders: DataLoaderSet = classifier_dataloaders
        self.alignment_dataloaders: DataLoaderSet = alignment_dataloaders
        self.file_folder: str = file_folder
        self.classifier_name: str = classifier_name

        self.classifier_trainer: ClassifierTrainer = ClassifierTrainer(
            classifier=self.classifier,
            alignment_model=self.alignment_model,  
            dataloaders=self.classifier_dataloaders,
        )

        self.alignment_trainer: ClassifierTrainer = AlignmentTrainer(
            classifier=self.classifier,
            alignment_model=self.alignment_model,
            dataloaders=self.alignment_dataloaders,
            alignment_loss=self.alignment_loss
        )

    def cascading_train_loop(self, classifier_fname, alignment_fname, examples_fname, device, num_epochs=100):
        num_layers = self.classifier.get_num_layers()
        layers = [i for i in range(num_layers)]

        best_layer_val_loss = 1e7
        best_layer_val_acc = 0

        best_layer_test_loss = 1e7
        best_layer_test_acc = 0
        
        best_layer = None

        inter_layer_distances = []
        intra_layer_distances = []
        classifier_val_accs = []
        classifier_test_accs = []

        ebsw_plotter: EBSW_Plotter = EBSW_Plotter(
            dataloaders=self.alignment_dataloaders,
            batch_size=self.classifier_dataloaders.train_loader.batch_size
        )

        # calculate divergence performance before any alignment
        if self.alignment_loss == "ebsw":
            inter = ebsw_plotter.run_isebsw(
                model=self.classifier,
                dataloader=self.alignment_dataloaders.val_loader,
                layers=[j for j in range(num_layers-1)],
                device=device,
                target_only=True,
                alignment_model=None,
                num_projections=256
            )

            intra = ebsw_plotter.run_isebsw(
                model=self.classifier,
                dataloader=self.alignment_dataloaders.val_loader,
                layers=[j for j in range(num_layers-1)],
                device=device,
                target_only=False,
                alignment_model=None,
                num_projections=256
            )

        else:
            inter = ebsw_plotter.run_mmdfuse(
                model=self.classifier,
                dataloader=self.alignment_dataloaders.val_loader,
                layers=[j for j in range(num_layers-1)],
                device=device,
                target_only=True,
                alignment_model=None,
            )

            intra = ebsw_plotter.run_mmdfuse(
                model=self.classifier,
                dataloader=self.alignment_dataloaders.val_loader,
                layers=[j for j in range(num_layers-1)],
                device=device,
                target_only=False,
                alignment_model=None,
            )

        inter_layer_distances.append(inter)
        intra_layer_distances.append(intra)

        for i in range(1, num_layers+1):
            layer_set: List[int] = layers[-i:]
            logger.info(f"COVERING LAYERS: {layer_set}\n")
            start_layer: int = layer_set[0]
            classifier_layer_fname: str = path.splitext(classifier_fname)[0] + f"-{start_layer}.pt"
            alignment_layer_fname: str = path.splitext(alignment_fname)[0] + f"-{start_layer}.pt"
            alignment_layer_examples_fname: str = path.splitext(examples_fname)[0] + f"-{start_layer}.pdf"

            alignment_val: float = self.alignment_trainer.cascade_alignment_train_loop(
                layers=layer_set,
                device=device,
                alignment_fname=alignment_layer_fname,
                epochs=num_epochs
            )

            plot_examples(
                dataset=self.classifier_dataloaders.val_loader.dataset,
                unet_model=self.alignment_model,
                filename=alignment_layer_examples_fname,
                device=device
            )

            _ = self.classifier_trainer.classification_train_loop(
                classifier_filename=classifier_layer_fname,
                device=device,
                num_epochs=num_epochs,
                use_alignment=True,
            )

            if self.alignment_loss == "ebsw":
                inter = ebsw_plotter.run_isebsw(
                    model=self.classifier,
                    dataloader=self.alignment_dataloaders.val_loader,
                    layers=[j for j in range(num_layers-1)],
                    device=device,
                    target_only=True,
                    alignment_model=self.alignment_model,
                    num_projections=256
                )

                intra = ebsw_plotter.run_isebsw(
                    model=self.classifier,
                    dataloader=self.alignment_dataloaders.val_loader,
                    layers=[j for j in range(num_layers-1)],
                    device=device,
                    target_only=False,
                    alignment_model=self.alignment_model,
                    num_projections=256
                )

            else:
                inter = ebsw_plotter.run_mmdfuse(
                    model=self.classifier,
                    dataloader=self.alignment_dataloaders.val_loader,
                    layers=[j for j in range(num_layers-1)],
                    device=device,
                    target_only=True,
                    alignment_model=self.alignment_model,
                )

                intra = ebsw_plotter.run_mmdfuse(
                    model=self.classifier,
                    dataloader=self.alignment_dataloaders.val_loader,
                    layers=[j for j in range(num_layers-1)],
                    device=device,
                    target_only=False,
                    alignment_model=self.alignment_model,
                )

            inter_layer_distances.append(inter)
            intra_layer_distances.append(intra)

            classifier_val_loss, classifier_val_acc = self.classifier_trainer.evaluate_model(
                device=device,
            )
            classifier_val_accs.append(classifier_val_acc)

            classifier_test_loss, classifier_test_acc = self.classifier_trainer.evaluate_model(
                device=device, test=True
            )
            classifier_test_accs.append(classifier_test_acc)

            if classifier_val_acc > best_layer_val_acc:
                best_layer_val_acc: float = classifier_val_acc
                best_layer_val_loss: float = classifier_val_loss

                best_layer_test_acc: float = classifier_test_acc
                best_layer_test_loss: float = classifier_test_loss
                
                best_layer: int = start_layer

        logger.info(f"Best Performing Layer Set: {best_layer}-{num_layers-1} : {best_layer_test_loss} | {best_layer_test_acc*100:.4f} ")
        classifier_best_fname: str = classifier_fname[:-3] + f"-{best_layer}.pt"
        alignment_best_fname: str = alignment_fname[:-3] + f"-{best_layer}.pt"

        self.classifier.load_state_dict(torch.load(classifier_best_fname, weights_only=True))
        self.alignment_model.load_state_dict(torch.load(alignment_best_fname, weights_only=True))

        inter_layer_distances = np.array(inter_layer_distances)
        intra_layer_distances = np.array(intra_layer_distances)
        classifier_val_accs = np.array(classifier_val_accs)

        model_name: str = self.classifier_name
        inter_fname: str = path.join(self.file_folder, f"{model_name}-inter_layer_distances.npy")
        intra_fname: str = path.join(self.file_folder, f"{model_name}-intra_layer_distances.npy")
        val_fname: str = path.join(self.file_folder, f"{model_name}-classifier_val_accs.npy")
        test_fname: str = path.join(self.file_folder, f"{model_name}-classifier_test_accs.npy")
        div_plots_fname: str = path.join(self.file_folder, f"{model_name}-divergence_plots")

        np.save(val_fname, classifier_val_accs)
        np.save(test_fname, classifier_test_accs)
        np.save(inter_fname, inter_layer_distances)
        np.save(intra_fname, intra_layer_distances)

        divergence_plots(
            inter_data=inter_layer_distances,
            intra_data=intra_layer_distances,
            val_acc_values=classifier_val_accs,
            fname=div_plots_fname
        )

class PreloaderTrainer:
    def __init__(self, autoencoder, alignment_model, classifier, cls_dataloaders: DataLoaderSet, align_dataloaders: DataLoaderSet):
        self.autoencoder = autoencoder
        self.classifier = classifier
        self.alignment_model = alignment_model

        self.train_loader: DataLoader = cls_dataloaders.train_loader
        self.test_loader: DataLoader = cls_dataloaders.test_loader
        self.val_loader: DataLoader = cls_dataloaders.val_loader

        self.align_train_loader: DataLoader = align_dataloaders.train_loader
        self.align_val_loader: DataLoader = align_dataloaders.val_loader
        self.align_test_loader: DataLoader = align_dataloaders.test_loader

        self.alignment_model.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _alignment_run(self, autoencoder, alignment_model, optimizer, dataloader, device, train=True):
        criterion = ISEBSW
        alignment_model.to(device)
        autoencoder.to(device)

        if train:
            alignment_model.train()
        else:
            alignment_model.eval()

        autoencoder.eval()
        running_loss = 0.0

        for target_samples, source_samples, _ in dataloader: 
            target_samples, source_samples = target_samples.to(device), source_samples.to(device)

            if train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                alignment_output = alignment_model(source_samples)[-1]
                target_reps = autoencoder(target_samples)
                source_reps = autoencoder(alignment_output)

            loss = 0

            for i in range(len(target_reps)//2):
                target_reshaped = target_reps[i].view(target_reps[i].size(0), -1)
                source_reshaped = source_reps[i].view(source_reps[i].size(0), -1)
                loss += criterion(target_reshaped, source_reshaped, L=256, device=device)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)

        return epoch_loss
    
    def _classification_run(self, optimizer, dataloader, device, target_only=False, train=True, use_alignment=False):
        """
        Run one epoch of training or evaluation on the given dataset.
        
        Args:
        - model: The neural network model
        - optimizer: The optimizer for the model (used only if train=True)
        - dataloader: DataLoader instance for the dataset
        - target_only: Boolean, if True, only use the samples from the target dataset
        - train: Boolean, if True, run in training mode; if False, run in evaluation mode
        - use_alignment: Boolean, if True, pass source samples through alignment model 
        
        Returns:
        - Average loss for the epoch
        - Accuracy for the epoch (for classification) or None (for contrastive)
        """

        criterion = nn.CrossEntropyLoss()
        self.classifier.to(device)

        if train:
            self.classifier.train()
        else:
            self.classifier.eval()

        self.alignment_model.to(device)
        self.alignment_model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for target_samples, source_samples, labels in dataloader:
            labels = labels.long()
            target_samples, source_samples, labels = (
                target_samples.to(device), 
                source_samples.to(device), 
                labels.to(device)
            )
            
            if train:
                optimizer.zero_grad()
            
            with torch.set_grad_enabled(train):
                if target_only:
                    outputs = self.classifier(target_samples)[-1]
                    loss = criterion(outputs, labels)
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                
                else:
                    if use_alignment:
                        source_samples = self.alignment_model(source_samples)[-1]
                        source_samples = source_samples.detach()

                    inputs = torch.cat((target_samples, source_samples), 0)

                    outputs = self.classifier(inputs)[-1]
                    loss = criterion(outputs, labels.repeat(2))
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0) * 2
                    correct += predicted.eq(labels.repeat(2)).sum().item()
            
            if train:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct / total
        
        return epoch_loss, epoch_accuracy

    def alignment_preloader_train_loop(self, ae_filename, alignment_filename, device, num_epochs=100, train_both=True):
        if self.autoencoder is None:
            logger.error("No autoencoder found in trainer")
            raise ValueError("No autoencoder found in trainer")

        if train_both:
            if self.alignment_model is None:
                logger.error("No alignment model found in trainer")
                raise ValueError("No alignment model found in trainer")
            
            best_ae_val = float('inf')
            ae_optimizer = optim.Adam(self.autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)

            self.alignment_model.to(device)
            self.autoencoder.to(device)

            for epoch in range(num_epochs):
                self.autoencoder.train()
                for target, source, _ in self.train_loader:
                    ae_optimizer.zero_grad()
                    target = target.to(device)
                    source = source.to(device)
                    combined_input = torch.cat((target, source), dim=0)
                    output = self.autoencoder(combined_input)[-1]
                    loss = nn.MSELoss()(output, combined_input)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1)
                    ae_optimizer.step()

                self.autoencoder.eval()
                total_loss = 0
                with torch.no_grad():
                    for target, source, _ in self.val_loader:
                        target = target.to(device)
                        source = source.to(device)
                        combined_input = torch.cat((target, source), dim=0)
                        output = self.autoencoder(combined_input)[-1]
                        loss = nn.MSELoss()(output, combined_input)
                        total_loss += loss.item()

                avg_val_loss = total_loss / max(1, len(self.val_loader))
                log_message = f"\tAE Epoch {epoch+1}: {avg_val_loss:.4f}"

                if avg_val_loss < best_ae_val:
                    log_message += " <- New Best"
                    best_ae_val = avg_val_loss
                    torch.save(self.autoencoder.state_dict(), ae_filename)

                logger.info(log_message)

        self.autoencoder.load_state_dict(torch.load(ae_filename, weights_only=True))

        alignment_optimizer = optim.Adam(self.alignment_model.parameters(), lr=3e-5, weight_decay=1e-5)
        best_alignment_val = float('inf')

        self.autoencoder.to(device)
        self.alignment_model.to(device)
        self.autoencoder.eval()
        for epoch in range(num_epochs):
            self.alignment_model.train()
            self._alignment_run(
                autoencoder=self.autoencoder,
                alignment_model=self.alignment_model,
                optimizer=alignment_optimizer,
                dataloader=self.align_train_loader, 
                device=device,
                train=True
            )
            torch.nn.utils.clip_grad_norm_(self.alignment_model.parameters(), 1)
            epoch_loss = self._alignment_run(
                autoencoder=self.autoencoder,
                alignment_model=self.alignment_model,
                optimizer=None,
                dataloader=self.align_val_loader, 
                device=device,
                train=False
            )
            log_message = f"\tAlignment Epoch {epoch+1}: {epoch_loss:.4f}"

            if epoch_loss < best_alignment_val:
                log_message += " <- New Best"
                best_alignment_val = epoch_loss
                torch.save(self.alignment_model.state_dict(), alignment_filename)

            logger.info(log_message)

        self.autoencoder.load_state_dict(torch.load(ae_filename, weights_only=True))
        self.alignment_model.load_state_dict(torch.load(alignment_filename, weights_only=True))


    def classification_train_loop(self, classifier_filename, device, num_epochs=100, target_only=False, use_alignment=False):
        """
        Trains the classifier using supervised learning, tracks best validation performance, and saves the best model checkpoint.

        Args:
            classifier_filename (str): Path to save the best model weights.
            device (torch.device): Device to perform training on.
            num_epochs (int, optional): Number of training epochs. Default is 100.
            target_only (bool, optional): If True, only target samples are used for training. Default is False.
            use_alignment (bool, optional): If True, processes inputs with the alignment model before classification. Default is False.

        Returns:
            float: Best validation loss achieved during training.
        """

        optimizer = optim.SGD(self.classifier.parameters(), lr=1e-2, momentum=.9, 
                              nesterov=True, weight_decay=1e-2)

        scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=20,
            T_mult=1,
            eta_min=1e-8
        )

        best_val_loss = 1e7

        self.classifier.to(device)
        self.alignment_model.to(device)
        self.alignment_model.eval()

        for epoch in range(num_epochs):
            train_loss, train_acc = self._classification_run(
                optimizer=optimizer,
                dataloader=self.train_loader,
                device=device,
                target_only=target_only,
                train=True,
                use_alignment=use_alignment
            )

            current_lr = optimizer.param_groups[0]['lr']
            logger.debug(f"Epoch {epoch+1}: Current LR: {current_lr:.2e}")

            scheduler.step() 

            val_loss, val_acc = self._classification_run(
                optimizer=None,
                dataloader=self.val_loader,
                device=device,
                target_only=True,
                train=False,
                use_alignment=use_alignment
            )

            test_loss, test_acc = self._classification_run(
                optimizer=None,
                dataloader=self.test_loader,
                device=device,
                target_only=True,
                train=False,
                use_alignment=use_alignment
            )

            log_message = f"\tEpoch {epoch+1}: Train- {train_loss:.4f} {train_acc*100:.2f}, Val- {val_loss:.4f} {val_acc*100:.2f}, Test - {test_loss:.4f} {test_acc*100:.2f}"

            if val_loss < best_val_loss:
                log_message += " <- New Best"
                best_val_loss = val_loss
                torch.save(self.classifier.state_dict(), classifier_filename)

            logger.info(log_message)
        
        self.classifier.load_state_dict(torch.load(classifier_filename, weights_only=True))

        return best_val_loss    

    def evaluate_model(self, device, use_alignment=False, test=False):
        """
        Evaluates the model on the validation or test set.

        Args:
            device: torch.device to evaluate on.
            use_alignment: Whether to use the alignment model for preprocessing.
            test: If True, evaluates on test set; else on validation set.

        Returns:
            loss (float), acc (float): Average loss and accuracy.
        """
        if test:
            loss, acc = self._classification_run(
                optimizer=None,
                dataloader=self.test_loader,
                device=device,
                target_only=True,
                train=False,
                use_alignment=use_alignment
            )
        else:
            loss, acc = self._classification_run(
                optimizer=None,
                dataloader=self.val_loader,
                device=device,
                target_only=True,
                train=False,
                use_alignment=use_alignment
            )
        return loss, acc

          