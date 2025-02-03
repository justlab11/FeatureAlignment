import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import logging

from models import DynamicCNN
from helpers import EarlyStopper
from losses import supervised_contrastive_loss
from type_defs import DataLoaderSet

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, classifier, dataloaders: DataLoaderSet, unet=None, 
                 contrastive=False, classifier_load_path=None, unet_load_path=None):
        
        if unet_load_path and not classifier_load_path:
            logger.warning("UNET weights were given, but no classifier weights were given")

        self.classifier = classifier
        if classifier_load_path:
            self.classifier.load_state_dict(torch.load(classifier_load_path, weights_only=True))

        self.train_loader: DataLoader = dataloaders.train_loader
        self.test_loader: DataLoader= dataloaders.test_loader
        self.val_loader: DataLoader = dataloaders.val_loader

        if contrastive:
            body_output_size = self.classifier.get_body_output_size()
            self.proj_head = nn.Sequential(
                nn.Linear(body_output_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        else:
            self.proj_head = None

        self.unet = unet
        if unet_load_path:
            self.unet.load_state_dict(torch.load(unet_load_path, weights_only=True))

    def classification_train_loop(self, filename, device, mode, num_epochs=50):
        optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=.001,
            weight_decay=1e-5
        )
        early_stopper = EarlyStopper(
            patience=5,
            min_delta=0
        )

        best_val = 1e7

        for epoch in range(num_epochs):
            train_loss, train_acc = self._classification_run(
                model = self.classifier,
                unet_model = self.unet,
                optimizer = optimizer,
                dataloader = self.train_loader,
                mode = mode,
                device = device
            )

            val_loss, val_acc = self._classification_run(
                model = self.classifier,
                unet_model = self.unet,
                optimizer = optimizer,
                dataloader = self.val_loader,
                mode = mode,
                device = device,
                train = False
            )

            logger.info(f"\n\tEpoch {epoch+1}:", round(train_loss, 4), round(train_acc*100, 2), round(val_loss, 4), round(val_acc*100, 2), end="")

            if val_loss < best_val:
                logger.info(" <- New Best", end="")
                best_val = val_loss
                torch.save(self.classifier.state_dict(), filename)

            if early_stopper(val_loss):
                logger.info("\n\t-- Stopped -- ")
                break

    def contrastive_train_loop(self, device, filename, temp_range=[0.05, 0.1, 0.15], num_epochs=200):
        # Step 1: Optimize temperature and train body
        best_temp, best_val_loss = self.optimize_temperature(temp_range, num_epochs, device, filename)
        
        logger.info(f"Best temperature: {best_temp:.4f}, Best validation loss: {best_val_loss:.4f}")
        
        # Load the best model
        self.classifier.load_state_dict(torch.load(filename))
        
        # Step 2: Freeze body and train head
        self.classifier.set_freeze_head(False)
        self.classifier.set_freeze_body(True)
        
        # Train the head
        head_filename = filename.replace('body', 'full')
        self.classification_train_loop(head_filename, device, mode='base_only', num_epochs=50)
        
        # Optionally, unfreeze everything for potential fine-tuning
        self.classifier.set_freeze_head(False)
        self.classifier.set_freeze_body(False)

        logger.info("Contrastive training and head training completed.")


    def _classification_run(self, model, optimizer, dataloader, device, mode, train=True, unet_model=None):
        """
        Run one epoch of training or evaluation on the given dataset.
        
        Args:
        - model: The neural network model
        - optimizer: The optimizer for the model (used only if train=True)
        - dataloader: DataLoader instance for the dataset
        - mode: 'base_only', 'base_and_aux', or 'contrastive'
        - train: Boolean, if True, run in training mode; if False, run in evaluation mode
        
        Returns:
        - Average loss for the epoch
        - Accuracy for the epoch (for classification) or None (for contrastive)
        """

        criterion = nn.CrossEntropyLoss()
        model.to(device)

        if train:
            model.train()
        else:
            model.eval()

        if unet_model:
            unet_model.to(device)
            unet_model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0

        if mode == "base_only":
            dataloader.dataset.unique_sources = True 
        else:
            dataloader.dataset.unique_sources = False
        
        for base_samples, aux_samples, labels in dataloader:
            base_samples, aux_samples, labels = base_samples.to(device), aux_samples.to(device), labels.to(device)
            
            if train:
                optimizer.zero_grad()
            
            with torch.set_grad_enabled(train):
                if mode == 'base_only' and unet_model == None:
                    outputs = model(base_samples)[-1]
                    loss = criterion(outputs, labels)
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                
                else:
                    if unet_model != None:
                        aux_samples = unet_model(aux_samples)

                    inputs = torch.cat((base_samples, aux_samples), 0)

                    outputs = model(inputs)[-1]
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

    def _optimize_temperature(self, temp_range, num_epochs, device, filename):
        best_temp = None
        best_val_loss = float('inf')
        
        for temp in temp_range:
            logger.info(f"Starting temp: {temp}")
            self.classifier.reset_parameters()
            self.proj_head.reset_parameters()
            
            optimizer = optim.Adam(
                list(self.classifier.parameters()) + list(self.proj_head.parameters()),
                lr=0.001, 
                weight_decay=1e-5
            )
            
            early_stopper = EarlyStopper(patience=10, min_delta=0)
            
            for epoch in range(num_epochs):
                train_loss = self._contrastive_run(
                    model=self.classifier,
                    proj_head=self.proj_head,
                    optimizer=optimizer,
                    dataloader=self.train_loader,
                    device=device,
                    temperature=temp
                )
                
                val_loss = self._contrastive_run(
                    model=self.classifier,
                    proj_head=self.proj_head,
                    dataloader=self.val_loader,
                    device=device,
                    train=False,
                    temperature=temp
                )
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_temp = temp
                    torch.save(self.classifier.state_dict(), filename)
                
                if early_stopper(val_loss):
                    break
        
        return best_temp, best_val_loss


    def _contrastive_run(model: DynamicCNN, proj_head, dataloader, device, optimizer=None, train=True, temperature=0.1, unet_model=None):
        """
        Run one epoch of contrastive learning training or evaluation.
        """
        criterion = supervised_contrastive_loss
        model.set_freeze_head(True)
        model.set_freeze_body(False)
        
        if train:
            model.train()
            proj_head.train()
        else:
            model.eval()
            proj_head.eval()
        
        running_loss = 0.0

        if unet_model:
            unet_model.to(device)
            unet_model.eval()
        
        for base_samples, aux_samples, labels in dataloader:
            base_samples, aux_samples, labels = base_samples.to(device), aux_samples.to(device), labels.to(device)
            
            if unet_model!=None:
                aux_samples = unet_model(aux_samples)
            if train:
                optimizer.zero_grad()
            
            with torch.set_grad_enabled(train):
                inputs = torch.cat((base_samples, aux_samples), 0)
                features = model(inputs)[-1]
                projected = proj_head(features)
                loss = criterion(projected, labels.repeat(2), temperature=temperature)
            
            if train:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        
        return epoch_loss

