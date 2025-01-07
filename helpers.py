import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from losses import supervised_contrastive_loss


def classification_run(model, optimizer, dataloader, device, mode='base_only', train=True):
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
    
    if train:
        model.train()
    else:
        model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for base_samples, aux_samples, labels in dataloader:
        base_samples, aux_samples, labels = base_samples.to(device), aux_samples.to(device), labels.to(device)
        
        if train:
            optimizer.zero_grad()
        
        with torch.set_grad_enabled(train):
            if mode == 'base_only':
                outputs = model(base_samples)
                loss = criterion(outputs, labels)
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            else:
                inputs = torch.cat((base_samples, aux_samples), 0)
                outputs = model(inputs)
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

def contrastive_run(model, proj_head, optimizer, dataloader, device, train=True, temperature=0.07):
    """
    Run one epoch of contrastive learning training or evaluation.
    """
    criterion = supervised_contrastive_loss
    
    if train:
        model.train()
        proj_head.train()
    else:
        model.eval()
        proj_head.eval()
    
    running_loss = 0.0
    
    for base_samples, aux_samples, labels in dataloader:
        base_samples, aux_samples, labels = base_samples.to(device), aux_samples.to(device), labels.to(device)
        
        if train:
            optimizer.zero_grad()
        
        with torch.set_grad_enabled(train):
            inputs = torch.cat((base_samples, aux_samples), 0)
            features = model(inputs)
            projected = proj_head(features)
            loss = criterion(projected, labels.repeat(2), temperature=temperature)
        
        if train:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    
    return epoch_loss

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        """
        Initialize the EarlyStopper.

        Args:
        patience (int): Number of epochs to wait before stopping after last improvement.
        min_delta (float): Minimum change in monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop
