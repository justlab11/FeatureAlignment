import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from losses import supervised_contrastive_loss


def one_run(model, optimizer, dataset, batch_size, mode='base_only'):
    """
    Run one epoch of training on the given dataset.
    
    Args:
    - model: The neural network model to train
    - optimizer: The optimizer for the model
    - dataset: PairedMNISTDataset instance
    - batch_size: Batch size for training
    - mode: 'base_only', 'base_and_aux', or 'contrastive'
    
    Returns:
    - Average loss for the epoch
    - Accuracy for the epoch (for classification) or None (for contrastive)
    """
    device = model.parameters().__next__().device
    
    if mode in ['base_only', 'base_and_aux']:
        criterion = nn.CrossEntropyLoss()
    elif mode == 'contrastive':
        criterion = supervised_contrastive_loss
    else:
        raise ValueError("Mode must be 'base_only', 'base_and_aux', or 'contrastive'")
    
    effective_batch_size = batch_size if mode == 'base_only' else batch_size // 2
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for base_samples, aux_samples, labels in dataloader:
        base_samples, aux_samples, labels = base_samples.to(device), aux_samples.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if mode == 'base_only':
            outputs = model(base_samples)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        elif mode == 'base_and_aux':
            inputs = torch.cat((base_samples, aux_samples), 0)
            outputs = model(inputs)
            loss = criterion(outputs, labels.repeat(2))
            
            _, predicted = outputs.max(1)
            total += labels.size(0) * 2
            correct += predicted.eq(labels.repeat(2)).sum().item()
        
        else:  # contrastive
            inputs = torch.cat((base_samples, aux_samples), 0)
            features = model(inputs)
            loss = criterion(features, labels.repeat(2))
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct / total if mode != 'contrastive' else None
    
    return epoch_loss, epoch_accuracy


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
