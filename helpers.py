import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import yaml

from losses import supervised_contrastive_loss, SlicedWasserstein, DSW
from models import ProjNet


def classification_run(model, optimizer, dataloader, device, mode='base_only', train=True, unet_model=None):
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

def contrastive_run(model, proj_head, dataloader, device, optimizer=None, train=True, temperature=0.1, unet_model=None):
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

def old_unet_run(unet_model, classifier, optimizer, dataloader, device, train=True):
    upscale_transform = transforms.Resize((256, 256))

    criterion = SlicedWasserstein(num_projections=256)
    unet_model.to(device)
    classifier.to(device)

    if train:
        unet_model.train()
    else:
        unet_model.eval()

    classifier.eval()

    running_loss = 0.0

    for base_samples, aux_samples, _ in dataloader: 
        aux_samples = upscale_transform(aux_samples)

        base_samples, aux_samples = base_samples.to(device), aux_samples.to(device)
        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            unet_output = unet_model(aux_samples)
            unet_images = nn.functional.interpolate(unet_output, size=(28, 28), mode='bilinear', align_corners=False)

            base_reps = classifier(base_samples)
            aux_reps = classifier(unet_images)
        
        loss = 0
        for i in range(len(base_reps)):
            loss += criterion(base_reps[i], aux_reps[i])

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss

def unet_run(unet_model, classifier, optimizer, dataloader, device, train=True):
    criterion = SlicedWasserstein(num_projections=256)
    unet_model.to(device)
    classifier.to(device)

    if train:
        unet_model.train()
    else:
        unet_model.eval()

    classifier.eval()

    running_loss = 0.0

    for base_samples, aux_samples, _ in dataloader: 
        base_samples, aux_samples = base_samples.to(device), aux_samples.to(device)
        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            unet_output = unet_model(aux_samples)
            base_reps = classifier(base_samples)
            aux_reps = classifier(unet_output)
        
        loss = 0
        for i in range(len(base_reps)):
            loss += criterion(base_reps[i], aux_reps[i])

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss

def run_dswd_classwise(model, dataloader, layers, device, base_only=True, unet_model=None, class_choice=None, num_projections=128, embedding_norm=1.0):
    dataloader.dataset.unique_sources = True
    dataloader.dataset.specific_class = class_choice
    model.to(device)
    model.eval()

    if unet_model != None:
        unet_model.to(device)
        unet_model.eval()

    if type(layers) == int:
        layers = [i for i in range(layers+1)]

    dswd_loss = np.zeros((len(dataloader), len(layers)))
    
    for i, (base, aux, _) in enumerate(dataloader):
        if base_only:
            dataset_1, dataset_2 = torch.split(base, base.size(0) // 2)
        else:
            dataset_1, _ = torch.split(base, base.size(0) // 2)
            dataset_2, _ = torch.split(aux, aux.size(0) // 2)

        dataset_1 = dataset_1.to(device)
        dataset_2 = dataset_2.to(device)

        if unet_model and not base_only:
            dataset_2 = unet_model(dataset_2)
            
        dataset_1_outputs = model(dataset_1)[:layers[-1]+1]
        dataset_2_outputs = model(dataset_2)[:layers[-1]+1]

        for j, layer in enumerate(layers):
            dataset_1_layer_flat = dataset_1_outputs[layer].view(dataset_1_outputs[layer].size(0), -1)
            dataset_2_layer_flat = dataset_2_outputs[layer].view(dataset_2_outputs[layer].size(0), -1)
    
            projnet = ProjNet(size=dataset_1_layer_flat.size(1)).to(device)
            op_projnet = optim.Adam(
                projnet.parameters(),
                lr=0.001, 
                weight_decay=1e-5
            )

            dsw = DSW(
                encoder=None,
                embedding_norm=embedding_norm,
                num_projections=num_projections,
                projnet=projnet,
                op_projnet=op_projnet
            )

            dswd_loss[i, j] += dsw(
                dataset_1_layer_flat,
                dataset_2_layer_flat
            ) / dataset_1_layer_flat.size(0)

    return dswd_loss

def run_dswd_all_classes(model, dataloader, layers, device, base_only=True, unet_model=None, num_projections=128, embedding_norm=1.0):
    unique_classes = torch.unique(dataloader.dataset.base_labels, sorted=True).tolist()
    class_hist = {str(i): 0 for i in unique_classes}
    class_loss = {str(i): 0 for i in unique_classes}

    for cls in unique_classes:
        dataloader.dataset.specific_class = cls
        class_hist[str(cls)] = len(dataloader)
        print(f"Class {cls} Size: {len(dataloader)*dataloader.batch_size} ({len(dataloader)} batches)")
        loss = run_dswd_classwise(
            model=model,
            dataloader=dataloader,
            layers=layers,
            device=device,
            base_only=base_only,
            num_projections=num_projections,
            unet_model=unet_model,
            embedding_norm=embedding_norm
        )
        class_loss[str(cls)] = loss

    return class_loss, class_hist

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

def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data