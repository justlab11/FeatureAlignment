import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np 
import logging
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

from models import DynamicCNN
from losses import supervised_contrastive_loss, ISEBSW
from datasets import CombinedDataset, HEIFFolder, IndexedDataset
from type_defs import DataLoaderSet

logger = logging.getLogger(__name__)

class Trainer:
    """
    The trainer is an all inclusive model trainer for this work. It's quite long, but here are all of the functions with their use case:
    * classification_train_loop - trains the classifier provided on the the dataset provided using CE for improving classification accuracy
    * evaluate_model - inferences the model on the validation set to get accuracy
    * unet_train_loop - the train loop to train the unet architecture to get source closer to target
    * unet_classifier_loop - the full loop to go from unet train to classifier retrain
    """
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

    def classification_train_loop(self, filename, device, mode, num_epochs=100):
        optimizer = optim.SGD(self.classifier.parameters(), lr=1e-2, momentum=.9, 
                              nesterov=True, weight_decay=1e-2)

        # def lr_lambda(epoch):
        #     if epoch < 30:  # Linear warmup for first 30 epochs
        #         return (epoch * (1e-1 - 1e-8) / 30) + 1e-8
        #     else:  # Cosine annealing for remaining epochs
        #         return 1e-1 * 0.5 * (1 + np.cos(np.pi * (epoch - 30) / (num_epochs - 30)))

        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=20,
            T_mult=1,
            eta_min=1e-8
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

            current_lr = optimizer.param_groups[0]['lr']
            logger.debug(f"Epoch {epoch+1}: Current LR: {current_lr:.2e}")

            scheduler.step() 

            val_loss, val_acc = self._classification_run(
                model = self.classifier,
                unet_model = self.unet,
                optimizer = optimizer,
                dataloader = self.val_loader,
                mode = mode,
                device = device,
                train = False
            )

            log_message = f"\tEpoch {epoch+1}: {train_loss:.4f} {train_acc*100:.2f} {val_loss:.4f} {val_acc*100:.2f}"

            if val_loss < best_val:
                log_message += " <- New Best"
                best_val = val_loss
                torch.save(self.classifier.state_dict(), filename)

            logger.info(log_message)

        self.classifier.load_state_dict(torch.load(filename, weights_only=True))

    def evaluate_model(self, device):
        _, val_acc = self._classification_run(
            model=self.classifier,
            unet_model=self.unet,
            optimizer=None,
            dataloader=self.val_loader,
            device=device,
            mode="base_only",
            train=False,
        )

        return val_acc
    
    def unet_train_loop(self, filename, device, num_epochs=20):
        optimizer = optim.Adam(
            self.unet.parameters(),
            lr = 3e-4,
            weight_decay = 1e-5
        )
        unet_val_loss = 1e7

        for epoch in range(num_epochs):
            train_loss = self._unet_run(
                unet_model=self.unet,
                classifier=self.classifier,
                optimizer=optimizer,
                dataloader=self.train_loader,
                device=device,
                train=True
            )

            val_loss = self._unet_run(
                unet_model=self.unet,
                classifier=self.classifier,
                optimizer=optimizer,
                dataloader=self.val_loader,
                device=device,
                train=False
            )

            _, classifier_val_acc = self._classification_run(
                model=self.classifier,
                optimizer=None,
                dataloader=self.val_loader,
                device=device,
                train=False,
                unet_model=self.unet
            )

            log_message = f"\tEpoch {epoch+1}: {train_loss:.4f} {val_loss:.4f}"

            if val_loss < unet_val_loss:
                log_message += " <- New Best"
                log_message += f"\n\tClassifier Acc: {round(classifier_val_acc*100), 2}\n"
                unet_val_loss = val_loss
                torch.save(self.unet.state_dict(), filename)

            logger.info(log_message)

        self.unet.load_state_dict(torch.load(filename, weights_only=True))

    def unet_classifier_train_loop(self, classifier_filename, unet_filename, device, batch_size, unet_epochs=100, classifier_epochs=50):
        unet_optimizer = optim.Adam(
            self.unet.parameters(),
            lr = 3e-3,
            weight_decay = 1e-5
        )

        unet_scheduler = CosineAnnealingLR(unet_optimizer, T_max=unet_epochs, eta_min=1e-6)

        best_unet_val_loss = float('inf')
        best_classifier_val_loss = float('inf')

        # Phase 1: U-Net Training
        logger.info("Warm Starting UNET Model")
        for ws_epoch in range(20):
            _ = self._unet_run(
                unet_model=self.unet,
                classifier=self.classifier,
                optimizer=unet_optimizer,
                dataloader=self.train_loader,
                device=device,
                train=True
            )

        logger.info("Starting UNET Model Training")
        for epoch in range(unet_epochs):
            train_loss = self._unet_run(
                unet_model=self.unet,
                classifier=self.classifier,
                optimizer=unet_optimizer,
                dataloader=self.train_loader,
                device=device,
                train=True
            )

            unet_scheduler.step()

            # Validate U-Net
            with torch.no_grad():
                val_loss = self._unet_run(
                    unet_model=self.unet,
                    classifier=self.classifier,
                    optimizer=None,
                    dataloader=self.val_loader,
                    device=device,
                    train=False
                )

            log_message = f"U-Net Epoch {epoch+1}/{unet_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"

            if val_loss < best_unet_val_loss:
                log_message += " <- New Best"
                best_unet_val_loss = val_loss
                torch.save(self.unet.state_dict(), unet_filename)

            logger.info(log_message)

        # Load best U-Net model
        self.unet.load_state_dict(torch.load(unet_filename, weights_only=True))

        train_ds = self._adjust_aux_dataset(
            dataloader=self.train_loader,
            classifier=self.classifier,
            unet=self.unet,
            batch_size=batch_size,
            device=device
        )

        val_ds = self._adjust_aux_dataset(
            dataloader=self.val_loader,
            classifier=self.classifier,
            unet=self.unet,
            batch_size=batch_size,
            device=device
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        classifier_optimizer = optim.Adam(
            self.classifier.parameters(),
            lr = 1e-4,
            weight_decay = 1e-5
        )

        classifier_scheduler = CosineAnnealingLR(classifier_optimizer, T_max=classifier_epochs, eta_min=1e-8)

        # reset head of the model
        self.classifier.reset_head_parameters()

        # Phase 2: Classifier Head Fine-tuning
        logger.info("Starting Classifier Head Fine-tuning")
        for epoch in range(classifier_epochs):
            self.classifier.set_freeze_head(False)
            self.classifier.set_freeze_body(False)

            train_loss, _ = self._classification_run(
                model = self.classifier,
                unet_model = self.unet,
                optimizer = classifier_optimizer,
                dataloader = self.train_loader,
                mode = "sa",
                device = device,
                train=True
            )

            classifier_scheduler.step()

            classifier_val_loss, classifier_val_acc = self._classification_run(
                model=self.classifier,
                unet_model=self.unet,
                optimizer=None,
                dataloader=self.val_loader,
                device=device,
                mode="base_only",
                train=False,
            )

            log_message = f"Classifier Epoch {epoch+1}/{classifier_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {classifier_val_loss:.4f}, Val Acc: {classifier_val_acc*100:.2f}%"

            if classifier_val_loss < best_classifier_val_loss:
                log_message += " <- New Best"
                best_classifier_val_loss = classifier_val_loss
                torch.save(self.classifier.state_dict(), classifier_filename)

            logger.info(log_message)

        # Load best classifier model
        self.classifier.load_state_dict(torch.load(classifier_filename, weights_only=True))

    def unet_contrastive_train_loop(self, classifier_filename, unet_filename, device, num_epochs=100, best_temp=0.05):
        unet_optimizer = optim.Adam(
            self.unet.parameters(),
            lr = 3e-4,
            weight_decay = 1e-5
        )

        body_optimizer = optim.Adam(
            list(self.classifier.parameters()) + list(self.proj_head.parameters()),
            lr=0.001, 
            weight_decay=1e-5
        )

        best_val_loss = 1e7

        for ws_epoch in range(10):
            # train unet once
            _ = self._unet_run(
                unet_model=self.unet,
                classifier=self.classifier,
                optimizer=unet_optimizer,
                dataloader=self.train_loader,
                device=device,
                train=True
            )

        for epoch in range(num_epochs):
            # train unet once
            _ = self._unet_run(
                unet_model=self.unet,
                classifier=self.classifier,
                optimizer=unet_optimizer,
                dataloader=self.train_loader,
                device=device,
                train=True
            )
            
            # train contrastive body with unet
            contrast_train_loss = self._contrastive_run(
                model=self.classifier,
                unet_model=self.unet,
                proj_head=self.proj_head,
                optimizer=body_optimizer,
                dataloader=self.train_loader,
                device=device,
                temperature=best_temp
            )

            self.classifier.set_freeze_head(False)
            self.classifier.set_freeze_body(True)
            
            # Train the head
            self.classification_train_loop(classifier_filename, device, mode='mixed', num_epochs=100)

            self.classifier.set_freeze_head(False)
            self.classifier.set_freeze_body(False)

            classifier_val_loss, classifier_val_acc = self._classification_run(
                model=self.classifier,
                unet_model=self.unet,
                optimizer=None,
                dataloader=self.val_loader,
                device=device,
                mode="base_only",
                train=False,
            )

            log_message = f"\tEpoch {epoch+1}: {classifier_val_loss:.4f} {classifier_val_acc*100:.2f}"

            if classifier_val_loss < best_val_loss:
                log_message += " <- New Best"
                log_message += f"\n\tClassifier Acc: {round(classifier_val_acc*100), 2}\n"
                best_val_loss = classifier_val_loss
                torch.save(self.classifier.state_dict(), classifier_filename)
                torch.save(self.unet.state_dict(), unet_filename)

            logger.info(log_message)

        self.classifier.load_state_dict(torch.load(classifier_filename, weights_only=True))
        self.unet.load_state_dict(torch.load(unet_filename, weights_only=True))
    
    def _unet_run(self, unet_model, classifier, optimizer, dataloader, device, train=True):
        criterion = ISEBSW
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
                unet_output = unet_model(aux_samples)[-1]
                base_reps = classifier(base_samples)
                aux_reps = classifier(unet_output)
            
            loss = 0
            for i in range(len(base_reps)):
                base_reshaped = base_reps[i].view(base_reps[i].size(0), -1)
                aux_reshaped = aux_reps[i].view(aux_reps[i].size(0), -1)
                loss += criterion(base_reshaped, aux_reshaped, L=256, device=device)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)

        return epoch_loss


    def contrastive_train_loop(self, device, filename, temp_range=[0.05, 0.1, 0.15], num_epochs=100, best_temp=None):
        # Step 1: Optimize temperature and train body
        if best_temp == None:
            best_temp, best_val_loss = self._optimize_temperature(temp_range, num_epochs, device, filename)
        
            logger.info(f"\tBest temperature: {best_temp:.4f}, Best validation loss: {best_val_loss:.4f}")
        
        # Load the best model
        self.classifier.load_state_dict(torch.load(filename, weights_only=True))
        
        # Step 2: Freeze body and train head
        self.classifier.set_freeze_head(False)
        self.classifier.set_freeze_body(True)
        
        # Train the head
        head_filename = filename.replace('body', 'full')
        self.classification_train_loop(head_filename, device, mode='mixed', num_epochs=100)
        
        # Optionally, unfreeze everything for potential fine-tuning
        self.classifier.set_freeze_head(False)
        self.classifier.set_freeze_body(False)

        logger.info("Contrastive training and head training completed.")
        return best_temp


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

        # if mode == "base_only":
        #     dataloader.dataset.dataset.unique_sources = True 
        # else:
        #     dataloader.dataset.dataset.unique_sources = True
        
        for base_samples, aux_samples, labels in dataloader:
            labels = labels.long()
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
                        aux_samples = unet_model(aux_samples)[-1]

                    # logger.info(base_samples.shape)
                    # logger.info(aux_samples.shape)
                    # print(base_samples.shape)
                    # print(aux_samples.shape)

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
            logger.info(f"\tStarting temp: {temp}")
            self.classifier.reset_parameters()
            self.reset_parameters(self.proj_head)
            
            optimizer = optim.Adam(
                list(self.classifier.parameters()) + list(self.proj_head.parameters()),
                lr=0.001, 
                weight_decay=1e-5
            )
                        
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
        
        return best_temp, best_val_loss


    def _contrastive_run(self, model: DynamicCNN, proj_head, dataloader, device, optimizer=None, train=True, temperature=0.1, unet_model=None):
        """
        Run one epoch of contrastive learning training or evaluation.
        """
        criterion = supervised_contrastive_loss
        model.set_freeze_head(True)
        model.set_freeze_body(False)
        model.to(device)
        proj_head.to(device)

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
            labels = labels.long()
            group_labels = torch.cat([
                torch.zeros(len(base_samples)),
                torch.ones(len(aux_samples)),
            ]).long()
            group_labels = group_labels.reshape(1, -1)
            base_samples, aux_samples, labels = base_samples.to(device), aux_samples.to(device), labels.to(device)
            
            if unet_model!=None:
                aux_samples = unet_model(aux_samples)[-1]
            if train:
                optimizer.zero_grad()
            
            with torch.set_grad_enabled(train):
                inputs = torch.cat((base_samples, aux_samples), 0)
                features = model(inputs)[-1]
                features = features.reshape(inputs.shape[0], -1)
                projected = proj_head(features)
                loss = criterion(projected, labels.repeat(2), group_labels, temperature=temperature)
            
            if train:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        
        return epoch_loss
    
    def _adjust_aux_dataset(self, dataloader, classifier, unet, device, batch_size):
        classifier.to(device)
        unet.to(device)

        classifier.eval()
        unet.eval()

        dataset: CombinedDataset = dataloader.dataset
        aux_dataset = dataset.aux_dataset

        init_len = len(aux_dataset)

        aux_dataloader = DataLoader(
            aux_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        indices_to_keep = []

        for batch_idx, (aux_samples, labels) in enumerate(aux_dataloader):
            labels = labels.long()
            aux_samples = aux_samples.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = classifier(aux_samples)[-1]
                _, predicted = torch.max(outputs, 1)

                # Compare predictions with labels
                correctly_classified = (predicted == labels)
                
                # Get the indices of correctly classified samples
                correct_indices = correctly_classified.nonzero().squeeze().cpu().numpy()
                
                # Adjust indices to global dataset indices
                global_indices = correct_indices + batch_idx * batch_size
                
                indices_to_keep.extend(global_indices.tolist())

        # Create a new IndexedDataset with only the correctly classified samples
        new_aux_dataset = IndexedDataset(aux_dataset.base_dataset, [aux_dataset.indices[i] for i in indices_to_keep])

        final_len = len(new_aux_dataset)

        # Update the CombinedDataset
        dataset.aux_dataset = new_aux_dataset

        removed_len = init_len - final_len

        logger.info(f"Removed {removed_len} ({round(removed_len/init_len*100, 2)}%) samples from the auxiliary dataset.")

        return dataset
                

    def reset_parameters(self, model):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class PreloaderTrainer:
    def __init__(self, autoencoder, dataloaders: DataLoaderSet, unet, classifier,
                 ae_load_path=None, unet_load_path=None, classifier_load_path=None):
        
        if unet_load_path and not ae_load_path:
            logger.warning("UNET weights were given, but no classifier weights were given")

        self.autoencoder = autoencoder
        if ae_load_path:
            self.autoencoder.load_state_dict(torch.load(ae_load_path, weights_only=True))

        self.classifier = classifier
        if classifier_load_path:
            self.classifier.load_state_dict(torch.load(classifier_load_path, weights_only=True))

        self.train_loader: DataLoader = dataloaders.train_loader
        self.test_loader: DataLoader= dataloaders.test_loader
        self.val_loader: DataLoader = dataloaders.val_loader

        self.unet = unet
        if unet_load_path:
            self.unet.load_state_dict(torch.load(unet_load_path, weights_only=True))

    def unet_preloader_train_loop(self, ae_filename, unet_filename, device, num_epochs=100, train_both=True):
        if self.autoencoder == None:
            logger.error("No classifier found in trainer")
            raise ValueError("No classifier found in trainer")
        
        if train_both:
            if self.unet == None:
                logger.error("No UNET found in trainer")
                raise ValueError("No UNET found in trainer")
            
            best_ae_val = 1e7
            best_unet_val = 1e7
            ae_optimizer = optim.Adam(self.autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)

            self.unet.to(device)
            self.autoencoder.to(device)

            for epoch in range(num_epochs):
                self.unet.train()
                for base, aux, _ in self.train_loader:
                    ae_optimizer.zero_grad()
                    base = base.to(device)
                    aux = aux.to(device)

                    combined_input = torch.cat((base, aux), dim=0)

                    output = self.autoencoder(combined_input)[-1]

                    loss = nn.MSELoss()(output, combined_input)

                    loss.backward()
                    ae_optimizer.step()

                self.unet.eval()
                total_loss = 0
                with torch.no_grad():
                    for base, aux, _ in self.val_loader:
                        base = base.to(device)
                        aux = aux.to(device)

                        combined_input = torch.cat((base, aux), dim=0)

                        output = self.autoencoder(combined_input)[-1]
                        loss = nn.MSELoss()(output, combined_input)
                        total_loss += loss

                total_loss /= len(self.val_loader)
                if total_loss < best_ae_val:
                    best_ae_val = total_loss
                    torch.save(self.autoencoder.state_dict(), ae_filename)

        self.autoencoder.load_state_dict(torch.load(ae_filename, weights_only=True))

        unet_optimizer = optim.Adam(self.unet.parameters(), lr=1e-3, weight_decay=1e-5)

        best_unet_val = 1e7

        self.autoencoder.to(device)
        self.unet.to(device)

        self.autoencoder.eval()
        for epoch in range(num_epochs):
            self.autoencoder.train()
            for base, aux, labels in self.train_loader:
                labels = labels.long()

                base = base.to(device)
                aux = aux.to(device)
                labels = labels.to(device)

                self._unet_run(
                    autoencoder=self.autoencoder,
                    unet_model=self.unet,
                    optimizer=unet_optimizer,
                    dataloader=self.train_loader,
                    device=device,
                    train=True
                )

                epoch_loss = self._unet_run(
                    autoencoder=self.autoencoder,
                    unet_model=self.unet,
                    optimizer=None,
                    dataloader=self.val_loader,
                    device=device,
                    train=False
                )

            log_message = f"\tEpoch {epoch+1}: {epoch_loss:.4f}"

            if epoch_loss < best_unet_val:
                log_message += " <- New Best"
                best_unet_val = epoch_loss
                torch.save(self.unet.state_dict(), unet_filename)

            logger.info(log_message)

        self.autoencoder.load_state_dict(torch.load(ae_filename, weights_only=True))
        self.unet.load_state_dict(torch.load(unet_filename, weights_only=True))

    def classification_train_loop(self, classifier_filename, device, num_epochs=100):
        optimizer = optim.SGD(self.classifier.parameters(), lr=1e-2, momentum=.9, 
                              nesterov=True, weight_decay=1e-2)
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=20,
            T_mult=1,
            eta_min=1e-8
        )

        best_val = 1e7

        for epoch in range(num_epochs):
            train_loss, train_acc = self._classification_run(
                model = self.classifier,
                unet_model = self.unet,
                optimizer = optimizer,
                dataloader = self.train_loader,
                mode = "both",
                device = device
            )

            current_lr = optimizer.param_groups[0]['lr']
            logger.debug(f"Epoch {epoch+1}: Current LR: {current_lr:.2e}")

            scheduler.step() 

            val_loss, val_acc = self._classification_run(
                model = self.classifier,
                unet_model = self.unet,
                optimizer = optimizer,
                dataloader = self.val_loader,
                mode = "both",
                device = device,
                train = False
            )

            log_message = f"\tEpoch {epoch+1}: {train_loss:.4f} {train_acc*100:.2f} {val_loss:.4f} {val_acc*100:.2f}"

            if val_loss < best_val:
                log_message += " <- New Best"
                best_val = val_loss
                torch.save(self.classifier.state_dict(), classifier_filename)

            logger.info(log_message)

        self.classifier.load_state_dict(torch.load(classifier_filename, weights_only=True))

    def evaluate_model(self, device):
        _, val_acc = self._classification_run(
            model=self.classifier,
            unet_model=self.unet,
            optimizer=None,
            dataloader=self.val_loader,
            device=device,
            mode="base_only",
            train=False,
        )

        return val_acc             

    def _unet_run(self, autoencoder, unet_model, optimizer, dataloader, device, train=True):
        criterion = ISEBSW
        unet_model.to(device)
        autoencoder.to(device)

        if train:
            unet_model.train()
        else:
            unet_model.eval()

        autoencoder.eval()

        running_loss = 0.0

        for base_samples, aux_samples, _ in dataloader: 
            base_samples, aux_samples = base_samples.to(device), aux_samples.to(device)
            if train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                unet_output = unet_model(aux_samples)[-1]
                base_reps = autoencoder(base_samples)
                aux_reps = autoencoder(unet_output)
            
            loss = 0
            for i in range(len(base_reps)//2):
                base_reshaped = base_reps[i].view(base_reps[i].size(0), -1)
                aux_reshaped = aux_reps[i].view(aux_reps[i].size(0), -1)
                loss += criterion(base_reshaped, aux_reshaped, L=256, device=device)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)

        return epoch_loss

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
            dataloader.dataset.dataset.unique_sources = True 
        else:
            dataloader.dataset.dataset.unique_sources = True
        
        for base_samples, aux_samples, labels in dataloader:
            labels = labels.long()
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
                        aux_samples = unet_model(aux_samples)[-1]

                    # logger.info(base_samples.shape)
                    # logger.info(aux_samples.shape)
                    # print(base_samples.shape)
                    # print(aux_samples.shape)

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