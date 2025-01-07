import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
import os

from datasets import DatasetGenerator, PairedMNISTDataset
from helpers import EarlyStopper, classification_run, contrastive_run
from models import TinyCNN, TinyCNN_Headless

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True)

x_train = mnist_train.data.numpy()
y_train = mnist_train.targets.numpy()

x_test = mnist_test.data.numpy()
y_test = mnist_test.targets.numpy()

x_test, x_val, y_test, y_val = tts(
    x_test, y_test, test_size=.5
)

ds_generator: DatasetGenerator = DatasetGenerator(
    base_ds="red"
)

base_images_train, base_labels_train = ds_generator.build_base_dataset(x_train, y_train)
base_images_test, base_labels_test = ds_generator.build_base_dataset(x_test, y_test)
base_images_val, base_labels_val = ds_generator.build_base_dataset(x_val, y_val)

aux_images_train, aux_labels_train = ds_generator.build_aux_dataset(x_train, y_train)
aux_images_test, aux_labels_test = ds_generator.build_aux_dataset(x_test, y_test)
aux_images_val, aux_labels_val = ds_generator.build_aux_dataset(x_val, y_val)

train_dataset: PairedMNISTDataset = PairedMNISTDataset(
    base_images=base_images_train,
    base_labels=base_labels_train,
    aux_images=aux_images_train,
    aux_labels=aux_labels_train
)

test_dataset: PairedMNISTDataset = PairedMNISTDataset(
    base_images=base_images_test,
    base_labels=base_labels_test,
    aux_images=aux_images_test,
    aux_labels=aux_labels_test
)

val_dataset: PairedMNISTDataset = PairedMNISTDataset(
    base_images=base_images_val,
    base_labels=base_labels_val,
    aux_images=aux_images_val,
    aux_labels=aux_labels_val
)

train_loader: DataLoader = DataLoader(
    dataset=train_dataset,
    batch_size=8
)

test_loader: DataLoader = DataLoader(
    dataset=test_dataset,
    batch_size=8
)

val_loader: DataLoader = DataLoader(
    dataset=val_dataset,
    batch_size=8
)

if not os.path.exists("models"):
    os.mkdir("models")

### Base model

if not os.path.exists("models/base_model.pt"):
    num_base_epochs = 20

    model = TinyCNN()
    model.to(DEVICE)
    optimizer = optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    base_early_stopper = EarlyStopper(
        patience=5,
        min_delta=0
    )

    base_best = {
        "val_loss": 1000,
        "val_acc": 0
    }

    for epoch in range(num_base_epochs):
        train_loss, train_acc = classification_run(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            device=DEVICE,
        )

        val_loss, val_acc = classification_run(
            model=model,
            optimizer=optimizer,
            dataloader=val_loader,
            device=DEVICE,
            train=False
        )

        print(f"Epoch {epoch+1}:", round(train_loss, 4), round(train_acc*100, 2), round(val_loss, 4), round(val_acc*100, 2))

        if val_loss < base_best["val_loss"]:
            base_best["val_loss"] = val_loss
            base_best["val_acc"] = val_acc
            torch.save(model.state_dict(), "models/base_model.pt")

        if base_early_stopper(val_loss):
            print("Stopped")
            break

    print("\n")

### Base + aux model

if not os.path.exists("models/aux_model.pt"):
    model = TinyCNN()
    model.to(DEVICE)
    optimizer = optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    base_early_stopper = EarlyStopper(
        patience=5,
        min_delta=0
    )

    base_aux_best = {
        "val_loss": 1000,
        "val_acc": 0
    }
    base_aux_model = TinyCNN()

    for epoch in range(num_base_epochs):
        train_loss, train_acc = classification_run(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            mode="base_and_aux",
            device=DEVICE,
        )

        val_loss, val_acc = classification_run(
            model=model,
            optimizer=optimizer,
            dataloader=val_loader,
            device=DEVICE,
            mode="base_and_aux",
            train=False
        )

        print(f"Epoch {epoch+1}:", round(train_loss, 4), round(train_acc*100, 2), round(val_loss, 4), round(val_acc*100, 2))

        if val_loss < base_aux_best["val_loss"]:
            base_aux_best["val_loss"] = val_loss
            base_aux_best["val_acc"] = val_acc
            torch.save(model.state_dict(), "models/aux_model.pt")

        if base_early_stopper(val_loss):
            print("Stopped")
            break

    print("\n")

### Comparison of models
optimizer = None # Set to None so it can't train

base_model = TinyCNN()
base_model.load_state_dict(torch.load("models/base_model.pt", weights_only=True))
base_model.to(DEVICE)
base_loss, base_acc = classification_run(
    model=base_model,
    optimizer=optimizer,
    dataloader=val_loader,
    device=DEVICE,
    mode="base_only",
    train=False
)

aux_model = TinyCNN()
aux_model.load_state_dict(torch.load("models/aux_model.pt", weights_only=True))
aux_model.to(DEVICE)
aux_loss, aux_acc = classification_run(
    model=aux_model,
    optimizer=optimizer,
    dataloader=val_loader,
    device=DEVICE,
    mode="base_only",
    train=False
)

print(f"Base: {round(base_loss, 4)}, {round(base_acc*100, 2)}")
print(f"Base + Aux: {round(aux_loss, 4)}, {round(aux_acc*100, 2)}")

### Contrastive learning model
num_contrast_epochs = 200
temp_range = np.linspace(0.05, .5, 10)
best_val_loss = 1000*np.ones(len(temp_range))

for i, temp in enumerate(temp_range):
    model = TinyCNN_Headless()
    proj_head = torch.nn.Linear(32, 128)
    class_head = torch.nn.Linear(32, 10)

    model.to(DEVICE)
    proj_head.to(DEVICE)
    class_head.to(DEVICE)

    contrast_optimizer = optim.Adam(
        list(model.parameters()) + list(proj_head.parameters()),
        lr=0.001, 
        weight_decay=1e-5
    )

    contrast_early_stopper = EarlyStopper(
        patience=10,
        min_delta=0
    )

    for epoch in range(num_contrast_epochs):
        train_loss = contrastive_run(
            model=model,
            proj_head=proj_head,
            optimizer=contrast_optimizer,
            dataloader=train_loader,
            device=DEVICE,
            temperature=temp
        )

        val_loss = contrastive_run(
            model=model,
            proj_head=proj_head,
            optimizer=contrast_optimizer,
            dataloader=val_loader,
            device=DEVICE,
            train=False,
            temperature=temp
        )

        if val_loss < best_val_loss[i]:
            best_val_loss[i] = val_loss
            torch.save(model.state_dict(), f"models/contrast_body_{temp}.pt")
            torch.save(proj_head.state_dict(), f"models/contrast_proj_{temp}.pt")

        if contrast_early_stopper(val_loss):
            print("Stopped")
            break

print(f"Best temp: {temp_range[np.argmin(best_val_loss)]}")

# TODO: add contrastive classification head training