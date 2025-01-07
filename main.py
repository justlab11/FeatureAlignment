import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts

from datasets import DatasetGenerator, PairedMNISTDataset
from helpers import EarlyStopper, one_run
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
base_model = TinyCNN()

for epoch in range(num_base_epochs):
    train_loss, train_acc = one_run(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        device=DEVICE,
    )

    val_loss, val_acc = one_run(
        model=model,
        optimizer=optimizer,
        dataloader=val_loader,
        device=DEVICE,
        train=False
    )

    print(f"Epoch {epoch+1}:", round(train_loss, 4), round(train_acc, 4)*100, round(val_loss, 4), round(val_acc, 4)*100)

    if val_loss < base_best["val_loss"]:
        base_best["val_loss"] = val_loss
        base_best["val_acc"] = val_acc
        base_model = model

    if base_early_stopper(val_loss):
        print("Stopped")
        break

print("\n\n\n\n\n")

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
    train_loss, train_acc = one_run(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        mode="base_and_aux",
        device=DEVICE,
    )

    val_loss, val_acc = one_run(
        model=model,
        optimizer=optimizer,
        dataloader=val_loader,
        device=DEVICE,
        mode="base_and_aux",
        train=False
    )

    print(f"Epoch {epoch+1}:", round(train_loss, 4), round(train_acc, 4)*100, round(val_loss, 4), round(val_acc, 4)*100)

    if val_loss < base_aux_best["val_loss"]:
        base_aux_best["val_loss"] = val_loss
        base_aux_best["val_acc"] = val_acc
        base_aux_model = model

    if base_early_stopper(val_loss):
        print("Stopped")
        break

base_loss, base_acc = one_run(
    model=base_model,
    optimizer=optimizer,
    dataloader=val_loader,
    device=DEVICE,
    mode="base_only",
    train=False
)

aux_loss, aux_acc = one_run(
    model=base_aux_model,
    optimizer=optimizer,
    dataloader=val_loader,
    device=DEVICE,
    mode="base_only",
    train=False
)

print(f"Base: {round(base_loss, 4)}, {round(base_acc, 4)*100}")
print(f"Base + Aux: {round(aux_loss, 4)}, {round(aux_acc, 4)*100}")

# TODO: contrastive learning part, already coded just needs implementation 