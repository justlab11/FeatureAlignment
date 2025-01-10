import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split as tts
import os

from datasets import DatasetGenerator, PairedMNISTDataset
from helpers import EarlyStopper, classification_run, contrastive_run
from models import TinyCNN, TinyCNN_Headless, WrapperModelTrainHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True)

x_train = mnist_train.data.numpy()
y_train = mnist_train.targets.numpy()

x_test = mnist_test.data.numpy()
y_test = mnist_test.targets.numpy()

x_test, x_val, y_test, y_val = tts(
    x_test, y_test, test_size=.5, random_state=71
)

train_ds_gen: DatasetGenerator = DatasetGenerator(
    images = x_train,
    labels = y_train,
    subset_ratio = .2,
)

base_images_train, base_labels_train = train_ds_gen.build_base_dataset()
aux_images_train, aux_labels_train = train_ds_gen.build_aux_dataset()

print(f"Train Dataset Base Size: {len(base_images_train)}")
print(f"Train Dataset Aux Size: {len(aux_images_train)}")

test_ds_gen: DatasetGenerator = DatasetGenerator(
    images = x_test,
    labels = y_test,
    subset_ratio = .95,
)

base_images_test, base_labels_test = test_ds_gen.build_base_dataset()
aux_images_test, aux_labels_test = test_ds_gen.build_aux_dataset()

print(f"Test Dataset Base Size: {len(base_images_test)}")
print(f"Test Dataset Aux Size: {len(aux_images_test)}")

val_ds_gen: DatasetGenerator = DatasetGenerator(
    images = x_val,
    labels = y_val,
    subset_ratio = .95,
)

base_images_val, base_labels_val = val_ds_gen.build_base_dataset()
aux_images_val, aux_labels_val = val_ds_gen.build_aux_dataset()

print(f"Validation Dataset Base Size: {len(base_images_val)}")
print(f"Validation Dataset Aux Size: {len(aux_images_val)}")

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
    batch_size=8,
    shuffle=True
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

if not os.path.exists("models/base_model_plain.pt"):
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
            torch.save(model.state_dict(), "models/base_model_plain.pt")

        if base_early_stopper(val_loss):
            print("Stopped")
            break

    print("\n")

### Base + aux model

if not os.path.exists("models/aux_model_plain+skips.pt"):
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
            torch.save(model.state_dict(), "models/aux_model_plain+skips.pt")

        if base_early_stopper(val_loss):
            print("Stopped")
            break

    print("\n")

### Comparison of models
optimizer = None # Set to None so it can't train

base_model = TinyCNN()
base_model.load_state_dict(torch.load("models/base_model_plain.pt", weights_only=True))
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
aux_model.load_state_dict(torch.load("models/aux_model_plain+skips.pt", weights_only=True))
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

    model.to(DEVICE)
    proj_head.to(DEVICE)

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
            torch.save(model.state_dict(), f"models/contrast_body_plain+skips_{round(temp, 2)}.pt")
            torch.save(proj_head.state_dict(), f"models/contrast_proj_plain+skips_{round(temp, 2)}.pt")

        if contrast_early_stopper(val_loss):
            print("\n")
            print("Best Val Loss:", best_val_loss[i])
            break

        if (epoch+1 == num_contrast_epochs):
            print("\n")
            print("Best Val Loss:", best_val_loss[i])

best_temp = round(temp_range[np.argmin(best_val_loss)], 2)
print(f"Best temp: {best_temp}")

# train contrastive learning classifier

num_class_epochs = 20

contrast_body = TinyCNN_Headless()
contrast_body.load_state_dict(torch.load(f"models/contrast_body_plain+skips_{best_temp}.pt", weights_only=True))

class_head = torch.nn.Sequential(
    torch.nn.Linear(32,32),
    torch.nn.Linear(32,10)
)

wrapped_model = WrapperModelTrainHead(
    body = contrast_body,
    head = class_head
)
wrapped_model.to(DEVICE)
optimizer = optim.Adam(
    wrapped_model.head.parameters(),
    lr = 0.001,
    weight_decay = 1e-5
)

contrast_early_stopper = EarlyStopper(
    patience=5,
    min_delta=0
)

contrast_best = {
    "val_loss": 1000,
    "val_acc": 0
}

for epoch in range(num_class_epochs):
    train_loss, train_acc = classification_run(
        model=wrapped_model,
        optimizer=optimizer,
        dataloader=train_loader,
        mode="base",
        device=DEVICE,
    )

    val_loss, val_acc = classification_run(
        model=wrapped_model,
        optimizer=optimizer,
        dataloader=val_loader,
        device=DEVICE,
        mode="base",
        train=False
    )

    print(f"Epoch {epoch+1}:", round(train_loss, 4), round(train_acc*100, 2), round(val_loss, 4), round(val_acc*100, 2))

    if val_loss < contrast_best["val_loss"]:
        contrast_best["val_loss"] = val_loss
        contrast_best["val_acc"] = val_acc
        torch.save(wrapped_model.state_dict(), "models/contrast_class_plain+skips.pt")

    if contrast_early_stopper(val_loss):
        print("\n")
        print("Best Val Loss:", contrast_best["val_loss"])
        print("Best Val Acc:", round(contrast_best["val_acc"]*100, 2))
        break

    if (epoch+1 == num_class_epochs):
        print("\n")
        print("Best Val Loss:", contrast_best["val_loss"])
        print("Best Val Acc:", round(contrast_best["val_acc"]*100, 2))