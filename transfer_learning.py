"""Part 3 placeholder.

Fine-tune ResNet-18 here after preparing a 4-class image dataset under
data/train/<class_name>/ and data/val/<class_name>/.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt


torch.manual_seed(42)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

DATA_DIR = Path("./data")


# -----------------------------
# ImageNet transforms
# -----------------------------
# ResNet-18 was pretrained on ImageNet, so we use ImageNet normalization.
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# -----------------------------
# Load custom dataset
# -----------------------------
train_ds = torchvision.datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
val_ds = torchvision.datasets.ImageFolder(DATA_DIR / "val", transform=val_tf)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

class_names = train_ds.classes
num_classes = len(class_names)

print(f"Classes: {class_names}")
print(f"Train images: {len(train_ds)}")
print(f"Validation images: {len(val_ds)}")

with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)


# -----------------------------
# Helper functions
# -----------------------------
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    return avg_loss


# -----------------------------
# Load pretrained ResNet-18
# -----------------------------
weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)

# Replace final classification layer.
# Original ResNet-18 head predicts 1000 ImageNet classes.
# Our new head predicts 4 custom classes.
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()


# -----------------------------
# Phase 1: train head only
# -----------------------------
print("\nPhase 1: training classification head only")

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

phase1_train_losses = []
phase1_val_losses = []
phase1_val_accs = []

PHASE1_EPOCHS = 10

for epoch in range(PHASE1_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    phase1_train_losses.append(train_loss)
    phase1_val_losses.append(val_loss)
    phase1_val_accs.append(val_acc)

    print(
        f"Phase 1 Epoch {epoch + 1:2d}/{PHASE1_EPOCHS} "
        f"train_loss={train_loss:.4f} "
        f"val_loss={val_loss:.4f} "
        f"val_acc={val_acc:.4f}"
    )


# -----------------------------
# Phase 2: fine-tune all layers
# -----------------------------
print("\nPhase 2: fine-tuning all layers")

for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-4,
    momentum=0.9,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10,
)

phase2_train_losses = []
phase2_val_losses = []
phase2_val_accs = []

PHASE2_EPOCHS = 10
best_acc = 0.0

for epoch in range(PHASE2_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    scheduler.step()

    phase2_train_losses.append(train_loss)
    phase2_val_losses.append(val_loss)
    phase2_val_accs.append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "resnet18_finetuned.pth")

    print(
        f"Phase 2 Epoch {epoch + 1:2d}/{PHASE2_EPOCHS} "
        f"train_loss={train_loss:.4f} "
        f"val_loss={val_loss:.4f} "
        f"val_acc={val_acc:.4f} "
        f"best={best_acc:.4f}"
    )

print(f"\nBest validation accuracy: {best_acc:.4f}")
print("Saved resnet18_finetuned.pth")
print("Saved class_names.json")


# -----------------------------
# Plot transfer learning curves
# -----------------------------
all_train_losses = phase1_train_losses + phase2_train_losses
all_val_losses = phase1_val_losses + phase2_val_losses
all_val_accs = phase1_val_accs + phase2_val_accs

plt.figure(figsize=(8, 5))
plt.plot(all_train_losses, label="train loss")
plt.plot(all_val_losses, label="val loss")
plt.axvline(PHASE1_EPOCHS - 1, linestyle="--", label="start fine-tuning")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transfer Learning Loss Curves")
plt.legend()
plt.tight_layout()
plt.savefig("resnet18_transfer_loss.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(all_val_accs)
plt.axvline(PHASE1_EPOCHS - 1, linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Transfer Learning Validation Accuracy")
plt.tight_layout()
plt.savefig("resnet18_transfer_accuracy.png", dpi=150)
plt.close()

print("Saved resnet18_transfer_loss.png")
print("Saved resnet18_transfer_accuracy.png")
