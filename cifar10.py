"""Part 2 placeholder.

Build the deeper CIFAR-10 CNN here, including batch norm, augmentation,
training curves, and validation accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


torch.manual_seed(42)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# -----------------------------
# Data transforms
# -----------------------------

# Training transform:
# Random changes teach the model to handle small shifts, flips, and colour changes.
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
])

# Validation/test transform:
# No randomness here. Evaluation should be stable.
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
])


# -----------------------------
# Dataset and loaders
# -----------------------------

train_set = torchvision.datasets.CIFAR10(
    "./data",
    train=True,
    download=True,
    transform=train_tf
)

test_set = torchvision.datasets.CIFAR10(
    "./data",
    train=False,
    download=True,
    transform=val_tf
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

CLASSES = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# -----------------------------
# VGG-style CNN
# -----------------------------

class VGGNet(nn.Module):
    """
    Three convolution blocks.

    Each block:
    Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout

    Spatial size:
    32x32 -> 16x16 -> 8x8 -> 4x4
    """

    def __init__(self):
        super().__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(2),
                nn.Dropout2d(0.1),
            )

        self.features = nn.Sequential(
            block(3, 64),
            block(64, 128),
            block(128, 256),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)           # -> (batch, 256, 4, 4)
        x = x.view(x.size(0), -1)      # -> (batch, 256*4*4)
        x = self.classifier(x)         # -> (batch, 10)
        return x


model = VGGNet().to(DEVICE)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[12, 17],
    gamma=0.1
)

criterion = nn.CrossEntropyLoss()

EPOCHS = 20

train_losses = []
val_losses = []
val_accs = []


# -----------------------------
# Training loop
# -----------------------------

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        running_loss += loss.item()

    scheduler.step()


    # -----------------------------
    # Validation each epoch
    # -----------------------------

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            val_loss += criterion(outputs, labels).item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)
    val_acc = correct / total

    train_losses.append(train_loss)
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch + 1:3d}/{EPOCHS} "
        f"train_loss={train_loss:.4f} "
        f"val_loss={avg_val_loss:.4f} "
        f"val_acc={val_acc:.4f}"
    )


# -----------------------------
# Save model
# -----------------------------

torch.save(model.state_dict(), "cifar10_vgg.pth")
print("\nSaved cifar10_vgg.pth")


# -----------------------------
# Plot curves
# -----------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, label="train loss")
axes[0].plot(val_losses, label="val loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("CIFAR-10 Loss Curves")
axes[0].legend()

axes[1].plot(val_accs)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation Accuracy")
axes[1].set_title("CIFAR-10 Validation Accuracy")

plt.tight_layout()
plt.savefig("cifar10_curves.png", dpi=150)
plt.close()

print("Saved cifar10_curves.png")
