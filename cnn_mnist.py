import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# -----------------------------
# Part A: Toy convolution example
# -----------------------------

image_2d = torch.tensor([
    [0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 0.],
    [0., 1., 1., 1., 0.],
    [0., 1., 1., 1., 0.],
    [0., 0., 0., 0., 0.]
])

image = image_2d.unsqueeze(0).unsqueeze(0)

h_edge_2d = torch.tensor([
    [1.,  2.,  1.],
    [0.,  0.,  0.],
    [-1., -2., -1.]
])

h_edge = h_edge_2d.unsqueeze(0).unsqueeze(0)

print("Image shape:", image.shape)
print("Filter shape:", h_edge.shape)

result = F.conv2d(image, h_edge, padding=1)

print("\nInput:")
print(image_2d.numpy())

print("\nHorizontal edge detector:")
print(h_edge_2d.numpy())

print("\nOutput feature map:")
print(result.squeeze().numpy())


# -----------------------------
# Part B: CNN on MNIST
# -----------------------------

torch.manual_seed(42)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\nUsing device: {DEVICE}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = torchvision.datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.MNIST(
    "./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=64,
    shuffle=False
)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Input: (batch, 1, 28, 28)

        x = self.conv1(x)        # -> (batch, 32, 28, 28)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)         # -> (batch, 32, 14, 14)

        x = self.conv2(x)        # -> (batch, 64, 14, 14)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)         # -> (batch, 64, 7, 7)

        x = self.drop(x)

        x = x.view(x.size(0), -1)  # -> (batch, 64*7*7)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)            # -> (batch, 10)

        return x


model = SimpleCNN().to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS = 5

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

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"loss={avg_loss:.4f} "
        f"accuracy={accuracy:.4f}"
    )

torch.save(model.state_dict(), "mnist_cnn.pth")
print("\nSaved mnist_cnn.pth")


# -----------------------------
# Part C: Visualize filters and feature maps
# -----------------------------

model.load_state_dict(torch.load("mnist_cnn.pth", map_location=DEVICE))
model.eval()

# Visualize 32 learned filters from conv1
filters = model.conv1.weight.detach().cpu()

fig, axes = plt.subplots(4, 8, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(filters[i, 0], cmap="gray", vmin=-1, vmax=1)
    ax.axis("off")

fig.suptitle("Conv1 learned filters: 32 filters of size 3x3", fontsize=13)
plt.tight_layout()
plt.savefig("conv1_filters.png", dpi=150)
plt.close()


# Visualize feature maps for one test image
sample_images, sample_labels = next(iter(test_loader))

sample = sample_images[:1].to(DEVICE)
sample_label = sample_labels[0].item()

with torch.no_grad():
    feat = model.conv1(sample)
    feat = model.bn1(feat)
    feat = F.relu(feat)
    feat = feat[0].cpu()

fig, axes = plt.subplots(4, 8, figsize=(14, 7))

for i, ax in enumerate(axes.flat):
    ax.imshow(feat[i], cmap="viridis")
    ax.axis("off")

fig.suptitle(f"Conv1 feature maps for one MNIST input, label={sample_label}", fontsize=13)
plt.tight_layout()
plt.savefig("conv1_features.png", dpi=150)
plt.close()

print("Saved conv1_filters.png and conv1_features.png")