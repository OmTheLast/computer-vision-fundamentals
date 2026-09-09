import json
import sys

from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_PATH = "resnet18_finetuned.pth"
CLASS_NAMES_PATH = "class_names.json"


# -----------------------------
# Load class names
# -----------------------------

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)


# -----------------------------
# Recreate ResNet-18 architecture
# -----------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# -----------------------------
# Validation / inference transform
# -----------------------------

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# -----------------------------
# Predict one image
# -----------------------------

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    pred_idx = probabilities.argmax().item()

    print(f"\nImage: {image_path}")
    print(f"Prediction: {class_names[pred_idx]}")
    print(f"Confidence: {probabilities[pred_idx].item():.4f}")

    print("\nProbabilities:")

    for i, probability in enumerate(probabilities):
        print(
            f"{class_names[i]:12s}: "
            f"{probability.item():.4f}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python predict_custom.py "
            "<image_path>"
        )
        sys.exit(1)

    predict_image(sys.argv[1])