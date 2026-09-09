# Project 4 Corrections

Use this alongside `260328 - project4_computer_vision.docx.pdf`.

This file gives setup fixes and safety notes. It does not provide completed CNN implementations.

## Folder Layout

Use the TARQ folder:

```bash
cd /Users/ompatnaik/Documents/Code/tarq/project4
```

Recommended layout:

```text
project4/
├── conv_scratch.py
├── cnn_mnist.py
├── cifar10.py
├── transfer_learning.py
├── yolo_demo.py
├── webcam_detect.py
├── data/
├── models/
└── outputs/
```

Save trained weights under `models/` and figures under `outputs/`.

## Device Selection

The PDF uses:

```python
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
```

For portability, use:

```python
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
```

If an operation fails on MPS, force:

```python
DEVICE = "cpu"
```

Correctness first, speed second.

## DataLoader Workers On macOS

The PDF uses `num_workers=2` for CIFAR-10. On macOS, multiprocessing can sometimes cause hangs or noisy errors inside scripts.

If that happens, use:

```python
num_workers=0
```

This is slower but simpler while learning.

## Reshape Instead Of View

The PDF uses `.view(...)` in model forward passes. It is usually fine here, but prefer:

```python
x = x.reshape(x.size(0), -1)
```

Why: `.reshape()` is safer if the tensor is non-contiguous.

## Saving Outputs

Prefer:

```python
from pathlib import Path

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
```

Then save:

```python
torch.save(model.state_dict(), MODEL_DIR / "mnist_cnn.pth")
plt.savefig(OUTPUT_DIR / "conv1_filters.png", dpi=150)
```

This keeps the project root clean.

## Loading PyTorch Weights

When loading your own saved `state_dict`, use:

```python
state = torch.load(MODEL_DIR / "mnist_cnn.pth", map_location=DEVICE)
model.load_state_dict(state)
```

If a future PyTorch version warns about `weights_only`, read the warning. For files you created yourself, the normal `state_dict` loading pattern is acceptable.

## MNIST Target

After 5 epochs, expect roughly 99% test accuracy. If accuracy is much lower:

- check `model.train()` before training
- check `model.eval()` before evaluation
- check labels and images both move to `DEVICE`
- check `optimizer.zero_grad()` before `loss.backward()`
- check the flattened shape before `fc1`

## CIFAR-10 Expectations

CIFAR-10 is much harder than MNIST. Do not expect 99%.

At 20 epochs, around 80-85% is plausible depending on hardware, transforms, and exact implementation. More epochs may improve it.

## Transfer Learning Dataset

If collecting your own dataset, prioritize variation:

- different backgrounds
- different lighting
- different object angles
- different distances

Avoid collecting one class in one background and another class in a different background. That creates a background shortcut.

## ResNet Weights Download

This line downloads pretrained ImageNet weights on first run:

```python
models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
```

If the download fails, rerun with working internet or manually cache the weights. Do not replace transfer learning with random initialization unless you are intentionally testing the difference.

## YOLO Downloads

`YOLO("yolov8n.pt")` downloads weights on first run. Keep the `.pt` file local after download.

If the download fails, manually download the weight file and place it in the project root or `models/`.

## Webcam On macOS

If OpenCV cannot open the webcam:

1. Open System Settings.
2. Go to Privacy and Security > Camera.
3. Enable camera access for Terminal, iTerm, VS Code, or the app launching Python.
4. Restart the terminal session.

## What To Write Up

Project 4 is complete only after a short writeup explaining:

- why convolutions beat fully connected layers for images
- what filters and feature maps showed in MNIST
- why CIFAR-10 is harder than MNIST
- what data augmentation changed
- what transfer learning did in Phase 1 vs Phase 2
- what YOLO adds beyond classification
- how this connects to robotics perception

