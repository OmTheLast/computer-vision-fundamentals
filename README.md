# TARQ Project 4 - Computer Vision Fundamentals

This project introduces computer vision for the TARQ sequence. The goal is to understand how models process images before using vision in robotics projects.

## Project Parts

1. MNIST CNN from scratch
2. CIFAR-10 deeper CNN with batch norm and data augmentation
3. ResNet-18 transfer learning on a custom 4-class image dataset
4. YOLO object detection on a static image and webcam
5. Robotics connection: sense, perceive, represent, plan, act, observe

## Setup

```bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Suggested File Order

Work through these in order:

```text
conv_scratch.py       # small convolution exercise from the PDF
cnn_mnist.py          # MNIST CNN from scratch
cifar10.py            # deeper CIFAR-10 CNN
transfer_learning.py  # ResNet-18 fine-tuning
yolo_demo.py          # YOLO on static image
webcam_detect.py      # YOLO webcam detection
```

The Python files are intentionally left as placeholders. Type the actual model code yourself from the PDF and `CORRECTIONS.md`.

## Expected Outputs

```text
models/mnist_cnn.pth
outputs/conv1_filters.png
outputs/conv1_features.png
models/cifar10_vgg.pth
outputs/cifar10_curves.png
models/resnet18_finetuned.pth
runs/detect/predict/
```

## Learning Rule

For Part 1 and Part 2, write the CNNs yourself. The point is to understand convolutions, feature maps, pooling, batch norm, training/eval mode, and data augmentation.

