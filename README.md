<h1 align="center">Computer Vision Fundamentals</h1>

**Work through convolution, image classification, transfer learning, and object detection with inspectable Python exercises.**

Python + PyTorch/torchvision. Dataset and YOLO weights download separately. Some exercises are incomplete; no unified application.

By Om Patnaik · repository created September 2026.

## Install

| Platform / path | Start here |
| --- | --- |
| Python | `pip install -r requirements.txt` in a virtual environment |

Read `CORRECTIONS.md`, then inspect `cnn_mnist.py`, `cifar10.py`, and `transfer_learning.py`: these contain implemented training paths. Review saved figures without starting training. `conv_scratch.py`, `yolo_demo.py`, and `webcam_detect.py` are still placeholders.

```bash
python -m compileall -q .  # inspect syntax without starting training
```

## Why

You need to understand how images become features before connecting vision to a robotics loop. This TARQ sequence covers convolution, MNIST/CIFAR-10 CNNs, ResNet-18 transfer, and YOLO detection. Completion varies by file; the earlier blanket claim that every Python file was a placeholder is stale.

## Inspect a convolution

Begin with the toy kernel and tensor dimensions, then follow the MNIST network. Read each later script before running it to establish what is implemented and which local inputs it expects.

## Connect predictions to observations

The YOLO and webcam exercises are unfinished scaffolds, so they do not yet produce detections. Saved filter, feature-map, and training plots document the classification work; they are not an independently reproduced benchmark.

## Notes from building it

I used separate exercises to connect convolution mechanics with classification and transfer learning. The implemented training files now go beyond the original scaffold description, while detection remains unfinished. I would finish and evaluate each exercise explicitly before claiming that the whole sequence is complete.

## Commands

```bash
python cnn_mnist.py              # convolution example and MNIST training
python cifar10.py                # inspect this exercise before running
python transfer_learning.py      # inspect data paths before running
python predict_custom.py         # inspect checkpoint/input settings before inference
python -m compileall -q .        # check syntax without training
```

## Verify it yourself

```bash
python -m compileall -q .
```

Compilation checks syntax only; no automated model-accuracy suite is present. Review each exercise, dataset split, and saved plot independently. Hardware, webcam, and training checks were not run during this README edit.

## Credits, license and pointers

Built with PyTorch, torchvision, Matplotlib, and the detection packages in `requirements.txt`; exercises build on TARQ course material and MNIST/CIFAR-10/ResNet/YOLO work. No source license file is present.

[Full setup, original notes, and project history](README-DETAILS.md).
