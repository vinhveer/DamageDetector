# Object Detection Datasets

Shared dataset manifests and helper logic for YOLO and StableDINO.

Suggested layout:

```text
object_detection/datasets/
  crack_dataset.yaml
  crack_dataset/
    train/images/
    train/labels/
    val/images/
    val/labels/
```

Example manifest:

```yaml
path: object_detection/datasets/crack_dataset
train: train/images
val: val/images
names:
  0: crack

Quick start (with the included example):

```bash
python -m object_detection.yolo train --data object_detection/datasets/example_crack.yaml
python -m object_detection.stable_dino.train --dataset object_detection/datasets/example_crack.yaml
```
```
