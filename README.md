# Urban ReID - Reproduction Guide

This repository contains the implementation of a person Re-ID system for urban environments using Vision Transformers with DINOv3 backbone and multi-layer feature extraction.

## Installation

### Prerequisites
- Linux/macOS (tested on Ubuntu)
- Python 3.10+
- CUDA 12.4 (or compatible GPU support)

### Environment Setup

We use `uv` for fast Python package management. Install the environment:

```bash
# Create and activate virtual environment with Python 3.10
uv venv urban-reid --python 3.10.0
source urban-reid/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Dataset Setup
- First download the Urban-reid dataset from the kaggle compition
    - [Challenge Kaggle Competition](https://www.kaggle.com/competitions/urban-elements-re-id-challenge-2026/data)

- Second download the uinfied version of the campus dataset from the official competition website
    - [Challenge Website](http://www-vpu.eps.uam.es/challenges/UrbanReIDChallenge2026/)

Then run 
```
python tools/merge_campus_dataset.py --main ./Urban2026 --secondary ./UAM_Unified --output ./Combined_dataset 
```

To Create the last version where the val set is combined into the trainig too run
```
python tools/merge_val.py 
```

The training uses the Urban ReID dataset. Ensure the dataset directory structure matches the configuration:

```
./Combined_dataset
├── train.csv              # Training split metadata
├── train_classes.csv      # Training class labels
├── val_query.csv          # Validation query set
├── val_test.csv           # Validation gallery set
├── val_query_classes.csv
├── val_test_classes.csv
├── query.csv              # Test query set
├── query_classes.csv
├── test.csv               # Test gallery set
└── test_classes.csv
    # Plus corresponding image files
```

Update the `dataset.root` path in the experiment config if your dataset is in a different location.

## Training

### Training the Model

To reproduce the results, train using the provided experiment configuration:

```bash
python tools/train.py --config configs/expirements/vit_large_dinov3_multilayer.yaml
```

### Monitoring Training

View training progress in TensorBoard:

```bash
tensorboard --logdir=outputs/
```

## Submission Generation

Once training completes, generate the competition submission CSV using the checkpoint at epoch 80:

```bash
python tools/submit.py \
  --config outputs/vit_large_dinov3_multilayer/config.yaml \
  test.weight=outputs/vit_large_dinov3_multilayer/checkpoint_ep80.pth
```
Or you can use the Checkpoint we provide at the following link:
- [Model Weights](https://drive.google.com/drive/my-drive)