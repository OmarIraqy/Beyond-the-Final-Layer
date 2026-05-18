# Beyond the Final Layer: Learned Transformer Feature Fusion for ReID

This is the official implementation of Beyond the Final Layer entry to the Urban-ReID 2026 Grand Challenge ICIP2026 Achieving first place on the Public test set.

## Installation

### Environment Setup

We use `uv` for fast Python package management. Install the environment:

```bash
# Clone repo 
git clone https://github.com/OmarIraqy/Beyond-the-Final-Layer
cd Beyond-the-Final-Layer

# Create and activate virtual environment with Python 3.10
uv venv urban-reid --python 3.10.0
source urban-reid/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Dataset Setup
- Download the Urban-reid dataset from the kaggle compition
    - [Challenge Kaggle Competition](https://www.kaggle.com/competitions/urban-elements-re-id-challenge-2026/data)

- Download the uinfied version of the campus dataset from the official competition website
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
  --config configs/expirements/vit_large_dinov3_multilayer.yaml \
  test.weight=outputs/vit_large_dinov3_multilayer/checkpoint_ep80.pth
```
Or you can use the Checkpoint we provide at the following link:
- [Model Weights](https://drive.google.com/file/d/1XoH1TrwkfMLlP69uKyxFb2AEcKhoog7E/view?usp=sharing)
```bash
python tools/submit.py \
  --config configs/expirements/vit_large_dinov3_multilayer.yaml \
  test.weight=./model_checkpoint.pth
```
