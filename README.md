# MRI-Only Workflow

A PyTorch-based pipeline for generating **synthetic CT (sCT)** from MRI using a **3D U-Net**, designed for preclinical MRI-only radiotherapy workflows. Includes training, inference, DICOM export (sCT, GT, MR), explainable AI (XAI) visualization, and baseline evaluation tools.

## Features

- **3D U-Net** with configurable depth, InstanceNorm, LeakyReLU, and residual connections
- **Reconstruction loss** (weighted sum of L1 and 3D SSIM)
- **Mixed-precision training** (AMP) with EMA and cosine annealing
- **Explainable AI**: gradient saliency, SmoothGrad, Grad-CAM, occlusion sensitivity
- **End-to-end pipeline**: NIfTI inference, DICOM export (HFS/RayStation) for sCT, GT, and MR
- **Evaluation**: bone Dice scores with rigid alignment (SimpleITK), publication-quality figures
- **YAML-based configuration** with typed dataclasses

## Project Structure

```
mri-only-workflow/
├── configs/
│   └── default.yaml              # Training/model/data configuration
├── simple_sct/                   # Core library
│   ├── config.py                 # Typed YAML config with dataclasses
│   ├── data/
│   │   └── volume_dataset.py     # NIfTI volume pair dataset
│   ├── losses/
│   │   └── basic.py              # L1, SSIM3D, ReconstructionLoss
│   ├── metrics/
│   │   └── metrics.py            # MAE, RMSE, PSNR, SSIM
│   ├── models/
│   │   └── simple_unet3d.py      # 3D U-Net architecture
│   ├── utils/
│   │   ├── logging.py            # TensorBoard helpers, directory setup
│   │   └── seed.py               # Reproducibility utilities
│   └── xai/
│       └── attribution.py        # Saliency, SmoothGrad, Grad-CAM, occlusion
├── train.py                      # Model training script
├── infer.py                      # Single-volume inference
├── pipeline.py                   # Full MRI -> sCT -> DICOM (sCT, GT, MR)
├── xai_figures.py                # Generate publication XAI figures
├── evaluate_alignment.py         # Rigid alignment + bone Dice evaluation
├── average_ct_baseline.py        # Average CT baseline generation + Dice
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/<your-username>/mri-only-workflow.git
cd mri-only-workflow
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch (with CUDA recommended)
- See `requirements.txt` for full dependency list

## Data Preparation

Organize your NIfTI volumes into MRI and CT directories:

```
data/
├── MR/
│   ├── subject001_MR_T1.nii.gz
│   ├── subject002_MR_T1.nii.gz
│   └── ...
└── CT/
    ├── subject001_CT_T1.nii.gz
    ├── subject002_CT_T1.nii.gz
    └── ...
```

All volumes must be co-registered and resampled to the same grid (configured via `enforce_shape` in the YAML config). Update `data.mri_dir` and `data.ct_dir` in `configs/default.yaml` to point to your data directories.

## Usage

### Training

```bash
python train.py --config configs/default.yaml
```

Training supports mixed-precision (AMP), EMA, gradient clipping, cosine annealing LR, and live XAI logging to TensorBoard. Checkpoints are saved to the directory specified in the config.

Monitor training:
```bash
tensorboard --logdir runs/
```

### Single-Volume Inference

```bash
python infer.py \
    --config configs/default.yaml \
    --ckpt checkpoints/best.pt \
    --mri data/MR/subject001_MR_T1.nii.gz \
    --out output/subject001_sCT.nii.gz
```

### Full Pipeline (Inference + DICOM Export)

```bash
python pipeline.py \
    --config configs/default.yaml \
    --ckpt checkpoints/best.pt \
    --mr data/MR/subject001_MR_T1.nii.gz \
    --gt data/CT/subject001_CT_T1.nii.gz \
    --ref reference.dcm \
    --out output/workflow_results
```

### XAI Figures

Generate publication-quality saliency, SmoothGrad, and Grad-CAM figures:

```bash
python xai_figures.py \
    --config configs/default.yaml \
    --ckpt checkpoints/best.pt \
    --out_dir output/xai_figures
```

### Average CT Baseline

Compute an average CT from the training set as a naive baseline:

```bash
# From a directory of co-registered CTs
python average_ct_baseline.py \
    --ct-dir data/CT \
    --out output/average_ct \
    --gt-dir data/validation/CT \
    --gt-subjects subject001_CT_T1 subject002_CT_T1

# With rigid alignment before averaging
python average_ct_baseline.py \
    --ct-dir data/CT \
    --out output/average_ct_aligned \
    --align

# From YAML config (uses training split)
python average_ct_baseline.py \
    --config configs/default.yaml \
    --out output/average_ct
```

### Rigid Alignment + Dice Evaluation

Evaluate bone Dice scores between the average CT baseline and ground truth CTs with rigid registration:

```bash
python evaluate_alignment.py \
    --avg-ct output/average_ct/average_ct.nii \
    --gt-dir data/validation/CT \
    --gt-subjects subject001_CT_T1 subject002_CT_T1 \
    --out output/aligned_dice_results \
    --save-aligned
```

## Configuration

All training, model, data, loss, and logging parameters are controlled via YAML. See `configs/default.yaml` for the full specification. Key sections:

| Section | Description |
|---------|-------------|
| `data` | Input/output directories, volume shape, scaling mode |
| `train` | Epochs, batch size, LR, AMP, EMA, XAI settings |
| `model` | U-Net architecture (channels, depth, activation) |
| `loss` | L1 and SSIM weights and parameters |
| `metrics` | Which metrics to compute (PSNR, SSIM) |
| `logging` | TensorBoard, checkpointing, image logging frequency |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
