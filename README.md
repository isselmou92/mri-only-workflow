# MRI-Only Workflow

Synthetic CT (sCT) generation from ultra-high-field mouse brain MRI with a 3D U-Net, for
MRI-only preclinical treatment planning (X-rays, protons and carbon ions). The repository
contains the complete code for training, cross-validation, ensemble inference on the
held-out test cohort, evaluation against micro-CT and a population-average CT baseline,
and DICOM export for treatment planning (RayStation).

Two model variants are provided:

| Variant | Config | MR scaling | Augmentation |
|---------|--------|------------|--------------|
| Published model | `configs/default.yaml` | z-score | translation, left/right flip |
| Intensity-augmented model (recommended) | `configs/aug_noN4.yaml` | foreground percentiles | as above + MR-specific intensity augmentation |

`configs/aug_full.yaml` is the N4 ablation of the augmented model (trained and tested on
N4 bias-corrected MRIs); it did not improve the results.

## Method in brief

- **Network**: 3D U-Net (32 base filters, depth 3, 256-channel bottleneck, residual
  double-convolution blocks, InstanceNorm, LeakyReLU, `tanh` output).
- **Input / target**: co-registered MR and micro-CT volumes of 128 x 128 x 112 voxels
  (0.14 mm isotropic). CT is mapped linearly from [-1000, 3000] HU to [-1, 1];
  predictions are converted back to HU and clamped to this range.
- **Training**: SSIM loss (L1 used as comparison), AdamW, cosine learning rate, AMP, EMA,
  500 epochs, subject-grouped 3-fold cross-validation.
- **Intensity augmentation** (`simple_sct/data/intensity.py`, MR only): smooth multiplicative
  bias fields, gamma and random monotonic intensity curves, blur / sharpening and Gaussian
  noise, applied on top of translation and flipping.
- **Test cohort**: K = 3 ensemble (one model per fold), predictions averaged in HU.
  Intra-strain (BALB/c) and inter-strain (C57BL/6) test animals, three acquisitions each.
- **Baseline**: population-average CT (sCT^avg) of the training CTs, rigidly registered
  to each test subject.

## Repository structure

```
mri-only-workflow/
├── configs/
│   ├── default.yaml              # published model
│   ├── aug_noN4.yaml             # intensity-augmented model
│   ├── aug_full.yaml             # augmented model with N4 bias correction (ablation)
│   └── test_strain_map.json      # mouse ID -> strain for the test cohort
├── simple_sct/                   # core library
│   ├── config.py                 # typed YAML config (dataclasses)
│   ├── data/volume_dataset.py    # NIfTI pair dataset, folds, spatial augmentation
│   ├── data/intensity.py         # MR normalisation, N4, intensity augmentation
│   ├── losses/basic.py           # L1, 3D SSIM, reconstruction loss
│   ├── metrics/metrics.py        # MAE, RMSE, PSNR, SSIM
│   ├── metrics/fov.py            # micro-CT field-of-view mask
│   ├── models/simple_unet3d.py   # 3D U-Net
│   ├── utils/                    # logging, seeding
│   └── xai/attribution.py        # saliency, SmoothGrad, Grad-CAM, occlusion
├── train.py                      # train one model (one fold)
├── infer.py                      # single-volume inference
├── pipeline.py                   # MR -> sCT -> DICOM (sCT, GT, MR)
├── average_ct_baseline.py        # average CT baseline
├── evaluate_alignment.py         # rigid registration + bone Dice
├── postprocess_sct.py            # optional sCT post-processing
├── xai_figures.py                # XAI figures
└── scripts/                      # sweeps, evaluation, analysis and figures
```

## Installation

```bash
git clone https://github.com/isselmou92/mri-only-workflow.git
cd mri-only-workflow
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ and a CUDA GPU are recommended (one training run uses about 9 GB of VRAM at
batch size 2). The code was run with Python 3.12, PyTorch 2.5 and SimpleITK 2.5.

## Data

The imaging data are not part of this repository. The expected layout is

```
dataset/
├── MR/          MZEI00018_MR_T1.nii.gz ...     training / cross-validation MRIs
├── CT/          MZEI00018_CT_T1.nii.gz ...     co-registered micro-CTs
└── Test_Data/
    ├── MR/      MZEI00027_MR_T1.nii.gz ...     held-out test MRIs
    └── CT/      MZEI00027_CT_T1.nii.gz ...
```

MR and CT are paired by replacing `MR` with `CT` in the file name; the leading token is the
mouse ID used for subject-grouped folds. All volumes must be co-registered and resampled to
the same grid. Paths in the configs are relative to the YAML file. Mouse IDs of the test
cohort are listed in `exclude_mouse_ids` so they never enter training.

The micro-CTs were cropped and rotated during alignment to the MR; voxels outside the
rotated crop box carry a constant fill value of -990 HU. `simple_sct/metrics/fov.py`
derives the CT field of view from this fill, and `--fov-mask` / `scripts/evaluate_test_fov.py`
restrict the evaluation to it.

## Training

Cross-validation sweep (one run per loss and fold, distributed over the available GPUs):

```bash
# published model: L1 and SSIM, folds 0-2
python scripts/run_sweep.py --base-config configs/default.yaml --out output/loss_sweep_final_22

# intensity-augmented model (SSIM)
python scripts/run_sweep.py --base-config configs/aug_noN4.yaml --losses ssim --out output/aug_noN4
```

A single run can also be started directly with `python train.py --config <config.yaml>`
(the fold is set by `data.fold`). For the N4 ablation, first write bias-corrected MRIs with
`python scripts/precompute_n4.py dataset/MR dataset/Test_Data/MR`.

## Evaluation

```bash
SWEEP=output/aug_noN4

# cross-validation metrics in HU per volume (add --fov-mask for the CT field of view)
python scripts/evaluate_sweep.py --sweep-root $SWEEP --use best

# loss comparison with bootstrap confidence intervals (L1 vs SSIM)
python scripts/compare_losses.py --per-volume-csv output/loss_sweep_final_22/eval_per_volume_best.csv

# K = 3 ensemble on the held-out test cohort, writes sCT NIfTIs and per-volume metrics
python scripts/ensemble_test.py --sweep-root $SWEEP --loss ssim \
    --test-mri-dir dataset/Test_Data/MR --out-dir $SWEEP/ensemble_test_ssim --save-volumes

# bone Dice of sCT and rigidly aligned average CT on the test cohort
python scripts/test_dice_table.py --sweep-root $SWEEP --avg-ct output/average_CT/average_ct.nii.gz

# bone surface distances (MSD, HD95) per strain
python scripts/strain_mm_analysis.py \
    --baseline-sct-dir output/loss_sweep_final_22/ensemble_test_ssim/sct --aug-sct-dir $SWEEP/ensemble_test_ssim/sct

# metrics over the whole volume and inside the CT field of view
python scripts/evaluate_sweep.py --sweep-root $SWEEP --use best --fov-mask --out-csv $SWEEP/eval_per_volume_best_fov.csv
python scripts/evaluate_test_fov.py --published-sct-dir output/loss_sweep_final_22/ensemble_test_ssim/sct \
    --aug-sct-dir $SWEEP/ensemble_test_ssim/sct --out output/fov_eval
python scripts/plot_fov_summary.py   # reads output/fov_eval/cv_{final_22,aug_noN4}_{whole,fov}.csv
```

The analysis scripts expect the ensemble sCTs of the published and the augmented model in
`output/test_sct/<model>/` by default; pass the directories explicitly as above otherwise.

Complete reproductions:

- `bash scripts/reproduce_paper.sh [SWEEP]`: cross-validation, test ensembles, bone Dice,
  figures and a markdown report for the published model.
- `bash scripts/evaluate_aug.sh [SWEEP] [BASELINE] [TEST_MRI_DIR]`: evaluates an augmented
  sweep and compares it with the published model (`scripts/compare_runs.py`).

Further analysis scripts:

| Script | Purpose |
|--------|---------|
| `scripts/predict_per_fold.py` | sCT of each ensemble member separately |
| `scripts/aggregate_sweep.py` | validation metrics stored in the checkpoints |
| `scripts/plot_training_curves.py` | training curves from TensorBoard logs |
| `scripts/make_combined_dice_figure.py` | bone-overlay figure (MR, micro-CT, sCT, average CT) |
| `scripts/visualize_surface_distance.py` | illustration of the surface-distance metrics |
| `scripts/test_dice_mr_registration.py` | average CT registered to micro-CT vs. to MR |
| `scripts/strain_morphometry.py`, `scripts/strain_pairwise.py` | skull morphometry BALB/c vs. C57BL/6 |
| `scripts/mr_intensity_check.py`, `scripts/quickcheck_harmonize_test_mr.py` | MR intensity distribution across strains |
| `scripts/plot_fov_mask.py`, `scripts/plot_fov_overlay.py`, `scripts/summarize_fov_cv.py` | CT field-of-view mask and summaries |

## Inference and DICOM export

```bash
# single volume
python infer.py --config configs/aug_noN4.yaml --ckpt <fold>/checkpoints/best.pt \
    --mri dataset/Test_Data/MR/MZEI00031_MR_T1.nii.gz --out MZEI00031_MR_T1_sCT.nii.gz

# MR -> sCT -> DICOM series (sCT, GT, MR) for one subject
python pipeline.py --config configs/aug_noN4.yaml --ckpt <fold>/checkpoints/best.pt \
    --mr <mr.nii.gz> --gt <ct.nii.gz> --ref reference.dcm --out output/workflow_results

# DICOM series (GT, MR, ensemble sCT, aligned average CT) for one test subject
python scripts/export_dicom.py --subject MZEI00031_MR_T1 --sct-dir $SWEEP/ensemble_test_ssim/sct \
    --sct-description "Synthetic CT - intensity-augmented U-Net" \
    --ref dataset/reference.dcm --out output/dicom_export
```

Use the config the checkpoint was trained with, since it defines the MR scaling.

### Pretrained weights

The three fold checkpoints of the intensity-augmented model are available in release
[v1.0.0](https://github.com/isselmou92/mri-only-workflow/releases/tag/v1.0.0).
Extract `aug_noN4_model.zip` in the repository root; it creates `output/aug_noN4/` with the
per-fold configs and checkpoints, so the commands above work directly:

```bash
python scripts/ensemble_test.py --sweep-root output/aug_noN4 --loss ssim \
    --test-mri-dir dataset/Test_Data/MR --out-dir output/test_sct/aug_noN4 --save-volumes
```

The model was trained on BALB/c mice and expects the original (not N4-corrected) MR.

## Average CT baseline

```bash
python average_ct_baseline.py --config configs/default.yaml --out output/average_ct
python evaluate_alignment.py --avg-ct output/average_ct/average_ct.nii \
    --gt-dir dataset/Test_Data/CT --gt-subjects MZEI00027_CT_T1 MZEI00031_CT_T1 \
    --out output/aligned_dice_results --save-aligned
```

## License

MIT License, see [LICENSE](LICENSE).
