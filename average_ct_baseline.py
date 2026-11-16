#!/usr/bin/env python
"""
Generate Average CT Baseline
=============================
Computes the voxel-wise average of co-registered CT images as a baseline
comparison for the sCT model. Exports as NIfTI and optionally runs Dice
score analysis against validation GT CTs.

Supports optional **SimpleITK rigid alignment** before averaging, so that
all CT volumes are brought into the same physical space even when mouse
positioning varies between scans.

Usage:
    # Directory-based mode (co-registered CTs) + Dice
    python average_ct_baseline.py \
        --ct-dir data/average_CT/CT \
        --out output/average_ct \
        --gt-dir data/validation_dataset/CT \
        --gt-subjects MZEI00027_CT_T1 MZEI00031_CT_T1 MZEI00066_CT_T1

    # With rigid alignment before averaging
    python average_ct_baseline.py \
        --ct-dir data/average_CT/CT \
        --out output/average_ct_aligned \
        --align \
        --gt-dir data/validation_dataset/CT \
        --gt-subjects MZEI00027_CT_T1 MZEI00031_CT_T1 MZEI00066_CT_T1

    # Config-based mode (uses training set from YAML config)
    python average_ct_baseline.py \
        --config configs/default.yaml \
        --out output/average_ct

    # Options
        --threshold 300       # Bone HU threshold for Dice (default: 300)
        --window 2000         # Display window width (default: 2000)
        --level 500           # Display window center (default: 500)
        --align               # Rigid-register all CTs to first before averaging
        --align-iterations 500 # Max registration iterations (default: 500)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib

from evaluate_alignment import (
    dice_score,
    generate_figure,
    load_sitk,
    sitk_to_numpy,
    rigid_register,
)


# ---------------------------------------------------------------------------
# NIfTI I/O
# ---------------------------------------------------------------------------

def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray, object]:
    """Load NIfTI and return (array, affine, header)."""
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)
    return np.asarray(arr, dtype=np.float32), img.affine, img.header


def load_nifti_array(path: Path) -> np.ndarray:
    """Load a NIfTI file and return only the float32 array."""
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(), dtype=np.float32)


def save_nifti(path: Path, volume: np.ndarray, affine=None, header=None) -> None:
    """Save NIfTI with affine/header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    ni = nib.Nifti1Image(volume.astype(np.float32), affine=affine, header=header)
    nib.save(ni, str(path))


# ---------------------------------------------------------------------------
# Config-based train split
# ---------------------------------------------------------------------------

def split_by_mouse_timepoint(pairs, seed: int = 1337):
    """
    Split dataset into train/val/test (same logic as train.py).

    Val set (held out): Mouse 66 T1,T2 + Mouse 30 T1,T2,T3 (5 samples)
    Train/Test: Remaining samples split 80/20
    """
    import random

    val_patterns = [
        ("66", ["T1", "T2"]),
        ("30", ["T1", "T2", "T3"]),
    ]

    val_indices = []
    remaining_indices = []

    for idx, pair in enumerate(pairs):
        filename = pair.mri.stem.upper()

        is_val = False
        for mouse_id, timepoints in val_patterns:
            if f"MZEI000{mouse_id}" in filename or f"MZEI00{mouse_id}" in filename:
                for tp in timepoints:
                    if filename.endswith(f"_{tp}") or filename.endswith(tp):
                        is_val = True
                        break
            if is_val:
                break

        if is_val:
            val_indices.append(idx)
        else:
            remaining_indices.append(idx)

    rng = random.Random(seed)
    rng.shuffle(remaining_indices)

    n_remaining = len(remaining_indices)
    n_train = int(0.8 * n_remaining)

    train_indices = remaining_indices[:n_train]
    test_indices = remaining_indices[n_train:]

    return train_indices, val_indices, test_indices


# ---------------------------------------------------------------------------
# Rigid alignment of CT volumes before averaging
# ---------------------------------------------------------------------------

def align_ct_volumes(
    ct_paths: List[Path],
    max_iterations: int = 500,
    ref_index: int = 0,
) -> Tuple[List[np.ndarray], object]:
    """
    Load CT volumes and rigidly align them all to a reference volume.

    Uses multi-start rigid registration (MOMENTS + GEOMETRY) from
    evaluate_alignment to handle varying mouse positions.

    Args:
        ct_paths: List of NIfTI file paths to load and align.
        max_iterations: Max optimizer iterations per registration level.
        ref_index: Index of the reference volume (all others align to this).

    Returns:
        (aligned_numpy_volumes, reference_sitk_image)
    """
    n = len(ct_paths)
    print(f"\n  Rigid alignment enabled (reference: [{ref_index + 1}/{n}] {ct_paths[ref_index].name})")

    sitk_images = []
    for i, path in enumerate(ct_paths):
        img = load_sitk(path)
        size = img.GetSize()
        spacing = img.GetSpacing()
        print(f"    [{i + 1}/{n}] {path.name}  size={size}  spacing={tuple(f'{s:.4f}' for s in spacing)}")
        sitk_images.append(img)

    ref_sitk = sitk_images[ref_index]
    aligned_volumes: List[np.ndarray] = []

    for i, (path, moving_sitk) in enumerate(zip(ct_paths, sitk_images)):
        if i == ref_index:
            arr = sitk_to_numpy(ref_sitk)
            aligned_volumes.append(arr)
            print(f"\n    [{i + 1}/{n}] {path.name} -- REFERENCE (no registration)")
            continue

        print(f"\n    [{i + 1}/{n}] {path.name} -- registering to reference...")
        resampled_sitk, transform, info = rigid_register(
            fixed=ref_sitk,
            moving=moving_sitk,
            max_iterations=max_iterations,
            verbose=True,
        )

        if hasattr(transform, "GetParameters"):
            params = transform.GetParameters()
            if len(params) >= 6:
                print(f"      Rotation (rad) : ({params[0]:.4f}, {params[1]:.4f}, {params[2]:.4f})")
                print(f"      Translation    : ({params[3]:.4f}, {params[4]:.4f}, {params[5]:.4f})")

        arr = sitk_to_numpy(resampled_sitk)
        aligned_volumes.append(arr)

    print(f"\n  Alignment complete -- {n} volumes aligned to reference.")
    return aligned_volumes, ref_sitk


# ---------------------------------------------------------------------------
# Dice analysis
# ---------------------------------------------------------------------------

def run_dice_analysis(
    avg_ct_path: Path,
    gt_dir: Path,
    gt_subjects: List[str],
    out_dir: Path,
    threshold: float = 300.0,
    window: float = 2000.0,
    level: float = 500.0,
) -> None:
    """
    Compute bone Dice scores between the average CT and each GT CT subject,
    generate publication-quality figures, and write a summary CSV.
    """
    print(f"\n{'=' * 62}")
    print("  Average CT -- Bone Dice Score Analysis")
    print(f"{'=' * 62}")
    print(f"  Threshold : {threshold:.0f} HU")
    print(f"  Subjects  : {len(gt_subjects)}")
    print(f"{'=' * 62}")

    avg_ct_vol = load_nifti_array(avg_ct_path)

    results: List[Dict] = []

    for gt_name in gt_subjects:
        gt_path: Optional[Path] = None
        for ext in [".nii.gz", ".nii"]:
            candidate = gt_dir / (gt_name + ext)
            if candidate.exists():
                gt_path = candidate
                break

        if gt_path is None:
            print(f"\n  [{gt_name}] WARNING: GT file not found in {gt_dir}, skipping.")
            continue

        print(f"\n  [{gt_name}]")
        print(f"    AvgCT : {avg_ct_path}")
        print(f"    GT    : {gt_path}")

        gt_vol = load_nifti_array(gt_path)

        if avg_ct_vol.shape != gt_vol.shape:
            print(f"    Warning: shape mismatch AvgCT {avg_ct_vol.shape} vs GT {gt_vol.shape}")

        d = dice_score(avg_ct_vol, gt_vol, threshold)
        print(f"    Dice  : {d:.4f}")

        results.append({"subject": f"AvgCT_vs_{gt_name}", "dice": d})

        generate_figure(
            sct_vol=avg_ct_vol,
            gt_vol=gt_vol,
            threshold=threshold,
            window=window,
            level=level,
            dice=d,
            subject_id=f"AvgCT_vs_{gt_name}",
            out_dir=out_dir,
            col_labels=["Ground Truth CT", "Average CT", "Difference (HU)", "Bone Overlay"],
            legend_labels=("GT bone", "Average CT bone"),
        )

    # ---- Summary table ----
    print(f"\n{'=' * 62}")
    print("  Summary")
    print(f"{'=' * 62}")
    print(f"  {'Subject':<40s}  {'Dice':>8s}")
    print("  " + "-" * 50)
    for r in results:
        print(f"  {r['subject']:<40s}  {r['dice']:>8.4f}")
    dice_values = [r["dice"] for r in results]
    if len(dice_values) > 1:
        mean_d = float(np.mean(dice_values))
        std_d = float(np.std(dice_values))
        print("  " + "-" * 50)
        print(f"  {'Mean +/- Std':<40s}  {mean_d:.4f} +/- {std_d:.4f}")
    print(f"{'=' * 62}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dice_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "dice"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
        if len(dice_values) > 1:
            writer.writerow({"subject": "Mean", "dice": f"{np.mean(dice_values):.4f}"})
            writer.writerow({"subject": "Std", "dice": f"{np.std(dice_values):.4f}"})
    print(f"\n  CSV saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate average CT baseline, export NIfTI, compute Dice."
    )

    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument(
        "--ct-dir", type=Path,
        help="Directory of co-registered CT NIfTI files to average (e.g., data/average_CT/CT)"
    )
    input_grp.add_argument(
        "--config", type=str,
        help="YAML config to load training CTs from dataset (config-based mode)"
    )

    parser.add_argument("--out", required=True, type=Path,
                        help="Output root directory (NIfTI and Dice results go here)")

    parser.add_argument("--gt-dir", type=Path, default=None,
                        help="Directory containing GT CT NIfTI files for Dice comparison")
    parser.add_argument("--gt-subjects", nargs="+", default=None,
                        help="GT subject stems to compare against (e.g., MZEI00027_CT_T1)")

    parser.add_argument("--threshold", type=float, default=300.0,
                        help="Bone HU threshold for Dice binarization (default: 300)")
    parser.add_argument("--window", type=float, default=2000.0,
                        help="Display window width in HU (default: 2000)")
    parser.add_argument("--level", type=float, default=500.0,
                        help="Display window center in HU (default: 500)")

    parser.add_argument("--align", action="store_true",
                        help="Rigidly register all CT volumes to the first before averaging "
                             "(SimpleITK Euler3D, multi-start MOMENTS+GEOMETRY)")
    parser.add_argument("--align-iterations", type=int, default=500,
                        help="Max optimizer iterations per resolution level (default: 500)")

    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed for train/val/test split (config mode only)")

    args = parser.parse_args()

    # ==================================================================
    # 1. Load CT volumes
    # ==================================================================
    ct_volumes: List[np.ndarray] = []
    ct_paths_for_align: List[Path] = []
    ref_affine = None
    ref_header = None

    if args.ct_dir is not None:
        if not args.ct_dir.exists():
            print(f"Error: CT directory not found: {args.ct_dir}")
            sys.exit(1)

        ct_files = sorted(list(args.ct_dir.glob("*.nii")) + list(args.ct_dir.glob("*.nii.gz")))
        if not ct_files:
            print(f"Error: No NIfTI files found in {args.ct_dir}")
            sys.exit(1)

        if args.align:
            print(f"Loading {len(ct_files)} CT volumes from {args.ct_dir}")
            print(f"  (Will rigidly align all volumes to reference before averaging)\n")
            ct_paths_for_align = ct_files
            _, ref_affine, ref_header = load_nifti(ct_files[0])
        else:
            print(f"Loading {len(ct_files)} co-registered CT volumes from {args.ct_dir}")
            print(f"  (Assuming all volumes are already aligned in voxel space)\n")
            for i, ct_path in enumerate(ct_files):
                arr, affine, header = load_nifti(ct_path)
                print(f"  [{i + 1}/{len(ct_files)}] {ct_path.name}  shape={arr.shape}")
                ct_volumes.append(arr)
                if ref_affine is None:
                    ref_affine = affine
                    ref_header = header

    else:
        from simple_sct.config import load_config
        from simple_sct.data.volume_dataset import VolumePairs

        cfg = load_config(args.config)
        ds = VolumePairs(cfg)
        train_indices, val_indices, test_indices = split_by_mouse_timepoint(
            ds.pairs, seed=args.seed
        )

        print(f"Dataset split:")
        print(f"  Training:   {len(train_indices)} samples")
        print(f"  Validation: {len(val_indices)} samples")
        print(f"  Test:       {len(test_indices)} samples")
        print()

        if args.align:
            print(f"Loading {len(train_indices)} training CT volume paths...")
            for i, idx in enumerate(train_indices):
                pair = ds.pairs[idx]
                ct_path = pair.ct
                print(f"  [{i + 1}/{len(train_indices)}] {ct_path.name}")
                ct_paths_for_align.append(ct_path)
            _, ref_affine, ref_header = load_nifti(ct_paths_for_align[0])
        else:
            print(f"Loading {len(train_indices)} training CT volumes...")
            for i, idx in enumerate(train_indices):
                pair = ds.pairs[idx]
                ct_path = pair.ct
                print(f"  [{i + 1}/{len(train_indices)}] {ct_path.name}")
                arr, affine, header = load_nifti(ct_path)
                ct_volumes.append(arr)
                if ref_affine is None:
                    ref_affine = affine
                    ref_header = header

    # ==================================================================
    # 1b. (Optional) Rigid alignment before averaging
    # ==================================================================
    if args.align and ct_paths_for_align:
        ct_volumes, _ref_sitk = align_ct_volumes(
            ct_paths=ct_paths_for_align,
            max_iterations=args.align_iterations,
            ref_index=0,
        )

    # ==================================================================
    # 2. Compute voxel-wise average
    # ==================================================================
    mode_str = "aligned" if args.align else "co-registered"
    print(f"\nComputing voxel-wise average of {len(ct_volumes)} {mode_str} CT volumes...")
    ct_stack = np.stack(ct_volumes, axis=0)
    average_ct = np.mean(ct_stack, axis=0).astype(np.float32)

    print(f"  Shape    : {average_ct.shape}")
    print(f"  HU range : [{average_ct.min():.1f}, {average_ct.max():.1f}]")
    print(f"  Mean HU  : {average_ct.mean():.1f}")

    # ==================================================================
    # 3. Save NIfTI
    # ==================================================================
    args.out.mkdir(parents=True, exist_ok=True)
    nifti_path = args.out / "average_ct.nii"
    print(f"\nSaving NIfTI: {nifti_path}")
    save_nifti(nifti_path, average_ct, affine=ref_affine, header=ref_header)

    # ==================================================================
    # 4. Dice analysis against validation GT CTs
    # ==================================================================
    if args.gt_dir is not None and args.gt_subjects is not None:
        if not args.gt_dir.exists():
            print(f"\nWarning: GT directory not found: {args.gt_dir}. Skipping Dice analysis.")
        else:
            dice_dir = args.out / "dice_results"
            run_dice_analysis(
                avg_ct_path=nifti_path,
                gt_dir=args.gt_dir,
                gt_subjects=args.gt_subjects,
                out_dir=dice_dir,
                threshold=args.threshold,
                window=args.window,
                level=args.level,
            )
    elif args.gt_dir is not None:
        print("\nWarning: --gt-dir provided but no --gt-subjects specified. Skipping Dice analysis.")
    else:
        print("\nSkipping Dice analysis (no --gt-dir / --gt-subjects provided).")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'=' * 62}")
    print("  Average CT Generation Complete")
    print(f"{'=' * 62}")
    print(f"  NIfTI  : {nifti_path}")
    if args.gt_dir is not None and args.gt_subjects is not None:
        print(f"  Dice   : {args.out / 'dice_results'}")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
