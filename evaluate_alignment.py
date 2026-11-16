#!/usr/bin/env python
"""
Aligned Average CT -- Bone Dice Score with SimpleITK Rigid Registration
=======================================================================
Aligns the average CT to each ground truth CT using rigid registration
(SimpleITK Euler3DTransform: 3 rotations + 3 translations) before
computing bone Dice scores and generating publication-quality figures.

This compensates for varying mouse positions across subjects.

Usage:
    python evaluate_alignment.py \
        --avg-ct output/average_ct/average_ct.nii \
        --gt-dir data/validation_dataset/CT \
        --gt-subjects MZEI00027_CT_T1 MZEI00031_CT_T1 MZEI00066_CT_T1 \
        --out output/aligned_dice_results \
        --threshold 300

    # Save aligned NIfTI volumes
    python evaluate_alignment.py \
        --avg-ct output/average_ct/average_ct.nii \
        --gt-dir data/validation_dataset/CT \
        --gt-subjects MZEI00027_CT_T1 MZEI00031_CT_T1 MZEI00066_CT_T1 \
        --out output/aligned_dice_results \
        --save-aligned

    # Options
        --threshold 300                 # Bone HU threshold (default: 300)
        --max-iterations 500            # Registration max iterations (default: 500)
        --bone-mask-threshold 0         # Mask to tissue/bone for registration focus
        --window 2000 --level 500       # Display window for figures
        --save-aligned                  # Export aligned average CT NIfTI per subject
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import ticker


# ---------------------------------------------------------------------------
# Dice computation
# ---------------------------------------------------------------------------

def dice_score(vol_a: np.ndarray, vol_b: np.ndarray, threshold: float) -> float:
    """
    Compute the Dice coefficient between two volumes after binarization.

    Dice = 2 * |A n B| / (|A| + |B|)

    Args:
        vol_a: First volume (e.g., sCT) in HU.
        vol_b: Second volume (e.g., GT CT) in HU.
        threshold: HU threshold for binarization (voxels > threshold -> 1).

    Returns:
        Dice coefficient in [0, 1].
    """
    mask_a = vol_a > threshold
    mask_b = vol_b > threshold
    intersection = np.sum(mask_a & mask_b)
    size_a = np.sum(mask_a)
    size_b = np.sum(mask_b)
    if size_a + size_b == 0:
        return 1.0
    return float(2.0 * intersection / (size_a + size_b))


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _get_slices(vol: np.ndarray) -> Dict[str, np.ndarray]:
    """Return mid-plane slices for axial, sagittal, coronal views.

    Convention (volume shape = X, Y, Z):
        Axial   : vol[:, :, z_mid]  -- XY plane
        Sagittal: vol[x_mid, :, :]  -- YZ plane
        Coronal : vol[:, y_mid, :]  -- XZ plane
    """
    x_mid = vol.shape[0] // 2
    y_mid = vol.shape[1] // 2
    z_mid = vol.shape[2] // 2
    return {
        "Axial": vol[:, :, z_mid].T,
        "Sagittal": vol[x_mid, :, :].T,
        "Coronal": vol[:, y_mid, :].T,
    }


def _window_level(arr: np.ndarray, window: float, level: float) -> np.ndarray:
    """Apply window/level (W/L) to an image array, returning values in [0, 1]."""
    lo = level - window / 2.0
    hi = level + window / 2.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _bone_contour(mask_2d: np.ndarray) -> np.ndarray:
    """
    Extract the boundary of a 2D binary mask using morphological erosion.
    Returns a boolean array where True = boundary pixel.
    """
    from scipy.ndimage import binary_erosion
    if not np.any(mask_2d):
        return np.zeros_like(mask_2d, dtype=bool)
    interior = binary_erosion(mask_2d, iterations=1)
    return mask_2d & ~interior


# ---------------------------------------------------------------------------
# Publication figure generation
# ---------------------------------------------------------------------------

def generate_figure(
    sct_vol: np.ndarray,
    gt_vol: np.ndarray,
    threshold: float,
    window: float,
    level: float,
    dice: float,
    subject_id: str,
    out_dir: Path,
    col_labels: Optional[List[str]] = None,
    legend_labels: Optional[Tuple[str, str]] = None,
) -> None:
    """
    Generate a publication-quality 4-column x 3-row figure.

    Columns : GT CT | sCT | Difference Map | Bone Overlay
    Rows    : Axial | Sagittal | Coronal
    """
    view_names = ["Axial", "Sagittal", "Coronal"]
    col_names = col_labels or ["GT CT", "sCT", "Difference (HU)", "Bone Overlay"]
    gt_legend, pred_legend = legend_labels or ("GT bone", "sCT bone")

    gt_slices = _get_slices(gt_vol)
    sct_slices = _get_slices(sct_vol)

    diff_vol = sct_vol - gt_vol
    diff_slices = _get_slices(diff_vol)

    gt_bone = gt_vol > threshold
    sct_bone = sct_vol > threshold
    gt_bone_slices = _get_slices(gt_bone.astype(np.float32))
    sct_bone_slices = _get_slices(sct_bone.astype(np.float32))

    diff_abs_max = max(
        abs(np.percentile(diff_vol[gt_vol > -500], 1)),
        abs(np.percentile(diff_vol[gt_vol > -500], 99)),
    ) if np.any(gt_vol > -500) else 500.0
    diff_abs_max = max(diff_abs_max, 1.0)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "mathtext.fontset": "dejavuserif",
    })

    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.2, 5.4),
        gridspec_kw={"wspace": 0.05, "hspace": 0.12},
    )

    diff_norm = TwoSlopeNorm(vmin=-diff_abs_max, vcenter=0, vmax=diff_abs_max)
    im_diff = None

    for row_idx, view in enumerate(view_names):
        gt_img = gt_slices[view]
        sct_img = sct_slices[view]
        diff_img = diff_slices[view]
        gt_bone_img = gt_bone_slices[view] > 0.5
        sct_bone_img = sct_bone_slices[view] > 0.5

        ax = axes[row_idx, 0]
        ax.imshow(
            _window_level(gt_img, window, level),
            cmap="gray", aspect="equal", interpolation="bilinear",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row_idx, 1]
        ax.imshow(
            _window_level(sct_img, window, level),
            cmap="gray", aspect="equal", interpolation="bilinear",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row_idx, 2]
        im_diff = ax.imshow(
            diff_img, cmap="RdBu_r", norm=diff_norm,
            aspect="equal", interpolation="bilinear",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row_idx, 3]
        ax.imshow(
            _window_level(gt_img, window, level),
            cmap="gray", aspect="equal", interpolation="bilinear",
        )
        gt_contour = _bone_contour(gt_bone_img)
        if np.any(gt_contour):
            overlay_gt = np.zeros((*gt_contour.shape, 4), dtype=np.float32)
            overlay_gt[gt_contour] = [1.0, 0.0, 0.0, 0.9]
            ax.imshow(overlay_gt, aspect="equal", interpolation="nearest")
        sct_contour = _bone_contour(sct_bone_img)
        if np.any(sct_contour):
            overlay_sct = np.zeros((*sct_contour.shape, 4), dtype=np.float32)
            overlay_sct[sct_contour] = [0.0, 0.8, 0.0, 0.9]
            ax.imshow(overlay_sct, aspect="equal", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

    for row_idx, view in enumerate(view_names):
        axes[row_idx, 0].set_ylabel(view, fontsize=10, fontweight="bold", rotation=90, labelpad=8)

    for col_idx, title in enumerate(col_names):
        axes[0, col_idx].set_title(title, fontsize=10, fontweight="bold", pad=6)

    cbar_ax = fig.add_axes([0.56, 0.04, 0.12, 0.015])
    cbar = fig.colorbar(im_diff, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("\u0394HU", fontsize=8, labelpad=2)
    cbar.ax.tick_params(labelsize=7)
    cbar.locator = ticker.MaxNLocator(nbins=5, symmetric=True)
    cbar.update_ticks()

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="red", linewidth=1.5, label=gt_legend),
        Line2D([0], [0], color=(0, 0.8, 0), linewidth=1.5, label=pred_legend),
    ]
    axes[2, 3].legend(
        handles=legend_elements, loc="lower center",
        fontsize=7, frameon=False, ncol=2,
        bbox_to_anchor=(0.5, -0.18),
    )

    fig.text(
        0.5, 0.97,
        f"{subject_id}  \u2014  Bone Dice (threshold {int(threshold)} HU): {dice:.4f}",
        ha="center", va="top", fontsize=11, fontweight="bold",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out_path = out_dir / f"{subject_id}_dice_figure.{ext}"
        fig.savefig(
            str(out_path),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
    plt.close(fig)
    print(f"  Figure saved: {out_dir / subject_id}_dice_figure.[pdf|png]")


# ---------------------------------------------------------------------------
# SimpleITK I/O helpers
# ---------------------------------------------------------------------------

def load_sitk(path: Path) -> sitk.Image:
    """Load a NIfTI (or any ITK-supported format) as a SimpleITK Image."""
    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    return img


def sitk_to_numpy(img: sitk.Image) -> np.ndarray:
    """
    Convert a SimpleITK image to a numpy array in (X, Y, Z) layout.

    SimpleITK returns (Z, Y, X) from GetArrayFromImage, so we transpose.
    """
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    return np.asarray(arr.transpose(2, 1, 0), dtype=np.float32)  # (X, Y, Z)


def numpy_to_sitk(arr: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """
    Convert a numpy array (X, Y, Z) back to a SimpleITK image,
    copying spatial metadata from a reference image.
    """
    img = sitk.GetImageFromArray(arr.transpose(2, 1, 0).astype(np.float32))
    img.SetSpacing(reference.GetSpacing())
    img.SetDirection(reference.GetDirection())
    img.SetOrigin(reference.GetOrigin())
    return img


# ---------------------------------------------------------------------------
# Rigid Registration
# ---------------------------------------------------------------------------

def _run_single_registration(
    reg_fixed: sitk.Image,
    reg_moving: sitk.Image,
    initial_transform: sitk.Transform,
    max_iterations: int = 500,
    verbose: bool = True,
    label: str = "",
) -> Tuple[sitk.Transform, float, Dict]:
    """
    Run a single rigid registration attempt and return the transform + final
    metric value.
    """
    reg = sitk.ImageRegistrationMethod()

    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.25, seed=42)

    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-6,
        numberOfIterations=max_iterations,
        gradientMagnitudeTolerance=1e-8,
        relaxationFactor=0.5,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetInitialTransform(initial_transform, inPlace=False)

    info: Dict = {"iterations": [], "metric_values": []}

    if verbose:
        def _iteration_callback(method):
            iteration = method.GetOptimizerIteration()
            metric = method.GetMetricValue()
            info["iterations"].append(iteration)
            info["metric_values"].append(metric)
            if iteration % 100 == 0 or iteration == 1:
                print(f"        Iter {iteration:4d}  |  MI = {metric:.6f}")

        reg.AddCommand(sitk.sitkIterationEvent, lambda: _iteration_callback(reg))

        def _level_callback(method):
            level = method.GetCurrentLevel()
            print(f"      --- Level {level} ---")

        reg.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                       lambda: _level_callback(reg))

    final_transform = reg.Execute(reg_fixed, reg_moving)
    final_metric = reg.GetMetricValue()

    if verbose:
        print(f"      Stopped: {reg.GetOptimizerStopConditionDescription()}")
        print(f"      Final MI: {final_metric:.6f}")

    return final_transform, final_metric, info


def rigid_register(
    fixed: sitk.Image,
    moving: sitk.Image,
    max_iterations: int = 500,
    bone_mask_threshold: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[sitk.Image, object, Dict]:
    """
    Perform rigid registration of *moving* image onto *fixed* image.

    Uses a **multi-start** strategy: tries both MOMENTS and GEOMETRY
    initializations, and picks the result with the best (lowest) MI metric.
    This prevents local-minimum failures when the mouse position differs
    substantially between volumes.

    Args:
        fixed:  Ground truth CT (defines the target grid).
        moving: Average CT to be aligned.
        max_iterations: Maximum optimizer iterations per resolution level.
        bone_mask_threshold: If set, mask both images above this HU value
            for the registration metric (e.g. 0 for tissue, 300 for bone).
        verbose: Print registration progress.

    Returns:
        (resampled_image, final_transform, info_dict)
    """
    reg_fixed = fixed
    reg_moving = moving
    if bone_mask_threshold is not None:
        reg_fixed = sitk.Threshold(fixed, lower=bone_mask_threshold, upper=1e6,
                                   outsideValue=bone_mask_threshold)
        reg_moving = sitk.Threshold(moving, lower=bone_mask_threshold, upper=1e6,
                                    outsideValue=bone_mask_threshold)

    init_methods = [
        ("MOMENTS", sitk.CenteredTransformInitializerFilter.MOMENTS),
        ("GEOMETRY", sitk.CenteredTransformInitializerFilter.GEOMETRY),
    ]

    best_transform = None
    best_metric = float("inf")
    best_info: Dict = {}
    best_label = ""

    for label, init_type in init_methods:
        if verbose:
            print(f"    [Start: {label}]")

        try:
            initial_transform = sitk.CenteredTransformInitializer(
                reg_fixed, reg_moving,
                sitk.Euler3DTransform(),
                init_type,
            )
        except RuntimeError as e:
            if verbose:
                print(f"      Initialization failed ({e}), skipping.")
            continue

        transform, metric, info = _run_single_registration(
            reg_fixed, reg_moving,
            initial_transform,
            max_iterations=max_iterations,
            verbose=verbose,
            label=label,
        )

        if metric < best_metric:
            best_metric = metric
            best_transform = transform
            best_info = info
            best_label = label

    if best_transform is None:
        raise RuntimeError("All registration initializations failed.")

    if verbose:
        print(f"    Best initialization: {best_label} (MI = {best_metric:.6f})")

    resampled = sitk.Resample(
        moving,
        fixed,
        best_transform,
        sitk.sitkLinear,
        -1000.0,
        moving.GetPixelID(),
    )

    euler: object = best_transform
    try:
        is_composite = isinstance(best_transform, sitk.CompositeTransform)
    except AttributeError:
        is_composite = (
            hasattr(best_transform, "GetName")
            and "Composite" in best_transform.GetName()
        )

    if is_composite and hasattr(best_transform, "GetNumberOfTransforms"):
        n = best_transform.GetNumberOfTransforms()
        for i in range(n):
            t = best_transform.GetNthTransform(i)
            if "Euler3D" in t.GetName():
                euler = sitk.Euler3DTransform(t)
                break
    else:
        try:
            euler = sitk.Euler3DTransform(best_transform)
        except RuntimeError:
            euler = best_transform

    return resampled, euler, best_info


# ---------------------------------------------------------------------------
# Dice analysis with alignment
# ---------------------------------------------------------------------------

def run_aligned_dice_analysis(
    avg_ct_path: Path,
    gt_dir: Path,
    gt_subjects: List[str],
    out_dir: Path,
    threshold: float = 300.0,
    window: float = 2000.0,
    level: float = 500.0,
    max_iterations: int = 500,
    bone_mask_threshold: Optional[float] = None,
    save_aligned: bool = False,
) -> None:
    """
    For each GT subject:
      1. Rigidly register average CT -> GT CT
      2. Compute bone Dice on aligned pair
      3. Generate publication figure
      4. Optionally save aligned NIfTI
    """
    print(f"\n{'=' * 70}")
    print("  Aligned Average CT -- Bone Dice Score Analysis (Rigid Registration)")
    print(f"{'=' * 70}")
    print(f"  Bone threshold   : {threshold:.0f} HU")
    print(f"  Max iterations   : {max_iterations}")
    print(f"  Mask threshold   : {bone_mask_threshold if bone_mask_threshold is not None else 'None (full image)'}")
    print(f"  Subjects         : {len(gt_subjects)}")
    print(f"  Save aligned     : {save_aligned}")
    print(f"{'=' * 70}")

    print(f"\n  Loading average CT: {avg_ct_path}")
    avg_ct_sitk = load_sitk(avg_ct_path)
    avg_ct_size = avg_ct_sitk.GetSize()
    avg_ct_spacing = avg_ct_sitk.GetSpacing()
    print(f"    Size    : {avg_ct_size}")
    print(f"    Spacing : {tuple(f'{s:.4f}' for s in avg_ct_spacing)}")

    results: List[Dict] = []
    aligned_dir = out_dir / "aligned_nifti"

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

        gt_sitk = load_sitk(gt_path)
        gt_size = gt_sitk.GetSize()
        gt_spacing = gt_sitk.GetSpacing()
        print(f"    GT size    : {gt_size}")
        print(f"    GT spacing : {tuple(f'{s:.4f}' for s in gt_spacing)}")

        avg_ct_np = sitk_to_numpy(avg_ct_sitk)
        gt_np = sitk_to_numpy(gt_sitk)

        if avg_ct_np.shape == gt_np.shape:
            dice_before = dice_score(avg_ct_np, gt_np, threshold)
            print(f"    Dice (before alignment) : {dice_before:.4f}")
        else:
            dice_before = None
            print(f"    Dice (before alignment) : N/A (shape mismatch "
                  f"{avg_ct_np.shape} vs {gt_np.shape})")

        print(f"    Running rigid registration...")
        aligned_sitk, transform, reg_info = rigid_register(
            fixed=gt_sitk,
            moving=avg_ct_sitk,
            max_iterations=max_iterations,
            bone_mask_threshold=bone_mask_threshold,
            verbose=True,
        )

        if hasattr(transform, "GetParameters"):
            params = transform.GetParameters()
            if len(params) >= 6:
                print(f"    Transform parameters:")
                print(f"      Rotation (rad) : ({params[0]:.4f}, {params[1]:.4f}, {params[2]:.4f})")
                print(f"      Translation    : ({params[3]:.4f}, {params[4]:.4f}, {params[5]:.4f})")
            else:
                print(f"    Transform parameters: {params}")
        else:
            print(f"    Transform: {transform}")

        aligned_np = sitk_to_numpy(aligned_sitk)

        dice_after = dice_score(aligned_np, gt_np, threshold)
        print(f"    Dice (after alignment)  : {dice_after:.4f}")

        if dice_before is not None:
            delta = dice_after - dice_before
            print(f"    Improvement             : {delta:+.4f}")

        results.append({
            "subject": f"AlignedAvgCT_vs_{gt_name}",
            "dice_before": f"{dice_before:.4f}" if dice_before is not None else "N/A",
            "dice_after": f"{dice_after:.4f}",
        })

        if save_aligned:
            aligned_dir.mkdir(parents=True, exist_ok=True)
            aligned_path = aligned_dir / f"AvgCT_aligned_to_{gt_name}.nii.gz"
            sitk.WriteImage(aligned_sitk, str(aligned_path))
            print(f"    Aligned NIfTI saved: {aligned_path}")

        generate_figure(
            sct_vol=aligned_np,
            gt_vol=gt_np,
            threshold=threshold,
            window=window,
            level=level,
            dice=dice_after,
            subject_id=f"AlignedAvgCT_vs_{gt_name}",
            out_dir=out_dir,
            col_labels=["Ground Truth CT", "Aligned Average CT",
                         "Difference (HU)", "Bone Overlay"],
            legend_labels=("GT bone", "Aligned AvgCT bone"),
        )

    # ---- Summary table ----
    print(f"\n{'=' * 70}")
    print("  Summary -- Rigid Alignment Results")
    print(f"{'=' * 70}")
    print(f"  {'Subject':<40s}  {'Before':>8s}  {'After':>8s}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['subject']:<40s}  {r['dice_before']:>8s}  {r['dice_after']:>8s}")

    dice_after_values = [float(r["dice_after"]) for r in results]
    if len(dice_after_values) > 1:
        mean_d = float(np.mean(dice_after_values))
        std_d = float(np.std(dice_after_values))
        print("  " + "-" * 60)
        print(f"  {'Mean +/- Std (after)':<40s}  {'':>8s}  {mean_d:.4f} +/- {std_d:.4f}")
    print(f"{'=' * 70}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "aligned_dice_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "dice_before", "dice_after"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
        if len(dice_after_values) > 1:
            writer.writerow({
                "subject": "Mean",
                "dice_before": "",
                "dice_after": f"{np.mean(dice_after_values):.4f}",
            })
            writer.writerow({
                "subject": "Std",
                "dice_before": "",
                "dice_after": f"{np.std(dice_after_values):.4f}",
            })
    print(f"\n  CSV saved: {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Align Average CT to GT CTs via rigid registration (SimpleITK) "
            "and compute bone Dice scores."
        )
    )
    ap.add_argument("--avg-ct", required=True, type=Path,
                    help="Path to the average CT NIfTI file")
    ap.add_argument("--gt-dir", required=True, type=Path,
                    help="Directory containing GT CT NIfTI files")
    ap.add_argument("--gt-subjects", required=True, nargs="+",
                    help="GT subject stems (e.g. MZEI00027_CT_T1)")

    ap.add_argument("--out", type=Path, default=Path("output/aligned_dice_results"),
                    help="Output directory (default: output/aligned_dice_results)")

    ap.add_argument("--threshold", type=float, default=300.0,
                    help="Bone HU threshold for Dice (default: 300)")
    ap.add_argument("--window", type=float, default=2000.0,
                    help="Display window width in HU (default: 2000)")
    ap.add_argument("--level", type=float, default=500.0,
                    help="Display window center in HU (default: 500)")

    ap.add_argument("--max-iterations", type=int, default=500,
                    help="Max optimizer iterations per resolution level (default: 500)")
    ap.add_argument("--bone-mask-threshold", type=float, default=None,
                    help="Mask images above this HU for registration focus "
                         "(e.g. 0 for tissue, 300 for bone). Default: None (full image)")

    ap.add_argument("--save-aligned", action="store_true",
                    help="Save aligned average CT NIfTI for each subject")

    return ap.parse_args()


def main():
    args = parse_args()

    if not args.avg_ct.exists():
        print(f"Error: Average CT not found: {args.avg_ct}")
        sys.exit(1)
    if not args.gt_dir.exists():
        print(f"Error: GT directory not found: {args.gt_dir}")
        sys.exit(1)

    run_aligned_dice_analysis(
        avg_ct_path=args.avg_ct,
        gt_dir=args.gt_dir,
        gt_subjects=args.gt_subjects,
        out_dir=args.out,
        threshold=args.threshold,
        window=args.window,
        level=args.level,
        max_iterations=args.max_iterations,
        bone_mask_threshold=args.bone_mask_threshold,
        save_aligned=args.save_aligned,
    )

    print(f"\n{'=' * 70}")
    print("  Aligned Dice Analysis Complete")
    print(f"{'=' * 70}")
    print(f"  Results : {args.out}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
