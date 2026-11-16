#!/usr/bin/env python
"""
Combined Bone-Dice Figure: Best sCT vs Average CT
=================================================
Builds a single publication-quality figure that summarises, side-by-side, the
ground-truth micro-CT, the best ensemble sCT, the population-average CT
(avCT) and the input MR slice for a chosen test mouse.

Pipeline (two clean steps, both written to disk so they can be reused):
  1. Build the population avCT by voxel-wise mean of the training CT volumes
     in ``output/average_CT/CT/*_Aligned.nii``.  Those volumes were aligned
     manually in 3D Slicer; the ``.tfm`` files under ``Transformations/`` are
     *not* re-applied.  The result is written to
     ``output/average_CT/average_ct.nii.gz``.
  2. Compute bone Dice (HU > threshold, default 300) of best sCT vs GT and
     avCT vs GT directly in the shared voxel grid (no further registration).
  3. Render (axial | sagittal | coronal) x (MR | GT | sCT | avCT |
     bone-contour overlay) with both DSC values in the suptitle.

Pass ``--rigid-align-to-gt`` to additionally rigid-register the avCT onto the
test GT (matches the old ``align_and_dice.py`` workflow).

Run
---
    python scripts/make_combined_dice_figure.py
    python scripts/make_combined_dice_figure.py \
        --subject MZEI00031_MR_T1 \
        --ensemble ssim \
        --out manuscript/CombinedDice_BALBC_Mouse2

The defaults pick the highest-Dice subject (MZEI00031_MR_T1) of the SSIM K=3
ensemble (the loss chosen in the main paper).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import nibabel as nib

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_alignment import (  # noqa: E402
    dice_score,
    load_sitk,
    numpy_to_sitk,
    rigid_register,
    sitk_to_numpy,
)

import SimpleITK as sitk  # noqa: E402  (after evaluate_alignment so the env is set)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_array(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)


def _pick_reference_index(ct_files: List[Path], preferred_stem: str | None) -> int:
    """Pick the index of the reference volume (default: MZEI00039 since the
    Slicer ``.tfm`` files in ``Transformations/`` are all ``*_to_MZEI00039.tfm``).
    Falls back to the first file if the preferred stem is missing.
    """
    if preferred_stem is not None:
        for i, p in enumerate(ct_files):
            if preferred_stem.lower() in p.name.lower():
                return i
    return 0


def _compute_average_ct(
    ct_dir: Path,
    pattern: str = "*_Aligned.nii*",
    reference_stem: str | None = "MZEI00039",
    default_value: float = -1000.0,
) -> Tuple[np.ndarray, sitk.Image, Path]:
    """Resample every NIfTI in ``ct_dir`` matching ``pattern`` onto a single
    reference grid (using each file's own affine -- *no* new transformation is
    applied) and return the voxel-wise mean.

    The Slicer-aligned volumes have *different* NIfTI affines (the rigid
    alignment was baked into the header, not the voxel data), so a naive
    ``np.mean`` across raw arrays mixes voxels that correspond to different
    anatomical points.  Resampling each volume onto the same grid via
    SimpleITK fixes this -- the identity transform plus each file's existing
    direction/origin/spacing is enough.
    """
    ct_files = sorted(ct_dir.glob(pattern))
    if not ct_files:
        sys.exit(
            f"No NIfTI files matching '{pattern}' found in {ct_dir}.\n"
            f"Pass --avg-ct-pattern '*.nii*' if you want to average every file."
        )
    ref_idx = _pick_reference_index(ct_files, reference_stem)
    ref_path = ct_files[ref_idx]
    print(
        f"Averaging {len(ct_files)} training CT volumes from {ct_dir} "
        f"(pattern: '{pattern}')"
    )
    print(f"  Reference grid: [{ref_idx + 1}/{len(ct_files)}] {ref_path.name}")
    print("  Resampling each volume onto the reference grid via SimpleITK")
    print("  (uses each file's existing affine; no new transform is applied).")

    ref_img = load_sitk(ref_path)
    identity = sitk.Transform(3, sitk.sitkIdentity)

    stack: List[np.ndarray] = []
    for i, p in enumerate(ct_files):
        moving = load_sitk(p)
        if i == ref_idx:
            arr = sitk_to_numpy(moving)
            print(f"  [{i + 1:2d}/{len(ct_files)}] {p.name}  REFERENCE")
        else:
            resampled = sitk.Resample(
                moving,
                ref_img,
                identity,
                sitk.sitkLinear,
                float(default_value),
                moving.GetPixelID(),
            )
            arr = sitk_to_numpy(resampled)
            print(f"  [{i + 1:2d}/{len(ct_files)}] {p.name}  resampled onto reference")
        stack.append(arr)

    avg = np.mean(np.stack(stack, axis=0), axis=0).astype(np.float32)
    print(f"  -> avCT shape={avg.shape}  HU range=[{avg.min():.1f}, {avg.max():.1f}]")
    return avg, ref_img, ref_path


def _align_avg_to_subject(
    avg_arr: np.ndarray,
    ref_template: sitk.Image,
    target_ct_path: Path,
    max_iterations: int = 500,
    bone_mask_threshold: float | None = None,
) -> np.ndarray:
    """Rigid AVG -> GT registration, matching the old ``align_and_dice.py``.

    The avg array is wrapped as a SimpleITK image using the spatial metadata
    of the resampling reference grid (its native frame), then rigidly
    registered onto the test subject's GT CT.  No transformation is applied
    to the individual training volumes -- those are already Slicer-aligned.
    """
    moving = numpy_to_sitk(avg_arr, reference=ref_template)
    fixed = load_sitk(target_ct_path)
    print(
        f"\nRigid AVG -> {target_ct_path.name}"
        f"  (bone_mask_threshold={bone_mask_threshold})"
    )
    resampled, transform, _ = rigid_register(
        fixed=fixed,
        moving=moving,
        max_iterations=max_iterations,
        bone_mask_threshold=bone_mask_threshold,
        verbose=True,
    )
    if hasattr(transform, "GetParameters"):
        params = transform.GetParameters()
        if len(params) >= 6:
            print(
                f"  Rotation (rad) : ({params[0]:+.4f}, {params[1]:+.4f}, {params[2]:+.4f})"
            )
            print(
                f"  Translation    : ({params[3]:+.4f}, {params[4]:+.4f}, {params[5]:+.4f})"
            )
    return sitk_to_numpy(resampled)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _window_level(arr: np.ndarray, window: float, level: float) -> np.ndarray:
    lo = level - window / 2.0
    hi = level + window / 2.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _mr_norm(arr: np.ndarray) -> np.ndarray:
    """Robust 1-99% display normalisation for MR (intensities are arbitrary)."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _bone_contour(mask_2d: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    if not np.any(mask_2d):
        return np.zeros_like(mask_2d, dtype=bool)
    interior = binary_erosion(mask_2d, iterations=1)
    return mask_2d & ~interior


def _get_slices(vol: np.ndarray) -> Dict[str, np.ndarray]:
    """Mid-plane axial / sagittal / coronal slices, oriented for display."""
    x_mid = vol.shape[0] // 2
    y_mid = vol.shape[1] // 2
    z_mid = vol.shape[2] // 2
    return {
        "Axial": vol[:, :, z_mid].T,
        "Sagittal": vol[x_mid, :, :].T,
        "Coronal": vol[:, y_mid, :].T,
    }


def _draw_contour_overlay(
    ax,
    base_img_w: np.ndarray,
    gt_mask: np.ndarray,
    sct_mask: np.ndarray,
    avg_mask: np.ndarray,
    color_gt: Tuple[float, float, float] = (1.0, 0.05, 0.05),
    color_sct: Tuple[float, float, float] = (0.0, 0.85, 0.10),
    color_avg: Tuple[float, float, float] = (0.10, 0.55, 1.00),
    alpha: float = 0.95,
) -> None:
    """Draw GT-CT grayscale background plus three colored bone contours."""
    ax.imshow(base_img_w, cmap="gray", aspect="equal", interpolation="bilinear")
    for mask, color in (
        (avg_mask, color_avg),
        (sct_mask, color_sct),
        (gt_mask, color_gt),
    ):
        contour = _bone_contour(mask)
        if np.any(contour):
            rgba = np.zeros((*contour.shape, 4), dtype=np.float32)
            rgba[contour] = (*color, alpha)
            ax.imshow(rgba, aspect="equal", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Master figure
# ---------------------------------------------------------------------------

def build_combined_figure(
    mr_vol: np.ndarray,
    gt_vol: np.ndarray,
    sct_vol: np.ndarray,
    avg_vol: np.ndarray,
    *,
    subject_id: str,
    ensemble_label: str,
    threshold: float,
    window: float,
    level: float,
    out_dir: Path,
    views: Tuple[str, ...] = ("Axial", "Sagittal", "Coronal"),
) -> Tuple[float, float]:
    """Return (sCT Dice, avCT Dice) and save the publication figure."""
    dice_sct = dice_score(sct_vol, gt_vol, threshold)
    dice_avg = dice_score(avg_vol, gt_vol, threshold)

    print(
        f"\nDice (HU > {threshold:.0f}): sCT={dice_sct:.4f}  avCT={dice_avg:.4f}"
    )

    mr_slices = _get_slices(mr_vol)
    gt_slices = _get_slices(gt_vol)
    sct_slices = _get_slices(sct_vol)
    avg_slices = _get_slices(avg_vol)

    gt_bone = (gt_vol > threshold).astype(np.float32)
    sct_bone = (sct_vol > threshold).astype(np.float32)
    avg_bone = (avg_vol > threshold).astype(np.float32)
    gt_bone_slices = _get_slices(gt_bone)
    sct_bone_slices = _get_slices(sct_bone)
    avg_bone_slices = _get_slices(avg_bone)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "mathtext.fontset": "dejavuserif",
        }
    )

    # Use Unicode superscripts to match manuscript abbreviation scheme:
    # sCT^UNet, sCT^avg, µCT
    col_titles = [
        "MR",
        "\u00b5CT",
        f"sCT$^{{\\mathregular{{UNet}}}}$\nDSC = {dice_sct:.3f}",
        f"sCT$^{{\\mathregular{{avg}}}}$\nDSC = {dice_avg:.3f}",
        "Bone contour overlay",
    ]
    n_rows = len(views)
    n_cols = len(col_titles)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.0 * n_cols, 2.05 * n_rows + 0.6),
        gridspec_kw={"wspace": 0.04, "hspace": 0.08},
        squeeze=False,
    )

    for r, view in enumerate(views):
        mr_img = _mr_norm(mr_slices[view])
        gt_img_w = _window_level(gt_slices[view], window, level)
        sct_img_w = _window_level(sct_slices[view], window, level)
        avg_img_w = _window_level(avg_slices[view], window, level)
        gt_bone_2d = gt_bone_slices[view] > 0.5
        sct_bone_2d = sct_bone_slices[view] > 0.5
        avg_bone_2d = avg_bone_slices[view] > 0.5

        for c, img in enumerate((mr_img, gt_img_w, sct_img_w, avg_img_w)):
            axes[r, c].imshow(img, cmap="gray", aspect="equal", interpolation="bilinear")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

        _draw_contour_overlay(
            axes[r, 4],
            base_img_w=gt_img_w,
            gt_mask=gt_bone_2d,
            sct_mask=sct_bone_2d,
            avg_mask=avg_bone_2d,
        )

        axes[r, 0].set_ylabel(view, fontsize=12, fontweight="bold", rotation=90, labelpad=8)

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=12, fontweight="bold", pad=6)

    legend_elements = [
        Line2D([0], [0], color=(1.0, 0.05, 0.05), linewidth=2.0, label="\u00b5CT bone"),
        Line2D([0], [0], color=(0.0, 0.85, 0.10), linewidth=2.0, label="sCT$^{\\mathregular{UNet}}$ bone"),
        Line2D([0], [0], color=(0.10, 0.55, 1.00), linewidth=2.0, label="sCT$^{\\mathregular{avg}}$ bone"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.005),
    )

    # No suptitle — caption is provided in the LaTeX figure environment

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{subject_id}_combined_dice_{ensemble_label}"
    for ext in ("pdf", "png"):
        out_path = out_dir / f"{stem}.{ext}"
        fig.savefig(
            str(out_path),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.12,
            facecolor="white",
        )
        print(f"  wrote {out_path}")
    plt.close(fig)
    return dice_sct, dice_avg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_subject_paths(
    test_root: Path,
    sweep_root: Path,
    ensemble: str,
    subject: str,
    sct_dir: Path | None = None,
) -> Tuple[Path, Path, Path]:
    mr_path = test_root / "MR" / f"{subject}.nii.gz"
    ct_subject = subject.replace("_MR_", "_CT_")
    gt_path = test_root / "CT" / f"{ct_subject}.nii.gz"
    sct_dir = sct_dir or sweep_root / f"ensemble_test_{ensemble}" / "sct"
    sct_path = sct_dir / f"{subject}_sCT.nii.gz"
    for label, p in (("MR", mr_path), ("GT CT", gt_path), ("sCT", sct_path)):
        if not p.exists():
            sys.exit(f"{label} not found: {p}")
    return mr_path, gt_path, sct_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--subject",
        default="MZEI00031_MR_T1",
        help="Test-subject stem matching dataset/Test_Data/MR/<subject>.nii.gz "
        "(default: MZEI00031_MR_T1, the highest-Dice test volume).",
    )
    ap.add_argument(
        "--ensemble",
        default="ssim",
        choices=("ssim", "l1"),
        help="Which K=3 ensemble to take the sCT from (default: ssim, the "
        "loss chosen in the manuscript).",
    )
    ap.add_argument(
        "--avg-ct-dir",
        default=REPO_ROOT / "output" / "average_CT" / "CT",
        type=Path,
        help="Directory of manually aligned training CTs (default: "
        "output/average_CT/CT).",
    )
    ap.add_argument(
        "--avg-ct-pattern",
        default="*_Aligned.nii*",
        help="Glob pattern for the manually aligned training CTs inside "
        "--avg-ct-dir (default: '*_Aligned.nii*' so any raw un-aligned "
        "siblings are skipped).",
    )
    ap.add_argument(
        "--avg-reference-stem",
        default="MZEI00039",
        help="Substring used to pick the resampling-reference volume among "
        "the matched aligned CTs (default: MZEI00039 -- matches the Slicer "
        "'*_to_MZEI00039.tfm' transforms).  Falls back to the first file.",
    )
    ap.add_argument(
        "--test-root",
        default=REPO_ROOT / "dataset" / "Test_Data",
        type=Path,
        help="Root with MR/ and CT/ subdirectories for the held-out test cohort.",
    )
    ap.add_argument(
        "--sweep-root",
        default=REPO_ROOT / "output" / "loss_sweep_final_22",
        type=Path,
        help="Sweep root containing ensemble_test_<loss>/sct/*_sCT.nii.gz.",
    )
    ap.add_argument(
        "--sct-dir",
        default=None,
        type=Path,
        help="Directory with <subject>_sCT.nii.gz; overrides --sweep-root/--ensemble "
        "(e.g. output/test_sct/aug_noN4 for the intensity-augmented model).",
    )
    ap.add_argument(
        "--figure-name",
        default=None,
        help="Additionally save the figure as <out>/<figure-name> (e.g. mouse_dice.png).",
    )
    ap.add_argument(
        "--out",
        default=REPO_ROOT / "manuscript",
        type=Path,
        help="Output directory for the figure (default: manuscript/).",
    )
    ap.add_argument("--threshold", type=float, default=300.0, help="Bone HU threshold.")
    ap.add_argument("--window", type=float, default=2000.0, help="CT display window width (HU).")
    ap.add_argument("--level", type=float, default=500.0, help="CT display window centre (HU).")
    ap.add_argument(
        "--avg-ct-out",
        default=REPO_ROOT / "output" / "average_CT" / "average_ct.nii.gz",
        type=Path,
        help="Where to write the population-average CT NIfTI "
        "(default: output/average_CT/average_ct.nii.gz).",
    )
    ap.add_argument(
        "--rigid-align-to-gt",
        dest="rigid_align_to_gt",
        action="store_true",
        default=True,
        help="Rigid-register the avCT onto the test GT before computing "
        "Dice (matches the manuscript / align_and_dice.py workflow). "
        "This is the default; pass --no-rigid-align-to-gt to disable.",
    )
    ap.add_argument(
        "--no-rigid-align-to-gt",
        dest="rigid_align_to_gt",
        action="store_false",
        help="Skip the rigid avCT -> GT registration; compare the raw "
        "population mean voxel-wise (yields the much lower DSC ~ 0.37 on "
        "MZEI00031 because the avCT is in MZEI00039's frame).",
    )
    ap.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="Max SITK optimizer iterations per resolution level "
        "(only used with --rigid-align-to-gt).",
    )
    ap.add_argument(
        "--bone-mask-threshold",
        type=float,
        default=None,
        help="HU threshold below which both images are clamped for the "
        "AVG -> GT registration metric (only used with --rigid-align-to-gt).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    mr_path, gt_path, sct_path = _resolve_subject_paths(
        args.test_root, args.sweep_root, args.ensemble, args.subject, args.sct_dir
    )
    print(f"Subject : {args.subject}")
    print(f"  MR    : {mr_path}")
    print(f"  GT CT : {gt_path}")
    print(f"  sCT   : {sct_path}  ({args.ensemble} K=3 ensemble)")

    mr_vol = _load_array(mr_path)
    gt_vol = _load_array(gt_path)
    sct_vol = _load_array(sct_path)
    if not (mr_vol.shape == gt_vol.shape == sct_vol.shape):
        sys.exit(
            f"Shape mismatch for {args.subject}: "
            f"MR {mr_vol.shape}, GT {gt_vol.shape}, sCT {sct_vol.shape}"
        )

    # ---- Step 1: build the population avCT and write it to disk first ----
    avg_vol, ref_sitk, ref_path = _compute_average_ct(
        args.avg_ct_dir,
        pattern=args.avg_ct_pattern,
        reference_stem=args.avg_reference_stem,
    )

    avg_out = args.avg_ct_out
    avg_out.parent.mkdir(parents=True, exist_ok=True)
    avg_sitk = numpy_to_sitk(avg_vol, reference=ref_sitk)
    sitk.WriteImage(avg_sitk, str(avg_out))
    print(f"\nWrote average CT NIfTI -> {avg_out}")
    print(f"  (uses the affine of {ref_path.name})")

    # If the avg-CT grid differs from the test subject, resample it onto the
    # subject grid via the identity transform (i.e. let the affines do the
    # work) so Dice / overlay can be computed voxel-wise.
    if avg_vol.shape == gt_vol.shape:
        avg_on_subject = avg_vol
    else:
        print(
            f"  avCT shape {avg_vol.shape} differs from subject {gt_vol.shape}; "
            f"resampling avCT onto subject grid for direct Dice."
        )
        target_sitk = load_sitk(gt_path)
        resampled = sitk.Resample(
            avg_sitk,
            target_sitk,
            sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkLinear,
            -1000.0,
            avg_sitk.GetPixelID(),
        )
        avg_on_subject = sitk_to_numpy(resampled)

    # ---- Step 2: (optional) rigid AVG -> GT, then Dice + figure ----
    if args.rigid_align_to_gt:
        avg_for_dice = _align_avg_to_subject(
            avg_arr=avg_vol,
            ref_template=ref_sitk,
            target_ct_path=gt_path,
            max_iterations=args.max_iterations,
            bone_mask_threshold=args.bone_mask_threshold,
        )
        aligned_out = args.avg_ct_out.with_name(
            args.avg_ct_out.stem.replace(".nii", "")
            + f"_aligned_to_{args.subject}.nii.gz"
        )
        ref_img = nib.load(str(gt_path))
        nib.save(
            nib.Nifti1Image(
                avg_for_dice.astype(np.float32),
                affine=ref_img.affine,
                header=ref_img.header,
            ),
            str(aligned_out),
        )
        print(f"Wrote aligned avCT NIfTI -> {aligned_out}")
    else:
        avg_for_dice = avg_on_subject

    ensemble_label = "SSIM" if args.ensemble == "ssim" else "$\\ell_1$"

    dice_sct, dice_avg = build_combined_figure(
        mr_vol=mr_vol,
        gt_vol=gt_vol,
        sct_vol=sct_vol,
        avg_vol=avg_for_dice,
        subject_id=args.subject,
        ensemble_label=ensemble_label,
        threshold=args.threshold,
        window=args.window,
        level=args.level,
        out_dir=args.out,
    )
    if args.figure_name:
        import shutil

        shutil.copyfile(args.out / f"{args.subject}_combined_dice_{ensemble_label}.png", args.out / args.figure_name)
        print(f"  wrote {args.out / args.figure_name}")

    print("\nSummary")
    print(f"  Subject               : {args.subject}")
    print(f"  Average CT NIfTI      : {avg_out}")
    print(f"  Best sCT DSC          : {dice_sct:.4f}")
    avg_label = "Aligned avCT DSC" if args.rigid_align_to_gt else "avCT DSC"
    print(f"  {avg_label:<21s} : {dice_avg:.4f}")
    print(f"  Threshold             : {int(args.threshold)} HU")
    print(f"  Figure                : {args.out}")


if __name__ == "__main__":
    main()
