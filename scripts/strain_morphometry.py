#!/usr/bin/env python
"""
Inter-strain Skull Morphometry -- BALB/c vs C57BL/6
===================================================
Quantifies whether the skull geometry of a C57BL/6 mouse can be matched to a
BALB/c reference by rigid alignment alone, or whether scaling / non-linear
deformation is required. The same pipeline is applied to the remaining BALB/c
animals to provide the within-strain reference range.

For every moving CT (registered onto the BALB/c reference CT):
  1. Registration hierarchy, each initialised from the previous stage:
       rigid (6 dof) -> similarity (7 dof) -> affine (12 dof) -> B-spline
  2. Bone agreement after each stage (Dice, mean surface distance, HD95),
     evaluated only inside the overlapping field of view, both for the full
     FOV and for a cranial ROI that excludes the posture-dependent caudal
     craniocervical region (--roi-z-min-mm)
  3. Affine decomposition: stretch along each image axis, volume ratio,
     anisotropy (ratio of largest to smallest principal stretch)
  4. Residual non-affine displacement of the B-spline stage on bone
  5. Slice-wise skull width / height / bone area profiles in the reference
     frame after rigid alignment (ratio moving / reference)

Usage:
    python scripts/strain_morphometry.py \
        --ref dataset/CT/MZEI00023_CT_T1.nii.gz \
        --ref-mr dataset/MR/MZEI00023_MR_T1.nii.gz \
        --target dataset/CT/MZEI00069_CT_T7.nii.gz \
        --target-mr dataset/MR/MZEI00069_MR_T7.nii.gz \
        --ct-dir dataset/CT \
        --balbc MZEI00018 MZEI00019 ... \
        --c57-timepoints dataset/CT/MZEI00069_CT_T*.nii.gz \
        --out output/strain_morphometry
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_alignment import load_sitk, sitk_to_numpy, rigid_register  # noqa: E402


BONE_HU = 300.0
STAGES = ["rigid", "similarity", "affine", "bspline"]
STAGE_LABELS = {
    "rigid": "Rigid\n(6 dof)",
    "similarity": "Similarity\n(7 dof)",
    "affine": "Affine\n(12 dof)",
    "bspline": "B-spline\n(non-linear)",
}
AXIS_NAMES = ["Lateral (x)", "Dorso-ventral (y)", "Rostro-caudal (z)"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _clamp(img: sitk.Image) -> sitk.Image:
    return sitk.Clamp(img, sitk.sitkFloat32, -1000.0, 3000.0)


def _linear_registration(fixed: sitk.Image, moving: sitk.Image,
                         initial: sitk.Transform, max_iterations: int) -> sitk.Transform:
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.25, seed=42)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-6, numberOfIterations=max_iterations,
        gradientMagnitudeTolerance=1e-8, relaxationFactor=0.5,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)
    out = reg.Execute(fixed, moving)
    if isinstance(out, sitk.CompositeTransform) and out.GetNumberOfTransforms() == 1:
        out = out.GetNthTransform(0)
    return out


def _bspline_registration(fixed: sitk.Image, moving: sitk.Image,
                          affine: sitk.Transform, mesh: int) -> sitk.Transform:
    bspline = sitk.BSplineTransformInitializer(fixed, [mesh] * 3, order=3)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.05, seed=42)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                             numberOfIterations=50,
                             maximumNumberOfCorrections=5,
                             maximumNumberOfFunctionEvaluations=500,
                             costFunctionConvergenceFactor=1e7)
    reg.SetShrinkFactorsPerLevel([2, 1])
    reg.SetSmoothingSigmasPerLevel([1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetMovingInitialTransform(affine)
    reg.SetInitialTransform(bspline, inPlace=True)
    reg.Execute(fixed, moving)
    return bspline


def register_hierarchy(fixed: sitk.Image, moving: sitk.Image,
                       max_iterations: int, mesh: int) -> Dict[str, sitk.Transform]:
    fixed_c, moving_c = _clamp(fixed), _clamp(moving)

    _, rigid, _ = rigid_register(fixed_c, moving_c, max_iterations=max_iterations,
                                 verbose=False)
    rigid = sitk.Euler3DTransform(rigid)

    sim0 = sitk.Similarity3DTransform()
    sim0.SetCenter(rigid.GetCenter())
    sim0.SetMatrix(rigid.GetMatrix())
    sim0.SetTranslation(rigid.GetTranslation())
    similarity = sitk.Similarity3DTransform(
        _linear_registration(fixed_c, moving_c, sim0, max_iterations))

    aff0 = sitk.AffineTransform(3)
    aff0.SetCenter(similarity.GetCenter())
    aff0.SetMatrix(similarity.GetMatrix())
    aff0.SetTranslation(similarity.GetTranslation())
    affine = sitk.AffineTransform(
        _linear_registration(fixed_c, moving_c, aff0, max_iterations))

    bspline = _bspline_registration(fixed_c, moving_c, affine, mesh)
    return {
        "rigid": rigid,
        "similarity": similarity,
        "affine": affine,
        "bspline": sitk.CompositeTransform([affine, bspline]),
        "_bspline_only": bspline,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask)


def bone_agreement(fixed_np: np.ndarray, moving_np: np.ndarray,
                   overlap: np.ndarray, spacing: float) -> Dict[str, float]:
    a = (fixed_np > BONE_HU) & overlap
    b = (moving_np > BONE_HU) & overlap
    dice = 2.0 * np.sum(a & b) / max(np.sum(a) + np.sum(b), 1)

    sa, sb = _surface(a), _surface(b)
    d_to_b = ndimage.distance_transform_edt(~sb, sampling=spacing)[sa]
    d_to_a = ndimage.distance_transform_edt(~sa, sampling=spacing)[sb]
    d = np.concatenate([d_to_b, d_to_a])
    return {
        "dice": float(dice),
        "msd_mm": float(d.mean()),
        "hd95_mm": float(np.percentile(d, 95)),
    }


def affine_decomposition(affine: sitk.AffineTransform) -> Dict[str, float]:
    """Stretches of the fixed->moving matrix (i.e. moving size relative to fixed)."""
    A = np.asarray(affine.GetMatrix()).reshape(3, 3)
    axis_stretch = np.linalg.norm(A, axis=0)
    principal = np.linalg.svd(A, compute_uv=False)
    return {
        "stretch_x": float(axis_stretch[0]),
        "stretch_y": float(axis_stretch[1]),
        "stretch_z": float(axis_stretch[2]),
        "volume_ratio": float(np.linalg.det(A)),
        "anisotropy": float(principal.max() / principal.min()),
    }


def slice_profiles(fixed_np: np.ndarray, moving_np: np.ndarray, overlap: np.ndarray,
                   spacing: float, min_voxels: int = 30) -> Dict[str, np.ndarray]:
    """Per rostro-caudal slice: lateral width, dorso-ventral height, bone area."""
    nz = fixed_np.shape[2]
    out = {k: np.full((2, nz), np.nan) for k in ["width", "height", "area"]}
    for vi, vol in enumerate([fixed_np, moving_np]):
        bone = (vol > BONE_HU) & overlap
        for z in range(nz):
            xs, ys = np.nonzero(bone[:, :, z])
            if xs.size < min_voxels:
                continue
            out["width"][vi, z] = (np.percentile(xs, 98) - np.percentile(xs, 2)) * spacing
            out["height"][vi, z] = (np.percentile(ys, 98) - np.percentile(ys, 2)) * spacing
            out["area"][vi, z] = xs.size * spacing ** 2
    return out


# ---------------------------------------------------------------------------
# Per-subject worker
# ---------------------------------------------------------------------------

def cranial_roi(shape: Tuple[int, int, int], spacing: float, z_min_mm: float) -> np.ndarray:
    """Excludes the caudal craniocervical region (cervical spine, posture dependent)."""
    roi = np.zeros(shape, dtype=bool)
    roi[:, :, int(round(z_min_mm / spacing)):] = True
    return roi


def analyse_subject(ref_path: str, moving_path: str, max_iterations: int,
                    mesh: int, keep_volumes: bool, roi_z_min_mm: float) -> Dict:
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(3)
    fixed = load_sitk(Path(ref_path))
    moving = load_sitk(Path(moving_path))
    spacing = float(fixed.GetSpacing()[0])
    fixed_np = sitk_to_numpy(fixed)
    roi = cranial_roi(fixed_np.shape, spacing, roi_z_min_mm)

    tx = register_hierarchy(fixed, moving, max_iterations, mesh)

    ones = sitk.Image(moving.GetSize(), sitk.sitkFloat32) + 1.0
    ones.CopyInformation(moving)

    result: Dict = {"moving": Path(moving_path).name.replace(".nii.gz", "")}
    volumes: Dict[str, np.ndarray] = {}
    for stage in STAGES:
        res = sitk_to_numpy(sitk.Resample(moving, fixed, tx[stage], sitk.sitkLinear, -1000.0))
        overlap = sitk_to_numpy(sitk.Resample(ones, fixed, tx[stage],
                                              sitk.sitkNearestNeighbor, 0.0)) > 0.5
        for k, v in bone_agreement(fixed_np, res, overlap, spacing).items():
            result[f"{stage}_{k}"] = v
        for k, v in bone_agreement(fixed_np, res, overlap & roi, spacing).items():
            result[f"cranial_{stage}_{k}"] = v
        if stage == "rigid":
            result["profiles"] = {k: v.tolist() for k, v in
                                  slice_profiles(fixed_np, res, overlap, spacing).items()}
        if keep_volumes:
            volumes[stage] = res
            volumes[f"{stage}_overlap"] = overlap

    result.update(affine_decomposition(tx["affine"]))
    result["similarity_scale"] = float(tx["similarity"].GetScale())

    field = sitk.TransformToDisplacementField(
        tx["_bspline_only"], sitk.sitkVectorFloat64, fixed.GetSize(),
        fixed.GetOrigin(), fixed.GetSpacing(), fixed.GetDirection())
    disp = np.linalg.norm(sitk.GetArrayFromImage(field), axis=-1).transpose(2, 1, 0)
    bone_ref = fixed_np > BONE_HU
    result["nonaffine_disp_bone_mean_mm"] = float(disp[bone_ref].mean())
    result["nonaffine_disp_bone_p95_mm"] = float(np.percentile(disp[bone_ref], 95))
    result["cranial_nonaffine_disp_bone_mean_mm"] = float(disp[bone_ref & roi].mean())

    if keep_volumes:
        total = sitk.TransformToDisplacementField(
            tx["bspline"], sitk.sitkVectorFloat64, fixed.GetSize(),
            fixed.GetOrigin(), fixed.GetSpacing(), fixed.GetDirection())
        volumes["log_jacobian"] = np.log2(np.clip(sitk_to_numpy(
            sitk.DisplacementFieldJacobianDeterminant(total)), 1e-3, None))
        volumes["nonaffine_disp"] = disp.astype(np.float32)
        result["_transforms"] = {s: tx[s] for s in ["rigid", "similarity", "affine"]}
        result["_volumes"] = volumes
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "mathtext.fontset": "dejavuserif",
    })


def _views(vol: np.ndarray, idx: Tuple[int, int, int]) -> List[np.ndarray]:
    """Anatomical cross-section (xy), sagittal (yz), horizontal (xz)."""
    return [vol[:, :, idx[2]].T, vol[idx[0], :, :].T, vol[:, idx[1], :].T]


VIEW_NAMES = ["Cross-section", "Sagittal", "Horizontal"]


def _wl(a, lo=-1000.0, hi=2500.0):
    return np.clip((a - lo) / (hi - lo), 0, 1)


def _contour_rgba(mask: np.ndarray, color) -> np.ndarray:
    c = _surface(mask)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[c] = [*color, 1.0]
    return rgba


def _save(fig, out: Path, name: str):
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"  Figure saved: {out / name}.[png|pdf]")


def figure_alignment_stages(fixed_np, main: Dict, idx, out: Path, ref_id: str, tgt_id: str):
    vols = main["_volumes"]
    fig, axes = plt.subplots(3, 4, figsize=(7.4, 6.0),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.06})
    red, green = (0.95, 0.1, 0.1), (0.1, 0.9, 0.2)
    for c, stage in enumerate(STAGES):
        overlap = vols[f"{stage}_overlap"]
        f_bone = (fixed_np > BONE_HU) & overlap
        m_bone = (vols[stage] > BONE_HU) & overlap
        for r, (bg, fb, mb) in enumerate(zip(_views(fixed_np, idx), _views(f_bone, idx),
                                             _views(m_bone, idx))):
            ax = axes[r, c]
            ax.imshow(_wl(bg), cmap="gray", origin="lower", interpolation="bilinear")
            ax.imshow(_contour_rgba(fb, red), origin="lower", interpolation="nearest")
            ax.imshow(_contour_rgba(mb, green), origin="lower", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(VIEW_NAMES[r], fontweight="bold")
        axes[0, c].set_title(
            f"{STAGE_LABELS[stage].replace(chr(10), ' ')}\n"
            f"Dice {main[f'{stage}_dice']:.3f} | MSD {main[f'{stage}_msd_mm']:.2f} mm",
            fontsize=8.5)
    fig.legend(handles=[Line2D([0], [0], color=red, lw=1.5, label=f"BALB/c bone ({ref_id})"),
                        Line2D([0], [0], color=green, lw=1.5, label=f"C57BL/6 bone ({tgt_id})")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))
    _save(fig, out, "fig1_alignment_stages")


def figure_mr_overlay(ref_mr_np, tgt_mr_rigid_np, fixed_np, rigid_ct_np, overlap, idx,
                      out: Path, ref_id: str, tgt_id: str):
    def norm(a):
        lo, hi = np.percentile(a[a > 0], [1, 99.5]) if np.any(a > 0) else (0, 1)
        return np.clip((a - lo) / (hi - lo), 0, 1)

    ref_n, tgt_n = norm(ref_mr_np), norm(tgt_mr_rigid_np) * overlap
    fig, axes = plt.subplots(3, 3, figsize=(6.0, 6.2),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.06})
    cols = [f"BALB/c\n{ref_id}", f"C57BL/6 (rigidly aligned)\n{tgt_id}",
            "Fusion\nBALB/c magenta, C57BL/6 green"]
    f_bone, m_bone = (fixed_np > BONE_HU) & overlap, (rigid_ct_np > BONE_HU) & overlap
    for r, (a, b, fb, mb) in enumerate(zip(_views(ref_n, idx), _views(tgt_n, idx),
                                           _views(f_bone, idx), _views(m_bone, idx))):
        axes[r, 0].imshow(a, cmap="gray", origin="lower")
        axes[r, 0].imshow(_contour_rgba(fb, (0.95, 0.1, 0.1)), origin="lower")
        axes[r, 1].imshow(b, cmap="gray", origin="lower")
        axes[r, 1].imshow(_contour_rgba(mb, (0.1, 0.9, 0.2)), origin="lower")
        axes[r, 2].imshow(np.stack([a, b, a], axis=-1), origin="lower")
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(VIEW_NAMES[r], fontweight="bold")
    for c, t in enumerate(cols):
        axes[0, c].set_title(t, fontsize=8.5)
    _save(fig, out, "fig2_mr_rigid_fusion")


def figure_jacobian(fixed_np, main: Dict, idx, out: Path):
    vols = main["_volumes"]
    head = ndimage.binary_fill_holes(ndimage.binary_closing(fixed_np > -300, iterations=3))
    head &= vols["bspline_overlap"]
    lj = np.where(head, vols["log_jacobian"], np.nan)
    disp = np.where(head, vols["nonaffine_disp"], np.nan)
    bone = fixed_np > BONE_HU

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.12})
    norm = TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)
    for c, (bg, j, d, b) in enumerate(zip(_views(fixed_np, idx), _views(lj, idx),
                                          _views(disp, idx), _views(bone, idx))):
        ax = axes[0, c]
        ax.imshow(_wl(bg), cmap="gray", origin="lower")
        im1 = ax.imshow(j, cmap="RdBu_r", norm=norm, origin="lower", alpha=0.75)
        ax.imshow(_contour_rgba(b, (0, 0, 0)), origin="lower")
        ax = axes[1, c]
        ax.imshow(_wl(bg), cmap="gray", origin="lower")
        im2 = ax.imshow(d, cmap="magma", vmin=0, vmax=np.nanpercentile(disp, 99),
                        origin="lower", alpha=0.8)
        ax.imshow(_contour_rgba(b, (0.3, 0.9, 1.0)), origin="lower")
        for r in range(2):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[0, c].set_title(VIEW_NAMES[c], fontweight="bold")
    axes[0, 0].set_ylabel("Local volume ratio\nC57BL/6 / BALB/c", fontsize=8.5)
    axes[1, 0].set_ylabel("Residual displacement\nbeyond affine", fontsize=8.5)
    cb = fig.colorbar(im1, ax=axes[0, :], fraction=0.025, pad=0.01)
    cb.set_ticks([-0.585, -0.263, 0, 0.263, 0.585])
    cb.set_ticklabels(["0.67", "0.83", "1.0", "1.2", "1.5"])
    cb2 = fig.colorbar(im2, ax=axes[1, :], fraction=0.025, pad=0.01)
    cb2.set_label("mm")
    _save(fig, out, "fig3_deformation_maps")


def figure_quantitative(balb: List[Dict], c57: List[Dict], main: Dict, other: List[Dict],
                        out: Path, prefix: str = "", name: str = "fig4_quantitative",
                        region: str = "full field of view"):
    fig = plt.figure(figsize=(7.4, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.42)
    rng = np.random.default_rng(0)

    def strip(ax, key_fn, xs, labels, ylabel, ref_line=None):
        for i, x in enumerate(xs):
            bv = np.array([key_fn(r, x) for r in balb])
            cv = np.array([key_fn(r, x) for r in c57])
            ax.scatter(i - 0.15 + rng.uniform(-0.05, 0.05, bv.size), bv, s=10,
                       color="tab:blue", alpha=0.7, lw=0)
            ax.errorbar(i - 0.15, bv.mean(), yerr=bv.std(ddof=1), fmt="_", color="navy",
                        capsize=3, ms=10)
            ax.scatter(i + 0.15 + rng.uniform(-0.05, 0.05, cv.size), cv, s=10,
                       color="tab:orange", alpha=0.6, lw=0)
            ax.scatter(i + 0.15, key_fn(main, x), s=40, marker="*", color="darkred", zorder=5)
            for o in other:
                ax.scatter(i, key_fn(o, x), s=18, marker="D", facecolor="none",
                           edgecolor="gray", zorder=4)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ylabel)
        if ref_line is not None:
            ax.axhline(ref_line, color="k", lw=0.6, ls=":")
        ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[0, 0:2])
    strip(ax, lambda r, s: r[f"{prefix}{s}_dice"], STAGES, [STAGE_LABELS[s] for s in STAGES],
          "Bone Dice to BALB/c reference")
    ax.set_title(f"(a) Bone overlap after each registration stage ({region})",
                 loc="left", fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    strip(ax, lambda r, s: r[f"{prefix}{s}_msd_mm"], ["rigid", "affine"],
          ["Rigid", "Affine"], "Mean surface distance (mm)")
    ax.set_title("(b) Skull surface distance", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[1, 0:2])
    strip(ax, lambda r, k: r[k], ["stretch_x", "stretch_y", "stretch_z", "volume_ratio"],
          ["Lateral", "Dorso-\nventral", "Rostro-\ncaudal", "Volume\n(det A)"],
          "Affine scale vs. BALB/c ref.", ref_line=1.0)
    ax.set_title("(c) Affine scaling per axis (1 = same size)", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[1, 2])
    strip(ax, lambda r, k: r[k], ["anisotropy", f"{prefix}nonaffine_disp_bone_mean_mm"],
          ["Anisotropy", "Non-affine\ndisp. (mm)"], "", ref_line=1.0)
    ax.set_title("(d) Shape beyond scaling", loc="left", fontsize=9)

    handles = [
        Line2D([0], [0], marker="o", ls="", color="tab:blue", label=f"BALB/c (n={len(balb)} mice)"),
        Line2D([0], [0], marker="o", ls="", color="tab:orange",
               label=f"C57BL/6, distinct CT sessions (n={len(c57)})"),
        Line2D([0], [0], marker="*", ls="", color="darkred", ms=9, label="C57BL/6 analysed scan"),
    ]
    if other:
        handles.append(Line2D([0], [0], marker="D", ls="", mfc="none", mec="gray",
                              label=", ".join(o["moving"] for o in other) + " (strain to confirm)"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=8)
    _save(fig, out, name)


def figure_profiles(balb: List[Dict], main: Dict, spacing: float, out: Path,
                    roi_z_min_mm: float):
    keys = [("width", "Lateral skull width"), ("height", "Dorso-ventral skull height"),
            ("area", "Bone cross-sectional area")]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), sharex=True,
                             gridspec_kw={"hspace": 0.12, "wspace": 0.35})

    def smooth(v, k=5):
        m = np.isfinite(v)
        num = np.convolve(np.where(m, v, 0), np.ones(k), "same")
        den = np.convolve(m.astype(float), np.ones(k), "same")
        return np.where(den > k / 2, num / np.maximum(den, 1), np.nan)

    for c, (k, title) in enumerate(keys):
        p = np.asarray(main["profiles"][k])
        z = np.arange(p.shape[1]) * spacing
        ratios_balb = np.array([smooth(np.asarray(r["profiles"][k])[1]) /
                                smooth(np.asarray(r["profiles"][k])[0]) for r in balb])
        lo, hi = np.nanpercentile(ratios_balb, [5, 95], axis=0)
        unit = "mm²" if k == "area" else "mm"
        axes[0, c].plot(z, smooth(p[0]), color="tab:red", label="BALB/c ref.")
        axes[0, c].plot(z, smooth(p[1]), color="tab:green", label="C57BL/6 (rigid)")
        axes[0, c].set_title(title, fontsize=9)
        axes[0, c].set_ylabel(unit)
        axes[1, c].fill_between(z, lo, hi, color="tab:blue", alpha=0.2,
                                label="BALB/c 5–95 %")
        axes[1, c].plot(z, smooth(p[1]) / smooth(p[0]), color="k", label="C57BL/6 / BALB/c")
        axes[1, c].axhline(1.0, color="k", lw=0.6, ls=":")
        axes[1, c].set_xlabel("Caudal → rostral (mm)")
        axes[1, c].set_ylabel("Ratio")
        for r in range(2):
            axes[r, c].grid(alpha=0.3)
            axes[r, c].axvspan(0, roi_z_min_mm, color="0.85", zorder=0)
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[1, 0].legend(fontsize=7, frameon=False)
    axes[0, 0].text(roi_z_min_mm / 2, axes[0, 0].get_ylim()[0], "cranio-\ncervical",
                    ha="center", va="bottom", fontsize=6.5, color="0.4")
    _save(fig, out, "fig5_slice_profiles")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _scalar_row(r: Dict) -> Dict:
    return {k: v for k, v in r.items() if not k.startswith("_") and k != "profiles"}


def _zscore(value: float, ref: np.ndarray) -> float:
    sd = ref.std(ddof=1)
    return float((value - ref.mean()) / sd) if sd > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description="BALB/c vs C57BL/6 skull morphometry")
    ap.add_argument("--ref", type=Path, required=True, help="BALB/c reference CT")
    ap.add_argument("--ref-mr", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True, help="C57BL/6 CT")
    ap.add_argument("--target-mr", type=Path, required=True)
    ap.add_argument("--ct-dir", type=Path, required=True)
    ap.add_argument("--balbc", nargs="+", required=True,
                    help="BALB/c subject IDs used as within-strain reference (T1 scan)")
    ap.add_argument("--other", nargs="*", default=[],
                    help="Subject IDs shown separately (strain unconfirmed)")
    ap.add_argument("--c57-timepoints", nargs="*", default=[],
                    help="Further CTs of the C57BL/6 animal (repeatability)")
    ap.add_argument("--out", type=Path, default=Path("output/strain_morphometry"))
    ap.add_argument("--max-iterations", type=int, default=300)
    ap.add_argument("--mesh", type=int, default=5, help="B-spline control mesh size")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--roi-z-min-mm", type=float, default=5.0,
                    help="Cranial ROI starts this far rostral of the caudal FOV edge")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ref_id = args.ref.name.split("_")[0]
    tgt_id = args.target.name.replace(".nii.gz", "")

    def t1(sid):
        return str(args.ct_dir / f"{sid}_CT_T1.nii.gz")

    balb_paths = [t1(s) for s in args.balbc if s != ref_id and Path(t1(s)).exists()]
    other_paths = [t1(s) for s in args.other if Path(t1(s)).exists()]

    # Several time points share an identical CT file; keep one per distinct scan.
    def content_hash(p: Path) -> str:
        return hashlib.md5(sitk.GetArrayFromImage(load_sitk(p)).tobytes()).hexdigest()

    seen = {content_hash(args.target)}
    c57_paths = []
    for p in sorted({str(Path(p)) for pat in args.c57_timepoints for p in glob.glob(pat)},
                    key=lambda p: int(re.search(r"_T(\d+)", p).group(1))):
        h = content_hash(Path(p))
        if h not in seen:
            seen.add(h)
            c57_paths.append(p)

    print(f"  Reference (BALB/c)  : {args.ref}")
    print(f"  Target (C57BL/6)    : {args.target}")
    print(f"  BALB/c references   : {len(balb_paths)}")
    print(f"  C57BL/6 other CTs   : {[Path(p).name for p in c57_paths]}")

    main_res = analyse_subject(str(args.ref), str(args.target), args.max_iterations,
                               args.mesh, True, args.roi_z_min_mm)
    print(f"  Main pair done: rigid Dice {main_res['rigid_dice']:.3f}, "
          f"affine Dice {main_res['affine_dice']:.3f}")

    jobs = balb_paths + other_paths + c57_paths
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(analyse_subject, str(args.ref), p, args.max_iterations,
                             args.mesh, False, args.roi_z_min_mm) for p in jobs]
        results = []
        for p, f in zip(jobs, futures):
            results.append(f.result())
            print(f"    done: {Path(p).name}  rigid Dice {results[-1]['rigid_dice']:.3f}")
    balb = results[:len(balb_paths)]
    other = results[len(balb_paths):len(balb_paths) + len(other_paths)]
    c57 = [main_res] + results[len(balb_paths) + len(other_paths):]

    with open(args.out / "morphometry_results.csv", "w", newline="", encoding="utf-8") as f:
        rows = ([{"group": "C57BL/6 (analysed)", **_scalar_row(main_res)}]
                + [{"group": "BALB/c", **_scalar_row(r)} for r in balb]
                + [{"group": "strain unconfirmed", **_scalar_row(r)} for r in other]
                + [{"group": "C57BL/6 other CT session", **_scalar_row(r)} for r in c57[1:]])
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    stage_keys = [f"{s}_{m}" for s in STAGES for m in ["dice", "msd_mm", "hd95_mm"]]
    summary_keys = (stage_keys + [f"cranial_{k}" for k in stage_keys]
                    + ["similarity_scale", "stretch_x", "stretch_y", "stretch_z",
                       "volume_ratio", "anisotropy", "nonaffine_disp_bone_mean_mm",
                       "cranial_nonaffine_disp_bone_mean_mm"])
    summary = {}
    print(f"\n  {'metric':<36s} {'BALB/c mean±SD':>18s} {'BALB/c range':>15s} "
          f"{'C57 ' + tgt_id:>20s} {'C57 sessions':>16s} {'z':>6s}")
    for k in summary_keys:
        bv = np.array([r[k] for r in balb])
        cv = np.array([r[k] for r in c57])
        summary[k] = {"balbc_mean": bv.mean(), "balbc_sd": bv.std(ddof=1),
                      "balbc_min": bv.min(), "balbc_max": bv.max(),
                      "c57_main": main_res[k], "c57_tp_mean": cv.mean(),
                      "c57_tp_sd": cv.std(ddof=1) if cv.size > 1 else 0.0,
                      "z_main": _zscore(main_res[k], bv)}
        for o in other:
            summary[k][o["moving"]] = o[k]
        print(f"  {k:<36s} {bv.mean():>9.3f} ± {bv.std(ddof=1):<6.3f} "
              f"{bv.min():>6.3f}–{bv.max():<6.3f} {main_res[k]:>20.3f} "
              f"{cv.min():>7.3f}–{cv.max():<7.3f} {summary[k]['z_main']:>6.1f}")
    for prefix in ["", "cranial_"]:
        gain = {g: np.array([r[f"{prefix}affine_dice"] - r[f"{prefix}rigid_dice"] for r in g_res])
                for g, g_res in [("balb", balb), ("c57", c57)]}
        summary[f"{prefix}affine_minus_rigid_dice"] = {
            "balbc_mean": gain["balb"].mean(), "balbc_sd": gain["balb"].std(ddof=1),
            "c57_main": main_res[f"{prefix}affine_dice"] - main_res[f"{prefix}rigid_dice"],
            "c57_tp_mean": gain["c57"].mean()}
    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    _style()
    fixed_np = sitk_to_numpy(load_sitk(args.ref))
    bone = fixed_np > BONE_HU
    com = ndimage.center_of_mass(bone)
    idx = (int(round(com[0])), int(round(com[1])), int(round(com[2])))

    figure_alignment_stages(fixed_np, main_res, idx, args.out, ref_id, tgt_id)

    ref_mr = load_sitk(args.ref_mr)
    tgt_mr_rigid = sitk_to_numpy(sitk.Resample(load_sitk(args.target_mr), ref_mr,
                                               main_res["_transforms"]["rigid"],
                                               sitk.sitkLinear, 0.0))
    figure_mr_overlay(sitk_to_numpy(ref_mr), tgt_mr_rigid, fixed_np,
                      main_res["_volumes"]["rigid"], main_res["_volumes"]["rigid_overlap"],
                      idx, args.out, ref_id, args.target_mr.name.replace(".nii.gz", ""))
    figure_jacobian(fixed_np, main_res, idx, args.out)
    figure_quantitative(balb, c57, main_res, other, args.out, prefix="cranial_",
                        name="fig4_quantitative_cranial", region="cranial ROI")
    figure_quantitative(balb, c57, main_res, other, args.out, prefix="",
                        name="fig4b_quantitative_full_fov", region="full FOV")
    figure_profiles(balb, main_res, float(load_sitk(args.ref).GetSpacing()[0]), args.out,
                    args.roi_z_min_mm)
    print(f"\n  Results: {args.out}")


if __name__ == "__main__":
    main()
