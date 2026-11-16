#!/usr/bin/env python
"""
Bone geometry errors in millimetres, BALB/c vs C57BL/6 test animals.

For every held-out test acquisition, the bone mask (CT > 300 HU) of each
candidate is compared with the ground-truth micro-CT:

  * sCT U-Net, original pipeline       (--baseline-sct-dir)
  * sCT U-Net, intensity-augmented     (--aug-sct-dir)
  * sCT avg, rigidly aligned average CT (--avg-ct, aligned as in test_dice_table.py)

Metrics: bone DSC, symmetric mean surface distance (MSD) and 95th-percentile
Hausdorff distance (HD95) in mm. The skull-shape difference between strains is
taken from output/strain_pairwise (scripts/strain_pairwise.py; pairwise cranial bone
surface distance after rigid alignment) and shown on the same mm scale.

    python scripts/strain_mm_analysis.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_alignment import load_sitk, rigid_register, sitk_to_numpy  # noqa: E402

BONE_HU = 300.0
METHODS = [("baseline", "U-Net (original)", "#9e9e9e"),
           ("augmented", "U-Net (intensity aug.)", "#1f77b4"),
           ("avg_ct", "Average CT (aligned)", "#ff7f0e")]


def surface_distances(a: np.ndarray, b: np.ndarray, spacing: tuple) -> np.ndarray:
    """Distances (mm) from every surface voxel of a to the surface of b and vice versa."""
    sa = a & ~ndimage.binary_erosion(a)
    sb = b & ~ndimage.binary_erosion(b)
    da = ndimage.distance_transform_edt(~sb, sampling=spacing)[sa]
    db = ndimage.distance_transform_edt(~sa, sampling=spacing)[sb]
    return np.concatenate([da, db])


def bone_metrics(pred: np.ndarray, gt: np.ndarray, spacing: tuple) -> dict:
    a, b = pred > BONE_HU, gt > BONE_HU
    d = surface_distances(a, b, spacing)
    return {"dsc": 2.0 * (a & b).sum() / (a.sum() + b.sum()),
            "msd_mm": float(d.mean()), "hd95_mm": float(np.percentile(d, 95))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-root", type=Path, default=REPO_ROOT / "dataset" / "Test_Data")
    ap.add_argument("--baseline-sct-dir", type=Path, default=REPO_ROOT / "output" / "test_sct" / "loss_sweep_final_22")
    ap.add_argument("--aug-sct-dir", type=Path, default=REPO_ROOT / "output" / "test_sct" / "aug_noN4")
    ap.add_argument("--avg-ct", type=Path, default=REPO_ROOT / "output" / "average_CT" / "average_ct.nii.gz")
    ap.add_argument("--pairwise-dir", type=Path,
                    default=REPO_ROOT / "output" / "strain_pairwise")
    ap.add_argument("--strain-map", type=Path, default=REPO_ROOT / "configs" / "test_strain_map.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "strain_mm_analysis")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    strains = {k.upper(): v for k, v in json.loads(args.strain_map.read_text()).items()}
    avg = load_sitk(args.avg_ct)
    rows = []
    for sct in sorted(args.aug_sct_dir.glob("*_sCT.nii.gz")):
        stem = sct.name.removesuffix("_sCT.nii.gz")
        gt_path = args.test_root / "CT" / f"{stem.replace('_MR_', '_CT_')}.nii.gz"
        gt_img = nib.load(str(gt_path))
        gt = gt_img.get_fdata(dtype=np.float32)
        spacing = tuple(float(s) for s in gt_img.header.get_zooms()[:3])
        aligned, _, _ = rigid_register(fixed=load_sitk(gt_path), moving=avg, max_iterations=300,
                                       bone_mask_threshold=None, verbose=False)
        preds = {"baseline": nib.load(str(args.baseline_sct_dir / sct.name)).get_fdata(dtype=np.float32),
                 "augmented": nib.load(str(sct)).get_fdata(dtype=np.float32),
                 "avg_ct": sitk_to_numpy(aligned)}
        for key, pred in preds.items():
            rows.append({"acquisition": stem, "strain": strains[stem.split("_")[0]], "method": key,
                         **bone_metrics(pred, gt, spacing)})
        print(f"  {stem}: " + "  ".join(
            f"{r['method']} MSD={r['msd_mm']:.3f} mm DSC={r['dsc']:.3f}" for r in rows[-3:]))

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "bone_mm_per_acquisition.csv", index=False)
    summary = (df.groupby(["strain", "method"])[["dsc", "msd_mm", "hd95_mm"]]
                 .agg(["mean", "std", "min", "max"]).round(3))
    summary.to_csv(args.out / "bone_mm_summary.csv")

    dm = pd.read_csv(args.pairwise_dir / "distance_matrix_rigid.csv", index_col=0)
    c57 = [m for m in dm.index if strains.get(m) == "C57BL/6"]
    balb = [m for m in dm.index if m not in c57]
    iu = np.triu_indices(len(balb), 1)
    within = dm.loc[balb, balb].to_numpy()[iu]
    between = dm.loc[balb, c57].to_numpy().ravel()
    voxel = 0.14

    # Figure 1: model errors per strain
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, label in [(axes[0], "msd_mm", "Mean bone surface distance [mm]"),
                              (axes[1], "dsc", "Bone DSC")]:
        for si, strain in enumerate(["BALB/c", "C57BL/6"]):
            for mi, (key, name, color) in enumerate(METHODS):
                v = df[(df.strain == strain) & (df.method == key)][metric].to_numpy()
                x = si + (mi - 1) * 0.25
                ax.bar(x, v.mean(), 0.22, color=color, alpha=0.85, label=name if si == 0 else None)
                ax.scatter(np.full(len(v), x) + np.linspace(-0.06, 0.06, len(v)), v,
                           s=12, color="k", zorder=3)
        ax.set_xticks([0, 1], ["BALB/c\n(trained strain)", "C57BL/6\n(unseen strain)"])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    axes[0].axhline(voxel, ls="--", lw=1, color="k")
    axes[0].text(0.5, voxel + 0.005, "1 voxel (0.14 mm)", va="bottom", ha="center", fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Bone accuracy on the held-out test animals (3 acquisitions each)")
    fig.tight_layout()
    fig.savefig(args.out / "fig_bone_accuracy_mm.png", dpi=200)
    fig.savefig(args.out / "fig_bone_accuracy_mm.pdf")

    # Figure 2: anatomy difference vs prediction error on one mm scale
    fig, ax = plt.subplots(figsize=(7.5, 4))
    groups = [("Skull shape\nBALB/c vs BALB/c", within, "#bdbdbd"),
              ("Skull shape\nBALB/c vs C57BL/6", between, "#8c564b")]
    for key, name, color in METHODS:
        groups.append((name.replace(" (", "\n("), df[(df.strain == "C57BL/6") & (df.method == key)]["msd_mm"].to_numpy(), color))
    for i, (name, v, color) in enumerate(groups):
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v,
                   s=14, color=color, alpha=0.8)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color="k", lw=2)
    ax.axvline(1.5, color="k", lw=0.8)
    ax.text(0.5, 0.97, "Anatomy (pairwise, after rigid alignment)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8)
    ax.text(3.0, 0.97, "Prediction error on C57BL/6", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8)
    ax.axhline(voxel, ls="--", lw=1, color="k")
    ax.text(len(groups) - 0.5, voxel, "1 voxel (0.14 mm)", va="bottom", ha="right", fontsize=8)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
    ax.set_ylabel("Mean bone surface distance [mm]")
    ax.set_ylim(0, None)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "fig_anatomy_vs_error_mm.png", dpi=200)
    fig.savefig(args.out / "fig_anatomy_vs_error_mm.pdf")

    anatomy = {"within_balbc_median_mm": float(np.median(within)),
               "within_balbc_range_mm": [float(within.min()), float(within.max())],
               "between_strain_median_mm": float(np.median(between)),
               "between_strain_range_mm": [float(between.min()), float(between.max())]}
    (args.out / "anatomy_mm.json").write_text(json.dumps(anatomy, indent=2))
    print(summary.to_string())
    print(json.dumps(anatomy, indent=2))
    print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
