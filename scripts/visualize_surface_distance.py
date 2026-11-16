#!/usr/bin/env python
"""
Illustrates how the bone surface distances of strain_mm_analysis.py are computed,
for one test acquisition, on coronal slices.

  1. Bone masks: CT > 300 HU in the ground-truth micro-CT and in the sCT.
  2. Surfaces: mask voxels that disappear after one binary erosion.
  3. For every surface voxel of one mask, the Euclidean distance (in mm, 3D)
     to the nearest surface voxel of the other mask, in both directions.
  4. MSD = mean of all these distances, HD95 = their 95th percentile.

    python scripts/visualize_surface_distance.py --subject MZEI00069_MR_T1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent
BONE_HU = 300.0


def coronal(vol: np.ndarray, y: int) -> np.ndarray:
    return vol[:, y, :].T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="MZEI00069_MR_T1")
    ap.add_argument("--sct-dir", type=Path, default=REPO_ROOT / "output" / "test_sct" / "aug_noN4")
    ap.add_argument("--label", default="U-Net (intensity aug.)")
    ap.add_argument("--test-root", type=Path, default=REPO_ROOT / "dataset" / "Test_Data")
    ap.add_argument("--slices", type=float, nargs="+", default=[0.40, 0.50, 0.60],
                    help="Coronal slice positions as fractions of the volume")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "strain_mm_analysis")
    args = ap.parse_args()

    gt_img = nib.load(str(args.test_root / "CT" / f"{args.subject.replace('_MR_', '_CT_')}.nii.gz"))
    gt = gt_img.get_fdata(dtype=np.float32)
    pred = nib.load(str(args.sct_dir / f"{args.subject}_sCT.nii.gz")).get_fdata(dtype=np.float32)
    spacing = tuple(float(s) for s in gt_img.header.get_zooms()[:3])

    a, b = gt > BONE_HU, pred > BONE_HU
    sa = a & ~ndimage.binary_erosion(a)
    sb = b & ~ndimage.binary_erosion(b)
    dist_to_pred = ndimage.distance_transform_edt(~sb, sampling=spacing)
    dist_to_gt = ndimage.distance_transform_edt(~sa, sampling=spacing)
    d_all = np.concatenate([dist_to_pred[sa], dist_to_gt[sb]])
    msd, hd95 = float(d_all.mean()), float(np.percentile(d_all, 95))

    mr = nib.load(str(args.test_root / "MR" / f"{args.subject}.nii.gz")).get_fdata(dtype=np.float32)
    mr_hi = float(np.percentile(mr, 99.5))

    ys = [int(f * gt.shape[1]) for f in args.slices]
    vmax = 1.0
    fig = plt.figure(figsize=(19, 3.9 * len(ys)))
    gs = fig.add_gridspec(len(ys), 5, width_ratios=[1, 1, 1, 1.15, 1.5])
    for r, y in enumerate(ys):
        ct, sc = coronal(gt, y), coronal(pred, y)
        ma, mb = coronal(a, y), coronal(b, y)

        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(coronal(mr, y), cmap="gray", vmin=0, vmax=mr_hi)
        ax.contour(ma, [0.5], colors="red", linewidths=0.6)
        ax.set_title("MRI (input) with micro-CT bone" if r == 0 else "")
        ax.set_ylabel(f"coronal slice {y}")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[r, 1])
        ax.imshow(ct, cmap="gray", vmin=-1000, vmax=2000)
        ax.contour(ma, [0.5], colors="red", linewidths=0.8)
        ax.set_title(f"micro-CT, bone > {BONE_HU:.0f} HU" if r == 0 else "")

        ax = fig.add_subplot(gs[r, 2])
        ax.imshow(sc, cmap="gray", vmin=-1000, vmax=2000)
        ax.contour(mb, [0.5], colors="deepskyblue", linewidths=0.8)
        ax.set_title(f"sCT {args.label}" if r == 0 else "")

        ax = fig.add_subplot(gs[r, 3])
        ax.imshow(ct, cmap="gray", vmin=-1000, vmax=2000, alpha=0.6)
        ys_a, xs_a = np.nonzero(coronal(sa, y))
        ys_b, xs_b = np.nonzero(coronal(sb, y))
        sc_plot = ax.scatter(np.r_[xs_a, xs_b], np.r_[ys_a, ys_b], s=3, marker="s",
                             c=np.r_[coronal(dist_to_pred, y)[ys_a, xs_a], coronal(dist_to_gt, y)[ys_b, xs_b]],
                             cmap="turbo", vmin=0, vmax=vmax)
        ax.set_xlim(0, ct.shape[1] - 1)
        ax.set_ylim(ct.shape[0] - 1, 0)
        ax.set_title("surface voxels coloured by\n3D distance to the other surface" if r == 0 else "")
        for axx in fig.axes[-3:]:
            axx.set_xticks([])
            axx.set_yticks([])
        if r == 0:
            cb = fig.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("distance [mm]")

    ax = fig.add_subplot(gs[:, 4])
    ax.hist(d_all, bins=np.arange(0, max(2.0, hd95 * 1.3), 0.07), color="0.6")
    ax.axvline(msd, color="k", lw=2)
    ax.axvline(hd95, color="k", lw=2, ls="--")
    ax.text(msd, ax.get_ylim()[1] * 0.95, f"  MSD = {msd:.2f} mm", va="top")
    ax.text(hd95, ax.get_ylim()[1] * 0.85, f"  HD95 = {hd95:.2f} mm", va="top")
    ax.set_xlabel("distance between bone surfaces [mm]\n(all surface voxels of the whole volume, both directions)")
    ax.set_ylabel("number of surface voxels")
    ax.set_title(f"{args.subject}, {args.label}\nall distances in the 3D volume")
    fig.legend(handles=[Line2D([], [], color="red", label="micro-CT bone surface"),
                        Line2D([], [], color="deepskyblue", label="sCT bone surface")],
               loc="lower left", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stem = f"fig_surface_distance_{args.subject}_{args.sct_dir.name}"
    fig.savefig(args.out / f"{stem}.png", dpi=170)
    print(f"{args.subject}: MSD={msd:.3f} mm  HD95={hd95:.3f} mm  voxel={spacing}  "
          f"surface voxels GT={int(sa.sum())} sCT={int(sb.sum())}")
    print(f"Wrote {args.out / (stem + '.png')}")


if __name__ == "__main__":
    main()
