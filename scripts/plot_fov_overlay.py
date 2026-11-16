"""Overlay the micro-CT field of view (FOV) on the MR in three planes for several test acquisitions."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("fov", REPO_ROOT / "simple_sct" / "metrics" / "fov.py")
_fov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fov)

OUTSIDE = (0.85, 0.15, 0.15)
EDGE = "#FFD400"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stems", nargs="+",
                    default=["MZEI00027_MR_T1", "MZEI00031_MR_T1", "MZEI00066_MR_T1", "MZEI00069_MR_T1"])
    ap.add_argument("--test-root", type=Path, default=REPO_ROOT / "dataset" / "Test_Data")
    ap.add_argument("--strain-map", type=Path, default=REPO_ROOT / "configs" / "test_strain_map.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "fov_eval" / "fov_overlay_T1")
    args = ap.parse_args()

    strains = {k.upper(): v for k, v in json.loads(args.strain_map.read_text()).items()}
    tint = ListedColormap([(0, 0, 0, 0), (*OUTSIDE, 0.45)])
    planes = ["Sagittal", "Coronal", "Axial"]

    fig, axes = plt.subplots(len(args.stems), 3, figsize=(9.6, 3.05 * len(args.stems)),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.16})
    axes = np.atleast_2d(axes)
    for r, stem in enumerate(args.stems):
        mr = nib.load(str(args.test_root / "MR" / f"{stem}.nii.gz")).get_fdata(dtype=np.float32)
        ct = nib.load(str(args.test_root / "CT" / f"{stem.replace('_MR_', '_CT_')}.nii.gz")).get_fdata(dtype=np.float32)
        fov = _fov.ct_fov_mask(ct)
        # Boundary line drawn on the largest component only; the tint shows the exact mask.
        labels, n = ndimage.label(fov)
        box = labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1) if n > 1 else fov
        box = ndimage.binary_fill_holes(ndimage.binary_opening(box, iterations=2))
        lo, hi = np.percentile(mr, [1, 99.5])
        c = [s // 2 for s in mr.shape]
        slices = [lambda a: a[c[0]], lambda a: a[:, c[1]], lambda a: a[:, :, c[2]]]
        for j, sl in enumerate(slices):
            ax = axes[r, j]
            ax.imshow(sl(mr).T, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="bilinear")
            ax.imshow(sl(~fov).T, cmap=tint, vmin=0, vmax=1, origin="lower", interpolation="nearest")
            b2, n2 = ndimage.label(ndimage.binary_opening(sl(box), iterations=2))
            b2 = b2 == (np.argmax(np.bincount(b2.ravel())[1:]) + 1) if n2 > 1 else b2 > 0
            m = np.pad(ndimage.binary_fill_holes(b2).T.astype(float), 1)
            ax.contour(np.arange(-1, m.shape[1] - 1), np.arange(-1, m.shape[0] - 1), m,
                       levels=[0.5], colors=EDGE, linewidths=1.3)
            ax.set_xlim(-0.5, sl(mr).shape[0] - 0.5)
            ax.set_ylim(-0.5, sl(mr).shape[1] - 0.5)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.text(0.02, 0.97, f"{100 * sl(fov).mean():.0f}% in FOV", transform=ax.transAxes, ha="left", va="top",
                    fontsize=8, color="white", bbox={"boxstyle": "round,pad=0.25", "fc": "black", "alpha": 0.55, "lw": 0})
            if r == 0:
                ax.set_title(planes[j], fontsize=11)
        mouse, _, tp = stem.split("_")
        axes[r, 0].set_ylabel(f"{mouse}  {tp}\n{strains[mouse]}  ·  FOV {100 * fov.mean():.0f}% of volume",
                              fontsize=9.5, labelpad=6)
        axes[r, 0].yaxis.set_visible(True)

    fig.legend(handles=[Patch(facecolor=(0.55, 0.12, 0.12), label="Outside CT FOV (excluded, −990 HU fill)"),
                        Line2D([], [], color=EDGE, lw=1.6, label="CT FOV boundary (rotated crop box)")],
               loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.96, bottom=0.05)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out}.png/.pdf")


if __name__ == "__main__":
    main()
