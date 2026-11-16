"""Visualise how the CT field-of-view mask is derived for one test acquisition."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("fov", REPO_ROOT / "simple_sct" / "metrics" / "fov.py")
_fov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fov)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", default="MZEI00069_MR_T2")
    ap.add_argument("--test-root", type=Path, default=REPO_ROOT / "dataset" / "Test_Data")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "fov_eval" / "fov_mask_example.png")
    args = ap.parse_args()

    ct = nib.load(str(args.test_root / "CT" / f"{args.stem.replace('_MR_', '_CT_')}.nii.gz")).get_fdata(dtype=np.float32)
    mr = nib.load(str(args.test_root / "MR" / f"{args.stem}.nii.gz")).get_fdata(dtype=np.float32)
    fill = np.abs(ct - _fov.CT_FILL_HU) < 0.5
    fov = _fov.ct_fov_mask(ct)
    inner_fill = fill & fov

    c = [s // 2 for s in ct.shape]
    planes = [("axis 0", lambda a: a[c[0]]), ("axis 1", lambda a: a[:, c[1]]), ("axis 2", lambda a: a[:, :, c[2]])]
    mr_lo, mr_hi = np.percentile(mr, [1, 99.5])
    red = ListedColormap([(0, 0, 0, 0), (0.9, 0.15, 0.15, 0.45)])
    blue = ListedColormap([(0, 0, 0, 0), (0.1, 0.5, 1.0, 0.9)])

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.5))
    titles = ["1  Ground-truth CT", "2  Voxels at exactly -990 HU (fill value)",
              "3  CT field of view on MR\n(red = excluded, outline = mask)"]
    for r, (pname, sl) in enumerate(planes):
        ax = axes[r, 0]
        ax.imshow(sl(ct).T, cmap="gray", vmin=-1000, vmax=2000, origin="lower")
        ax = axes[r, 1]
        ax.imshow(sl(ct).T, cmap="gray", vmin=-1000, vmax=2000, origin="lower")
        ax.imshow(sl(fill & ~fov).T, cmap=red, vmin=0, vmax=1, origin="lower", interpolation="nearest")
        ax.imshow(sl(inner_fill).T, cmap=blue, vmin=0, vmax=1, origin="lower", interpolation="nearest")
        ax = axes[r, 2]
        ax.imshow(sl(mr).T, cmap="gray", vmin=mr_lo, vmax=mr_hi, origin="lower")
        ax.imshow(sl(~fov).T, cmap=red, vmin=0, vmax=1, origin="lower", interpolation="nearest")
        ax.contour(sl(fov).T.astype(float), levels=[0.5], colors="yellow", linewidths=1.2, origin="lower")
        axes[r, 0].set_ylabel(f"central slice, {pname}", fontsize=10)
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{args.stem}: CT field of view = {fov.mean():.0%} of the volume, "
                 f"{inner_fill.sum()} fill-valued voxels inside the box kept (blue)", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"FOV fraction {fov.mean():.3f}; border fill {(fill & ~fov).sum()}; inner fill kept {inner_fill.sum()}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
