#!/usr/bin/env python
"""
Write N4 bias-corrected copies of MR volumes into a sibling folder.

The output folder sits next to the input (``dataset/MR`` -> ``dataset/MR_N4``)
so the dataset still finds the ground-truth CT in the sibling ``CT`` folder.
Point ``data.mri_dir`` / ``data.test_mri_dir`` (or ``--test-mri-dir``) at the
new folders to train and evaluate on corrected MRIs.

    python scripts/precompute_n4.py dataset/MR dataset/Test_Data/MR --workers 8
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simple_sct.data.intensity import n4_correct  # noqa: E402


def _process(args: tuple[Path, Path]) -> str:
    src, dst = args
    if dst.exists():
        return f"skip {src.name}"
    img = nib.load(str(src))
    x = img.get_fdata(dtype=np.float32)
    nib.save(nib.Nifti1Image(n4_correct(x), img.affine, img.header), str(dst))
    return f"done {src.name}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mr_dirs", nargs="+", type=Path)
    ap.add_argument("--suffix", default="_N4", help="Appended to the input folder name")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = []
    for d in args.mr_dirs:
        out = d.parent / f"{d.name}{args.suffix}"
        out.mkdir(exist_ok=True)
        jobs += [(p, out / p.name) for p in sorted(d.glob("*.nii*"))]
    print(f"{len(jobs)} volumes")
    with ProcessPoolExecutor(args.workers) as ex:
        for msg in ex.map(_process, jobs):
            print(" ", msg, flush=True)


if __name__ == "__main__":
    main()
