#!/usr/bin/env python
"""
Inference-only check: how much of the C57BL/6 gap is MR intensity?

Writes harmonised copies of the held-out test MRIs next to the original
``Test_Data/MR`` folder, so ``scripts/ensemble_test.py --test-mri-dir`` can be
pointed at them unchanged (the ground-truth CT is found in the sibling ``CT``
folder):

  MR_hm      histogram of the z-scored volume matched to the pooled BALB/c
             training distribution
  MR_n4hm    N4 bias-field correction, then the same histogram matching

The reference distribution is built from the training MRIs only (test mice
and any ``--exclude`` acquisitions are left out).

    python scripts/quickcheck_harmonize_test_mr.py \
        --train-mr-dir dataset/MR --test-mr-dir dataset/Test_Data/MR \
        --exclude-mice MZEI00027 MZEI00031 MZEI00066 MZEI00069 \
        --exclude-keys MZEI00029_MR_T1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simple_sct.data.intensity import n4_correct  # noqa: E402

N_QUANTILES = 2001


def _stem(p: Path) -> str:
    return p.name.split(".")[0]


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def match_to_reference(x: np.ndarray, ref_q: np.ndarray) -> np.ndarray:
    z = _zscore(x)
    qs = np.linspace(0.0, 1.0, N_QUANTILES)
    src_q = np.quantile(z, qs)
    src_q = np.maximum.accumulate(src_q + np.arange(N_QUANTILES) * 1e-9)
    return np.interp(z, src_q, ref_q).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-mr-dir", type=Path, required=True)
    ap.add_argument("--test-mr-dir", type=Path, required=True)
    ap.add_argument("--exclude-mice", nargs="*", default=[])
    ap.add_argument("--exclude-keys", nargs="*", default=[])
    args = ap.parse_args()

    excl_mice = {m.upper() for m in args.exclude_mice}
    excl_keys = {k.upper() for k in args.exclude_keys}
    train = [p for p in sorted(args.train_mr_dir.glob("*.nii*"))
             if _stem(p).split("_")[0].upper() not in excl_mice
             and _stem(p).upper() not in excl_keys]
    print(f"Reference: {len(train)} training MR volumes")

    qs = np.linspace(0.0, 1.0, N_QUANTILES)
    ref_q = np.mean([np.quantile(_zscore(nib.load(str(p)).get_fdata(dtype=np.float32)), qs)
                     for p in train], axis=0)

    out_hm = args.test_mr_dir.parent / "MR_hm"
    out_n4hm = args.test_mr_dir.parent / "MR_n4hm"
    out_hm.mkdir(exist_ok=True)
    out_n4hm.mkdir(exist_ok=True)
    for p in sorted(args.test_mr_dir.glob("*.nii*")):
        img = nib.load(str(p))
        x = img.get_fdata(dtype=np.float32)
        nib.save(nib.Nifti1Image(match_to_reference(x, ref_q), img.affine, img.header),
                 str(out_hm / p.name))
        nib.save(nib.Nifti1Image(match_to_reference(n4_correct(x), ref_q), img.affine, img.header),
                 str(out_n4hm / p.name))
        print(f"  {p.name}")
    print(f"Wrote {out_hm} and {out_n4hm}")


if __name__ == "__main__":
    main()
