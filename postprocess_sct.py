#!/usr/bin/env python
"""
Post-process predicted sCT NIfTI volumes.
==========================================

Applies sharpening, histogram matching, and bone calibration to make
sCT output look more like ground-truth CT.

Usage (with reference CT -- best quality):

    python postprocess_sct.py \
        --sct  output/sct_nifti/MZEI00069_T9_sCT.nii.gz \
        --ref  dataset/Validation_Data/CT/MZEI00069_CT_T9.nii.gz \
        --out  output/sct_nifti/MZEI00069_T9_sCT_sharp.nii.gz

Usage (sharpen only -- no reference needed):

    python postprocess_sct.py \
        --sct  output/sct_nifti/MZEI00069_T9_sCT.nii.gz \
        --out  output/sct_nifti/MZEI00069_T9_sCT_sharp.nii.gz \
        --sharpen-only

Batch mode (process all sCT files in a directory):

    python postprocess_sct.py \
        --sct-dir  output/sct_nifti/ \
        --ref-dir  dataset/Validation_Data/CT/ \
        --out-dir  output/sct_postprocessed/

Tuning:
    --sharpen-sigma 1.0    Gaussian blur sigma (larger = coarser sharpening)
    --sharpen-alpha 0.4    Sharpening strength (0 = none, 1 = very aggressive)
    --bone-thr 200         HU threshold for bone calibration
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None

from simple_sct.postprocess import postprocess_sct, unsharp_mask_3d


def _load(path: Path):
    if nib is None:
        raise ImportError("nibabel is required.")
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.affine, img.header


def _save(path: Path, vol: np.ndarray, affine, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    ni = nib.Nifti1Image(vol.astype(np.float32), affine=affine, header=header)
    nib.save(ni, str(path))


def _pair_sct_ref(sct_dir: Path, ref_dir: Path):
    """Pair sCT files with reference CTs by normalised stem."""
    sct_files = sorted(list(sct_dir.glob("*.nii.gz")) + list(sct_dir.glob("*.nii")))
    ref_files = sorted(list(ref_dir.glob("*.nii.gz")) + list(ref_dir.glob("*.nii")))

    def _key(p: Path) -> str:
        stem = p.name.split(".")[0].upper()
        for tag in ("_SCT", "_CT", "_MR"):
            if stem.endswith(tag):
                stem = stem[: -len(tag)]
            stem = stem.replace(tag + "_", "_")
        return stem

    ref_map = {_key(p): p for p in ref_files}
    pairs = []
    for sp in sct_files:
        k = _key(sp)
        pairs.append((sp, ref_map.get(k), k))
    return pairs


def process_single(
    sct_path: Path,
    out_path: Path,
    ref_path: Optional[Path] = None,
    sharpen_only: bool = False,
    sigma: float = 1.0,
    alpha: float = 0.4,
    bone_thr: float = 200.0,
    hu_min: float = -1024.0,
    hu_max: float = 5000.0,
) -> None:
    sct_np, affine, header = _load(sct_path)

    if sharpen_only or ref_path is None:
        result = unsharp_mask_3d(sct_np, sigma=sigma, alpha=alpha,
                                 hu_min=hu_min, hu_max=hu_max)
    else:
        ref_np, _, _ = _load(ref_path)
        result = postprocess_sct(
            sct_np, reference=ref_np,
            sharpen=True, sharpen_sigma=sigma, sharpen_alpha=alpha,
            match_histogram=True, calibrate_bone=True,
            bone_thr=bone_thr, hu_min=hu_min, hu_max=hu_max,
        )

    _save(out_path, result, affine, header)
    print(f"  [OK] {out_path.name}")


def main():
    ap = argparse.ArgumentParser(
        description="Post-process predicted sCT volumes (sharpen + histogram match + bone calibrate)."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sct", type=Path, help="Single sCT NIfTI file")
    g.add_argument("--sct-dir", type=Path, help="Directory of sCT NIfTI files (batch mode)")

    ap.add_argument("--ref", type=Path, default=None,
                    help="Reference CT NIfTI (single mode)")
    ap.add_argument("--ref-dir", type=Path, default=None,
                    help="Directory of reference CT NIfTI files (batch mode)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output NIfTI path (single mode)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (batch mode)")
    ap.add_argument("--sharpen-only", action="store_true",
                    help="Only apply unsharp mask (no reference needed)")
    ap.add_argument("--sharpen-sigma", type=float, default=1.0,
                    help="Gaussian sigma for unsharp mask (default: 1.0)")
    ap.add_argument("--sharpen-alpha", type=float, default=0.4,
                    help="Sharpening strength 0-1 (default: 0.4)")
    ap.add_argument("--bone-thr", type=float, default=200.0,
                    help="Bone HU threshold for calibration (default: 200)")
    ap.add_argument("--hu-min", type=float, default=-1024.0)
    ap.add_argument("--hu-max", type=float, default=5000.0)
    args = ap.parse_args()

    if args.sct:
        if not args.sct.exists():
            print(f"Error: {args.sct} not found"); sys.exit(1)
        out_path = args.out or args.sct.parent / (args.sct.name.replace(".nii", "_pp.nii"))
        ref = args.ref
        if ref and not ref.exists():
            print(f"Error: reference {ref} not found"); sys.exit(1)
        if not args.sharpen_only and ref is None:
            print("Warning: no --ref provided, applying sharpen-only mode.")

        process_single(
            args.sct, out_path, ref_path=ref,
            sharpen_only=args.sharpen_only,
            sigma=args.sharpen_sigma, alpha=args.sharpen_alpha,
            bone_thr=args.bone_thr,
            hu_min=args.hu_min, hu_max=args.hu_max,
        )
    else:
        sct_dir = args.sct_dir
        if not sct_dir.exists():
            print(f"Error: {sct_dir} not found"); sys.exit(1)
        out_dir = args.out_dir or sct_dir.parent / (sct_dir.name + "_postprocessed")
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.ref_dir and args.ref_dir.exists():
            pairs = _pair_sct_ref(sct_dir, args.ref_dir)
        else:
            sct_files = sorted(
                list(sct_dir.glob("*.nii.gz")) + list(sct_dir.glob("*.nii"))
            )
            pairs = [(f, None, f.stem) for f in sct_files]
            if not args.sharpen_only:
                print("Warning: no --ref-dir provided, applying sharpen-only mode.")

        print(f"Processing {len(pairs)} files -> {out_dir}")
        for sct_path, ref_path, key in pairs:
            out_path = out_dir / sct_path.name
            process_single(
                sct_path, out_path, ref_path=ref_path,
                sharpen_only=args.sharpen_only,
                sigma=args.sharpen_sigma, alpha=args.sharpen_alpha,
                bone_thr=args.bone_thr,
                hu_min=args.hu_min, hu_max=args.hu_max,
            )


if __name__ == "__main__":
    main()
