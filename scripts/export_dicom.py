#!/usr/bin/env python
"""
Export GT / MR / sCT / avCT DICOM series for the held-out test cohort.
======================================================================
Drives ``pipeline.NiftiSeriesMeta`` + ``pipeline.export_ray_orientation`` to
write four RayStation-friendly DICOM series per subject, all sharing the
same StudyInstanceUID, FrameOfReferenceUID and PatientID so they line up in
any viewer.

Layout mirrors ``output/sCT_newmodel_v3/<subject>/dicom/``:

    <out-root>/<subject_stem>/
        nifti_intermediate/
            avCT_aligned_to_<subject>.nii.gz   # avCT rigidly aligned to GT
        dicom/
            GT_rayOrientation_slices/    CT_0001.dcm ...
            MR_rayOrientation_slices/    MR_0001.dcm ...
            sCT_rayOrientation_slices/   CT_0001.dcm ...
            avCT_rayOrientation_slices/  CT_0001.dcm ...

Defaults assume:
  * The sCT NIfTI was produced by the SSIM K=3 ensemble and sits in
    ``output/loss_sweep_final_22/ensemble_test_ssim/sct/``.
  * The population avCT lives at ``output/average_CT/average_ct.nii.gz``.
  * A reference DICOM (for geometry) is available at
    ``dataset/reference.dcm`` (copied from the historical MRIOnlyWorkflow
    project).

Run
---
    # Default subject (MZEI00031_MR_T1), SSIM ensemble:
    python scripts/export_dicom.py

    # Explicit subject + L1 ensemble + custom output root:
    python scripts/export_dicom.py \
        --subject MZEI00027_MR_T1 \
        --ensemble l1 \
        --out output/dicom_export

    # Skip the avCT -> GT rigid alignment (raw population frame):
    python scripts/export_dicom.py --no-avg-align
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pydicom

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import (  # noqa: E402
    NiftiSeriesMeta,
    SITK_AVAILABLE,
    export_ray_orientation,
    extract_mouse_id,
    rigid_align,
)


# ---------------------------------------------------------------------------
# Path resolution
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--subject",
        default="MZEI00031_MR_T1",
        help="Test-subject stem matching dataset/Test_Data/MR/<subject>.nii.gz "
        "(default: MZEI00031_MR_T1).",
    )
    ap.add_argument(
        "--ensemble",
        default="ssim",
        choices=("ssim", "l1"),
        help="Which K=3 ensemble's sCT to export (default: ssim).",
    )
    ap.add_argument(
        "--test-root",
        default=REPO_ROOT / "dataset" / "Test_Data",
        type=Path,
        help="Test-cohort root containing MR/ and CT/ subfolders.",
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
        help="Folder with <subject>_sCT.nii.gz; overrides --sweep-root/--ensemble.",
    )
    ap.add_argument(
        "--sct-description",
        default="Synthetic CT",
        help="SeriesDescription of the sCT series (max. 64 characters).",
    )
    ap.add_argument(
        "--avg-ct",
        default=REPO_ROOT / "output" / "average_CT" / "average_ct.nii.gz",
        type=Path,
        help="Population avCT NIfTI (write it first with "
        "scripts/make_combined_dice_figure.py).",
    )
    ap.add_argument(
        "--ref",
        default=REPO_ROOT / "dataset" / "reference.dcm",
        type=Path,
        help="Reference DICOM for geometry tags (default: dataset/reference.dcm).",
    )
    ap.add_argument(
        "--out",
        default=REPO_ROOT / "output" / "dicom_export",
        type=Path,
        help="Output root; per-subject folders are created beneath it.",
    )
    ap.add_argument(
        "--no-avg-align",
        action="store_true",
        help="Skip the rigid avCT -> GT alignment step; export the raw "
        "population avCT NIfTI as-is (it will sit in the training frame, "
        "not the subject frame).",
    )
    ap.add_argument(
        "--align-iterations",
        type=int,
        default=500,
        help="Max SITK optimizer iterations for the avCT -> GT registration "
        "(default: 500).",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.ref.exists():
        sys.exit(f"Reference DICOM not found: {args.ref}")
    if not args.avg_ct.exists():
        sys.exit(
            f"Average CT NIfTI not found: {args.avg_ct}\n"
            f"Create it first with `python scripts/make_combined_dice_figure.py`."
        )

    mr_path, gt_path, sct_path = _resolve_subject_paths(
        args.test_root, args.sweep_root, args.ensemble, args.subject, args.sct_dir
    )

    print("=" * 64)
    print(f"  Subject : {args.subject}  (ensemble: {args.ensemble})")
    print(f"  MR      : {mr_path}")
    print(f"  GT CT   : {gt_path}")
    print(f"  sCT     : {sct_path}")
    print(f"  avCT    : {args.avg_ct}")
    print(f"  Ref DCM : {args.ref}")
    print("=" * 64)

    subject_dir = args.out / args.subject
    nifti_dir = subject_dir / "nifti_intermediate"
    dicom_dir = subject_dir / "dicom"
    nifti_dir.mkdir(parents=True, exist_ok=True)
    dicom_dir.mkdir(parents=True, exist_ok=True)

    # ---- avCT -> GT rigid alignment (optional) ----
    if args.no_avg_align:
        print("\n[--no-avg-align] Exporting avCT NIfTI as-is, no rigid step.")
        avg_export_path = args.avg_ct
    else:
        if not SITK_AVAILABLE:
            sys.exit(
                "SimpleITK is required for the avCT -> GT alignment. "
                "Re-install it or pass --no-avg-align."
            )
        avg_export_path = nifti_dir / f"avCT_aligned_to_{args.subject}.nii.gz"
        print("\n  Rigidly aligning avCT -> GT...")
        rigid_align(
            fixed_path=gt_path,
            moving_path=args.avg_ct,
            output_path=avg_export_path,
            max_iterations=args.align_iterations,
            verbose=True,
        )
        print(f"  Saved aligned avCT NIfTI: {avg_export_path}")

    # ---- Shared study-level UIDs ----
    # Include the timepoint so each acquisition becomes a separate patient
    # in RayStation (e.g. "MZEI00027_T1" rather than just "MZEI00027").
    _parts = args.subject.split("_")  # e.g. ["MZEI00027", "MR", "T1"]
    patient_id = f"{_parts[0]}_{_parts[-1]}" if len(_parts) >= 3 else extract_mouse_id(args.subject)
    study_uid = pydicom.uid.generate_uid()
    frame_of_reference_uid = pydicom.uid.generate_uid()
    print(f"\n  PatientID         : {patient_id}")
    print(f"  StudyInstanceUID  : {study_uid}")
    print(f"  FrameOfRefUID     : {frame_of_reference_uid}")

    # ---- Series exports ----
    series = [
        ("GT", gt_path, "CT", "Ground truth CT", False),
        ("MR", mr_path, "MR", "MR", True),
        ("sCT", sct_path, "CT", args.sct_description, False),
        ("avCT", avg_export_path, "CT", "Average CT (population)", False),
    ]

    for prefix, nifti_path, modality, descr, is_mr in series:
        print(f"\n  Exporting {prefix} series ({modality}) from {nifti_path.name}")
        meta = NiftiSeriesMeta(
            nifti_path=nifti_path,
            reference_dicom=args.ref,
            patient_id=patient_id,
            study_uid=study_uid,
            frame_of_reference_uid=frame_of_reference_uid,
            is_mr=is_mr,
        )
        out_dir = export_ray_orientation(
            meta,
            dicom_dir,
            series_prefix=prefix,
            modality=modality,
            series_description=descr,
        )
        n = sum(1 for _ in out_dir.glob("*.dcm"))
        print(f"    -> {out_dir}  ({n} slices)")

    print("\n" + "=" * 64)
    print(f"  Done. Open this folder in RayStation: {dicom_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
