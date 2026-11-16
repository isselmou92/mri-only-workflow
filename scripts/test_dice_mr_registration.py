#!/usr/bin/env python
"""
sCT^avg Dice: µCT-registration vs MR-registration comparison
=============================================================
Evaluates bone Dice for the population-average CT (sCT^avg) under two
registration targets:

    * avCT -> µCT  (direct CT-to-CT, used in the paper evaluation)
    * avCT -> MR   (cross-modality, mirrors the actual MRI-only workflow)

Both registrations use mutual-information-based rigid alignment (Euler3D,
6 DOF) as implemented in evaluate_alignment.rigid_register.  After either
registration the Dice is computed against the ground-truth µCT so the two
numbers are directly comparable.

Because the µCT and MR are already co-registered (CT resampled to MR
frame-of-reference during preprocessing), both registrations land in the
same voxel space and Dice can be computed voxel-wise.

Run
---
    python scripts/test_dice_mr_registration.py
    python scripts/test_dice_mr_registration.py --threshold 400
    python scripts/test_dice_mr_registration.py --out manuscript/dice_mr_vs_ct_reg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_alignment import (  # noqa: E402
    dice_score,
    load_sitk,
    rigid_register,
    sitk_to_numpy,
    numpy_to_sitk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_array(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)


def _discover_subjects(test_root: Path) -> List[str]:
    """All subjects that have both MR and GT µCT on disk."""
    mr_dir = test_root / "MR"
    ct_dir = test_root / "CT"
    subjects: List[str] = []
    for mr in sorted(mr_dir.glob("*.nii.gz")):
        stem = mr.name.removesuffix(".nii.gz")
        gt = ct_dir / f"{stem.replace('_MR_', '_CT_')}.nii.gz"
        if gt.exists():
            subjects.append(stem)
    return subjects


def _align_avg(
    avg_sitk: sitk.Image,
    fixed_path: Path,
    max_iterations: int,
    label: str,
) -> np.ndarray:
    """Register avg_sitk onto fixed_path; return numpy array in fixed space."""
    fixed = load_sitk(fixed_path)
    print(f"    rigid avCT -> {label} ({fixed_path.name}) ...", end="", flush=True)
    resampled, transform, _ = rigid_register(
        fixed=fixed,
        moving=avg_sitk,
        max_iterations=max_iterations,
        bone_mask_threshold=None,
        verbose=False,
    )
    if hasattr(transform, "GetParameters"):
        params = transform.GetParameters()
        if len(params) >= 6:
            print(
                f"  rot=({params[0]:+.3f},{params[1]:+.3f},{params[2]:+.3f})"
                f"  trans=({params[3]:+.2f},{params[4]:+.2f},{params[5]:+.2f})"
            )
        else:
            print()
    else:
        print()
    return sitk_to_numpy(resampled)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_markdown(rows: List[Dict], threshold: float) -> str:
    header = (
        f"| Subject | avCT->µCT DSC | avCT->MR DSC | Delta (MR - µCT) |\n"
        f"|---|---:|---:|---:|"
    )
    lines = [header]
    for r in rows:
        delta = r["mr"] - r["ct"]
        lines.append(
            f"| {r['subject']} "
            f"| {r['ct']:.3f} "
            f"| {r['mr']:.3f} "
            f"| {delta:+.3f} |"
        )
    ct_arr  = np.array([r["ct"] for r in rows])
    mr_arr  = np.array([r["mr"] for r in rows])
    d_arr   = mr_arr - ct_arr
    lines.append(
        f"| **mean +/- sd (n={len(rows)})** "
        f"| **{ct_arr.mean():.3f} +/- {ct_arr.std(ddof=1):.3f}** "
        f"| **{mr_arr.mean():.3f} +/- {mr_arr.std(ddof=1):.3f}** "
        f"| **{d_arr.mean():+.3f} +/- {d_arr.std(ddof=1):.3f}** |"
    )
    return (
        f"Bone Dice (HU > {int(threshold)}): avCT registration target comparison\n\n"
        + "\n".join(lines)
    )


def _save_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("subject,avct_ct_reg_dsc,avct_mr_reg_dsc,delta\n")
        for r in rows:
            fh.write(
                f"{r['subject']},{r['ct']:.6f},{r['mr']:.6f},{r['mr']-r['ct']:+.6f}\n"
            )


def _save_plot(rows: List[Dict], path: Path, threshold: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )

    subjects = [r["subject"].replace("_MR_", " ") for r in rows]
    ct_vals  = [r["ct"] for r in rows]
    mr_vals  = [r["mr"] for r in rows]
    x = np.arange(len(subjects))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(7.5, 0.8 * len(subjects) + 2.5), 4.5))
    ax.bar(x - w / 2, ct_vals, width=w, label=r"sCT$^{\mathrm{avg}}$ $\rightarrow$ $\mu$CT reg.",
           color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, mr_vals, width=w, label=r"sCT$^{\mathrm{avg}}$ $\rightarrow$ MR reg.",
           color="#ff7f0e", edgecolor="black", linewidth=0.5)

    ax.set_ylabel(f"Bone Dice (HU > {int(threshold)})")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=30, ha="right", fontsize=9)
    ax.set_title(
        r"sCT$^{\mathrm{avg}}$ Dice: $\mu$CT-registration vs. MR-registration"
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, ncol=2, fontsize=10)

    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--test-root",
        default=REPO_ROOT / "dataset" / "Test_Data",
        type=Path,
    )
    ap.add_argument(
        "--avg-ct",
        default=REPO_ROOT / "output" / "average_CT" / "average_ct.nii.gz",
        type=Path,
        help="Population avCT NIfTI.",
    )
    ap.add_argument(
        "--out",
        default=REPO_ROOT / "manuscript" / "dice_mr_vs_ct_reg",
        type=Path,
        help="Output stem (CSV + PNG/PDF written here).",
    )
    ap.add_argument("--threshold", type=float, default=300.0)
    ap.add_argument("--max-iterations", type=int, default=500)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not args.avg_ct.exists():
        sys.exit(
            f"avCT NIfTI not found: {args.avg_ct}\n"
            f"Create it first with `python scripts/make_combined_dice_figure.py`."
        )

    subjects = _discover_subjects(args.test_root)
    if not subjects:
        sys.exit(f"No subjects found under {args.test_root}")

    print(
        f"Comparing avCT registration targets (HU > {int(args.threshold)}) "
        f"on {len(subjects)} subjects.\n"
        f"  avCT: {args.avg_ct}\n"
    )

    avg_sitk = load_sitk(args.avg_ct)

    rows: List[Dict] = []
    for i, stem in enumerate(subjects, 1):
        gt_name  = stem.replace("_MR_", "_CT_")
        gt_path  = args.test_root / "CT"  / f"{gt_name}.nii.gz"
        mr_path  = args.test_root / "MR"  / f"{stem}.nii.gz"

        print(f"[{i:2d}/{len(subjects)}] {stem}")

        # --- avCT registered to µCT (paper evaluation) ---
        avg_on_ct = _align_avg(avg_sitk, gt_path, args.max_iterations, label="µCT")
        gt_vol    = _load_array(gt_path)
        d_ct      = dice_score(avg_on_ct, gt_vol, args.threshold)

        # --- avCT registered to MR (actual MRI-only workflow) ---
        # After MR registration the result lands in MR space.
        # MR and µCT share the same frame-of-reference (µCT was resampled to
        # MR space during preprocessing), so Dice vs gt_vol is valid directly.
        avg_on_mr = _align_avg(avg_sitk, mr_path, args.max_iterations, label="MR")
        d_mr      = dice_score(avg_on_mr, gt_vol, args.threshold)

        print(
            f"    DSC: avCT->µCT = {d_ct:.3f}   avCT->MR = {d_mr:.3f}"
            f"   delta = {d_mr - d_ct:+.3f}\n"
        )
        rows.append({"subject": stem, "ct": float(d_ct), "mr": float(d_mr)})

    table = _format_markdown(rows, args.threshold)
    print("\n" + table + "\n")

    csv_path = args.out.with_suffix(".csv")
    _save_csv(rows, csv_path)
    print(f"CSV  -> {csv_path}")

    _save_plot(rows, args.out, args.threshold)
    print(f"PNG  -> {args.out.with_suffix('.png')}")
    print(f"PDF  -> {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
