#!/usr/bin/env python
"""
Bone-Dice scoreboard across the full Test_Data cohort.
======================================================
For every test subject that has both a ground-truth CT and an ensemble sCT
on disk, compute the bone Dice coefficient (HU > 300) of:

    * sCT (SSIM K=3 ensemble)  vs GT
    * sCT (L1   K=3 ensemble)  vs GT
    * rigidly aligned population-average CT (avCT) vs GT

For the aligned avCT, MAE / RMSE / PSNR in HU are reported as well, using the
same convention as scripts/ensemble_test.py for the sCT (prediction clamped to
[-1000, 3000] HU, raw GT, whole volume, PSNR data range 4000 HU).

The avCT is rigid-registered to each subject independently (matches the
manuscript / supplementary protocol).  Results are printed as a markdown
table, dropped into a CSV next to the figures, and plotted as a grouped bar
chart.

Run
---
    python scripts/test_dice_table.py
    python scripts/test_dice_table.py --threshold 400 --out manuscript/dice_test_cohort
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_alignment import (  # noqa: E402
    dice_score,
    load_sitk,
    rigid_register,
    sitk_to_numpy,
)

import SimpleITK as sitk  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_array(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)


def _discover_subjects(
    test_root: Path,
    sweep_root: Path,
    ensembles: Tuple[str, ...],
) -> List[str]:
    """All subjects with MR + GT + sCT in *every* requested ensemble."""
    mr_dir = test_root / "MR"
    ct_dir = test_root / "CT"
    if not mr_dir.is_dir():
        sys.exit(f"Test MR directory not found: {mr_dir}")

    subjects: List[str] = []
    for mr in sorted(mr_dir.glob("*.nii.gz")):
        stem = mr.name.removesuffix(".nii.gz")
        gt = ct_dir / f"{stem.replace('_MR_', '_CT_')}.nii.gz"
        if not gt.exists():
            continue
        sct_ok = True
        for ens in ensembles:
            sct = sweep_root / f"ensemble_test_{ens}" / "sct" / f"{stem}_sCT.nii.gz"
            if not sct.exists():
                sct_ok = False
                break
        if sct_ok:
            subjects.append(stem)
    return subjects


def _align_avg_to_gt(
    avg_sitk: sitk.Image,
    gt_path: Path,
    max_iterations: int,
) -> np.ndarray:
    fixed = load_sitk(gt_path)
    resampled, _, _ = rigid_register(
        fixed=fixed,
        moving=avg_sitk,
        max_iterations=max_iterations,
        bone_mask_threshold=None,
        verbose=False,
    )
    return sitk_to_numpy(resampled)


HU_MIN, HU_MAX = -1000.0, 3000.0


def _intensity_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    diff = np.clip(pred, HU_MIN, HU_MAX).astype(np.float64) - gt.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": math.sqrt(mse),
        "psnr": 20.0 * math.log10(HU_MAX - HU_MIN) - 10.0 * math.log10(mse),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_markdown(rows: List[Dict[str, float]], threshold: float) -> str:
    header = (
        f"| Subject | Strain | sCT SSIM DSC | sCT L1 DSC | Aligned avCT DSC | avCT MAE [HU] |\n"
        f"|---|---|---:|---:|---:|---:|"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['subject']} "
            f"| {r['strain']} "
            f"| {r['ssim']:.3f} "
            f"| {r['l1']:.3f} "
            f"| {r['avct']:.3f} "
            f"| {r['avct_mae']:.1f} |"
        )
    # Aggregate row
    ssim = np.array([r["ssim"] for r in rows], dtype=np.float64)
    l1 = np.array([r["l1"] for r in rows], dtype=np.float64)
    avct = np.array([r["avct"] for r in rows], dtype=np.float64)
    avct_mae = np.array([r["avct_mae"] for r in rows], dtype=np.float64)
    lines.append(
        f"| **mean +/- sd (n={len(rows)})** | "
        f"| **{ssim.mean():.3f} +/- {ssim.std(ddof=1):.3f}** "
        f"| **{l1.mean():.3f} +/- {l1.std(ddof=1):.3f}** "
        f"| **{avct.mean():.3f} +/- {avct.std(ddof=1):.3f}** "
        f"| **{avct_mae.mean():.1f} +/- {avct_mae.std(ddof=1):.1f}** |"
    )
    return f"Bone Dice (HU > {int(threshold)}) on Test_Data\n\n" + "\n".join(lines)


def _save_csv(rows: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("subject,strain,sct_ssim_dsc,sct_l1_dsc,aligned_avct_dsc,"
                 "aligned_avct_mae_hu,aligned_avct_rmse_hu,aligned_avct_psnr_hu\n")
        for r in rows:
            fh.write(
                f"{r['subject']},{r['strain']},{r['ssim']:.6f},{r['l1']:.6f},"
                f"{r['avct']:.6f},{r['avct_mae']:.6f},{r['avct_rmse']:.6f},"
                f"{r['avct_psnr']:.6f}\n"
            )


def _save_bar_chart(rows: List[Dict[str, float]], path: Path, threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
        }
    )

    subjects = [r["subject"].replace("_MR_", " ") for r in rows]
    ssim = [r["ssim"] for r in rows]
    l1 = [r["l1"] for r in rows]
    avct = [r["avct"] for r in rows]

    x = np.arange(len(subjects))
    w = 0.27

    fig, ax = plt.subplots(figsize=(max(7.5, 0.7 * len(subjects) + 2.2), 4.0))
    ax.bar(x - w, ssim, width=w, label="sCT (SSIM ensemble)",
           color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.bar(x, l1, width=w, label="sCT (L1 ensemble)",
           color="#2ca02c", edgecolor="black", linewidth=0.4)
    ax.bar(x + w, avct, width=w, label="Aligned avCT",
           color="#d62728", edgecolor="black", linewidth=0.4)

    ax.set_ylabel(f"Bone Dice (HU > {int(threshold)})")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=30, ha="right", fontsize=8.5)
    ax.set_title("Bone Dice across Test_Data: sCT ensembles vs. aligned avCT")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, ncol=3, fontsize=8.5)

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
        "--sweep-root",
        default=REPO_ROOT / "output" / "loss_sweep_final_22",
        type=Path,
    )
    ap.add_argument(
        "--avg-ct",
        default=REPO_ROOT / "output" / "average_CT" / "average_ct.nii.gz",
        type=Path,
        help="Population avCT NIfTI (run scripts/make_combined_dice_figure.py "
        "once to create it).",
    )
    ap.add_argument(
        "--out",
        default=REPO_ROOT / "manuscript" / "dice_test_cohort",
        type=Path,
        help="Output stem (CSV + PNG/PDF written here).",
    )
    ap.add_argument("--threshold", type=float, default=300.0)
    ap.add_argument("--max-iterations", type=int, default=500)
    ap.add_argument(
        "--strain-map",
        default=REPO_ROOT / "configs" / "test_strain_map.json",
        type=Path,
        help="JSON mapping mouse ID -> strain label.",
    )
    ap.add_argument(
        "--ensembles",
        nargs="+",
        default=["ssim", "l1"],
        help="Which K=3 ensembles to score (default: ssim l1).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not args.avg_ct.exists():
        sys.exit(
            f"avCT NIfTI not found: {args.avg_ct}\n"
            f"Create it first with `python scripts/make_combined_dice_figure.py`."
        )

    ensembles: Tuple[str, ...] = tuple(args.ensembles)
    subjects = _discover_subjects(args.test_root, args.sweep_root, ensembles)
    if not subjects:
        sys.exit(
            f"No subjects found with MR + GT + sCT in {ensembles} under {args.test_root}"
        )

    print(
        f"Scoring bone Dice (HU > {int(args.threshold)}) on "
        f"{len(subjects)} test subjects across ensembles {ensembles}.\n"
    )

    avg_sitk = load_sitk(args.avg_ct)
    strain_map: Dict[str, str] = {}
    if args.strain_map.exists():
        strain_map = {k.upper(): v for k, v in json.loads(args.strain_map.read_text()).items()}

    rows: List[Dict[str, float]] = []
    for i, stem in enumerate(subjects, 1):
        gt_name = stem.replace("_MR_", "_CT_")
        gt_path = args.test_root / "CT" / f"{gt_name}.nii.gz"

        gt_vol = _load_array(gt_path)
        sct_ssim = _load_array(
            args.sweep_root / "ensemble_test_ssim" / "sct" / f"{stem}_sCT.nii.gz"
        )
        sct_l1 = _load_array(
            args.sweep_root / "ensemble_test_l1" / "sct" / f"{stem}_sCT.nii.gz"
        )

        d_ssim = dice_score(sct_ssim, gt_vol, args.threshold)
        d_l1 = dice_score(sct_l1, gt_vol, args.threshold)

        print(f"[{i:2d}/{len(subjects)}] {stem}  rigid avCT -> GT ...", end="", flush=True)
        avg_on_gt = _align_avg_to_gt(avg_sitk, gt_path, args.max_iterations)
        d_avct = dice_score(avg_on_gt, gt_vol, args.threshold)
        m_avct = _intensity_metrics(avg_on_gt, gt_vol)
        print(
            f"  sCT_SSIM={d_ssim:.3f}  sCT_L1={d_l1:.3f}  avCT={d_avct:.3f}"
            f"  avCT_MAE={m_avct['mae']:.1f} HU"
        )

        rows.append(
            {
                "subject": stem,
                "strain": strain_map.get(stem.split("_", 1)[0].upper(), "unknown"),
                "ssim": float(d_ssim),
                "l1": float(d_l1),
                "avct": float(d_avct),
                "avct_mae": m_avct["mae"],
                "avct_rmse": m_avct["rmse"],
                "avct_psnr": m_avct["psnr"],
            }
        )

    table = _format_markdown(rows, args.threshold)
    print("\n" + table + "\n")

    csv_path = args.out.with_suffix(".csv")
    _save_csv(rows, csv_path)
    print(f"CSV  -> {csv_path}")

    _save_bar_chart(rows, args.out, args.threshold)
    print(f"PNG  -> {args.out.with_suffix('.png')}")
    print(f"PDF  -> {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
