#!/usr/bin/env python
"""
Held-out test set metrics restricted to the micro-CT field of view.

The co-registered micro-CT holds a constant fill value outside its rotated crop
box (see simple_sct/metrics/fov.py). This script evaluates every test
acquisition for

  * the published SSIM ensemble          (--published-sct-dir)
  * the intensity-augmented SSIM ensemble (--aug-sct-dir)
  * the rigidly aligned average CT        (--avg-ct, aligned as in test_dice_table.py)

both over the whole volume (for reference) and inside the CT field of view:
MAE, RMSE, PSNR, bone DSC (300 HU), mean surface distance and HD95.

    python scripts/evaluate_test_fov.py --out output/fov_eval
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_alignment import load_sitk, rigid_register, sitk_to_numpy  # noqa: E402
from strain_mm_analysis import surface_distances  # noqa: E402

# simple_sct.metrics imports torch; load the numpy-only module directly.
_spec = importlib.util.spec_from_file_location("fov", REPO_ROOT / "simple_sct" / "metrics" / "fov.py")
_fov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fov)
ct_fov_mask = _fov.ct_fov_mask

HU_MIN, HU_MAX, BONE_HU = -1000.0, 3000.0, 300.0
METHODS = ["published", "augmented", "avg_ct"]


def metrics(pred: np.ndarray, gt: np.ndarray, region: np.ndarray, spacing: tuple) -> dict:
    p = np.clip(pred, HU_MIN, HU_MAX).astype(np.float64)
    diff = (p - gt)[region]
    mse = float(np.mean(diff ** 2))
    a, b = (p > BONE_HU) & region, (gt > BONE_HU) & region
    d = surface_distances(a, b, spacing)
    return {"mae_hu": float(np.mean(np.abs(diff))), "rmse_hu": math.sqrt(mse),
            "psnr_db": 20.0 * math.log10(HU_MAX - HU_MIN) - 10.0 * math.log10(mse),
            "dice_bone": 2.0 * (a & b).sum() / (a.sum() + b.sum()),
            "msd_mm": float(d.mean()), "hd95_mm": float(np.percentile(d, 95))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-root", type=Path, default=REPO_ROOT / "dataset" / "Test_Data")
    ap.add_argument("--published-sct-dir", type=Path, default=REPO_ROOT / "output" / "test_sct" / "loss_sweep_final_22")
    ap.add_argument("--aug-sct-dir", type=Path, default=REPO_ROOT / "output" / "test_sct" / "aug_noN4")
    ap.add_argument("--avg-ct", type=Path, default=REPO_ROOT / "output" / "average_CT" / "average_ct.nii.gz")
    ap.add_argument("--strain-map", type=Path, default=REPO_ROOT / "configs" / "test_strain_map.json")
    ap.add_argument("--max-iterations", type=int, default=500)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "fov_eval")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    strains = {k.upper(): v for k, v in json.loads(args.strain_map.read_text()).items()}
    avg = load_sitk(args.avg_ct)
    rows = []
    for sct in sorted(args.aug_sct_dir.glob("*_sCT.nii.gz")):
        stem = sct.name.removesuffix("_sCT.nii.gz")
        gt_path = args.test_root / "CT" / f"{stem.replace('_MR_', '_CT_')}.nii.gz"
        gt_img = nib.load(str(gt_path))
        gt = gt_img.get_fdata(dtype=np.float32).astype(np.float64)
        spacing = tuple(float(s) for s in gt_img.header.get_zooms()[:3])
        fov = ct_fov_mask(gt)
        aligned, _, _ = rigid_register(fixed=load_sitk(gt_path), moving=avg, max_iterations=args.max_iterations,
                                       bone_mask_threshold=None, verbose=False)
        preds = {"published": nib.load(str(args.published_sct_dir / sct.name)).get_fdata(dtype=np.float32),
                 "augmented": nib.load(str(sct)).get_fdata(dtype=np.float32),
                 "avg_ct": sitk_to_numpy(aligned)}
        for method, pred in preds.items():
            for region_name, region in (("whole_volume", np.ones_like(fov)), ("ct_fov", fov)):
                rows.append({"acquisition": stem, "strain": strains[stem.split("_")[0]], "method": method,
                             "region": region_name, "fov_fraction": float(fov.mean()),
                             **metrics(pred, gt, region, spacing)})
        fov_rows = [r for r in rows[-6:] if r["region"] == "ct_fov"]
        print(f"  {stem} (CT FOV {fov.mean():.2f}): " +
              "  ".join(f"{r['method']} MAE={r['mae_hu']:.0f} DSC={r['dice_bone']:.3f}" for r in fov_rows))

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "test_per_acquisition.csv", index=False)
    cols = ["mae_hu", "rmse_hu", "psnr_db", "dice_bone", "msd_mm", "hd95_mm"]
    summary = df.groupby(["region", "strain", "method"])[cols].agg(["mean", "std", "min", "max"])
    summary.to_csv(args.out / "test_summary.csv")
    print(summary.xs("mean", axis=1, level=1).round(3).to_string())
    print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
