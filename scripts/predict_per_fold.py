#!/usr/bin/env python
"""
Per-fold sCT prediction on the held-out test cohort.
=====================================================
Runs each ensemble member (one checkpoint per fold) independently and
saves the sCT NIfTI for every test volume to a separate subfolder:

    <out-dir>/
        ssim_seed1337_fold0/sct/  MZEI00027_MR_T1_sCT.nii.gz ...
        ssim_seed1337_fold1/sct/  MZEI00027_MR_T1_sCT.nii.gz ...
        ssim_seed1337_fold2/sct/  MZEI00027_MR_T1_sCT.nii.gz ...

Useful for analysing inter-fold variability or running on a different
machine where only a single checkpoint is available.

Example
-------
    python scripts/predict_per_fold.py \\
        --sweep-root output/loss_sweep_final_22 \\
        --loss ssim \\
        --test-mri-dir dataset/Test_Data/MR \\
        --out-dir output/per_fold_predictions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ensemble_test import (  # noqa: E402
    RUN_DIR_RE, FOLD_DIR_RE,
    _build_and_load, _load_cfg_for_run, _override_for_test,
    _save_sct_nifti, _to_hu,
    discover_checkpoints,
)
from simple_sct.data.volume_dataset import VolumePairs  # noqa: E402
from train import _extract_mouse_id, _sample_filename_stems  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sweep-root",
                    default=REPO_ROOT / "output" / "loss_sweep_final_22",
                    type=Path,
                    help="Sweep root produced by run_sweep.py "
                         "(default: output/loss_sweep_final_22).")
    ap.add_argument("--loss", default="ssim",
                    help="Loss name (default: ssim).")
    ap.add_argument("--use", default="best", choices=["latest", "best"],
                    help="Checkpoint flavour (default: best).")
    ap.add_argument("--test-mri-dir",
                    default=REPO_ROOT / "dataset" / "Test_Data" / "MR",
                    type=Path,
                    help="Directory of test MRI NIfTI volumes "
                         "(default: dataset/Test_Data/MR).")
    ap.add_argument("--out-dir",
                    default=REPO_ROOT / "output" / "per_fold_predictions",
                    type=Path,
                    help="Root output directory; one subfolder per fold "
                         "(default: output/per_fold_predictions).")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Inference device (default: cuda if available).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    test_mri_dir = args.test_mri_dir.resolve()
    out_dir = args.out_dir.resolve()
    device = torch.device(args.device)

    ckpts = discover_checkpoints(sweep_root, args.loss, args.use)
    if not ckpts:
        sys.exit(f"No '{args.use}.pt' checkpoints found for loss='{args.loss}' "
                 f"under {sweep_root}")

    print(f"Found {len(ckpts)} checkpoint(s) for loss='{args.loss}':")
    for loss, seed, fold, ckpt in ckpts:
        print(f"  seed={seed}  fold={fold}  {ckpt.relative_to(sweep_root)}")

    # Build dataset once from the first checkpoint's config.
    loss_0, seed_0, fold_0, _ = ckpts[0]
    base_cfg = _load_cfg_for_run(sweep_root, loss_0, fold_0, seed_0)
    base_cfg = _override_for_test(base_cfg, test_mri_dir)
    ds = VolumePairs(base_cfg)
    ds.augment = False
    sample_keys = _sample_filename_stems(ds)

    hu_min = float(base_cfg.data.hu_min)
    hu_max = float(base_cfg.data.hu_max)
    scale_target = base_cfg.data.scale_target

    print(f"\nTest volumes: {len(ds)}")
    print(f"HU range: [{hu_min}, {hu_max}]  scale_target: {scale_target}\n")

    for loss, seed, fold, ckpt_path in ckpts:
        tag = f"{loss}_seed{seed}_fold{fold}"
        fold_out = out_dir / tag / "sct"
        fold_out.mkdir(parents=True, exist_ok=True)

        print(f"{'='*60}")
        print(f"  Checkpoint: {tag}")
        print(f"  Output:     {fold_out}")
        print(f"{'='*60}")

        cfg = _load_cfg_for_run(sweep_root, loss, fold, seed)
        model = _build_and_load(cfg, ckpt_path, device)

        with torch.no_grad():
            for ds_idx in range(len(ds)):
                sample = ds[ds_idx]
                x = sample["mri"].unsqueeze(0).to(device, non_blocking=True)
                pred = model(x).float()
                pred_hu = _to_hu(pred, hu_min, hu_max, scale_target)
                pred_hu = pred_hu.clamp_(hu_min, hu_max)

                vol_np = pred_hu.squeeze(0).squeeze(0).detach().cpu().numpy()
                key = sample_keys[ds_idx]
                out_path = fold_out / f"{key}_sCT.nii.gz"
                _save_sct_nifti(out_path, vol_np, ds.files[ds_idx])
                print(f"  [{ds_idx+1:>2d}/{len(ds)}] {key}  "
                      f"HU=[{vol_np.min():.0f}, {vol_np.max():.0f}]  -> {out_path.name}")

        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
        print()

    print("Done. Per-fold predictions written to:")
    for loss, seed, fold, _ in ckpts:
        print(f"  {out_dir / f'{loss}_seed{seed}_fold{fold}' / 'sct'}")


if __name__ == "__main__":
    main()
