#!/usr/bin/env python
"""
K=9 ensemble inference on the external held-out test cohort.

Given a sweep root and a chosen loss, this script:

  1. Discovers every <sweep>/<loss>_seed<S>/fold<F>/checkpoints/<which>.pt
     for the selected loss (by default the 3 folds x 3 seeds = 9 checkpoints).
  2. Loads all of them onto the chosen device and freezes them in eval mode.
  3. Iterates over the external test cohort (separate MRI/CT directories that
     the CV split never touched) and, for each MRI volume, runs every model
     and averages the predictions in HU space to obtain a single ensemble
     sCT per volume.
  4. Computes MAE / RMSE / PSNR / bone-Dice in HU space against the ground
     truth micro-CT and writes a per-volume CSV.
  5. Optionally writes the ensemble sCT volumes to NIfTI (carrying over the
     input MR affine/header so they line up in any viewer) and a manifest
     describing the exact ensemble composition.

Protocol notes
--------------
* The held-out test cohort is touched exactly once, with the loss that won
  on the CV validation folds. No model/loss selection happens here.
* Predictions are averaged in HU space (after un-scaling).  Averaging in
  the normalised [-1, 1] space and then un-scaling is mathematically
  equivalent because the un-scaling is affine; HU-space averaging is just
  more interpretable.
* Models are clamped collectively to [hu_min, hu_max] *after* averaging,
  not before, so single-model out-of-range excursions are smoothed by the
  ensemble before clipping.
* The dataset is reused from ``simple_sct.data.volume_dataset.VolumePairs``
  with ``crop_prob`` and ``flip_prob`` forced to zero so inference is
  deterministic.

Example
-------
    python scripts/ensemble_test.py \\
      --sweep-root output/loss_sweep_v2 \\
      --loss l1 \\
      --test-mri-dir /path/to/test_cohort/MR \\
      --out-dir output/ensemble_test_l1 \\
      --save-volumes \\
      --strain-map test_strain_map.json

The strain map (optional) is a JSON ``{ "MZEI00088": "BALB/c", ... }`` keyed
by uppercase mouse ID (the leading underscore-separated token of the MRI
filename stem, matching ``train._extract_mouse_id``).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simple_sct.config import load_config  # noqa: E402
from simple_sct.data.volume_dataset import VolumePairs, unnormalize  # noqa: E402
from simple_sct.models.simple_unet3d import UNet3D  # noqa: E402
from train import _extract_mouse_id, _sample_filename_stems  # noqa: E402

try:
    import nibabel as nib  # type: ignore
except Exception:  # pragma: no cover - nibabel is a soft dep here
    nib = None


RUN_DIR_RE = re.compile(r"^(?P<loss>[a-z0-9]+)_seed(?P<seed>\d+)$")
FOLD_DIR_RE = re.compile(r"^fold(?P<fold>\d+)$")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sweep-root", required=True, type=Path,
                    help="Sweep root produced by run_sweep.py")
    ap.add_argument("--loss", required=True,
                    help="Loss name to ensemble (must match the sweep directory naming, "
                         "e.g. 'l1' for output/<sweep>/l1_seed*/fold*/...)")
    ap.add_argument("--use", default="best", choices=["latest", "best"],
                    help="Checkpoint flavour to ensemble (default: best)")
    ap.add_argument("--test-mri-dir", required=False, type=Path, default=None,
                    help="Directory of held-out test MRI NIfTI volumes. If "
                         "omitted, falls back to data.test_mri_dir from the "
                         "first discovered per-run YAML. The sibling 'CT' "
                         "directory of this path is used for the ground-truth "
                         "CT (filename convention MR->CT).")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Where to write per_volume_metrics.csv, summary.csv, "
                         "manifest.json and (optionally) ensemble sCT NIfTIs.")
    ap.add_argument("--save-volumes", action="store_true",
                    help="Also save each ensemble sCT as NIfTI under <out-dir>/sct/")
    ap.add_argument("--bone-threshold", type=float, default=300.0,
                    help="HU threshold for bone Dice (default: 300)")
    ap.add_argument("--strain-map", type=Path, default=None,
                    help="Optional JSON mapping uppercase mouse_id -> strain "
                         "label, used to group the summary table. If omitted, "
                         "tries <repo>/configs/test_strain_map.json.")
    ap.add_argument("--max-models", type=int, default=None,
                    help="Optional cap on number of ensemble members "
                         "(default: use every discovered checkpoint = 9 for "
                         "3 folds x 3 seeds).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device for inference (default: cuda if available)")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Metrics (HU space) - duplicated from evaluate_sweep.py to keep this script
# standalone; if you change one, change both.
# ---------------------------------------------------------------------------
def _mae(p: torch.Tensor, t: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(p - t)).item())


def _rmse(p: torch.Tensor, t: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((p - t) ** 2)).item())


def _psnr_hu(p: torch.Tensor, t: torch.Tensor, hu_range: float) -> float:
    mse = float(torch.mean((p - t) ** 2).item())
    if not math.isfinite(mse) or mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(float(hu_range)) - 10.0 * math.log10(mse)


def _dice(pred_hu: torch.Tensor, target_hu: torch.Tensor, thr: float) -> float:
    a = pred_hu > thr
    b = target_hu > thr
    a_sum = float(a.sum().item())
    b_sum = float(b.sum().item())
    if a_sum + b_sum == 0:
        return 1.0
    inter = float((a & b).sum().item())
    return (2.0 * inter) / (a_sum + b_sum)


def _to_hu(t: torch.Tensor, hu_min: float, hu_max: float, scale_target: str) -> torch.Tensor:
    if scale_target == "hu":
        return t
    return unnormalize(t, hu_min, hu_max)


# ---------------------------------------------------------------------------
# Sweep / checkpoint discovery
# ---------------------------------------------------------------------------
def discover_checkpoints(sweep_root: Path, loss: str, which: str
                         ) -> list[tuple[str, int, int, Path]]:
    """Return [(loss, seed, fold, ckpt_path), ...] for the requested loss."""
    found: list[tuple[str, int, int, Path]] = []
    for run_dir in sorted(sweep_root.iterdir()):
        if not run_dir.is_dir():
            continue
        m = RUN_DIR_RE.match(run_dir.name)
        if not m or m.group("loss") != loss:
            continue
        seed = int(m.group("seed"))
        for fold_dir in sorted(run_dir.iterdir()):
            mf = FOLD_DIR_RE.match(fold_dir.name)
            if not mf:
                continue
            fold = int(mf.group("fold"))
            ckpt = fold_dir / "checkpoints" / f"{which}.pt"
            if not ckpt.exists():
                print(f"  [miss] {ckpt}")
                continue
            found.append((loss, seed, fold, ckpt))
    return found


def _load_cfg_for_run(sweep_root: Path, loss: str, fold: int, seed: int):
    cfg_path = sweep_root / "configs" / f"{loss}_fold{fold}_seed{seed}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Per-run YAML not found: {cfg_path}")
    return load_config(cfg_path)


def _build_and_load(cfg, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    model = UNet3D(**cfg.model.__dict__).to(device)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = state.get("model", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [warn] {ckpt_path.name}: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Test-cohort dataset construction
# ---------------------------------------------------------------------------
def _override_for_test(cfg, test_mri_dir: Path):
    """Repoint VolumePairs at the test cohort and disable any augmentation."""
    cfg.data.mri_dir = str(test_mri_dir)
    # The dataset derives the CT path from the sibling 'CT' dir (filename
    # convention MR->CT); ct_dir on cfg is informational only but we sync it
    # so future readers aren't misled.
    cfg.data.ct_dir = str(test_mri_dir.parent / "CT")
    # The exclusion list keeps the test mice *out* of training. For the test
    # cohort itself we obviously want them in, so clear the list.
    cfg.data.exclude_mouse_ids = []
    # Copy the declared test-cohort acquisition cap into the active dataset
    # cap. None means "use everything on disk".
    cfg.data.max_acquisitions_per_mouse = getattr(
        cfg.data, "test_max_acquisitions_per_mouse", None
    )
    cfg.train.crop_prob = 0.0
    cfg.train.flip_prob = 0.0
    return cfg


def _save_sct_nifti(out_path: Path, volume_hu: np.ndarray, ref_mri_path: Path) -> None:
    if nib is None:
        raise RuntimeError("nibabel is required to save NIfTI output. "
                           "Install it or omit --save-volumes.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4, dtype=np.float32)
    header = None
    try:
        ref = nib.load(str(ref_mri_path))
        ref = nib.as_closest_canonical(ref)
        affine = ref.affine
        header = ref.header
    except Exception as e:
        print(f"  [warn] could not read affine from {ref_mri_path}: {e}")
    img = nib.Nifti1Image(volume_hu.astype(np.float32), affine=affine, header=header)
    nib.save(img, str(out_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # --- 1) Discover and validate the checkpoint ensemble ------------------
    ckpts = discover_checkpoints(sweep_root, args.loss, args.use)
    if not ckpts:
        sys.exit(f"No '{args.use}.pt' checkpoints found for loss='{args.loss}' "
                 f"under {sweep_root}")
    if args.max_models is not None:
        ckpts = ckpts[: args.max_models]

    print(f"Ensembling {len(ckpts)} model(s) for loss='{args.loss}', "
          f"checkpoint='{args.use}.pt'")
    for loss, seed, fold, ckpt in ckpts:
        print(f"  - fold={fold} seed={seed:>4d}  {ckpt.relative_to(sweep_root)}")

    # --- 2) Pick a base config and sanity-check that preprocessing agrees ---
    loss_0, seed_0, fold_0, _ = ckpts[0]
    base_cfg = _load_cfg_for_run(sweep_root, loss_0, fold_0, seed_0)
    preprocessing_fields = (
        "scale_input", "scale_target", "hu_min", "hu_max",
        "enforce_shape", "reorient_to_canonical",
    )
    for loss, seed, fold, _ in ckpts[1:]:
        c = _load_cfg_for_run(sweep_root, loss, fold, seed)
        for f in preprocessing_fields:
            v0 = getattr(base_cfg.data, f, None)
            vi = getattr(c.data, f, None)
            if v0 != vi:
                sys.exit(
                    f"Inconsistent data.{f} across ensemble members: "
                    f"fold{fold_0}_seed{seed_0}={v0!r} vs "
                    f"fold{fold}_seed{seed}={vi!r}. Refusing to ensemble."
                )

    # --- 3) Build all models on the requested device -----------------------
    models: list[torch.nn.Module] = []
    for loss, seed, fold, ckpt in ckpts:
        cfg = _load_cfg_for_run(sweep_root, loss, fold, seed)
        models.append(_build_and_load(cfg, ckpt, device))

    # --- 4) Construct the test-cohort dataset ------------------------------
    # Resolve the test MRI directory: CLI argument wins, otherwise fall back
    # to data.test_mri_dir from the per-run YAML.
    test_mri_dir = args.test_mri_dir
    if test_mri_dir is None:
        cfg_test = getattr(base_cfg.data, "test_mri_dir", None)
        if not cfg_test:
            sys.exit("--test-mri-dir was not given and data.test_mri_dir is "
                     "not set in the per-run YAML. Aborting.")
        test_mri_dir = Path(cfg_test)
    test_mri_dir = test_mri_dir.resolve()
    print(f"Test cohort MRI dir: {test_mri_dir}")

    base_cfg = _override_for_test(base_cfg, test_mri_dir)
    ds = VolumePairs(base_cfg)
    ds.augment = False  # deterministic test-cohort evaluation
    sample_keys = _sample_filename_stems(ds)
    mouse_ids = [_extract_mouse_id(s) for s in sample_keys]

    cap = getattr(ds, "_cap_per_mouse", None)
    capped_n = getattr(ds, "_capped_sample_count", 0)
    if cap is not None:
        print(f"Per-mouse acquisition cap: {cap} (dropped {capped_n} extra acquisition(s))")

    # Resolve strain map: CLI argument wins, then <repo>/configs/test_strain_map.json.
    strain_map_path = args.strain_map
    if strain_map_path is None:
        candidate = REPO_ROOT / "configs" / "test_strain_map.json"
        if candidate.exists():
            strain_map_path = candidate
    strain_map: dict[str, str] = {}
    if strain_map_path is not None:
        with open(strain_map_path) as f:
            strain_map = {k.upper(): v for k, v in json.load(f).items()}
        print(f"Strain map: {strain_map_path}  ({len(strain_map)} entries)")

    hu_min = float(base_cfg.data.hu_min)
    hu_max = float(base_cfg.data.hu_max)
    hu_range = hu_max - hu_min
    scale_target = base_cfg.data.scale_target

    # --- 5) Per-volume ensemble inference + metrics ------------------------
    per_volume_csv = out_dir / "per_volume_metrics.csv"
    fieldnames = [
        "mouse_id", "strain", "sample_key", "k",
        "mae_hu", "rmse_hu", "psnr_hu", "dice_bone", "bone_threshold_hu",
    ]
    rows: list[dict] = []

    print(f"\nRunning ensemble on {len(ds)} test volume(s)\n")
    t0 = time.time()
    with torch.no_grad():
        for ds_idx in range(len(ds)):
            sample = ds[ds_idx]
            x = sample["mri"].unsqueeze(0).to(device, non_blocking=True)
            y = sample["ct"].unsqueeze(0).to(device, non_blocking=True)

            # Accumulate predictions in HU space (avg of K members).
            ensemble_hu: Optional[torch.Tensor] = None
            for model in models:
                pred = model(x).float()
                pred_hu = _to_hu(pred, hu_min, hu_max, scale_target)
                ensemble_hu = pred_hu if ensemble_hu is None else ensemble_hu + pred_hu
            assert ensemble_hu is not None  # for type-checkers; guarded by len(ckpts)>0
            ensemble_hu = (ensemble_hu / float(len(models))).clamp_(hu_min, hu_max)
            y_hu = _to_hu(y.float(), hu_min, hu_max, scale_target)

            mid = mouse_ids[ds_idx]
            row = {
                "mouse_id": mid,
                "strain": strain_map.get(mid, "unknown"),
                "sample_key": sample_keys[ds_idx],
                "k": len(models),
                "mae_hu": _mae(ensemble_hu, y_hu),
                "rmse_hu": _rmse(ensemble_hu, y_hu),
                "psnr_hu": _psnr_hu(ensemble_hu, y_hu, hu_range),
                "dice_bone": _dice(ensemble_hu, y_hu, args.bone_threshold),
                "bone_threshold_hu": args.bone_threshold,
            }
            rows.append(row)
            print(f"  [{ds_idx + 1:>2d}/{len(ds)}] {sample_keys[ds_idx]}  "
                  f"mae={row['mae_hu']:.2f} HU  dice={row['dice_bone']:.3f}")

            if args.save_volumes:
                vol_np = ensemble_hu.squeeze(0).squeeze(0).detach().cpu().numpy()
                ref = ds.files[ds_idx]
                _save_sct_nifti(out_dir / "sct" / f"{sample_keys[ds_idx]}_sCT.nii.gz",
                                vol_np, ref)

    # --- 6) Write per-volume CSV and summary table -------------------------
    with open(per_volume_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {per_volume_csv}")

    summary_csv = out_dir / "summary.csv"
    metric_keys = ["mae_hu", "rmse_hu", "psnr_hu", "dice_bone"]
    groups: dict[str, list[dict]] = {"all": rows}
    for r in rows:
        groups.setdefault(r["strain"], []).append(r)

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n"] + [f"{k}_mean" for k in metric_keys]
                                  + [f"{k}_std" for k in metric_keys])
        for name in sorted(groups.keys(), key=lambda s: (s != "all", s)):
            grp = groups[name]
            n = len(grp)
            means, stds = [], []
            for k in metric_keys:
                vals = [r[k] for r in grp if math.isfinite(r[k])]
                means.append(f"{statistics.mean(vals):.6f}" if vals else "")
                stds.append(f"{statistics.stdev(vals):.6f}" if len(vals) > 1 else
                            ("0.000000" if vals else ""))
            w.writerow([name, n] + means + stds)
    print(f"Wrote {summary_csv}")

    # --- 7) Manifest --------------------------------------------------------
    manifest = {
        "sweep_root": str(sweep_root),
        "loss": args.loss,
        "checkpoint": args.use,
        "k": len(models),
        "members": [
            {"fold": fold, "seed": seed,
             "ckpt": str(ckpt.relative_to(sweep_root))}
            for (loss, seed, fold, ckpt) in ckpts
        ],
        "test_mri_dir": str(test_mri_dir),
        "test_ct_dir": str((test_mri_dir.parent / "CT").resolve()),
        "hu_min": hu_min, "hu_max": hu_max, "scale_target": scale_target,
        "bone_threshold_hu": args.bone_threshold,
        "n_test_volumes": len(ds),
        "max_acquisitions_per_mouse": cap,
        "dropped_acquisitions": capped_n,
        "strain_map_path": str(strain_map_path) if strain_map_path else None,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out_dir / 'manifest.json'}")

    # --- 8) Console summary -------------------------------------------------
    print("\nEnsemble summary (mean +/- std):")
    for name in sorted(groups.keys(), key=lambda s: (s != "all", s)):
        grp = groups[name]
        if not grp:
            continue
        parts = []
        for k in metric_keys:
            vals = [r[k] for r in grp if math.isfinite(r[k])]
            if not vals:
                parts.append(f"{k}=n/a")
                continue
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            unit = " HU" if k in ("mae_hu", "rmse_hu") else (" dB" if k == "psnr_hu" else "")
            parts.append(f"{k}={m:.2f}{unit} +/- {s:.2f}")
        print(f"  {name} (n={len(grp)}): " + ", ".join(parts))


if __name__ == "__main__":
    main()
