#!/usr/bin/env python
"""
Re-evaluate every trained model in a sweep on the held-out CV val fold, in
Hounsfield-Unit space, and compute bone-Dice scores. Writes a per-volume CSV
that downstream comparison/statistics scripts consume.

For each <sweep>/<loss>_seed<seed>/fold<N>/checkpoints/<which>.pt:
  1. Load the frozen per-run YAML from <sweep>/configs/<loss>_fold<N>_seed<seed>.yaml
  2. Rebuild the dataset and split with the same seed/fold as training
  3. Forward each held-out val volume through the model
  4. Un-scale the prediction (and target if scale_target="minmax") to HU
  5. Compute MAE_HU, RMSE_HU, PSNR_HU, SSIM_HU, dice_bone (threshold 300 HU)
  6. Append one row per volume to <sweep>/eval_per_volume_<which>.csv

Example
-------
    python scripts/evaluate_sweep.py --sweep-root output/loss_sweep_v2
    python scripts/evaluate_sweep.py --sweep-root output/loss_sweep_v2 --use best
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simple_sct.config import load_config  # noqa: E402
from simple_sct.data.volume_dataset import VolumePairs, unnormalize  # noqa: E402
from simple_sct.metrics.fov import ct_fov_mask  # noqa: E402
from simple_sct.models.simple_unet3d import UNet3D  # noqa: E402

# Re-use the exact split function the training script uses, so val indices
# match training byte-for-byte.
from train import split_kfold, _sample_filename_stems, _extract_mouse_id  # noqa: E402


RUN_DIR_RE = re.compile(r"^(?P<loss>[a-z0-9]+)_seed(?P<seed>\d+)$")
FOLD_DIR_RE = re.compile(r"^fold(?P<fold>\d+)$")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-root", required=True, type=Path,
                    help="Sweep root produced by run_sweep.py")
    ap.add_argument("--use", default="best", choices=["latest", "best"],
                    help="Which checkpoint to evaluate per fold (default: best)")
    ap.add_argument("--bone-threshold", type=float, default=300.0,
                    help="HU threshold for bone Dice (default: 300)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device for inference (default: cuda if available)")
    ap.add_argument("--out-csv", default=None, type=Path,
                    help="Override per-volume CSV path "
                         "(default: <sweep>/eval_per_volume_<use>.csv)")
    ap.add_argument("--fov-mask", action="store_true",
                    help="Restrict all metrics to the micro-CT field of view "
                         "(exclude the constant fill outside the rotated CT crop box).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="If the output CSV already contains a (loss,seed,fold,split,sample_key) "
                         "row, do not recompute it. Useful to resume after a crash.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Metrics (HU space)
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
    """Bone Dice: binarise both volumes at `thr` HU and compute 2|A n B|/(|A|+|B|)."""
    a = pred_hu > thr
    b = target_hu > thr
    a_sum = float(a.sum().item())
    b_sum = float(b.sum().item())
    if a_sum + b_sum == 0:
        # Both empty: by convention Dice == 1 (perfect agreement on absence).
        return 1.0
    inter = float((a & b).sum().item())
    return (2.0 * inter) / (a_sum + b_sum)


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------
def _load_cfg_for_run(sweep_root: Path, loss: str, fold: int, seed: int):
    """Load the frozen per-run YAML produced by run_sweep.py."""
    cfg_path = sweep_root / "configs" / f"{loss}_fold{fold}_seed{seed}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Per-run YAML not found: {cfg_path}")
    return load_config(cfg_path)


def _build_model(cfg, device: torch.device) -> torch.nn.Module:
    model = UNet3D(**cfg.model.__dict__).to(device)
    return model


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> dict:
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = state.get("model", state)  # tolerate raw state_dicts
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [warn] {ckpt_path.name}: missing={len(missing)}, unexpected={len(unexpected)}")
    return state


def _to_hu(t: torch.Tensor, hu_min: float, hu_max: float, scale_target: str) -> torch.Tensor:
    """Convert a normalised tensor in [-1, 1] back to HU. Pass through if HU."""
    if scale_target == "hu":
        return t
    return unnormalize(t, hu_min, hu_max)


def _iter_indices(ds: VolumePairs, indices: list[int], mouse_ids: list[str]) -> Iterable[tuple[int, str]]:
    """Yield (dataset_index, mouse_id) pairs in deterministic order."""
    for i in indices:
        yield i, mouse_ids[i]


def evaluate_run(
    cfg,
    ckpt_path: Path,
    device: torch.device,
    bone_threshold: float,
    fov_mask: bool = False,
) -> list[dict]:
    """Return a list of per-volume metric rows for the held-out val fold."""
    ds = VolumePairs(cfg)
    ds.augment = False  # deterministic, un-augmented evaluation
    train_idx, val_idx, _ = split_kfold(
        ds, fold=int(cfg.data.fold), seed=int(cfg.seed), k=3
    )
    sample_keys = _sample_filename_stems(ds)
    mouse_ids = [_extract_mouse_id(k) for k in sample_keys]

    model = _build_model(cfg, device)
    _load_checkpoint(model, ckpt_path, device)
    model.eval()

    hu_min = float(cfg.data.hu_min)
    hu_max = float(cfg.data.hu_max)
    hu_range = hu_max - hu_min
    scale_target = cfg.data.scale_target

    rows: list[dict] = []
    with torch.no_grad():
        for split_name, idxs in (("val", val_idx),):
            for ds_idx, mid in _iter_indices(ds, idxs, mouse_ids):
                sample = ds[ds_idx]
                x = sample["mri"] if isinstance(sample, dict) else sample[0]
                y = sample["ct"] if isinstance(sample, dict) else sample[1]
                x = x.unsqueeze(0).to(device, non_blocking=True)  # (1,1,D,H,W)
                y = y.unsqueeze(0).to(device, non_blocking=True)
                pred = model(x)

                pred_hu = _to_hu(pred.float(), hu_min, hu_max, scale_target).clamp_(hu_min, hu_max)
                y_hu = _to_hu(y.float(), hu_min, hu_max, scale_target)
                if fov_mask:
                    m = torch.from_numpy(ct_fov_mask(y_hu[0, 0].cpu().numpy())).to(y_hu.device)
                    pred_hu, y_hu = pred_hu[m[None, None]], y_hu[m[None, None]]

                rows.append({
                    "split": split_name,
                    "mouse_id": mid,
                    "sample_key": sample_keys[ds_idx],
                    "mae_hu": _mae(pred_hu, y_hu),
                    "rmse_hu": _rmse(pred_hu, y_hu),
                    "psnr_hu": _psnr_hu(pred_hu, y_hu, hu_range),
                    "dice_bone": _dice(pred_hu, y_hu, bone_threshold),
                    "bone_threshold_hu": bone_threshold,
                })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    device = torch.device(args.device)

    out_csv = args.out_csv or (sweep_root / f"eval_per_volume_{args.use}.csv")
    out_csv = out_csv.resolve()

    # Discover runs (loss, seed, fold) directly from the on-disk layout.
    runs: list[tuple[str, int, int, Path]] = []
    for run_dir in sorted(sweep_root.iterdir()):
        if not run_dir.is_dir():
            continue
        m = RUN_DIR_RE.match(run_dir.name)
        if not m:
            continue
        loss = m.group("loss")
        seed = int(m.group("seed"))
        for fold_dir in sorted(run_dir.iterdir()):
            mf = FOLD_DIR_RE.match(fold_dir.name)
            if not mf:
                continue
            fold = int(mf.group("fold"))
            ckpt = fold_dir / "checkpoints" / f"{args.use}.pt"
            if not ckpt.exists():
                print(f"[miss] {ckpt}")
                continue
            runs.append((loss, seed, fold, ckpt))

    if not runs:
        print(f"No runs found under {sweep_root}")
        return

    # Existing rows for resumability.
    seen: set[tuple] = set()
    if args.skip_existing and out_csv.exists():
        with open(out_csv, newline="") as f:
            for r in csv.DictReader(f):
                seen.add((r["loss"], int(r["seed"]), int(r["fold"]),
                          r["split"], r["sample_key"]))
        print(f"[resume] {len(seen)} rows already present in {out_csv}")

    # Write incrementally so a crash mid-sweep doesn't lose work.
    fieldnames = [
        "loss", "seed", "fold", "split", "mouse_id", "sample_key",
        "mae_hu", "rmse_hu", "psnr_hu", "dice_bone", "bone_threshold_hu",
    ]
    write_header = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(out_csv, "a", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    try:
        print(f"Evaluating {len(runs)} run(s) -> {out_csv}\n")
        t0 = time.time()
        for i, (loss, seed, fold, ckpt) in enumerate(runs, start=1):
            tag = f"[{i}/{len(runs)}] {loss}_fold{fold}_seed{seed}"
            try:
                cfg = _load_cfg_for_run(sweep_root, loss, fold, seed)
            except FileNotFoundError as e:
                print(f"{tag}  [skip] {e}")
                continue

            t_run = time.time()
            try:
                rows = evaluate_run(cfg, ckpt, device, args.bone_threshold, args.fov_mask)
            except Exception as e:
                print(f"{tag}  [error] {e}")
                continue

            new_rows = 0
            for r in rows:
                key = (loss, seed, fold, r["split"], r["sample_key"])
                if key in seen:
                    continue
                r.update({"loss": loss, "seed": seed, "fold": fold})
                writer.writerow(r)
                seen.add(key)
                new_rows += 1
            f_out.flush()
            took = time.time() - t_run
            print(f"{tag}  +{new_rows} row(s)  ({took:.1f}s)")

        elapsed = time.time() - t0
        print(f"\nDone in {elapsed/60:.1f} min. Per-volume CSV: {out_csv}")
        print("Next: python scripts/compare_losses.py --per-volume-csv",
              str(out_csv))
    finally:
        f_out.close()


if __name__ == "__main__":
    main()
