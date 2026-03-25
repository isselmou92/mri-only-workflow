from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

# prefer nibabel; fallback to SimpleITK
try:
    import nibabel as nib
except Exception:
    nib = None
try:
    import SimpleITK as sitk
except Exception:
    sitk = None

from simple_sct.config import load_config, choose_device
from simple_sct.models.simple_unet3d import UNet3D


# -------------------------
# I/O helpers
# -------------------------

def load_nifti(path: Path, reorient_to_canonical: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[object]]:
    """
    Returns:
      arr: (D,H,W) float32 array
      affine/header: only when using nibabel; else (None, None)
    """
    if nib is not None:
        img = nib.load(str(path))
        if reorient_to_canonical:
            img = nib.as_closest_canonical(img)
        arr = img.get_fdata(dtype=np.float32)
        return np.asarray(arr, dtype=np.float32), img.affine, img.header
    if sitk is not None:
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (D,H,W)
        return arr, None, img
    raise ImportError("Install nibabel or SimpleITK to read NIfTI.")


def save_nifti(path: Path, volume: np.ndarray, affine=None, header=None) -> None:
    """
    Save NIfTI, copying MR affine/header when available (nibabel path).
    For SimpleITK, writes with default identity direction unless a reference was passed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if nib is not None:
        if affine is None:
            affine = np.eye(4, dtype=np.float32)
        ni = nib.Nifti1Image(volume.astype(np.float32), affine=affine, header=header)
        nib.save(ni, str(path))
        return
    if sitk is not None:
        img = sitk.GetImageFromArray(volume.astype(np.float32))
        # If header is a sitk.Image (from load), copy spacing/direction/origin when possible
        if header is not None and isinstance(header, sitk.Image):
            img.SetSpacing(header.GetSpacing())
            img.SetDirection(header.GetDirection())
            img.SetOrigin(header.GetOrigin())
        sitk.WriteImage(img, str(path))
        return
    raise ImportError("Install nibabel or SimpleITK to write NIfTI.")


# -------------------------
# Scaling helpers (match training)
# -------------------------

def scale_input(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "zscore":
        m = float(x.mean()); s = float(x.std()) + 1e-8
        return (x - m) / s
    if mode == "minmax":
        xmin = float(x.min()); xmax = float(x.max()) + 1e-8
        u = (x - xmin) / (xmax - xmin)
        return u * 2.0 - 1.0
    if mode == "none":
        return x.astype(np.float32)
    raise ValueError(f"Unknown scale_input: {mode}")


def invert_target_scaling(y_net: np.ndarray, mode: str, hu_min: float, hu_max: float) -> np.ndarray:
    """
    Convert network output back to interpretable units.
    - If training used 'hu', prediction is already HU (identity).
    - If training used 'minmax', map from [-1,1] back to [hu_min,hu_max].
    """
    if mode == "hu":
        return y_net.astype(np.float32)
    if mode == "minmax":
        # [-1,1] -> [0,1] -> [hu_min, hu_max]
        u = 0.5 * (y_net + 1.0)
        return (u * (hu_max - hu_min) + hu_min).astype(np.float32)
    raise ValueError(f"Unknown scale_target: {mode}")


# -------------------------
# Inference
# -------------------------

@torch.no_grad()
def run_inference(
    cfg_path: str,
    ckpt_path: str,
    mri_path: str,
    out_path: str,
    device_override: Optional[str] = None,
    amp: bool = False,
    clamp_hu: bool = True,
):
    cfg = load_config(cfg_path)
    device = choose_device(device_override or cfg.train.device_preference)

    # Use same reorientation setting as training (default: False for standardized data)
    reorient = getattr(cfg.data, 'reorient_to_canonical', False)
    
    # Load MR volume
    mri_np, affine, header = load_nifti(Path(mri_path), reorient_to_canonical=reorient)
    exp_shape = tuple(cfg.data.enforce_shape)
    if tuple(mri_np.shape) != exp_shape:
        raise ValueError(f"MR shape {mri_np.shape} != expected {exp_shape}. Please pre-align to training shape.")

    # Scale input same as training
    x_np = scale_input(mri_np, cfg.data.scale_input)
    x_t = torch.from_numpy(x_np[None, None, ...]).to(device=device, dtype=torch.float32)

    # Model
    model = UNet3D(**cfg.model.__dict__).to(device)
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd)
    model.eval()

    # Inference (whole volume)
    if amp and device.type == "cuda":
        from torch.amp import autocast
        with autocast(device_type="cuda", enabled=True):
            pred = model(x_t)
    else:
        pred = model(x_t)

    y_pred = pred.squeeze(0).squeeze(0).detach().cpu().numpy()  # (D,H,W)

    # Invert target scaling to HU if needed
    y_hu = invert_target_scaling(
        y_pred,
        mode=cfg.data.scale_target,
        hu_min=float(cfg.data.hu_min),
        hu_max=float(cfg.data.hu_max),
    )
    if cfg.data.scale_target == "hu" and clamp_hu:
        y_hu = np.clip(y_hu, float(cfg.data.hu_min), float(cfg.data.hu_max))

    # Save as NIfTI (reuse MR affine/header if available)
    save_nifti(Path(out_path), y_hu, affine=affine, header=header)
    print(f"[OK] Wrote predicted CT to: {out_path}")


# -------------------------
# CLI
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Infer a synthetic CT from a single MR NIfTI volume.")
    ap.add_argument("--config", required=True, help="Path to YAML (e.g., configs/default.yaml)")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt (e.g., out/checkpoints/best.pt)")
    ap.add_argument("--mri", required=True, help="Input MR NIfTI path")
    ap.add_argument("--out", required=True, help="Output CT NIfTI path")
    ap.add_argument("--device", default=None, help='Override device ("cuda","cpu","auto")')
    ap.add_argument("--amp", action="store_true", help="Use CUDA autocast during inference")
    ap.add_argument("--no-clamp", dest="clamp_hu", action="store_false", help="Do not clamp HU to [hu_min,hu_max]")
    ap.set_defaults(clamp_hu=True)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        cfg_path=args.config,
        ckpt_path=args.ckpt,
        mri_path=args.mri,
        out_path=args.out,
        device_override=args.device,
        amp=bool(args.amp),
        clamp_hu=bool(args.clamp_hu),
    )
