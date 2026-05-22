# simple_sct/data/volume_dataset.py
from __future__ import annotations

"""
Directory-based pairing dataset (no CSV).
- Scans MRI and CT directories using glob patterns.
- Pairs files by filename stem (with optional suffix stripping), or by sorted order.
- Enforces a fixed (D,H,W) shape and applies simple scaling policies.
- Returns whole volumes (C=1).

Windows/PyCharm friendly: uses pathlib; relative paths are resolved against YAML file dir by the config loader.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Prefer nibabel; fallback to SimpleITK if missing.
try:
    import nibabel as nib  # type: ignore
except Exception:
    nib = None

try:
    import SimpleITK as sitk  # type: ignore
except Exception:
    sitk = None


@dataclass
class Pair:
    mri: Path
    ct: Path
    key: str  # normalized pairing key (e.g., stem)


# -----------------------------
# Helpers
# -----------------------------


def _load_nifti(path: Path, reorient_to_canonical: bool = True) -> np.ndarray:
    """
    Load NIfTI and return float32 array of shape (D,H,W).
    
    If reorient_to_canonical=True, reorients the image to RAS+ (Right-Anterior-Superior)
    canonical orientation using the NIfTI header. This ensures MRI and CT are 
    anatomically aligned regardless of how they were originally saved.
    """
    if nib is not None:
        img = nib.load(str(path))
        
        if reorient_to_canonical:
            # Reorient to canonical (RAS+) orientation
            # This uses the affine/qform/sform to determine current orientation
            # and applies the necessary transpose/flips
            img = nib.as_closest_canonical(img)
        
        arr = img.get_fdata(dtype=np.float32)
        return np.asarray(arr, dtype=np.float32)
    
    if sitk is not None:
        img = sitk.ReadImage(str(path))
        # SimpleITK: reorient to LPS (standard) then convert to RAS
        if reorient_to_canonical:
            img = sitk.DICOMOrient(img, 'RAS')
        arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (D,H,W)
        return arr
    
    raise ImportError("Please install nibabel or SimpleITK to read NIfTI volumes.")


def _try_transpose_to_shape(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Attempt to transpose array to match target_shape.
    If shapes already match, return as-is.
    If dimensions are the same but reordered, find and apply the correct transpose.
    Raises ValueError if shapes are incompatible.
    """
    current_shape = arr.shape[:3]
    
    if current_shape == target_shape:
        return arr
    
    # Check if we can transpose to match (same dimensions, different order)
    if sorted(current_shape) != sorted(target_shape):
        raise ValueError(
            f"Shape {current_shape} incompatible with target {target_shape} - "
            f"dimensions don't match even with transposition"
        )
    
    # Find the permutation needed
    # For each position in target, find which position in current has that size
    current_list = list(current_shape)
    used = [False] * 3
    axes = []
    
    for target_dim in target_shape:
        for i, cur_dim in enumerate(current_list):
            if cur_dim == target_dim and not used[i]:
                axes.append(i)
                used[i] = True
                break
    
    if len(axes) != 3:
        raise ValueError(f"Could not determine transpose axes for {current_shape} -> {target_shape}")
    
    return np.transpose(arr, tuple(axes))


def _zscore(x: np.ndarray) -> np.ndarray:
    m = float(x.mean())
    s = float(x.std()) + 1e-8
    return (x - m) / s


def _minmax_to_unit(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    return (x - x_min) / (x_max - x_min + 1e-8)


def _unit_to_pm1(u: np.ndarray) -> np.ndarray:
    return u * 2.0 - 1.0


def unnormalize(x:torch.tensor, x_min: float, x_max: float) -> torch.Tensor:
    norm_01 = (x + 1.0) / 2.0
    return norm_01 * (x_max - x_min) + x_min


# -----------------------------
# Dataset
# -----------------------------

class VolumePairs(Dataset):
    """
    Whole-volume dataset for MRI→CT with directory pairing.

    cfg.data fields used:
      - mri_dir, ct_dir
      - mri_glob, ct_glob
      - pairing: {mode, case_insensitive, mri_strip_suffixes, ct_strip_suffixes}
      - enforce_shape (D,H,W)
      - scale_input, scale_target, hu_min, hu_max
    """

    def __init__(self, cfg):
        dcfg = cfg.data

        # 1) Scan filesystem
        # mri_files = _scan(dcfg.mri_dir, dcfg.mri_glob)
        # ct_files = _scan(dcfg.ct_dir, dcfg.ct_glob)
        self.files = sorted(list(Path(dcfg.mri_dir).iterdir()))

        def _mouse_id(p: Path) -> str:
            return p.name.split(".")[0].split("_", 1)[0].upper()

        def _timepoint_key(p: Path) -> tuple:
            """Natural sort key so T1<T2<...<T10 instead of T1<T10<T2."""
            stem = p.name.split(".")[0]
            import re
            m = re.search(r"T(\d+)$", stem)
            if m:
                return (0, int(m.group(1)), stem)
            return (1, 0, stem)

        # Drop any sample whose mouse ID is in the explicit exclusion list
        # (used to keep the external test cohort out of training/CV even when
        # those volumes physically live in the same directory).
        excluded = getattr(dcfg, "exclude_mouse_ids", None) or []
        if excluded:
            excluded_upper = {str(m).upper() for m in excluded}
            before = len(self.files)
            self.files = [p for p in self.files if _mouse_id(p) not in excluded_upper]
            self._excluded_mouse_ids: list[str] = sorted(excluded_upper)
            self._excluded_sample_count: int = before - len(self.files)
        else:
            self._excluded_mouse_ids = []
            self._excluded_sample_count = 0

        # Optional per-mouse acquisition cap: keep the first N timepoints of
        # every mouse so the dataset is symmetric across mice with different
        # numbers of on-disk acquisitions.
        cap = getattr(dcfg, "max_acquisitions_per_mouse", None)
        if cap is not None and int(cap) > 0:
            cap_n = int(cap)
            by_mouse: dict[str, list[Path]] = {}
            for p in self.files:
                by_mouse.setdefault(_mouse_id(p), []).append(p)
            kept: list[Path] = []
            dropped = 0
            for mid in sorted(by_mouse.keys()):
                paths = sorted(by_mouse[mid], key=_timepoint_key)
                kept.extend(paths[:cap_n])
                dropped += max(0, len(paths) - cap_n)
            self.files = sorted(kept)
            self._cap_per_mouse: Optional[int] = cap_n
            self._capped_sample_count: int = dropped
        else:
            self._cap_per_mouse = None
            self._capped_sample_count = 0

        self.shape: Tuple[int, int, int] = tuple(dcfg.enforce_shape)
        self.reorient_to_canonical: bool = getattr(dcfg, 'reorient_to_canonical', True)

        # 3) Scaling policy
        self.scale_input: str = dcfg.scale_input
        self.scale_target: str = dcfg.scale_target
        self.hu_min: float = float(dcfg.hu_min)
        self.hu_max: float = float(dcfg.hu_max)

        self.crop_prob = cfg.train.crop_prob
        self.flip_prob = cfg.train.flip_prob

        # Master switch for stochastic augmentation. Set to False on any
        # instance used for validation/test evaluation so per-epoch metrics
        # are computed on the deterministic, un-augmented volumes.
        self.augment: bool = True

        if len(self.files) == 0:
            raise RuntimeError("No paired MRI/CT volumes found after pairing.")
    

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        file = self.files[idx]
        mri = file
        ct = file.parent.parent / "CT" / (file.stem.replace("MR", "CT") + file.suffix)  # naive pairing by directory structure
        #ct = file.with_name(file.stem.replace("MR", "CT") + file.suffix)  # naive pairing by filename
        
        # Load with optional canonical reorientation (RAS+)
        # This ensures MRI and CT have consistent anatomical orientation
        x = _load_nifti(mri, reorient_to_canonical=self.reorient_to_canonical)  # (D,H,W)
        y = _load_nifti(ct, reorient_to_canonical=self.reorient_to_canonical)   # (D,H,W)

        # 4) Auto-transpose to match expected shape (handles remaining dimension order differences)
        try:
            x = _try_transpose_to_shape(x, self.shape)
        except ValueError as e:
            raise ValueError(f"MRI {mri}: {e}")
        
        try:
            y = _try_transpose_to_shape(y, self.shape)
        except ValueError as e:
            raise ValueError(f"CT {ct}: {e}")

        # 5) Scaling
        if self.scale_input == "zscore":
            x = _zscore(x)
        elif self.scale_input == "minmax":
            x = _unit_to_pm1(_minmax_to_unit(x, float(x.min()), float(x.max())))
        elif self.scale_input == "none":
            x = x.astype(np.float32)
        else:
            raise ValueError(f"Unknown scale_input: {self.scale_input}")

        if self.scale_target == "hu":
            y = y.astype(np.float32)  # keep HU for interpretable metrics
        elif self.scale_target == "minmax":
            y = _unit_to_pm1(_minmax_to_unit(y, self.hu_min, self.hu_max))
        else:
            raise ValueError(f"Unknown scale_target: {self.scale_target}")

        # 6) Add channel dimension (C=1)
        x = torch.from_numpy(x[None, ...].astype(np.float32))
        y = torch.from_numpy(y[None, ...].astype(np.float32))


        if self.augment and self.crop_prob > 0.0 and np.random.rand() < self.crop_prob:
            pad = 46
            x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad, pad, pad), mode='constant', value=-1)
            y_pad = torch.nn.functional.pad(y, (pad, pad, pad, pad, pad, pad), mode='constant', value=-1)
            d, h, w = x.shape[1:]
            start_d = np.random.randint(0, pad * 2)
            start_h = np.random.randint(0, pad * 2)
            start_w = np.random.randint(0, pad * 2)
            x = x_pad[:, start_d:start_d + d, start_h:start_h + h, start_w:start_w + w]
            y = y_pad[:, start_d:start_d + d, start_h:start_h + h, start_w:start_w + w]

        if self.augment and self.flip_prob > 0.0 and np.random.rand() < self.flip_prob:
            # Anatomical left/right flip. For our (C, D, H, W) tensor under
            # canonical RAS orientation, dim 1 (D) is the R-L axis, so
            # torch.fliplr flips along that axis for both MRI and CT.
            x = torch.fliplr(x)
            y = torch.fliplr(y)


        return {
            "mri": x,     # (1,D,H,W)
            "ct": y,      # (1,D,H,W)
            "idx": torch.tensor(idx, dtype=torch.long),
        }
