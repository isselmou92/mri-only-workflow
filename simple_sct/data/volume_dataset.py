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
from typing import Dict, List, Tuple

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

def _scan(dir_path: str, pattern: str) -> List[Path]:
    root = Path(dir_path)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    files = sorted(root.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {root} with glob '{pattern}'")
    return [p.resolve() for p in files]


def _normalize_stem(stem: str, strip_suffixes: List[str], case_insensitive: bool) -> str:
    s = stem
    for suf in strip_suffixes:
        if case_insensitive:
            if s.lower().endswith(suf.lower()):
                s = s[: -len(suf)]
        else:
            if s.endswith(suf):
                s = s[: -len(suf)]
    return s.lower() if case_insensitive else s


def _pair_by_stem(
    mri_files: List[Path],
    ct_files: List[Path],
    mri_strip: List[str],
    ct_strip: List[str],
    case_insensitive: bool,
) -> List[Pair]:
    # Build dicts from normalized stems to files (last wins if duplicates; warn via comments)
    mri_map: Dict[str, Path] = {}
    for p in mri_files:
        key = _normalize_stem(p.stem, mri_strip, case_insensitive)
        mri_map[key] = p

    ct_map: Dict[str, Path] = {}
    for p in ct_files:
        key = _normalize_stem(p.stem, ct_strip, case_insensitive)
        ct_map[key] = p

    # Intersect keys deterministically
    keys = sorted(set(mri_map.keys()) & set(ct_map.keys()))
    if not keys:
        raise RuntimeError(
            "No MRI/CT pairs matched by stem.\n"
            "Tips: adjust data.pairing.{mri_strip_suffixes,ct_strip_suffixes} "
            "or switch to pairing.mode: sorted"
        )
    pairs = [Pair(mri=mri_map[k], ct=ct_map[k], key=k) for k in keys]
    return pairs


def _pair_by_sorted(mri_files: List[Path], ct_files: List[Path]) -> List[Pair]:
    n = min(len(mri_files), len(ct_files))
    if n == 0:
        raise RuntimeError("Empty MRI or CT list for sorted pairing.")
    # Already sorted by _scan()
    pairs = [Pair(mri=mri_files[i], ct=ct_files[i], key=f"idx_{i:05d}") for i in range(n)]
    return pairs


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
        mri_files = _scan(dcfg.mri_dir, dcfg.mri_glob)
        ct_files = _scan(dcfg.ct_dir, dcfg.ct_glob)

        # 2) Pair
        mode = dcfg.pairing.mode
        if mode == "stem":
            pairs = _pair_by_stem(
                mri_files=mri_files,
                ct_files=ct_files,
                mri_strip=dcfg.pairing.mri_strip_suffixes,
                ct_strip=dcfg.pairing.ct_strip_suffixes,
                case_insensitive=dcfg.pairing.case_insensitive,
            )
        elif mode == "sorted":
            pairs = _pair_by_sorted(mri_files, ct_files)
        else:
            raise ValueError(f"Unknown pairing.mode: {mode}")

        self.pairs: List[Pair] = pairs
        self.shape: Tuple[int, int, int] = tuple(dcfg.enforce_shape)
        self.reorient_to_canonical: bool = getattr(dcfg, 'reorient_to_canonical', True)

        # 3) Scaling policy
        self.scale_input: str = dcfg.scale_input
        self.scale_target: str = dcfg.scale_target
        self.hu_min: float = float(dcfg.hu_min)
        self.hu_max: float = float(dcfg.hu_max)

        # Optional: store for debugging/reporting
        self._pair_keys = [p.key for p in pairs]

        if len(self.pairs) == 0:
            raise RuntimeError("No paired MRI/CT volumes found after pairing.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        
        # Load with optional canonical reorientation (RAS+)
        # This ensures MRI and CT have consistent anatomical orientation
        x = _load_nifti(pair.mri, reorient_to_canonical=self.reorient_to_canonical)  # (D,H,W)
        y = _load_nifti(pair.ct, reorient_to_canonical=self.reorient_to_canonical)   # (D,H,W)

        # 4) Auto-transpose to match expected shape (handles remaining dimension order differences)
        try:
            x = _try_transpose_to_shape(x, self.shape)
        except ValueError as e:
            raise ValueError(f"MRI {pair.mri}: {e}")
        
        try:
            y = _try_transpose_to_shape(y, self.shape)
        except ValueError as e:
            raise ValueError(f"CT {pair.ct}: {e}")

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
        x = x[None, ...].astype(np.float32)
        y = y[None, ...].astype(np.float32)

        return {
            "mri": torch.from_numpy(x),     # (1,D,H,W)
            "ct": torch.from_numpy(y),      # (1,D,H,W)
            "idx": torch.tensor(idx, dtype=torch.long),
        }

    # Optional: expose pair keys (useful for debugging/logging which files were used)
    @property
    def pair_keys(self) -> List[str]:
        return list(self._pair_keys)
