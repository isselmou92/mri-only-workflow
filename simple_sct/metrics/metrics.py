# simple_sct/metrics/metrics.py
from __future__ import annotations

"""
Simple, interpretable metrics for sCT training/evaluation.

All functions expect tensors with shape (N,C,D,H,W) or (C,D,H,W) or (D,H,W).
They will internally promote to (N,C,D,H,W).
- MAE / RMSE are returned as Python floats (detach+item()).
- PSNR defaults to HU range width 4095 ([-1024, 3071]) unless overridden.
- SSIM3D reuses the same implementation as the loss (returns 1 - loss).

Note:
Keep targets in HU if you want MAE/RMSE in HU (recommended).
"""

from typing import Tuple
import math
import torch

from simple_sct.losses.basic import SSIM3D


def _ensure_5d(x: torch.Tensor) -> torch.Tensor:
    """Coerce (D,H,W) or (C,D,H,W) to (N,C,D,H,W) without copy when possible."""
    if x.dim() == 5:
        return x
    if x.dim() == 4:  # (C,D,H,W)
        return x.unsqueeze(0)
    if x.dim() == 3:  # (D,H,W)
        return x.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Expected 3D/4D/5D tensor, got {tuple(x.shape)}")


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = _ensure_5d(pred)
    t = _ensure_5d(target)
    return torch.mean(torch.abs(p - t)).item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = _ensure_5d(pred)
    t = _ensure_5d(target)
    return torch.sqrt(torch.mean((p - t) ** 2)).item()


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 4095.0) -> float:
    """
    PSNR = 20*log10(MAX_I) - 10*log10(MSE)
    Default MAX_I = 4095 for HU width [-1024, 3071].
    """
    p = _ensure_5d(pred)
    t = _ensure_5d(target)
    mse = torch.mean((p - t) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(float(data_range)) - 10.0 * math.log10(mse)


# Reuse the SSIM3D loss with L = 1 - SSIM → SSIM = 1 - L
# Cache SSIM3D instances per data_range to avoid re-creating every call.
_SSIM_CACHE: dict[float, SSIM3D] = {}


def _get_ssim(data_range: float = 8000.0) -> SSIM3D:
    if data_range not in _SSIM_CACHE:
        _SSIM_CACHE[data_range] = SSIM3D(data_range=data_range)
    return _SSIM_CACHE[data_range]


def ssim3d(pred: torch.Tensor, target: torch.Tensor, data_range: float = 8000.0) -> float:
    p = _ensure_5d(pred)
    t = _ensure_5d(target)
    return 1.0 - _get_ssim(data_range)(p, t).item()


def compute_all(pred: torch.Tensor, target: torch.Tensor,
                include_psnr: bool = True, include_ssim: bool = True,
                data_range: float = 8000.0) -> dict:
    """
    Convenience aggregator returning a dict of common metrics.
    """
    out = {
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
    }
    if include_psnr:
        out["psnr"] = psnr(pred, target, data_range=data_range)
    if include_ssim:
        out["ssim"] = ssim3d(pred, target, data_range=data_range)
    return out
