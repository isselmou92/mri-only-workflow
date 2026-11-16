# simple_sct/metrics/__init__.py
"""
Metrics for simple_sct.

Exports
-------
- mae, rmse, psnr, ssim3d
- compute_all : convenience aggregator returning a dict
"""
from .metrics import mae, rmse, psnr, ssim3d, compute_all  # noqa: F401

__all__ = ["mae", "rmse", "psnr", "ssim3d", "compute_all"]
