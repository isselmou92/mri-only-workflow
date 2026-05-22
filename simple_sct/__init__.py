# simple_sct/__init__.py
"""
simple_sct: Minimal, explainable 3D sCT (MRI→CT HU) training kit.

Modules
-------
- config        : YAML-backed typed config + CUDA-first device selection
- data          : Directory-based MRI/CT pairing (no CSV), whole-volume loader
- models        : Tiny 3D U-Net (InstanceNorm + ReLU)
- losses        : L1 (MAE) and 3D-SSIM (transparent math)
- metrics       : MAE, RMSE, PSNR, SSIM3D
- utils.logging : Console & TensorBoard logging (slices + XAI maps)
- utils.seed    : Reproducible seeding & CUDA summary
- xai           : Gradient saliency & occlusion sensitivity

This package is designed for PyCharm/Windows workflows and mirrors a CUDA-first setup.
"""

__version__ = "0.1.0"

# Re-export common entry points for convenience
from .config import CFG, load_config, choose_device  # noqa: F401
from .models.simple_unet3d import UNet3D            # noqa: F401
from .losses.basic import ReconstructionLoss        # noqa: F401
from .metrics.metrics import compute_all            # noqa: F401
