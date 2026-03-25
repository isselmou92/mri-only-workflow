# simple_sct/models/__init__.py
"""
Model zoo (intentionally tiny) for simple_sct.

Exports
-------
- UNet3D : Minimal 3D U-Net with InstanceNorm + ReLU
"""
from .simple_unet3d import UNet3D  # noqa: F401

__all__ = ["UNet3D"]
