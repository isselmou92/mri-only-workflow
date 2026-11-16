# simple_sct/data/__init__.py
"""
Data loading for simple_sct.

Exports
-------
- VolumePairs : Directory-based whole-volume MRI→CT dataset (no CSV), with strict
                (D,H,W) enforcement and simple scaling policies.

Usage
-----
from simple_sct.data import VolumePairs
ds = VolumePairs(cfg)
"""
from .volume_dataset import VolumePairs  # noqa: F401

__all__ = ["VolumePairs"]
