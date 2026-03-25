# simple_sct/xai/__init__.py
"""
Explainability (XAI) utilities for simple_sct.

Exports
-------
- gradient_saliency     : |d sum(pred) / d x| saliency maps
- occlusion_sensitivity : sliding-window occlusion ΔMAE/Δsum heatmaps
"""
from .attribution import gradient_saliency, occlusion_sensitivity  # noqa: F401

__all__ = ["gradient_saliency", "occlusion_sensitivity"]
