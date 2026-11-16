# simple_sct/metrics/fov.py
"""
Field of view of the co-registered micro-CT.

The micro-CT was cropped, rotated onto the MRI and resampled. Voxels of the MRI
frame that lie outside the rotated crop box carry a constant fill value instead
of measured data. Metrics restricted to the CT field of view ignore these voxels.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

CT_FILL_HU = -990.0


def ct_fov_mask(ct_hu: np.ndarray, fill_hu: float = CT_FILL_HU, tol: float = 0.5) -> np.ndarray:
    """True inside the CT field of view; False for fill voxels connected to the volume border."""
    fill = np.abs(np.asarray(ct_hu, dtype=np.float32) - fill_hu) < tol
    labels, _ = ndimage.label(fill)
    faces = [labels[0], labels[-1], labels[:, 0], labels[:, -1], labels[:, :, 0], labels[:, :, -1]]
    border = np.unique(np.concatenate([f.ravel() for f in faces]))
    return ~np.isin(labels, border[border > 0])
