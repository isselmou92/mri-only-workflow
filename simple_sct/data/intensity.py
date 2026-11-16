# simple_sct/data/intensity.py
from __future__ import annotations

"""
MR-specific intensity handling: foreground-percentile normalisation, N4 bias
correction and stochastic intensity augmentation.

The augmentations model acquisition differences between scanners, coils and
sequences (receive-coil sensitivity, tissue contrast, blur/sharpness, noise) and act
on the MR only; the CT target is never modified. They operate on a
non-negative, scale-free copy of the MR (divided by its foreground 99.5th
percentile), so the parameters do not depend on the raw scanner units.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage


# -----------------------------
# Foreground and normalisation
# -----------------------------

def foreground_mask(x: np.ndarray, rel_threshold: float = 0.05, sigma: float = 2.0) -> np.ndarray:
    """Rough head mask from the MR alone (usable at inference)."""
    sm = ndimage.gaussian_filter(np.clip(x, 0.0, None), sigma)
    mask = sm > rel_threshold * float(np.percentile(sm, 99.9))
    return mask if mask.any() else np.ones_like(mask, dtype=bool)


def fg_percentile_normalize(x: np.ndarray, p_lo: float = 0.5, p_hi: float = 99.5) -> np.ndarray:
    """Map the foreground [p_lo, p_hi] percentile range to [-1, 1], clipping outside."""
    fg = foreground_mask(x)
    lo, hi = np.percentile(x[fg], [p_lo, p_hi])
    u = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (u * 2.0 - 1.0).astype(np.float32)


# -----------------------------
# N4 bias-field correction
# -----------------------------

def n4_correct(x: np.ndarray, shrink: int = 2) -> np.ndarray:
    """N4 bias-field correction; returns the corrected volume on the input scale."""
    import SimpleITK as sitk

    pos = np.clip(x, 0.0, None).astype(np.float32) + 1e-3
    img = sitk.GetImageFromArray(pos)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    small = sitk.Shrink(img, [shrink] * 3)
    small_mask = sitk.Shrink(mask, [shrink] * 3)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30])
    corrector.Execute(small, small_mask)
    log_field = sitk.GetArrayFromImage(corrector.GetLogBiasFieldAsImage(img))
    return (pos / np.exp(log_field)).astype(np.float32)


# -----------------------------
# Stochastic augmentation
# -----------------------------

@dataclass
class IntensityAugParams:
    prob: float = 0.8                  # chance that a sample is augmented at all
    bias_prob: float = 0.5
    bias_max_log: float = 1.0          # field spans up to exp(+-1) ~ x0.37 .. x2.7
    bias_grid: int = 4                 # control points per axis (low order = smooth)
    gamma_prob: float = 0.5
    gamma_range: Tuple[float, float] = (0.5, 2.0)
    curve_prob: float = 0.5
    curve_knots: int = 5
    curve_jitter: float = 0.15         # max knot displacement on the [0, 1] scale
    blur_prob: float = 0.15
    blur_sigma: Tuple[float, float] = (0.25, 1.0)
    sharpen_prob: float = 0.15
    sharpen_amount: Tuple[float, float] = (0.3, 1.0)
    noise_prob: float = 0.3
    noise_std: Tuple[float, float] = (0.0, 0.05)


def _random_log_bias_field(shape: Tuple[int, ...], grid: int, max_log: float,
                           rng: np.random.Generator) -> np.ndarray:
    coarse = rng.standard_normal((grid,) * len(shape))
    field = ndimage.zoom(coarse, [s / grid for s in shape], order=3, mode="nearest")
    field = field[tuple(slice(0, s) for s in shape)]
    field -= field.mean()
    peak = float(np.abs(field).max()) + 1e-8
    return field * (rng.uniform(0.0, max_log) / peak)


def _random_monotone_curve(u: np.ndarray, knots: int, jitter: float,
                           rng: np.random.Generator) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, knots)
    ys = xs.copy()
    ys[1:-1] += rng.uniform(-jitter, jitter, knots - 2)
    ys = np.maximum.accumulate(np.clip(ys, 0.0, 1.0))
    out = np.interp(u, xs, ys)
    above = u > 1.0
    out[above] = u[above]
    return out


def augment_intensity(x: np.ndarray, p: IntensityAugParams,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply random MR acquisition variations to a raw MR volume."""
    rng = rng or np.random.default_rng(np.random.randint(0, 2**31 - 1))
    if rng.random() >= p.prob:
        return x
    fg = foreground_mask(x)
    scale = float(np.percentile(x[fg], 99.5)) + 1e-8
    u = np.clip(x, 0.0, None).astype(np.float32) / scale

    if rng.random() < p.bias_prob:
        u = u * np.exp(_random_log_bias_field(u.shape, p.bias_grid, p.bias_max_log, rng)).astype(np.float32)
    if rng.random() < p.gamma_prob:
        u = np.power(u, rng.uniform(*p.gamma_range)).astype(np.float32)
    if rng.random() < p.curve_prob:
        u = _random_monotone_curve(u, p.curve_knots, p.curve_jitter, rng).astype(np.float32)
    r = rng.random()
    if r < p.blur_prob:
        u = ndimage.gaussian_filter(u, rng.uniform(*p.blur_sigma))
    elif r < p.blur_prob + p.sharpen_prob:
        u = np.clip(u + rng.uniform(*p.sharpen_amount) * (u - ndimage.gaussian_filter(u, 1.0)), 0.0, None)
    if rng.random() < p.noise_prob:
        std = rng.uniform(*p.noise_std)
        n1 = rng.normal(0.0, std, u.shape).astype(np.float32)
        n2 = rng.normal(0.0, std, u.shape).astype(np.float32)
        u = np.sqrt((u + n1) ** 2 + n2 ** 2)
    return u.astype(np.float32)
