# simple_sct/xai/attribution.py
from __future__ import annotations

"""
Attribution (XAI) utilities for 3D volumes.

Methods
-------
1) Gradient Saliency:
   S = | d sum(pred) / d x |
   - Highlights where input voxels most affect the output globally.

2) SmoothGrad:
   Average of N gradient saliency maps computed on noisy copies of the input.
   Produces dramatically cleaner maps than vanilla gradients.

3) GradCAM (3D):
   Gradient-weighted Class Activation Mapping adapted for 3D regression.
   Hooks into a target convolutional layer to produce smooth, spatially
   interpretable heatmaps highlighting which regions drive the prediction.

4) Occlusion Sensitivity:
   Slide a 3D patch over x, replace with fill value (mean(x) by default),
   re-run model, and record ΔMAE w.r.t. baseline.
   - Highlights causal regions: larger Δ ⇒ region important.

Notes
-----
- These functions are intentionally minimal and transparent.
- They assume the model outputs a single-channel prediction (N,1,D,H,W).
- No orientation logic here; we operate on tensors already aligned upstream.
"""

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _mae(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over all dims."""
    return torch.mean(torch.abs(a - b))


def gradient_saliency(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute |d sum(pred) / d x| as a saliency map.

    Args:
      model: nn.Module in eval mode recommended.
      x: (N,1,D,H,W) input tensor (requires_grad will be set on a clone)

    Returns:
      saliency: (N,1,D,H,W) non-negative tensor
    """
    was_training = model.training
    model.eval()

    x_req = x.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        y = model(x_req)
        s = y.sum()
        s.backward()
    sal = x_req.grad.detach().abs()

    if was_training:
        model.train()
    return sal


def smooth_grad(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_samples: int = 50,
    noise_std: float = 0.15,
) -> torch.Tensor:
    """
    SmoothGrad: average gradient saliency over noisy copies of the input.

    Produces much cleaner, less noisy attribution maps than vanilla gradients.
    Reference: Smilkov et al., "SmoothGrad: removing noise by adding noise", 2017.

    Args:
      model: nn.Module (eval mode recommended).
      x: (N,1,D,H,W) input tensor.
      n_samples: number of noisy samples to average (default 50).
      noise_std: standard deviation of Gaussian noise, relative to
                 the range of x (default 0.15 = 15% of input range).

    Returns:
      saliency: (N,1,D,H,W) non-negative tensor (averaged absolute gradients).
    """
    was_training = model.training
    model.eval()

    # Scale noise to input range
    x_range = (x.max() - x.min()).item()
    sigma = noise_std * x_range

    accum = torch.zeros_like(x)

    for i in range(n_samples):
        x_noisy = x.clone().detach() + torch.randn_like(x) * sigma
        x_noisy.requires_grad_(True)

        with torch.enable_grad():
            y = model(x_noisy)
            y.sum().backward()

        accum += x_noisy.grad.detach().abs()
        if (i + 1) % max(1, n_samples // 5) == 0:
            print(f"  SmoothGrad: {i + 1}/{n_samples}")

    accum /= n_samples

    if was_training:
        model.train()
    return accum


def grad_cam_3d(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_layer: nn.Module,
) -> torch.Tensor:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for 3D regression.

    Hooks into `target_layer` to capture activations and gradients, then
    produces a smooth heatmap showing which spatial regions drive the output.

    Reference: Selvaraju et al., "Grad-CAM", ICCV 2017.

    Args:
      model: nn.Module (eval mode recommended).
      x: (N,1,D,H,W) input tensor.
      target_layer: the convolutional layer to hook into
                    (e.g., model.downs[-1].conv.block[-1] for the bottleneck).

    Returns:
      cam: (N,1,D,H,W) non-negative heatmap, upsampled to input resolution
           and normalised to [0, 1].
    """
    was_training = model.training
    model.eval()

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    # Register hooks
    def _fwd_hook(module, inp, out):
        activations.append(out.detach())

    def _bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h_fwd = target_layer.register_forward_hook(_fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(_bwd_hook)

    # Forward + backward
    x_req = x.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        y = model(x_req)
        y.sum().backward()

    h_fwd.remove()
    h_bwd.remove()

    # Compute Grad-CAM weights: global-average-pool the gradients over spatial dims
    act = activations[0]   # (N, C_feat, D', H', W')
    grad = gradients[0]    # (N, C_feat, D', H', W')

    # α_k = 1/(D'H'W') * Σ_{d,h,w} ∂y/∂A^k_{d,h,w}
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)  # (N, C_feat, 1, 1, 1)

    # Weighted combination of activations
    cam = (weights * act).sum(dim=1, keepdim=True)  # (N, 1, D', H', W')

    # ReLU: only keep positive contributions
    cam = F.relu(cam)

    # Upsample to input resolution
    cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)

    # Normalise to [0, 1] per sample
    N = cam.shape[0]
    for i in range(N):
        c = cam[i]
        c_min, c_max = c.min(), c.max()
        if c_max - c_min > 1e-8:
            cam[i] = (c - c_min) / (c_max - c_min)
        else:
            cam[i] = torch.zeros_like(c)

    if was_training:
        model.train()
    return cam


def occlusion_sensitivity(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: Optional[torch.Tensor] = None,
    patch: Tuple[int, int, int] = (9, 9, 9),
    stride: Tuple[int, int, int] = (8, 8, 8),
    fill: Optional[float] = None,
) -> torch.Tensor:
    """
    Sliding-window occlusion: replace a 3D region with a constant fill
    and measure ΔMAE vs. baseline (if y_true provided) or Δsum(pred).

    Args:
      model: nn.Module
      x: (N,1,D,H,W) input
      y_true: optional ground-truth (N,1,D,H,W). If provided, ΔMAE is used.
              If None, we compute Δsum(pred) magnitude as proxy.
      patch: (pd,ph,pw) occlusion box size
      stride: (sd,sh,sw) sliding strides
      fill: fill value; default is x.mean()

    Returns:
      heat: (N,1,D,H,W) occlusion heatmap (non-negative)
    """
    was_training = model.training
    model.eval()
    device = x.device
    x = x.detach()
    N, C, D, H, W = x.shape
    assert C == 1, "Expected single-channel input for occlusion"

    # Baseline
    with torch.no_grad():
        base_pred = model(x)
        if y_true is not None:
            base_score = _mae(base_pred, y_true).item()
        else:
            base_score = base_pred.sum().item()

    pd, ph, pw = map(int, patch)
    sd, sh, sw = map(int, stride)
    heat = torch.zeros_like(x)

    # Fill value
    if fill is None:
        fill_val = float(x.mean().item())
    else:
        fill_val = float(fill)

    # Slide window over volume
    for z in range(0, max(1, D - pd + 1), sd):
        for i in range(0, max(1, H - ph + 1), sh):
            for j in range(0, max(1, W - pw + 1), sw):
                xo = x.clone()
                xo[..., z:z + pd, i:i + ph, j:j + pw] = fill_val
                with torch.no_grad():
                    pred = model(xo)
                    if y_true is not None:
                        score = _mae(pred, y_true).item()
                    else:
                        score = pred.sum().item()
                delta = score - base_score
                # Accumulate delta over the occluded region
                heat[..., z:z + pd, i:i + ph, j:j + pw] += float(delta)

    # Ensure non-negative (if using Δsum, the sign is informative; for TB we keep magnitude)
    heat = heat.abs()

    if was_training:
        model.train()
    return heat
