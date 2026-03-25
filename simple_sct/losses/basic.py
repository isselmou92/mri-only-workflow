# simple_sct/losses/basic.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .nmi import NMILoss

def gaussian_kernel_3d(kernel_size: int, sigma: float, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g1 = g1 / g1.sum()
    g3 = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    return g3

class L1(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target))

class SSIM3D(nn.Module):
    """
    3D SSIM as a loss (L = 1 - SSIM).

    For numerical stability the inputs are **normalised to [0, 1]** inside
    ``forward()`` using ``data_range`` before any statistics are computed.
    This avoids float32 precision loss that otherwise occurs when computing
    ``E[x^2] - (E[x])^2`` on large-valued HU volumes (magnitudes ~ 10^3-10^4).

    Args:
        data_range: Dynamic range of the input data (e.g. ``hu_max - hu_min``).
                    Used to rescale inputs to [0, 1] internally.
                    Default 1.0 keeps the original behaviour for [0, 1] data.
    """
    def __init__(self, kernel_size: int = 7, sigma: float = 1.5, data_range: float = 1.0):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for symmetric padding"
        self.ks = int(kernel_size)
        self.sigma = float(sigma)
        self.data_range = float(data_range)
        # Standard SSIM constants for data normalised to [0, 1]
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == y.shape, f"SSIM3D expects same shape, got {x.shape} vs {y.shape}"

        # --- float32 for safety under AMP ---
        dev = x.device
        x32 = x.to(dtype=torch.float32, device=dev)
        y32 = y.to(dtype=torch.float32, device=dev)

        # --- normalise to ~[0, 1] so intermediate values stay O(1) ---
        if self.data_range != 1.0:
            lo = torch.minimum(x32.min(), y32.min()).detach()
            x32 = (x32 - lo) / self.data_range
            y32 = (y32 - lo) / self.data_range

        N, C, D, H, W = x32.shape

        g = gaussian_kernel_3d(self.ks, self.sigma, device=dev, dtype=torch.float32)[None, None, ...]
        pad = self.ks // 2

        def conv(z: torch.Tensor) -> torch.Tensor:
            zc = []
            for c in range(C):
                zc.append(F.conv3d(z[:, c:c+1], g, padding=pad))
            return torch.cat(zc, dim=1)

        mu_x = conv(x32)
        mu_y = conv(y32)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sig_x2 = conv(x32 * x32) - mu_x2
        sig_y2 = conv(y32 * y32) - mu_y2
        sig_xy = conv(x32 * y32) - mu_xy

        # Clamp variances to 0 (can go slightly negative from float arithmetic)
        sig_x2 = sig_x2.clamp(min=0.0)
        sig_y2 = sig_y2.clamp(min=0.0)

        C1, C2 = self.C1, self.C2
        num = (2.0 * mu_xy + C1) * (2.0 * sig_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)

        ssim_map = (num / (den + 1e-8)).clamp(-1.0, 1.0)
        return 1.0 - ssim_map.mean()

class CompositeLoss(nn.Module):
    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.0,
                 ssim_kernel: int = 7, ssim_sigma: float = 1.5,
                 ssim_data_range: float = 1.0):
        super().__init__()
        self.w_l1 = float(l1_weight)
        self.w_ssim = float(ssim_weight)
        self.l1 = L1()
        self.ssim = SSIM3D(kernel_size=int(ssim_kernel), sigma=float(ssim_sigma),
                           data_range=float(ssim_data_range)) if self.w_ssim > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.w_l1 > 0:
            loss = loss + self.w_l1 * self.l1(pred, target)
        if self.ssim is not None:
            # SSIM internally computes in fp32; safe under AMP
            loss = loss + self.w_ssim * self.ssim(pred, target)
        return loss

class ReconstructionLoss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=0.0,
                 ssim_kernel=7, ssim_sigma=1.5, ssim_data_range=1.0,
                 nmi_weight=0.0, nmi_bins=64,
                 nmi_value_min=-1.0, nmi_value_max=1.0, nmi_sigma=None):
        super().__init__()
        self.w_l1 = float(l1_weight)
        self.w_ssim = float(ssim_weight)
        self.w_nmi = float(nmi_weight)

        self.l1 = L1()
        self.ssim = SSIM3D(kernel_size=int(ssim_kernel), sigma=float(ssim_sigma),
                           data_range=float(ssim_data_range)) if self.w_ssim > 0 else None
        self.nmi = NMILoss(
            bins=int(nmi_bins),
            value_min=float(nmi_value_min),
            value_max=float(nmi_value_max),
            sigma=nmi_sigma
        ) if self.w_nmi > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.w_l1 > 0:
            loss = loss + self.w_l1 * self.l1(pred, target)
        if self.ssim is not None:
            loss = loss + self.w_ssim * self.ssim(pred, target)
        if self.nmi is not None:
            loss = loss + self.w_nmi * self.nmi(pred, target)
        return loss