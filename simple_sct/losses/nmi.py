# simple_sct/losses/nmi.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class NMILoss(nn.Module):
    """
    Differentiable NMI loss for 3D single-channel volumes.
    Computes a soft joint histogram using Gaussian binning.

    Args:
        bins: number of histogram bins (e.g., 64)
        value_min, value_max: value range for histogram (match your training scale)
            - if CT/HU: e.g., (-1000.0, 3000.0) or your clip range
            - if normalized [-1,1]: (-1.0, 1.0)
        sigma: Gaussian width in *value units* (≈ 0.5 * bin_width is a good start)
        reduce: "mean" (default) averages NMI over the batch
        eps: numerical stability
    """
    def __init__(
        self,
        bins: int = 64,
        value_min: float = -1.0,
        value_max: float = 1.0,
        sigma: float | None = None,
        reduce: str = "mean",
        eps: float = 1e-7
    ):
        super().__init__()
        assert bins > 1
        self.bins = int(bins)
        self.register_buffer("vmin", torch.tensor(float(value_min)))
        self.register_buffer("vmax", torch.tensor(float(value_max)))
        self.reduce = reduce
        self.eps = float(eps)

        # default sigma: half a bin width
        bw = (float(value_max) - float(value_min)) / float(bins)
        self.sigma = float(sigma) if sigma is not None else 0.5 * bw

        # bin centers buffer (created lazily on first forward when device is known)
        self.register_buffer("centers", torch.empty(0), persistent=False)

    @torch.no_grad()
    def _make_centers(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.centers.numel() != self.bins:
            centers = torch.linspace(self.vmin.item(), self.vmax.item(), self.bins, device=device, dtype=dtype)
            # shift to bin centers
            step = (self.vmax - self.vmin) / (self.bins - 1)
            centers = centers - 0.5 * step + 0.5 * (self.vmax - self.vmin) / self.bins
            self.centers = centers
        return self.centers

    def _soft_assign(self, v: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        # v: [N], centers: [K] -> weights: [N,K]
        # compute Gaussian distances in float32 for stability
        v32 = v.float().unsqueeze(-1)           # [N,1]
        c32 = centers.float().unsqueeze(0)      # [1,K]
        dist2 = (v32 - c32).pow(2)              # [N,K]
        weights = torch.exp(-dist2 / (2.0 * (self.sigma ** 2) + self.eps))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)
        return weights  # [N,K], rows sum to 1

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B, 1, D, H, W] (or broadcastable to same shape)
        returns: 1 - NMI in [0,1]
        """
        assert pred.shape[:2] == target.shape[:2]
        assert pred.shape[1] == 1 and target.shape[1] == 1, "Use single-channel volumes."

        B = pred.shape[0]
        device = pred.device
        centers = self._make_centers(device, pred.dtype)

        # flatten per item
        p = pred.reshape(B, -1)
        t = target.reshape(B, -1)

        losses = []
        for b in range(B):
            pv = p[b]
            tv = t[b]

            # clamp into histogram range (no grad clipping; safe)
            pv = pv.clamp(self.vmin.item(), self.vmax.item())
            tv = tv.clamp(self.vmin.item(), self.vmax.item())

            Wx = self._soft_assign(pv, centers)  # [N,K]
            Wy = self._soft_assign(tv, centers)  # [N,K]

            # marginals
            hx = Wx.sum(dim=0)     # [K]
            hy = Wy.sum(dim=0)     # [K]

            # joint via outer-product accumulation (N-by-K matrices -> K-by-K)
            # Hxy = Wx^T Wy  (each voxel contributes outer product of its soft bins)
            Hxy = Wx.t().matmul(Wy)  # [K,K]

            # normalize
            px = (hx + self.eps) / (hx.sum() + self.eps)
            py = (hy + self.eps) / (hy.sum() + self.eps)
            pxy = (Hxy + self.eps) / (Hxy.sum() + self.eps)

            # entropies
            Hx  = -(px * (px + self.eps).log()).sum()
            Hy  = -(py * (py + self.eps).log()).sum()
            Hxy = -(pxy * (pxy + self.eps).log()).sum()

            MI = Hx + Hy - Hxy
            NMI = 2.0 * MI / (Hx + Hy + self.eps)   # in [0,1]
            losses.append(1.0 - NMI)

        loss = torch.stack(losses).mean() if self.reduce == "mean" else torch.stack(losses).sum()
        return loss
