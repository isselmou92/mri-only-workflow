# simple_sct/models/simple_unet3d.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(c: int) -> nn.Module:
    # InstanceNorm3d with affine parameters → stable across batch_size=1
    return nn.InstanceNorm3d(c, affine=True, eps=1e-5)


def _act() -> nn.Module:
    return nn.LeakyReLU(negative_slope=0.01, inplace=True)


class DoubleConv(nn.Module):
    """
    Two 3×3×3 convolutions with InstanceNorm + LeakyReLU and a residual shortcut.
    Keeps the spatial size (same padding).
    When in_ch != out_ch a 1×1×1 projection aligns the skip path.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch), _act(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch),
        )
        # 1×1 projection when channel counts differ; identity otherwise
        self.skip = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )
        self.act = _act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.skip(x))


class Down(nn.Module):
    """
    2× downsample via MaxPool followed by DoubleConv.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """
    2× upsample via ConvTranspose3d followed by DoubleConv.
    Handles odd dimension alignment by padding the upsampled tensor
    before concatenating with the corresponding skip connection.
    """
    def __init__(self, in_ch: int, out_ch: int):
        """
        Args:
            in_ch: channels *before* concatenation (i.e., channels of encoder feature),
                   the transposed conv upsamples to in_ch // 2, then we cat with skip
                   to reach 'in_ch' channels again for DoubleConv.
        """
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # spatial alignment (odd dimensions): compute diff and pad
        diffD = skip.size(2) - x.size(2)
        diffH = skip.size(3) - x.size(3)
        diffW = skip.size(4) - x.size(4)
        x = F.pad(
            x,
            [diffW // 2, diffW - diffW // 2,
             diffH // 2, diffH - diffH // 2,
             diffD // 2, diffD - diffD // 2],
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Minimal 3D U-Net for MRI → CT (HU).

    Architecture:
      inc -> [down]*depth -> [up]*depth -> outc
    All blocks are 3×3×3 convs with InstanceNorm + ReLU.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 16,
        depth: int = 3,
        final_act: str = "identity",  # {"identity","tanh"}
    ):
        super().__init__()
        self.final_act = final_act
        f = base_filters

        self.inc = DoubleConv(in_channels, f)

        # Encoder
        self.downs = nn.ModuleList()
        ch = f
        for _ in range(depth):
            self.downs.append(Down(ch, ch * 2))
            ch *= 2

        # Decoder
        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(Up(ch, ch // 2))
            ch //= 2

        # 1×1×1 output conv
        self.outc = nn.Conv3d(ch, out_channels, kernel_size=1)

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        # Kaiming normal for convs, zeros for biases
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skip collection
        x1 = self.inc(x)
        skips = [x1]
        h = x1
        for d in self.downs:
            h = d(h)
            skips.append(h)

        # Decoder with symmetric skips
        for u in self.ups:
            skip = skips.pop(-2)  # last available skip (exclude current h)
            h = u(h, skip)

        y = self.outc(h)
        if self.final_act == "tanh":
            y = torch.tanh(y)
        return y
