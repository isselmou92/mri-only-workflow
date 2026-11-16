# simple_sct/utils/logging.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    print("""WARNING: TensorBoard is not available. Install torch with 'pip install torch[tensorboard]' to enable TB logging.""")
    SummaryWriter = None  # type: ignore


# -------------------------------------------------------------------------
# basic logger + dirs
# -------------------------------------------------------------------------
def get_logger(name: str = "simple_sct", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def prepare_dirs(
    out_dir: Union[str, Path],
    experiment_name: str,
    make_tb: bool,
    log_root: Union[str, Path] = "runs",
) -> Tuple[Path, Path, Path, Optional[Path]]:
    out = Path(out_dir).expanduser().resolve()
    ckpt = out / "checkpoints"
    preds = out / "preds"
    tb = Path(log_root).expanduser().resolve() / experiment_name if make_tb else None

    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    preds.mkdir(parents=True, exist_ok=True)
    if tb:
        tb.mkdir(parents=True, exist_ok=True)
    return out, ckpt, preds, tb


def create_tb_writer(tb_dir: Optional[Union[str, Path]]) -> Optional["SummaryWriter"]:
    if tb_dir is None or SummaryWriter is None:
        return None
    return SummaryWriter(log_dir=str(tb_dir))


# -------------------------------------------------------------------------
# tensor helpers
# -------------------------------------------------------------------------
def _ensure_volume(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 5:      # (N,C,D,H,W)
        t = t[0, 0]
    elif t.dim() == 4:    # (C,D,H,W)
        t = t[0]
    assert t.dim() == 3, f"Expected a 3D volume, got shape {tuple(t.shape)}"
    return t.contiguous().float()


def _normalize01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize tensor to [0,1] using 0.5th/99.5th percentiles (robust to outliers)."""
    flat = x.reshape(-1).float()
    lo = torch.quantile(flat, 0.005)
    hi = torch.quantile(flat, 0.995)
    denom = torch.clamp(hi - lo, min=eps)
    return ((x.float() - lo) / denom).clamp(0.0, 1.0)


def _extract_plane(vol_DHW: torch.Tensor, axis: str, index: int) -> torch.Tensor:
    D, H, W = vol_DHW.shape
    if axis == "axial":
        assert 0 <= index < D
        return vol_DHW[index, :, :]
    elif axis == "sagittal":
        assert 0 <= index < W
        return vol_DHW[:, :, index].transpose(0, 1)  # (H,D)
    elif axis == "coronal":
        assert 0 <= index < H
        return vol_DHW[:, index, :].transpose(0, 1)  # (W,D)
    else:
        raise ValueError(f"Unknown axis '{axis}'")


def _to_chw_gray(img2d_01: torch.Tensor) -> torch.Tensor:
    return img2d_01.unsqueeze(0).repeat(3, 1, 1).clamp(0.0, 1.0)


def _maybe_resize(img_chw: torch.Tensor, target_hw: Optional[Tuple[int, int]]) -> torch.Tensor:
    if target_hw is None:
        return img_chw
    Ht, Wt = target_hw
    if img_chw.shape[-2:] == (Ht, Wt):
        return img_chw
    img = img_chw.unsqueeze(0)
    img = F.interpolate(img, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return img.squeeze(0)


# -------------------------------------------------------------------------
# simple colormaps (pure PyTorch, no matplotlib)
# -------------------------------------------------------------------------
def _apply_cmap(img2d_01: torch.Tensor, cmap: Optional[str]) -> torch.Tensor:
    """
    img2d_01: (H,W) in [0,1]
    returns: (3,H,W) in [0,1]
    """
    if cmap is None or cmap == "" or cmap.lower() == "gray":
        return _to_chw_gray(img2d_01)

    c = cmap.lower()

    if c == "hot":
        # like a poor man's MATLAB hot: black -> red -> yellow -> white
        x = img2d_01.clamp(0, 1)
        r = x
        g = (x - 0.33).clamp(0, 1) / 0.67
        b = (x - 0.66).clamp(0, 1) / 0.34
        return torch.stack([r, g, b], dim=0).clamp(0, 1)

    if c in ("cool", "cold"):
        # cyan -> blue
        x = img2d_01.clamp(0, 1)
        r = torch.zeros_like(x)
        g = 1.0 - x * 0.3
        b = 0.4 + 0.6 * x
        return torch.stack([r, g, b], dim=0).clamp(0, 1)

    if c in ("redblue", "bwr", "blue-red"):
        # center at 0.5 -> white
        x = img2d_01.clamp(0, 1)
        # map 0->blue, 0.5->white, 1->red
        low = (0.5 - x).clamp(min=0) * 2  # blue amount
        high = (x - 0.5).clamp(min=0) * 2  # red amount
        r = high + (1 - high) * 0.9   # keep a bit of white
        g = 1.0 - (low * 0.5 + high * 0.5)
        b = low + (1 - low) * 0.9
        return torch.stack([r, g, b], dim=0).clamp(0, 1)

    if c == "magma":
        # very rough magma-ish
        x = img2d_01.clamp(0, 1)
        r = (x ** 0.7) * 1.0
        g = (x ** 1.1) * 0.5
        b = (1.0 - x) * 0.8
        return torch.stack([r, g, b], dim=0).clamp(0, 1)

    # fallback: gray
    return _to_chw_gray(img2d_01)


# -------------------------------------------------------------------------
# TB: images
# -------------------------------------------------------------------------
def log_volume_slices(
    writer: Optional["SummaryWriter"],
    tag_prefix: str,
    volume: torch.Tensor,
    axes: Sequence[str],
    slice_indices: dict,
    global_step: int,
    normalize: bool = True,
    resize_to: Optional[Tuple[int, int]] = None,
):
    if writer is None:
        return
    vol = _ensure_volume(volume)
    base = _normalize01(vol) if normalize else vol
    for axis in axes:
        idxs: List[int] = list(slice_indices.get(axis, []))
        for idx in idxs:
            plane = _extract_plane(base, axis, int(idx))
            chw = _to_chw_gray(plane)
            chw = _maybe_resize(chw, resize_to)
            writer.add_image(f"{tag_prefix}/{axis}/{idx:03d}", chw, global_step)


def log_triplet_panels(
    writer: Optional["SummaryWriter"],
    tag_prefix: str,
    mri: torch.Tensor,
    pred: torch.Tensor,
    ct: torch.Tensor,
    axes: Sequence[str],
    slice_indices: dict,
    global_step: int,
    normalize_input: bool = True,
    normalize_targets: bool = True,
    resize_to: Optional[Tuple[int, int]] = (256, 256),
):
    if writer is None:
        return
    v_x = _ensure_volume(mri)
    v_p = _ensure_volume(pred)
    v_y = _ensure_volume(ct)

    img_x = _normalize01(v_x) if normalize_input else v_x
    img_p = _normalize01(v_p) if normalize_targets else v_p
    img_y = _normalize01(v_y) if normalize_targets else v_y

    for axis in axes:
        idxs: List[int] = list(slice_indices.get(axis, []))
        for idx in idxs:
            px = _maybe_resize(_to_chw_gray(_extract_plane(img_x, axis, int(idx))), resize_to)
            pp = _maybe_resize(_to_chw_gray(_extract_plane(img_p, axis, int(idx))), resize_to)
            py = _maybe_resize(_to_chw_gray(_extract_plane(img_y, axis, int(idx))), resize_to)
            writer.add_image(f"{tag_prefix}/MRI/{axis}/{idx:03d}", px, global_step)
            writer.add_image(f"{tag_prefix}/PRED/{axis}/{idx:03d}", pp, global_step)
            writer.add_image(f"{tag_prefix}/CT/{axis}/{idx:03d}", py, global_step)


def log_xai_maps(
    writer: Optional["SummaryWriter"],
    tag_prefix: str,
    xai: torch.Tensor,
    axes: Sequence[str],
    slice_indices: dict,
    global_step: int,
    normalize: bool = True,
    resize_to: Optional[Tuple[int, int]] = (256, 256),
    cmap: Optional[str] = None,
):
    """
    Log XAI maps as true color images to TensorBoard.
    We output HWC uint8 + dataformats="HWC" so TB must render color.
    """
    if writer is None:
        return

    vol = _ensure_volume(xai)
    base = _normalize01(vol) if normalize else vol

    for axis in axes:
        idxs: List[int] = list(slice_indices.get(axis, []))
        for idx in idxs:
            plane = _extract_plane(base, axis, int(idx))            # (H,W)
            rgb = _apply_cmap(plane, cmap)                          # (3,H,W), float32 0..1
            rgb = _maybe_resize(rgb, resize_to)                     # (3,H,W)

            # to HWC uint8
            img_hwc = (rgb.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            writer.add_image(
                f"{tag_prefix}/{axis}/{idx:03d}",
                img_hwc,
                global_step,
                dataformats="HWC",
            )



def log_pred_vs_gt_grid(
    writer: Optional["SummaryWriter"],
    tag_prefix: str,
    pred: torch.Tensor,
    ct: torch.Tensor,
    axes: Sequence[str],
    slice_indices: dict,
    global_step: int,
    resize_to: Optional[Tuple[int, int]] = (256, 256),
    normalize_pairwise: bool = True,
):
    if writer is None:
        return

    vol_p = _ensure_volume(pred)
    vol_y = _ensure_volume(ct)

    def _pair(axis: str, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p_plane = _extract_plane(vol_p, axis, idx)
        y_plane = _extract_plane(vol_y, axis, idx)
        if normalize_pairwise:
            mn = torch.min(torch.min(p_plane), torch.min(y_plane))
            mx = torch.max(torch.max(p_plane), torch.max(y_plane))
            denom = torch.clamp(mx - mn, min=1e-8)
            p01 = (p_plane - mn) / denom
            y01 = (y_plane - mn) / denom
        else:
            p01 = _normalize01(p_plane)
            y01 = _normalize01(y_plane)
        return _to_chw_gray(p01), _to_chw_gray(y01)

    sel_axes = list(axes)
    sel_idxs: List[int] = []
    for ax in sel_axes:
        idxs = list(slice_indices.get(ax, []))
        if idxs:
            sel_idxs.append(int(idxs[0]))
        else:
            D, H, W = vol_p.shape
            mid = {"axial": D // 2, "sagittal": W // 2, "coronal": H // 2}[ax]
            sel_idxs.append(mid)

    tiles_pred = []
    tiles_gt = []
    for ax, idx in zip(sel_axes, sel_idxs):
        p_chw, y_chw = _pair(ax, idx)
        p_chw = _maybe_resize(p_chw, resize_to)
        y_chw = _maybe_resize(y_chw, resize_to)
        tiles_pred.append(p_chw)
        tiles_gt.append(y_chw)

    row_pred = torch.cat(tiles_pred, dim=-1)
    row_gt = torch.cat(tiles_gt, dim=-1)
    panel = torch.cat([row_pred, row_gt], dim=-2)

    writer.add_image(f"{tag_prefix}/pred_vs_gt", panel, global_step)


def log_scalars(
    writer: Optional["SummaryWriter"],
    scalars: dict,
    global_step: int,
    prefix: Optional[str] = None,
):
    if writer is None:
        return
    for k, v in scalars.items():
        tag = f"{prefix}/{k}" if prefix else k
        writer.add_scalar(tag, float(v), global_step)
