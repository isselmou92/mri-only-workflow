# xai_figures.py
"""
Generate publication-quality XAI figures from the trained 3D U-Net.

Runs gradient saliency, SmoothGrad, and GradCAM on a validation sample
and produces multi-panel figures for the paper.

Usage:
    python xai_figures.py --config configs/default.yaml --ckpt checkpoints/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from simple_sct.config import load_config, choose_device
from simple_sct.models.simple_unet3d import UNet3D
from simple_sct.data.volume_dataset import VolumePairs
from simple_sct.xai.attribution import gradient_saliency, smooth_grad, grad_cam_3d

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.linewidth": 0.6,
}


def normalize01(arr: np.ndarray, plow: float = 0.5, phigh: float = 99.5) -> np.ndarray:
    """Robust percentile normalization to [0, 1]."""
    lo = np.percentile(arr, plow)
    hi = np.percentile(arr, phigh)
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def extract_slices(vol: np.ndarray, axial_idx: int, sag_idx: int, cor_idx: int):
    """Extract 2D slices from a 3D volume (D, H, W), rotated 180° for correct orientation."""
    return [
        np.rot90(vol[axial_idx, :, :], 2),       # axial
        np.rot90(vol[:, :, sag_idx], 2),         # sagittal
        np.rot90(vol[:, cor_idx, :], 2),         # coronal
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--out_dir", default="output/xai_figures")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoothgrad_n", type=int, default=50)
    parser.add_argument("--smoothgrad_std", type=float, default=0.15)
    parser.add_argument("--cmap", default="inferno")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)

    # ---- Load config, model, data ----
    cfg = load_config(args.config)
    device = choose_device(args.device or cfg.train.device_preference)
    print(f"Device: {device}")

    model = UNet3D(**cfg.model.__dict__).to(device)
    state = torch.load(args.ckpt, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded model from {args.ckpt}")

    # Load a validation sample
    ds = VolumePairs(cfg)
    # Find a validation subject (e.g., MZEI00066 or MZEI00030)
    sample_idx = 0
    for i, pair in enumerate(ds.pairs):
        stem = pair.mri.stem.upper()
        if "MZEI00066" in stem or "MZEI00030" in stem:
            sample_idx = i
            break
    
    print(f"Using sample {sample_idx}: {ds.pairs[sample_idx].mri.stem}")
    sample = ds[sample_idx]
    x = sample["mri"].unsqueeze(0).to(device)  # (1, 1, D, H, W)
    y = sample["ct"].unsqueeze(0).to(device)

    # Slice indices (from config)
    ax_idx, sag_idx, cor_idx = 64, 64, 56
    axes_labels = ["Axial", "Sagittal", "Coronal"]

    # ---- Run inference ----
    print("Running inference...")
    with torch.no_grad():
        pred = model(x)

    # ---- Run XAI methods ----
    print("Computing vanilla gradient saliency...")
    sal_vanilla = gradient_saliency(model, x)

    print(f"Computing SmoothGrad (n={args.smoothgrad_n}, std={args.smoothgrad_std})...")
    sal_smooth = smooth_grad(model, x, n_samples=args.smoothgrad_n, noise_std=args.smoothgrad_std)

    print("Computing GradCAM...")
    # Target the last encoder block (deepest features = most semantically rich)
    target_layer = model.downs[-1].conv.block  # the Sequential in the bottleneck-adjacent encoder
    cam = grad_cam_3d(model, x, target_layer)

    # ---- Convert to numpy ----
    x_np = x[0, 0].cpu().numpy()
    pred_np = pred[0, 0].cpu().numpy()
    y_np = y[0, 0].cpu().numpy()
    sal_v_np = sal_vanilla[0, 0].cpu().numpy()
    sal_s_np = sal_smooth[0, 0].cpu().numpy()
    cam_np = cam[0, 0].cpu().numpy()

    # Normalize for display
    x_disp = normalize01(x_np)
    pred_disp = normalize01(pred_np)
    y_disp = normalize01(y_np)
    sal_v_disp = normalize01(sal_v_np, plow=0, phigh=99)
    sal_s_disp = normalize01(sal_s_np, plow=0, phigh=99)
    # GradCAM is already [0,1] from the function
    cam_disp = cam_np

    # ======================================================================
    # MAIN FIGURE: Comparison of XAI methods (4 rows × 3 cols)
    #   Row 0: Input MRI
    #   Row 1: Vanilla Gradient Saliency
    #   Row 2: SmoothGrad
    #   Row 3: Grad-CAM
    # ======================================================================
    print("Generating main saliency comparison figure...")
    fig = plt.figure(figsize=(7.16, 8.5))
    gs = gridspec.GridSpec(4, 3, hspace=0.06, wspace=0.04,
                           left=0.10, right=0.93, top=0.95, bottom=0.02)

    cmap_xai = plt.get_cmap(args.cmap)

    data_rows = [
        ("Input MRI", [extract_slices(x_disp, ax_idx, sag_idx, cor_idx)], "gray", False),
        ("Gradient\nsaliency", [extract_slices(sal_v_disp, ax_idx, sag_idx, cor_idx)], args.cmap, True),
        ("SmoothGrad", [extract_slices(sal_s_disp, ax_idx, sag_idx, cor_idx)], args.cmap, True),
        ("Grad-CAM", [extract_slices(cam_disp, ax_idx, sag_idx, cor_idx)], args.cmap, True),
    ]

    im_ref = None
    for row, (label, slice_data, cmap, is_xai) in enumerate(data_rows):
        slices = slice_data[0]
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(slices[col].T if col > 0 else slices[col],
                          cmap=cmap, vmin=0, vmax=1,
                          origin="lower", aspect="auto")
            ax.axis("off")
            if row == 0:
                ax.set_title(axes_labels[col], fontweight="bold", fontsize=11)
            if is_xai and col == 2:
                im_ref = im

    # Row labels
    row_y_positions = [0.85, 0.63, 0.40, 0.18]
    for label, y_pos in zip([d[0] for d in data_rows], row_y_positions):
        fig.text(0.03, y_pos, label, va="center", ha="center",
                 fontsize=9, fontweight="bold", rotation=90)

    # Colorbar
    if im_ref is not None:
        cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.35])
        cb = fig.colorbar(im_ref, cax=cbar_ax)
        cb.set_label("Attribution", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    p = out / "saliency_maps.png"
    fig.savefig(p, dpi=300)
    print(f"Saved  {p}")
    plt.close(fig)

    # ======================================================================
    # COMPACT FIGURE: MRI + SmoothGrad + Predicted sCT (3 rows × 3 cols)
    # ======================================================================
    print("Generating compact figure (MRI / SmoothGrad / sCT)...")
    fig2 = plt.figure(figsize=(7.16, 6.8))
    gs2 = gridspec.GridSpec(3, 3, hspace=0.06, wspace=0.04,
                            left=0.10, right=0.93, top=0.95, bottom=0.02)

    compact_rows = [
        ("Input MRI", extract_slices(x_disp, ax_idx, sag_idx, cor_idx), "gray"),
        ("SmoothGrad", extract_slices(sal_s_disp, ax_idx, sag_idx, cor_idx), args.cmap),
        ("Predicted sCT", extract_slices(pred_disp, ax_idx, sag_idx, cor_idx), "gray"),
    ]

    for row, (label, slices, cmap) in enumerate(compact_rows):
        for col in range(3):
            ax = fig2.add_subplot(gs2[row, col])
            im = ax.imshow(slices[col].T if col > 0 else slices[col],
                          cmap=cmap, vmin=0, vmax=1,
                          origin="lower", aspect="auto")
            ax.axis("off")
            if row == 0:
                ax.set_title(axes_labels[col], fontweight="bold", fontsize=11)

    # Row labels
    for i, (label, _, _) in enumerate(compact_rows):
        y_pos = 0.82 - i * 0.32
        fig2.text(0.03, y_pos, label, va="center", ha="center",
                  fontsize=10, fontweight="bold", rotation=90)

    # Colorbar for SmoothGrad row
    cbar_ax2 = fig2.add_axes([0.94, 0.38, 0.012, 0.25])
    sm = plt.cm.ScalarMappable(cmap=args.cmap, norm=plt.Normalize(0, 1))
    cb2 = fig2.colorbar(sm, cax=cbar_ax2)
    cb2.set_label("Saliency", fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    p2 = out / "saliency_smoothgrad.png"
    fig2.savefig(p2, dpi=300)
    print(f"Saved  {p2}")
    plt.close(fig2)

    # ======================================================================
    # GRADCAM-specific figure: MRI + GradCAM overlay (2 rows × 3 cols)
    # ======================================================================
    print("Generating GradCAM overlay figure...")
    fig3 = plt.figure(figsize=(7.16, 4.8))
    gs3 = gridspec.GridSpec(2, 3, hspace=0.06, wspace=0.04,
                            left=0.10, right=0.93, top=0.93, bottom=0.02)

    mri_slices = extract_slices(x_disp, ax_idx, sag_idx, cor_idx)
    cam_slices = extract_slices(cam_disp, ax_idx, sag_idx, cor_idx)

    for col in range(3):
        # Row 0: MRI
        ax = fig3.add_subplot(gs3[0, col])
        s = mri_slices[col].T if col > 0 else mri_slices[col]
        ax.imshow(s, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto")
        ax.set_title(axes_labels[col], fontweight="bold", fontsize=11)
        ax.axis("off")

        # Row 1: GradCAM overlaid on MRI
        ax = fig3.add_subplot(gs3[1, col])
        ax.imshow(s, cmap="gray", vmin=0, vmax=1, origin="lower", aspect="auto")
        c = cam_slices[col].T if col > 0 else cam_slices[col]
        im3 = ax.imshow(c, cmap="jet", vmin=0, vmax=1, alpha=0.5,
                        origin="lower", aspect="auto")
        ax.axis("off")

    fig3.text(0.03, 0.72, "Input MRI", va="center", ha="center",
              fontsize=10, fontweight="bold", rotation=90)
    fig3.text(0.03, 0.28, "Grad-CAM\noverlay", va="center", ha="center",
              fontsize=10, fontweight="bold", rotation=90)

    cbar_ax3 = fig3.add_axes([0.94, 0.15, 0.012, 0.35])
    cb3 = fig3.colorbar(im3, cax=cbar_ax3)
    cb3.set_label("Grad-CAM", fontsize=8)
    cb3.ax.tick_params(labelsize=7)

    p3 = out / "gradcam_overlay.png"
    fig3.savefig(p3, dpi=300)
    print(f"Saved  {p3}")
    plt.close(fig3)

    print("\nDone! Generated:")
    print(f"  {out / 'saliency_maps.png'}        — 4-method comparison")
    print(f"  {out / 'saliency_smoothgrad.png'}   — MRI / SmoothGrad / sCT")
    print(f"  {out / 'gradcam_overlay.png'}       — GradCAM overlaid on MRI")


if __name__ == "__main__":
    main()
