"""
Regenerate manuscript/training_curves.png from TensorBoard event files.

Reads the 3 SSIM folds from output/loss_sweep_final_22/tb/sct_ssim_seed1337_fold{0,1,2}
and produces a 6-panel figure:
  (a) Training loss vs. optimisation step
  (b) Batch-level training MAE [HU]
  (c) Validation MAE [HU] per epoch
  (d) Validation RMSE [HU] per epoch
  (e) Validation PSNR [dB] per epoch
  (f) Validation SSIM per epoch

Each panel shows the mean over folds as a solid line with a ±1 std shaded band.
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_scalar(run_dir: str, tag: str):
    """Return (steps, values) numpy arrays for a single run/tag."""
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def interpolate_to_common(steps_list, values_list):
    """
    Interpolate all (steps, values) pairs onto a common x-axis
    (the union of all steps for the tag, limited to the overlap range).
    Returns (common_steps, matrix) where matrix[i] = fold i values.
    """
    # Use the steps from the fold with most points as the reference
    ref_idx = int(np.argmax([len(s) for s in steps_list]))
    common = steps_list[ref_idx]

    matrix = []
    for s, v in zip(steps_list, values_list):
        if len(s) == 0:
            matrix.append(np.full(len(common), np.nan))
        else:
            matrix.append(np.interp(common, s, v))
    return common, np.array(matrix)


def smooth(values, window: int = 10):
    """Simple moving-average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot training curves from TensorBoard logs.")
    ap.add_argument(
        "--tb-dir",
        default="output/loss_sweep_final_22/tb",
        help="Root TensorBoard directory containing per-run sub-folders.",
    )
    ap.add_argument(
        "--loss", default="ssim", choices=["ssim", "l1"],
        help="Which loss to plot (ssim or l1).",
    )
    ap.add_argument(
        "--seed", default="1337",
        help="Seed string used in run directory names.",
    )
    ap.add_argument(
        "--n-folds", type=int, default=3,
        help="Number of cross-validation folds.",
    )
    ap.add_argument(
        "--smooth-window", type=int, default=20,
        help="Moving-average window for step-level curves (0 = no smoothing).",
    )
    ap.add_argument(
        "--out", default="manuscript/training_curves.png",
        help="Output figure path.",
    )
    ap.add_argument(
        "--dpi", type=int, default=200,
        help="Output resolution in DPI.",
    )
    args = ap.parse_args()

    tb_root = Path(args.tb_dir)
    run_dirs = [
        str(tb_root / f"sct_{args.loss}_seed{args.seed}_fold{f}")
        for f in range(args.n_folds)
    ]
    for d in run_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Run directory not found: {d}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading TensorBoard events …")
    tags = {
        "train_loss": "train/loss",
        "train_mae":  "train/batch_metrics/mae",
        "val_mae":    "val/mae",
        "val_rmse":   "val/rmse",
        "val_psnr":   "val/psnr",
        "val_ssim":   "val/ssim",
    }

    data = {}
    for key, tag in tags.items():
        steps_list, values_list = [], []
        for d in run_dirs:
            s, v = load_scalar(d, tag)
            steps_list.append(s)
            values_list.append(v)
        common, matrix = interpolate_to_common(steps_list, values_list)
        data[key] = (common, matrix)
        print(f"  {key}: {len(common)} points, {matrix.shape[0]} folds")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    COLOR = "#1f77b4"   # blue
    ALPHA = 0.25

    panels = [
        ("train_loss", "Optimisation step", "Training loss", "(a) SSIM training loss", True),
        ("train_mae",  "Optimisation step", "MAE [HU]",      "(b) Batch training MAE", True),
        ("val_mae",    "Epoch",             "MAE [HU]",      "(c) Validation MAE",     False),
        ("val_rmse",   "Epoch",             "RMSE [HU]",     "(d) Validation RMSE",    False),
        ("val_psnr",   "Epoch",             "PSNR [dB]",     "(e) Validation PSNR",    False),
        ("val_ssim",   "Epoch",             "SSIM",          "(f) Validation SSIM",    False),
    ]

    for ax, (key, xlabel, ylabel, title, do_smooth) in zip(axes, panels):
        common, matrix = data[key]
        if len(common) == 0:
            ax.set_title(title)
            continue

        mean = np.nanmean(matrix, axis=0)
        std  = np.nanstd(matrix, axis=0)

        if do_smooth and args.smooth_window > 1:
            mean = smooth(mean, args.smooth_window)
            std  = smooth(std,  args.smooth_window)

        ax.plot(common, mean, color=COLOR, linewidth=1.5, label="mean")
        ax.fill_between(common, mean - std, mean + std,
                        color=COLOR, alpha=ALPHA, label="±1 std")

        # Mark best value
        if ylabel.startswith("PSNR") or ylabel == "SSIM":
            best_idx = int(np.nanargmax(mean))
        else:
            best_idx = int(np.nanargmin(mean))
        ax.scatter(common[best_idx], mean[best_idx],
                   color="green", zorder=5, s=40, label=f"best: {mean[best_idx]:.3g}")

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    fig.suptitle(
        f"Training dynamics — SSIM loss, K={args.n_folds} folds (mean ± 1 std)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
