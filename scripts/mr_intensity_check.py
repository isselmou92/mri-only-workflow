#!/usr/bin/env python
"""
MR Intensity Distribution -- BALB/c vs C57BL/6
==============================================
Tests whether the MR input of the C57BL/6 test animals lies within the
intensity distribution seen during training (BALB/c).

Every MR volume is z-score normalised over the whole volume, as for network
input. Tissue classes are taken from the co-registered CT:
    bone        CT > 300 HU
    soft tissue -100 HU < CT < 100 HU
    head        filled CT > -400 HU

Per animal (averaged over its acquisitions):
  * intensity histogram inside the head
  * mean z-score in bone and soft tissue, bone/soft-tissue contrast
  * 99th percentile inside the head (bright-signal tail)

Group comparison: pairwise 1-Wasserstein distances between the animal
histograms, PERMANOVA pseudo-F with an exact permutation p-value, and the
distance of every animal to the pooled training distribution.

Usage:
    python scripts/mr_intensity_check.py \
        --mr-dir dataset/MR --ct-dir dataset/CT \
        --c57 MZEI00066 MZEI00069 \
        --test MZEI00027 MZEI00031 MZEI00066 MZEI00069 \
        --out output/mr_intensity
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import nibabel as nib
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from strain_pairwise import exact_permutation_test
from strain_morphometry import _style


BINS = np.linspace(-1.5, 8.0, 191)
CENTERS = 0.5 * (BINS[1:] + BINS[:-1])
GROUP_STYLE = {
    "BALB/c training": ("tab:blue", 0.35),
    "BALB/c test": ("navy", 1.0),
    "C57BL/6 test": ("tab:orange", 1.0),
}
FEATURES = [("p99_head", "99th perc.\n(head)"),
            ("bone_z", "Mean\nbone"),
            ("soft_z", "Mean soft\ntissue"),
            ("contrast", "Bone −\nsoft tissue")]


def acquisition_features(mr_path: str, ct_path: str) -> Dict:
    mr = nib.load(mr_path).get_fdata()
    ct = nib.load(ct_path).get_fdata()
    z = (mr - mr.mean()) / mr.std()
    head = ndimage.binary_fill_holes(ndimage.binary_closing(ct > -400, iterations=3))
    bone = (ct > 300) & head
    soft = (ct > -100) & (ct < 100) & head
    hist, _ = np.histogram(np.clip(z[head], BINS[0], BINS[-1]), bins=BINS)
    return {
        "hist": hist / hist.sum(),
        "p99_head": float(np.percentile(z[head], 99)),
        "bone_z": float(z[bone].mean()),
        "soft_z": float(z[soft].mean()),
        "contrast": float(z[bone].mean() - z[soft].mean()),
    }


def wasserstein(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))) * (BINS[1] - BINS[0]))


def make_figure(animals: List[str], group: Dict[str, str], per_animal: Dict[str, Dict],
                per_acq: Dict[str, List[Dict]], stats: Dict, out: Path):
    _style()
    fig = plt.figure(figsize=(7.4, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.35)
    rng = np.random.default_rng(0)
    order = list(GROUP_STYLE)

    ax = fig.add_subplot(gs[0, :])
    for g in order:
        for a in animals:
            if group[a] == g:
                col, alpha = GROUP_STYLE[g]
                ax.plot(CENTERS, per_animal[a]["hist"], color=col, alpha=alpha,
                        lw=0.8 if g == "BALB/c training" else 1.6)
    ax.set_xlabel("MR intensity inside the head (z-score, per-volume normalisation)")
    ax.set_ylabel("Fraction of voxels")
    ax.set_xlim(-1.5, 6.0)
    ax.grid(alpha=0.3)
    ax.legend(handles=[Line2D([0], [0], color=GROUP_STYLE[g][0], lw=1.6,
                              alpha=max(GROUP_STYLE[g][1], 0.6),
                              label=f"{g} (n={sum(group[a] == g for a in animals)})")
                       for g in order], fontsize=7.5, frameon=False)
    ax.set_title("(a) MR intensity distribution per animal", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    for fi, (key, _) in enumerate(FEATURES):
        train_vals = [per_animal[a][key] for a in animals if group[a] == "BALB/c training"]
        ax.vlines(fi, min(train_vals), max(train_vals), color="tab:blue", lw=8, alpha=0.2)
        for gi, g in enumerate(order):
            col, _ = GROUP_STYLE[g]
            for a in animals:
                if group[a] != g:
                    continue
                acq = [r[key] for r in per_acq[a]]
                x = fi + (gi - 1) * 0.22 + rng.uniform(-0.05, 0.05)
                ax.scatter(x, per_animal[a][key], s=14 if gi == 0 else 26, color=col,
                           edgecolor="k", lw=0.3, zorder=3)
                if gi > 0:
                    ax.vlines(x, min(acq), max(acq), color=col, lw=0.8, zorder=2)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([lbl for _, lbl in FEATURES], fontsize=7)
    ax.set_ylabel("z-score")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("(b) Tissue intensities (bars: training range)", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    for gi, g in enumerate(order):
        col, _ = GROUP_STYLE[g]
        vals = [stats["distance_to_training"][a] for a in animals if group[a] == g]
        ax.scatter(gi + rng.uniform(-0.12, 0.12, len(vals)), vals, s=20 if gi == 0 else 34,
                   color=col, edgecolor="k", lw=0.3)
        if gi > 0:
            for a in animals:
                if group[a] == g:
                    ax.annotate(a.replace("MZEI000", ""), (gi, stats["distance_to_training"][a]),
                                xytext=(8, -2), textcoords="offset points", fontsize=7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([g.replace(" ", "\n", 1) for g in order], fontsize=7.5)
    ax.set_ylabel("Wasserstein distance to\ntraining distribution (z-score)")
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"(c) Strain: pseudo-F = {stats['pseudo_F']:.1f}, p = {stats['p_exact']:.3f}",
                 loc="left", fontsize=9)

    for ext in ["png", "pdf"]:
        fig.savefig(out / f"fig_mr_intensity.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"  Figure saved: {out / 'fig_mr_intensity'}.[png|pdf]")


def main():
    ap = argparse.ArgumentParser(description="MR intensity distribution BALB/c vs C57BL/6")
    ap.add_argument("--mr-dir", type=Path, required=True)
    ap.add_argument("--ct-dir", type=Path, required=True)
    ap.add_argument("--c57", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("output/mr_intensity"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per_acq: Dict[str, List[Dict]] = defaultdict(list)
    for mr in sorted(glob.glob(str(args.mr_dir / "MZEI*_MR_T*.nii.gz"))):
        sid, tp = re.search(r"(MZEI\d+)_MR_(T\d+)", mr).groups()
        ct = args.ct_dir / f"{sid}_CT_{tp}.nii.gz"
        if ct.exists():
            per_acq[sid].append({"tp": tp, **acquisition_features(mr, str(ct))})

    animals = sorted(per_acq)
    group = {a: ("C57BL/6 test" if a in args.c57 else
                 "BALB/c test" if a in args.test else "BALB/c training") for a in animals}
    per_animal = {}
    for a in animals:
        acq = per_acq[a]
        per_animal[a] = {"hist": np.mean([r["hist"] for r in acq], axis=0),
                         **{k: float(np.mean([r[k] for r in acq])) for k, _ in FEATURES}}

    n = len(animals)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = wasserstein(per_animal[animals[i]]["hist"],
                                            per_animal[animals[j]]["hist"])
    strain = np.array(["C57BL/6" if a in args.c57 else "BALB/c" for a in animals])
    F, p, n_perm = exact_permutation_test(D, strain)

    train = [a for a in animals if group[a] == "BALB/c training"]
    dist_to_train = {}
    for a in animals:
        ref = [t for t in train if t != a]
        pooled = np.mean([per_animal[t]["hist"] for t in ref], axis=0)
        dist_to_train[a] = wasserstein(per_animal[a]["hist"], pooled)

    stats = {"pseudo_F": F, "p_exact": p, "n_permutations": n_perm,
             "distance_to_training": dist_to_train, "features": {}}
    print(f"  Animals: {n} ({len(train)} BALB/c training, "
          f"{sum(g == 'BALB/c test' for g in group.values())} BALB/c test, "
          f"{len(args.c57)} C57BL/6 test)")
    print(f"  Histogram distance, strain effect: pseudo-F {F:.2f}, exact p = {p:.4f} "
          f"({n_perm} relabellings)\n")
    print(f"  {'feature':<10s} {'training range':>16s}   " +
          "   ".join(f"{a.replace('MZEI000', '')}" + " (acq. range)" for a in args.test))
    for key, _ in FEATURES + [("dist", "")]:
        if key == "dist":
            tr = [dist_to_train[a] for a in train]
            vals = {a: (dist_to_train[a], None) for a in args.test}
        else:
            tr = [per_animal[a][key] for a in train]
            vals = {a: (per_animal[a][key], (min(r[key] for r in per_acq[a]),
                                             max(r[key] for r in per_acq[a])))
                    for a in args.test}
        stats["features"][key] = {"training_range": [min(tr), max(tr)],
                                  **{a: v[0] for a, v in vals.items()}}
        cells = []
        for a in args.test:
            v, rr = vals[a]
            cells.append(f"{v:6.2f}" + (f" ({rr[0]:.2f}–{rr[1]:.2f})" if rr else " " * 14))
        print(f"  {key:<10s} {min(tr):7.2f} – {max(tr):<6.2f}   " + "   ".join(cells))

    with open(args.out / "mr_intensity_per_acquisition.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["animal", "group", "timepoint"] + [k for k, _ in FEATURES])
        for a in animals:
            for r in per_acq[a]:
                w.writerow([a, group[a], r["tp"]] + [f"{r[k]:.4f}" for k, _ in FEATURES])
    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    make_figure(animals, group, per_animal, per_acq, stats, args.out)


if __name__ == "__main__":
    main()
