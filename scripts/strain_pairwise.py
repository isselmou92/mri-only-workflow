#!/usr/bin/env python
"""
Pairwise Inter-strain Skull Comparison -- BALB/c vs C57BL/6
===========================================================
Every animal is registered to every other animal (one baseline CT each), so
no single reference animal biases the result. For each pair the skull-surface
distance in the cranial ROI is recorded after rigid and after affine alignment.

Outputs:
  * Distance matrices (rigid / affine) and a group test per alignment level:
    PERMANOVA pseudo-F with an exact permutation p-value (all relabellings)
  * Classical MDS embedding of the affine-residual distances
  * Bone DSC per strain for sCT^avg and sCT^UNet as reported in the manuscript

Bone threshold (300 HU) and CT clipping (-1000 to 3000 HU) follow the manuscript.

Usage:
    python scripts/strain_pairwise.py \
        --ct-dir dataset/CT \
        --balbc MZEI00018 ... MZEI00039 \
        --c57 MZEI00066 MZEI00069 \
        --test MZEI00027 MZEI00031 MZEI00066 MZEI00069 \
        --out output/strain_pairwise
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_alignment import load_sitk, sitk_to_numpy, rigid_register  # noqa: E402
from strain_morphometry import _clamp, _linear_registration, bone_agreement, cranial_roi, _style


LEVELS = ["rigid", "affine"]
COLORS = {"BALB/c": "tab:blue", "C57BL/6": "tab:orange"}
MR_VOXEL_MM = 0.14
DOSE_GRID_MM = 0.2

# Bone DSC ranges on the test set as reported in the manuscript (Section 3.2).
PAPER_BONE_DSC = {
    "avg": {"BALB/c": (0.79, 0.80), "C57BL/6": (0.70, 0.77)},
    "unet": {"BALB/c": (0.73, 0.78), "C57BL/6": (0.53, 0.59)},
}
METHOD_LABELS = {"avg": r"sCT$^{\mathrm{avg}}$ (rigid)", "unet": r"sCT$^{\mathrm{UNet}}$"}


# ---------------------------------------------------------------------------
# Pairwise registration
# ---------------------------------------------------------------------------

def register_pair(fixed_path: str, moving_path: str, max_iterations: int,
                  roi_z_min_mm: float) -> Dict[str, float]:
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(3)
    fixed, moving = load_sitk(Path(fixed_path)), load_sitk(Path(moving_path))
    fixed_c, moving_c = _clamp(fixed), _clamp(moving)
    spacing = float(fixed.GetSpacing()[0])
    fixed_np = sitk_to_numpy(fixed)
    roi = cranial_roi(fixed_np.shape, spacing, roi_z_min_mm)

    _, rigid, _ = rigid_register(fixed_c, moving_c, max_iterations=max_iterations,
                                 verbose=False)
    rigid = sitk.Euler3DTransform(rigid)
    aff0 = sitk.AffineTransform(3)
    aff0.SetCenter(rigid.GetCenter())
    aff0.SetMatrix(rigid.GetMatrix())
    aff0.SetTranslation(rigid.GetTranslation())
    affine = sitk.AffineTransform(_linear_registration(fixed_c, moving_c, aff0, max_iterations))

    ones = sitk.Image(moving.GetSize(), sitk.sitkFloat32) + 1.0
    ones.CopyInformation(moving)
    out = {}
    for level, tx in [("rigid", rigid), ("affine", affine)]:
        res = sitk_to_numpy(sitk.Resample(moving, fixed, tx, sitk.sitkLinear, -1000.0))
        overlap = sitk_to_numpy(sitk.Resample(ones, fixed, tx, sitk.sitkNearestNeighbor, 0.0)) > 0.5
        m = bone_agreement(fixed_np, res, overlap & roi, spacing)
        out[f"{level}_msd_mm"] = m["msd_mm"]
        out[f"{level}_dice"] = m["dice"]
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def permanova_f(D: np.ndarray, labels: np.ndarray) -> float:
    """Pseudo-F of Anderson (2001) on a distance matrix."""
    n = len(labels)
    groups = np.unique(labels)
    d2 = D ** 2
    ss_total = d2[np.triu_indices(n, 1)].sum() / n
    ss_within = 0.0
    for g in groups:
        idx = np.where(labels == g)[0]
        sub = d2[np.ix_(idx, idx)]
        ss_within += sub[np.triu_indices(len(idx), 1)].sum() / len(idx)
    a = len(groups)
    return float(((ss_total - ss_within) / (a - 1)) / (ss_within / (n - a)))


def exact_permutation_test(D: np.ndarray, labels: np.ndarray) -> Tuple[float, float, int]:
    """Exact p-value over every assignment of the minority-group size."""
    n = len(labels)
    minority = min(np.unique(labels), key=lambda g: np.sum(labels == g))
    k = int(np.sum(labels == minority))
    f_obs = permanova_f(D, labels)
    count, total = 0, 0
    for combo in itertools.combinations(range(n), k):
        perm = np.array(["a"] * n, dtype=object)
        perm[list(combo)] = "b"
        count += permanova_f(D, perm) >= f_obs - 1e-12
        total += 1
    return f_obs, count / total, total


def classical_mds(D: np.ndarray, dims: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    explained = np.clip(w, 0, None) / np.clip(w, 0, None).sum()
    return v[:, :dims] * np.sqrt(np.clip(w[:dims], 0, None)), explained[:dims]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(ids: List[str], strain: np.ndarray, mats: Dict[str, np.ndarray],
                stats: Dict[str, Dict], out: Path):
    _style()
    fig = plt.figure(figsize=(7.4, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.55)
    order = np.argsort(strain == "C57BL/6", kind="stable")
    short = [i.replace("MZEI000", "") for i in ids]

    ax = fig.add_subplot(gs[0, 0])
    D = mats["affine"][np.ix_(order, order)]
    im = ax.imshow(D, cmap="viridis")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([short[i] for i in order], rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([short[i] for i in order], fontsize=5.5)
    nb = int(np.sum(strain == "BALB/c"))
    for p in [nb - 0.5]:
        ax.axhline(p, color="w", lw=0.8)
        ax.axvline(p, color="w", lw=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Surface distance after affine (mm)", fontsize=7.5)
    ax.set_title("(a) Pairwise skull distance (affine)", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    Y, expl = classical_mds(mats["affine"])
    for s in COLORS:
        m = strain == s
        ax.scatter(Y[m, 0], Y[m, 1], color=COLORS[s], s=28, label=s, edgecolor="k", lw=0.3)
    for i in np.where(strain == "C57BL/6")[0]:
        ax.annotate(short[i], Y[i], xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel(f"MDS 1 ({expl[0] * 100:.0f} %)")
    ax.set_ylabel(f"MDS 2 ({expl[1] * 100:.0f} %)")
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_title("(b) Embedding of affine-residual distances", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(0)
    iu = np.triu_indices(len(ids), 1)
    cats = [("BALB/c–BALB/c", "BALB/c", "BALB/c", "tab:blue"),
            ("BALB/c–C57BL/6", "BALB/c", "C57BL/6", "tab:purple"),
            ("C57BL/6–C57BL/6", "C57BL/6", "C57BL/6", "tab:orange")]
    for li, level in enumerate(LEVELS):
        d = mats[level][iu]
        a, b = strain[iu[0]], strain[iu[1]]
        for ci, (_, g1, g2, col) in enumerate(cats):
            m = ((a == g1) & (b == g2)) | ((a == g2) & (b == g1))
            x = li * 4 + ci
            ax.scatter(x + rng.uniform(-0.18, 0.18, m.sum()), d[m], s=6, color=col, alpha=0.6, lw=0)
            ax.hlines(np.median(d[m]), x - 0.3, x + 0.3, color="k", lw=1.2)
        s = stats[level]
        ax.annotate(f"pseudo-F = {s['pseudo_F']:.1f}, p = {s['p_exact']:.3f}",
                    (li * 4 + 1, 1.02), xycoords=("data", "axes fraction"), ha="center",
                    va="bottom", fontsize=7)
    ax.set_xticks([1, 5])
    ax.set_xticklabels(["After rigid", "After affine"])
    ax.set_ylabel("Skull surface distance (mm)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-0.7, 6.7)
    ax.set_ylim(0, None)
    for y, txt in [(MR_VOXEL_MM, "MR voxel"), (DOSE_GRID_MM, "dose grid")]:
        ax.axhline(y, color="0.3", lw=0.7, ls="--")
        ax.text(6.65, y, txt, ha="right", va="bottom", fontsize=6.5, color="0.3")
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=c, label=l)
                       for l, _, _, c in cats], fontsize=6.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, handletextpad=0.2, columnspacing=0.8)
    ax.set_title("(c) Within- vs. between-strain distances", loc="left", fontsize=9, pad=14)

    ax = fig.add_subplot(gs[1, 1])
    for mi, (method, label) in enumerate(METHOD_LABELS.items()):
        for si, s in enumerate(COLORS):
            lo, hi = PAPER_BONE_DSC[method][s]
            x = mi + (si - 0.5) * 0.28
            ax.vlines(x, lo, hi, color=COLORS[s], lw=7, alpha=0.85,
                      label=f"{s} test" if mi == 0 else None)
            ax.text(x, hi + 0.012, f"{lo:.2f}–{hi:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(METHOD_LABELS)))
    ax.set_xticklabels(list(METHOD_LABELS.values()))
    ax.set_xlim(-0.6, len(METHOD_LABELS) - 0.4)
    ax.set_ylim(0.45, 0.88)
    ax.set_ylabel(r"Bone DSC ($\tau$ = 300 HU)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_title("(d) Bone DSC per strain (Section 3.2)", loc="left", fontsize=9)

    for ext in ["png", "pdf"]:
        fig.savefig(out / f"fig_strain_pairwise.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"  Figure saved: {out / 'fig_strain_pairwise'}.[png|pdf]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pairwise BALB/c vs C57BL/6 skull comparison")
    ap.add_argument("--ct-dir", type=Path, required=True)
    ap.add_argument("--balbc", nargs="+", required=True)
    ap.add_argument("--c57", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True,
                    help="Held-out test animals; all others are treated as training animals")
    ap.add_argument("--timepoint", default="T1")
    ap.add_argument("--out", type=Path, default=Path("output/strain_pairwise"))
    ap.add_argument("--max-iterations", type=int, default=300)
    ap.add_argument("--roi-z-min-mm", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ids = list(args.balbc) + list(args.c57)
    strain = np.array(["BALB/c"] * len(args.balbc) + ["C57BL/6"] * len(args.c57))
    paths = [str(args.ct_dir / f"{i}_CT_{args.timepoint}.nii.gz") for i in ids]
    n = len(ids)

    cache = args.out / "pairwise_raw.json"
    raw: Dict[str, Dict] = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [(i, j) for i in range(n) for j in range(i + 1, n) if f"{ids[i]}|{ids[j]}" not in raw]
    print(f"  Animals: {n} ({len(args.balbc)} BALB/c, {len(args.c57)} C57BL/6); "
          f"pairs to register: {len(todo)}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(register_pair, paths[i], paths[j], args.max_iterations,
                             args.roi_z_min_mm): (i, j) for i, j in todo}
        for k, (f, (i, j)) in enumerate(futures.items(), 1):
            raw[f"{ids[i]}|{ids[j]}"] = f.result()
            if k % 10 == 0 or k == len(todo):
                cache.write_text(json.dumps(raw, indent=1))
                print(f"    {k}/{len(todo)} pairs done", flush=True)

    mats = {}
    for level in LEVELS:
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = M[j, i] = raw[f"{ids[i]}|{ids[j]}"][f"{level}_msd_mm"]
        mats[level] = M
        with open(args.out / f"distance_matrix_{level}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + ids)
            for i in range(n):
                w.writerow([ids[i]] + [f"{v:.4f}" for v in M[i]])

    iu = np.triu_indices(n, 1)
    stats = {}
    for level in LEVELS:
        F, p, n_perm = exact_permutation_test(mats[level], strain)
        d, a, b = mats[level][iu], strain[iu[0]], strain[iu[1]]
        within = d[(a == "BALB/c") & (b == "BALB/c")]
        between = d[a != b]
        c57_pair = d[(a == "C57BL/6") & (b == "C57BL/6")]
        nn_same = []
        for i in np.where(strain == "C57BL/6")[0]:
            row = mats[level][i].copy()
            row[i] = np.inf
            nn_same.append(bool(strain[np.argmin(row)] == "C57BL/6"))
        stats[level] = {
            "pseudo_F": F, "p_exact": p, "n_permutations": n_perm,
            "within_balbc_median": float(np.median(within)),
            "within_balbc_range": [float(within.min()), float(within.max())],
            "between_median": float(np.median(between)),
            "between_range": [float(between.min()), float(between.max())],
            "c57_c57": c57_pair.tolist(),
            "frac_between_above_within_max": float(np.mean(between > within.max())),
            "c57_nearest_neighbour_is_c57": nn_same,
        }

    train = [k for k, i in enumerate(ids) if i not in args.test]
    test = []
    for tid in args.test:
        k = ids.index(tid)
        test.append({"id": tid, "strain": str(strain[k]),
                     "dist_to_train_rigid_mm": float(mats["rigid"][k, train].mean()),
                     "dist_to_train_affine_mm": float(mats["affine"][k, train].mean())})
    stats["test_animals"] = test
    stats["paper_bone_dsc"] = PAPER_BONE_DSC

    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    for level in LEVELS:
        s = stats[level]
        print(f"\n  [{level}] pseudo-F {s['pseudo_F']:.2f}, exact p = {s['p_exact']:.4f} "
              f"({s['n_permutations']} relabellings)")
        print(f"    BALB/c–BALB/c   median {s['within_balbc_median']:.3f} mm "
              f"(range {s['within_balbc_range'][0]:.3f}–{s['within_balbc_range'][1]:.3f})")
        print(f"    BALB/c–C57BL/6  median {s['between_median']:.3f} mm "
              f"(range {s['between_range'][0]:.3f}–{s['between_range'][1]:.3f}); "
              f"{s['frac_between_above_within_max'] * 100:.0f} % above max within-BALB/c")
        print(f"    C57BL/6–C57BL/6 {s['c57_c57']}; nearest neighbour is C57BL/6: "
              f"{s['c57_nearest_neighbour_is_c57']}")
    print("\n  Test animals, mean distance to the training animals:")
    for t in test:
        print(f"    {t['id']} {t['strain']:<8s} rigid {t['dist_to_train_rigid_mm']:.3f} mm | "
              f"affine {t['dist_to_train_affine_mm']:.3f} mm")

    make_figure(ids, strain, mats, stats, args.out)


if __name__ == "__main__":
    main()
