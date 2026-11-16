"""Summary table and figure for the CT field-of-view (FOV) evaluation.

Reads the CSVs in output/fov_eval (written by evaluate_test_fov.py and evaluate_sweep.py --fov-mask)
and writes fov_summary_table.{csv,tex} and fov_summary.{png,pdf}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
METHODS = {"published": "Published sCT", "augmented": "Augmented sCT", "avg_ct": "Average CT"}
COLORS = {"published": "#7f7f7f", "augmented": "#0072B2", "avg_ct": "#E69F00"}
STRAINS = ["BALB/c", "C57BL/6"]
REGIONS = {"whole_volume": "Whole volume", "ct_fov": "CT FOV"}


def load_cv(root: Path) -> pd.DataFrame:
    frames = []
    for model, method in (("final_22", "published"), ("aug_noN4", "augmented")):
        for region, suffix in (("whole_volume", "whole"), ("ct_fov", "fov")):
            df = pd.read_csv(root / f"cv_{model}_{suffix}.csv")
            df = df[df["loss"] == "ssim"].drop_duplicates("sample_key", keep="last")
            df["method"], df["region"] = method, region
            frames.append(df)
    df = pd.concat(frames)
    return df[df["sample_key"] != "MZEI00029_MR_T1"]


def fmt(s: pd.Series, nd: int) -> str:
    return f"{s.mean():.{nd}f} ± {s.std():.{nd}f}"


TABLE_METRICS = [("MAE (HU)", "mae_hu", 0), ("Bone DSC", "dice_bone", 2), ("HD95 (mm)", "hd95_mm", 2)]


def table_row(data: str, method: str, w: pd.DataFrame, f: pd.DataFrame) -> dict:
    row = {"Data": data, "Method": METHODS[method], "n": len(f)}
    for label, col, nd in TABLE_METRICS:
        for region, d in (("whole", w), ("FOV", f)):
            row[f"{label} {region}"] = fmt(d[col], nd) if col in d else "–"
        if col in f:
            delta = round(float((f[col].values - w[col].values).mean()), nd)
            row[f"{label} Δ"] = f"{delta:+.{nd}f}" if delta != 0 else f"{0:.{nd}f}"
        else:
            row[f"{label} Δ"] = "–"
    return row


def build_table(test: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strain in STRAINS:
        for method in METHODS:
            d = test[(test.strain == strain) & (test.method == method)]
            rows.append(table_row(f"Test {strain}", method, d[d.region == "whole_volume"], d[d.region == "ct_fov"]))
    for method in ("published", "augmented"):
        d = cv[cv.method == method]
        w = d[d.region == "whole_volume"].set_index("sample_key")
        f = d[d.region == "ct_fov"].set_index("sample_key").loc[w.index]
        rows.append(table_row("Cross-validation", method, w, f))
    return pd.DataFrame(rows)


def write_latex(table: pd.DataFrame, path: Path) -> None:
    cols = ["Method", "n"] + [f"{label} {r}" for label, _, _ in TABLE_METRICS for r in ("whole", "FOV", "Δ")]
    group = " & ".join(f"\\multicolumn{{3}}{{c}}{{{label}}}" for label, _, _ in TABLE_METRICS)
    rules = " ".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(TABLE_METRICS)))
    sub = " & ".join(["Whole", "FOV", "$\\Delta$"] * len(TABLE_METRICS))
    lines = ["\\begin{table}[ht]", "\\centering", "\\small",
             "\\caption{MAE, bone Dice similarity coefficient (DSC, $\\geq$300\\,HU) and 95th-percentile Hausdorff "
             "distance (HD95) of the synthetic CTs and the average CT, evaluated over the whole volume and restricted "
             "to the micro-CT field of view (FOV). The FOV excludes the constant fill outside the rotated CT crop box. "
             "Mean $\\pm$ SD; $\\Delta$ = FOV $-$ whole volume, averaged over paired volumes.}",
             "\\label{tab:fov}", "\\begin{tabular}{lc" + "ccc" * len(TABLE_METRICS) + "}", "\\toprule",
             f"Method & $n$ & {group} \\\\", rules, f" & & {sub} \\\\", "\\midrule"]
    for data, block in table.groupby("Data", sort=False):
        lines.append(f"\\multicolumn{{{len(cols)}}}{{l}}{{\\textit{{{data}}}}} \\\\")
        for _, r in block.iterrows():
            lines.append(" & ".join(str(r[c]).replace("±", "$\\pm$").replace("–", "--").replace("-0", "$-$0").replace("-1", "$-$1") for c in cols) + " \\\\")
        lines.append("\\addlinespace")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dumbbell(ax, test: pd.DataFrame, metric: str, ylabel: str, title: str) -> None:
    rng = np.random.default_rng(0)
    offsets = dict(zip(METHODS, (-0.26, 0.0, 0.26)))
    for i, strain in enumerate(STRAINS):
        for method in METHODS:
            d = test[(test.strain == strain) & (test.method == method)]
            w = d[d.region == "whole_volume"][metric]
            f = d[d.region == "ct_fov"][metric]
            x = i + offsets[method]
            xw, xf = x - 0.06, x + 0.03
            ax.scatter(xf + 0.075 + rng.uniform(-0.02, 0.02, len(f)), f, s=9, color=COLORS[method],
                       alpha=0.4, linewidths=0, zorder=2)
            ax.annotate("", xy=(xf, f.mean()), xytext=(xw, w.mean()), zorder=3,
                        arrowprops={"arrowstyle": "-", "color": COLORS[method], "lw": 1.1, "alpha": 0.7})
            ax.scatter(xw, w.mean(), s=40, facecolor="white", edgecolor=COLORS[method], lw=1.6, zorder=4)
            ax.errorbar(xf, f.mean(), yerr=f.std(), fmt="o", ms=6.5, color=COLORS[method],
                        elinewidth=1.1, capsize=2.5, zorder=5)
    ax.set_xticks(range(len(STRAINS)), STRAINS)
    ax.set_xlim(-0.55, len(STRAINS) - 0.45)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "output" / "fov_eval")
    args = ap.parse_args()

    test = pd.read_csv(args.root / "test_per_acquisition.csv")
    cv = load_cv(args.root)

    table = build_table(test, cv)
    table.to_csv(args.root / "fov_summary_table.csv", index=False)
    write_latex(table, args.root / "fov_summary_table.tex")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9.5,
                         "axes.titlepad": 8, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))

    dumbbell(axes[0, 0], test, "mae_hu", "MAE (HU)", "a   Test set: MAE")
    dumbbell(axes[0, 1], test, "dice_bone", "Bone Dice (≥ 300 HU)", "b   Test set: bone Dice")
    dumbbell(axes[0, 2], test, "msd_mm", "Mean surface distance (mm)", "c   Test set: bone surface distance")

    # d: cross-validation, per-volume whole vs FOV MAE
    ax = axes[1, 0]
    for method in ("published", "augmented"):
        d = cv[cv.method == method]
        w = d[d.region == "whole_volume"].set_index("sample_key").mae_hu
        f = d[d.region == "ct_fov"].set_index("sample_key").mae_hu.loc[w.index]
        ax.scatter(w, f, s=16, color=COLORS[method], alpha=0.75, linewidths=0, label=METHODS[method])
    lim = [140, 300]
    ax.plot(lim, lim, color="0.55", lw=1, ls="--")
    ax.text(296, 282, "y = x", color="0.45", fontsize=8, ha="right", va="top")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("MAE whole volume (HU)")
    ax.set_ylabel("MAE inside CT FOV (HU)")
    ax.set_title("d   Cross-validation: per volume (n = 68)", loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)

    # e: cross-validation, paired gain of the augmented model in each region
    ax = axes[1, 1]
    gains = []
    for region in REGIONS:
        p = cv[(cv.method == "published") & (cv.region == region)].set_index("sample_key").mae_hu
        a = cv[(cv.method == "augmented") & (cv.region == region)].set_index("sample_key").mae_hu.loc[p.index]
        gains.append((p - a).values)
    bp = ax.boxplot(gains, positions=[0, 1], widths=0.45, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "lw": 1.3})
    for patch in bp["boxes"]:
        patch.set(facecolor="#d6e6f2", edgecolor=COLORS["augmented"])
    rng = np.random.default_rng(1)
    for i, g in enumerate(gains):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(g)), g, s=9, color=COLORS["augmented"], alpha=0.5, linewidths=0)
        ax.text(i, g.max() + 4, f"mean {g.mean():+.1f} HU\n"
                f"{(g > 0).mean():.0%} of volumes improved", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xticks([0, 1], list(REGIONS.values()))
    ax.set_ylabel("MAE published − augmented (HU)\n(> 0: augmented better)")
    ax.set_title("e   Cross-validation: augmentation gain", loc="left", fontsize=10.5, fontweight="bold")
    ax.set_ylim(top=max(g.max() for g in gains) + 30)

    # f: test set, MAE increase vs fraction of volume outside the CT FOV
    ax = axes[1, 2]
    markers = {"BALB/c": "o", "C57BL/6": "s"}
    for method in METHODS:
        d = test[test.method == method]
        w = d[d.region == "whole_volume"].set_index("acquisition")
        f = d[d.region == "ct_fov"].set_index("acquisition").loc[w.index]
        for strain, m in markers.items():
            sel = w.strain == strain
            ax.scatter(100 * (1 - w.fov_fraction[sel]), (f.mae_hu - w.mae_hu)[sel], s=30, marker=m,
                       color=COLORS[method], alpha=0.85, linewidths=0)
    ax.set_xlabel("Volume outside the CT FOV (%)")
    ax.set_ylabel("MAE increase inside FOV (HU)")
    ax.set_title("f   Test set: effect of the excluded fill", loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(handles=[Line2D([], [], marker=m, ls="", color="0.3", label=s) for s, m in markers.items()],
              frameon=False, loc="upper left", fontsize=8.5)

    for ax in axes.ravel():
        style(ax)
    handles = [Line2D([], [], marker="o", ls="", ms=7, color=COLORS[m], label=l) for m, l in METHODS.items()]
    handles += [Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec="0.3", label="Whole volume (mean)"),
                Line2D([], [], marker="o", ls="", ms=7, color="0.3", label="Inside CT FOV (mean ± SD, dots = acquisitions)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=2.5, w_pad=2.0)
    for ext in ("png", "pdf"):
        fig.savefig(args.root / f"fov_summary.{ext}", dpi=200)

    # compact numbers for reporting
    summary = {
        "test": {s: {m: {r: {k: round(float(test[(test.strain == s) & (test.method == m) & (test.region == r)][k].mean()), 3)
                             for k in ("mae_hu", "dice_bone", "msd_mm", "hd95_mm", "fov_fraction")}
                         for r in REGIONS} for m in METHODS} for s in STRAINS},
        "cv_gain_mean": {r: round(float(g.mean()), 2) for r, g in zip(REGIONS, gains)},
    }
    (args.root / "fov_summary.json").write_text(json.dumps(summary, indent=1))
    print(table.to_string(index=False))
    print(json.dumps(summary["cv_gain_mean"]))


if __name__ == "__main__":
    main()
