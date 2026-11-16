#!/usr/bin/env python
"""
Side-by-side comparison of two sweeps (e.g. the paper baseline and the
intensity-augmented model) on cross-validation and the held-out test set.

Cross-validation compares the same validation volumes in both runs (the
intersection of sample keys, so excluded acquisitions such as MZEI00029_MR_T1
drop out of both). Test-set rows come from ``ensemble_test.py`` per-volume CSVs.

    python scripts/compare_runs.py \
        --a output/loss_sweep_final_22 --a-name baseline \
        --b output/aug_noN4 --b-name aug_noN4 --loss ssim \
        --out output/aug_noN4/comparison.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRICS = [("mae_hu", "MAE [HU]", "{:.0f}"), ("rmse_hu", "RMSE [HU]", "{:.0f}"),
           ("psnr_hu", "PSNR [dB]", "{:.2f}"), ("dice_bone", "Bone DSC", "{:.3f}")]


def _fmt(s: pd.Series, f: str) -> str:
    return f"{f.format(s.mean())} ± {f.format(s.std())}"


def _table(groups: dict[str, pd.DataFrame], key: str, na: str, nb: str) -> list[str]:
    head = "| Group | n | " + " | ".join(f"{lab} {na} | {lab} {nb} | Δ" for _, lab, _ in METRICS) + " |"
    lines = [head, "|---|---:|" + "---:|" * (3 * len(METRICS))]
    for name, d in groups.items():
        cells = []
        for m, _, f in METRICS:
            diff = d[f"{m}_b"] - d[f"{m}_a"]
            cells += [_fmt(d[f"{m}_a"], f), _fmt(d[f"{m}_b"], f), ("{:+" + f[2:]).format(diff.mean())]
        lines.append(f"| {name} | {len(d)} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--loss", default="ssim")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    na, nb = args.a_name, args.b_name
    keep = ["sample_key"] + [m for m, _, _ in METRICS]

    def cv(root: Path) -> pd.DataFrame:
        d = pd.read_csv(root / "eval_per_volume_best.csv")
        return d[(d["loss"] == args.loss) & (d["split"] == "val")]

    ca, cb = cv(args.a), cv(args.b)
    c = ca[keep + ["mouse_id"]].merge(cb[keep], on="sample_key", suffixes=("_a", "_b"))
    c["acq"] = c["sample_key"].str.extract(r"_T(\d+)$")[0].astype(int)
    per_mouse = c.groupby("mouse_id")[[col for col in c.columns if col.endswith(("_a", "_b"))]].mean()

    def test(root: Path) -> pd.DataFrame:
        return pd.read_csv(root / f"ensemble_test_{args.loss}" / "per_volume_metrics.csv")

    t = test(args.a)[keep + ["strain"]].merge(test(args.b)[keep], on="sample_key", suffixes=("_a", "_b"))

    lines = [f"# {nb} vs. {na} ({args.loss} loss)", "",
             "Δ = mean paired difference (" + nb + " − " + na + ").", "",
             "## Cross-validation (identical validation volumes)", ""]
    lines += _table({"all volumes": c, "T1 only": c[c["acq"] == 1], "per mouse": per_mouse},
                    "cv", na, nb)
    lines += ["", "## Held-out test set (K = 3 ensemble)", ""]
    lines += _table({s: g for s, g in t.groupby("strain")}, "test", na, nb)
    lines += ["", "### Per acquisition", "",
              f"| Acquisition | Strain | MAE {na} | MAE {nb} | DSC {na} | DSC {nb} |",
              "|---|---|---:|---:|---:|---:|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['sample_key']} | {r['strain']} | {r['mae_hu_a']:.1f} | {r['mae_hu_b']:.1f} "
                     f"| {r['dice_bone_a']:.3f} | {r['dice_bone_b']:.3f} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport -> {args.out}")


if __name__ == "__main__":
    main()
