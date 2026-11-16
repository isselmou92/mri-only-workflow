#!/usr/bin/env python
"""
Side-by-side report: manuscript values vs. values reproduced from a sweep.

Reads the outputs of scripts/reproduce_paper.sh and writes one markdown file
covering every non-dosimetric number of the paper:

  * Table 1      CV loss comparison (T1-only subset as in the paper, and all
                 validation volumes)
  * Section 3.2  Test-set MAE / bone DSC of the SSIM ensemble and of the avCT
  * Table S1     Per-acquisition test metrics

Usage:
    python scripts/reproduce_report.py --sweep-root output/loss_sweep_final_22 \
        --dice-csv output/loss_sweep_final_22/paper_repro/dice_test_cohort.csv \
        --out output/loss_sweep_final_22/paper_repro/report.md
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

PAPER_TABLE1 = {
    "l1": {"mae_hu": (202, 47), "rmse_hu": (470, 75), "psnr_hu": (18.7, 1.2), "dice_bone": (0.61, 0.12)},
    "ssim": {"mae_hu": (203, 53), "rmse_hu": (471, 82), "psnr_hu": (18.7, 1.3), "dice_bone": (0.63, 0.12)},
}
PAPER_TEST = {
    "BALB/c": {"sct_mae": (150, 12), "sct_dsc": (0.73, 0.78), "avct_mae": (154, 5), "avct_dsc": (0.79, 0.80)},
    "C57BL/6": {"sct_mae": (285, 32), "sct_dsc": (0.53, 0.59), "avct_mae": (188, 17), "avct_dsc": (0.70, 0.77)},
}
# Suppl. Table S1: MAE, PSNR, sCT DSC, avCT DSC
PAPER_S1 = {
    "MZEI00027_MR_T1": (138.9, 22.07, 0.759, 0.786),
    "MZEI00027_MR_T2": (138.0, 22.25, 0.766, 0.786),
    "MZEI00027_MR_T3": (141.5, 22.08, 0.757, 0.786),
    "MZEI00031_MR_T1": (152.2, 21.07, 0.776, 0.796),
    "MZEI00031_MR_T2": (165.8, 20.65, 0.729, 0.796),
    "MZEI00031_MR_T3": (160.5, 20.83, 0.750, 0.796),
    "MZEI00066_MR_T1": (251.1, 18.64, 0.588, 0.766),
    "MZEI00066_MR_T2": (259.3, 18.43, 0.526, 0.757),
    "MZEI00066_MR_T3": (257.9, 18.43, 0.529, 0.757),
    "MZEI00069_MR_T1": (322.3, 16.80, 0.526, 0.715),
    "MZEI00069_MR_T2": (312.7, 17.08, 0.531, 0.742),
    "MZEI00069_MR_T3": (306.8, 16.93, 0.531, 0.704),
}
METRICS = [("mae_hu", "MAE [HU]", 0), ("rmse_hu", "RMSE [HU]", 0),
           ("psnr_hu", "PSNR [dB]", 1), ("dice_bone", "Bone DSC", 2)]


def _read(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ms(vals: List[float], nd: int) -> str:
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):.{nd}f} ± {sd:.{nd}f}"


def _rng(vals: List[float]) -> str:
    return f"{min(vals):.2f}–{max(vals):.2f}"


def table1(rows: List[Dict[str, str]]) -> List[str]:
    out = ["## Table 1: loss comparison (subject-grouped 3-fold CV)", ""]
    subsets = [("T1 only (as in the paper)", lambda r: r["sample_key"].endswith("_T1")),
               ("All validation volumes", lambda r: True)]
    for title, keep in subsets:
        out += [f"**{title}**", "",
                "| Loss | n | " + " | ".join(f"{lbl} (paper)" + f" | {lbl} (repro)"
                                             for _, lbl, _ in METRICS) + " |",
                "|---|---:|" + "---:|---:|" * len(METRICS)]
        for loss in ("l1", "ssim"):
            sel = [r for r in rows if r["loss"] == loss and r["split"] == "val" and keep(r)]
            if not sel:
                continue
            cells = []
            for key, _, nd in METRICS:
                pm, ps = PAPER_TABLE1[loss][key]
                cells += [f"{pm:.{nd}f} ± {ps:.{nd}f}", _ms([float(r[key]) for r in sel], nd)]
            out.append(f"| {loss} | {len(sel)} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def test_summary(sct_rows: List[Dict[str, str]], dice_rows: List[Dict[str, str]]) -> List[str]:
    sct_by = defaultdict(list)
    for r in sct_rows:
        sct_by[r["strain"]].append(r)
    avct_by = defaultdict(list)
    for r in dice_rows:
        avct_by[r["strain"]].append(r)
    out = ["## Section 3.2: held-out test set (SSIM ensemble, K = 3)", "",
           "| Strain | sCT MAE paper | sCT MAE repro | sCT DSC paper | sCT DSC repro "
           "| avCT MAE paper | avCT MAE repro | avCT DSC paper | avCT DSC repro |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for strain, p in PAPER_TEST.items():
        s, a = sct_by.get(strain, []), avct_by.get(strain, [])
        sct_mae = _ms([float(r["mae_hu"]) for r in s], 0) if s else "–"
        sct_dsc = _rng([float(r["dice_bone"]) for r in s]) if s else "–"
        av_mae = _ms([float(r["aligned_avct_mae_hu"]) for r in a], 0) if a else "–"
        av_dsc = _rng([float(r["aligned_avct_dsc"]) for r in a]) if a else "–"
        out.append(f"| {strain} | {p['sct_mae'][0]} ± {p['sct_mae'][1]} | {sct_mae} "
                   f"| {p['sct_dsc'][0]:.2f}–{p['sct_dsc'][1]:.2f} | {sct_dsc} "
                   f"| {p['avct_mae'][0]} ± {p['avct_mae'][1]} | {av_mae} "
                   f"| {p['avct_dsc'][0]:.2f}–{p['avct_dsc'][1]:.2f} | {av_dsc} |")
    return out + [""]


def table_s1(sct_rows: List[Dict[str, str]], dice_rows: List[Dict[str, str]]) -> List[str]:
    sct = {r["sample_key"]: r for r in sct_rows}
    avct = {r["subject"]: r for r in dice_rows}
    out = ["## Supplementary Table S1: per acquisition", "",
           "| Acquisition | MAE paper | MAE repro | PSNR paper | PSNR repro "
           "| sCT DSC paper | sCT DSC repro | avCT DSC paper | avCT DSC repro |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, (mae, psnr, dsc, av) in PAPER_S1.items():
        s, a = sct.get(key), avct.get(key)
        rep = (f"{float(s['mae_hu']):.1f} | {psnr:.2f} | {float(s['psnr_hu']):.2f} | {dsc:.3f} "
               f"| {float(s['dice_bone']):.3f}") if s else f"– | {psnr:.2f} | – | {dsc:.3f} | –"
        rep_av = f"{float(a['aligned_avct_dsc']):.3f}" if a else "–"
        out.append(f"| {key} | {mae:.1f} | {rep} | {av:.3f} | {rep_av} |")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-root", required=True, type=Path)
    ap.add_argument("--dice-csv", required=True, type=Path,
                    help="CSV written by scripts/test_dice_table.py")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    eval_rows = _read(args.sweep_root / "eval_per_volume_best.csv")
    sct_rows = _read(args.sweep_root / "ensemble_test_ssim" / "per_volume_metrics.csv")
    dice_rows = _read(args.dice_csv)

    lines = [f"# Reproduction report: `{args.sweep_root.name}` vs. manuscript", "",
             "Paper values are copied from `manuscript/main.tex` and `manuscript/suplementary.tex`. "
             "The average CT does not depend on the U-Net, so its values should match exactly. "
             "The U-Net values come from a retrained sweep and are expected to be close but not identical.",
             ""]
    lines += table1(eval_rows) + test_summary(sct_rows, dice_rows) + table_s1(sct_rows, dice_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport -> {args.out}")


if __name__ == "__main__":
    main()
