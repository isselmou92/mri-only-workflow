"""Summarise cross-validation metrics (whole volume vs. CT field of view) in output/fov_eval."""
from pathlib import Path

import pandas as pd

OUT = Path("output/fov_eval")
METRICS = ["mae_hu", "rmse_hu", "psnr_hu", "dice_bone"]

frames = []
for model in ("final_22", "aug_noN4"):
    for region, suffix in (("whole_volume", "whole"), ("ct_fov", "fov")):
        df = pd.read_csv(OUT / f"cv_{model}_{suffix}.csv")
        df = df.drop_duplicates(["loss", "sample_key"], keep="last")
        df["model"], df["region"] = model, region
        frames.append(df)
df = pd.concat(frames)

# Table 1: all volumes; augmentation comparison: same 68 volumes (without MZEI00029_MR_T1).
rows = []
for subset, d in (("all", df), ("68_common", df[df.sample_key != "MZEI00029_MR_T1"])):
    g = d.groupby(["model", "loss", "region"])[METRICS]
    s = g.mean().round(3).add_suffix("_mean").join(g.std().round(3).add_suffix("_sd"))
    s["n"] = g.size()
    s.insert(0, "subset", subset)
    rows.append(s.reset_index())
summary = pd.concat(rows)
summary.to_csv(OUT / "cv_summary.csv", index=False)

pd.set_option("display.width", 200)
for _, r in summary.iterrows():
    print(f"{r.subset:9s} {r.model:8s} {r.loss:4s} {r.region:12s} n={r.n:2d}  "
          + "  ".join(f"{m}={r[m + '_mean']:.2f}±{r[m + '_sd']:.2f}" for m in METRICS))
