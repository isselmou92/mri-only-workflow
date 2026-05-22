#!/usr/bin/env python
"""
Compare loss functions (L1 vs SSIM) on the held-out CV val fold produced by
``scripts/evaluate_sweep.py``. Designed for publication tables.

Statistics
----------
For every pair of losses (loss_a, loss_b) and every metric, this script
computes the paired mean difference at the **model** level. Each (seed, fold)
pair produces one paired observation (one model trained with loss_a vs the
matching model trained with loss_b on the same fold and seed). With 3 folds
and 1 seed (the default sweep), N = 3 paired observations per loss-pair.

The 95% confidence interval is produced by **non-parametric bootstrap**
(default 10000 resamples, percentile method). This avoids any normality
assumption on the paired differences, works for any metric including bounded
ones like Dice, and is robust to outliers. If the CI excludes 0, the two
losses differ at the 5% level on that metric.

Inputs
------
``--per-volume-csv``: the CSV written by ``evaluate_sweep.py``, with one row per
(loss, seed, fold, split, mouse_id) and columns
``mae_hu, rmse_hu, psnr_hu, dice_bone``.

Outputs
-------
For each split in {val} (only val is produced now; test is the external
ensemble run by scripts/ensemble_test.py):
1. ``summary_per_loss_<split>.csv``: one row per loss with mean/std across all
   per-volume measurements in that split.
2. ``summary_per_model_<split>.csv``: one row per (loss, seed, fold) with the
   within-split mean per metric. These are the inputs to the paired bootstrap.
3. ``paired_ci_<split>.csv``: every (loss_a, loss_b, metric) pair with the
   paired mean difference, bootstrap 95% CI, win rate, and a plain-English
   verdict.
4. A paper-ready markdown table printed to stdout with best-per-column bolded
   and asterisks for cells whose paired CI vs the baseline excludes 0.
5. A "simple paired comparison" block printed to stdout with one verdict line
   per (loss_a, loss_b, metric).

Example
-------
    python scripts/compare_losses.py \\
        --per-volume-csv output/loss_sweep_v2/eval_per_volume_best.csv

    python scripts/compare_losses.py \\
        --per-volume-csv output/eval_per_volume_best.csv \\
        --baseline ssim --bootstrap-n 50000
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252 which can't encode the arrow / delta
# glyphs we print. Switch stdout/stderr to UTF-8 so the markdown table renders
# correctly everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding and _stream.encoding.lower() != "utf-8":
            _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# Metric direction: True means "higher is better".
HIGHER_BETTER = {
    "mae_hu": False,
    "rmse_hu": False,
    "psnr_hu": True,
    "dice_bone": True,
}
METRIC_ORDER = ["mae_hu", "rmse_hu", "psnr_hu", "dice_bone"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-volume-csv", required=True, type=Path,
                    help="CSV produced by scripts/evaluate_sweep.py")
    ap.add_argument("--baseline", default="l1",
                    help="Loss treated as the baseline column for the markdown "
                         "table's significance markers (default: l1)")
    ap.add_argument("--bootstrap-n", type=int, default=10000,
                    help="Number of bootstrap resamples for paired CIs "
                         "(default: 10000)")
    ap.add_argument("--bootstrap-seed", type=int, default=1337,
                    help="RNG seed for the bootstrap (default: 1337). Fixed by "
                         "default so the CIs are reproducible across runs.")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="CI level (default: 0.05 -> 95%% CI)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write summary CSVs (default: alongside the input CSV)")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Load + clean
# ---------------------------------------------------------------------------
def _to_float(s: str) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            parsed = {
                "loss": r["loss"],
                "seed": int(r["seed"]),
                "fold": int(r["fold"]),
                "split": r["split"],
                "mouse_id": r["mouse_id"],
            }
            for k in METRIC_ORDER:
                parsed[k] = _to_float(r.get(k, ""))
            rows.append(parsed)
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def per_loss_summary(rows: list[dict], split: str
                      ) -> dict[str, dict[str, tuple[float, float, int]]]:
    """Return {loss: {metric: (mean, std, n_finite)}} across all volumes in split."""
    by_loss: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["split"] == split:
            by_loss[r["loss"]].append(r)
    out: dict[str, dict[str, tuple[float, float, int]]] = {}
    for loss, items in by_loss.items():
        out[loss] = {}
        for m in METRIC_ORDER:
            vals = [r[m] for r in items if r[m] is not None]
            if vals:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                out[loss][m] = (mean, std, len(vals))
            else:
                out[loss][m] = (float("nan"), float("nan"), 0)
    return out


def per_model_summary(rows: list[dict], split: str) -> list[dict]:
    """One row per (loss, seed, fold): mean of each metric across volumes."""
    by_model: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["split"] == split:
            by_model[(r["loss"], r["seed"], r["fold"])].append(r)
    out: list[dict] = []
    for (loss, seed, fold), items in sorted(by_model.items()):
        row = {"loss": loss, "seed": seed, "fold": fold, "n_volumes": len(items)}
        for m in METRIC_ORDER:
            vals = [r[m] for r in items if r[m] is not None]
            row[f"{m}_mean"] = statistics.mean(vals) if vals else float("nan")
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Paired bootstrap at the model level
# ---------------------------------------------------------------------------
def bootstrap_paired_ci(
    diffs: list[float],
    n_resamples: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return (ci_lo, ci_hi) for the mean of paired diffs by percentile bootstrap.

    Each resample draws ``len(diffs)`` indices with replacement and recomputes
    the mean. The 100*alpha/2 and 100*(1-alpha/2) percentiles of those means
    are the CI endpoints.
    """
    if len(diffs) == 0:
        return float("nan"), float("nan")
    arr = np.asarray(diffs, dtype=np.float64)
    n = arr.size
    # idx shape: (n_resamples, n). Vectorised mean over axis=1.
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return lo, hi


def paired_diffs_at_model_level(
    rows: list[dict], split: str, loss_a: str, loss_b: str, metric: str,
) -> tuple[list[float], int]:
    """Compute paired model-level diffs (loss_a - loss_b) for a metric in a split.

    Pairing key is (seed, fold) -- one model per loss per (seed, fold). The
    per-volume averaging happens *inside* each pair so within-mouse correlation
    is absorbed, not double-counted.
    Returns (diffs, wins_a) where wins_a is how many pairs loss_a beat loss_b.
    """
    bucket: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["split"] != split or r["loss"] not in (loss_a, loss_b):
            continue
        v = r[metric]
        if v is None:
            continue
        bucket[(r["loss"], r["seed"], r["fold"])].append(v)
    model_mean: dict[tuple, float] = {
        k: statistics.mean(vs) for k, vs in bucket.items() if vs
    }

    diffs: list[float] = []
    wins_a = 0
    for (loss, seed, fold), val_a in model_mean.items():
        if loss != loss_a:
            continue
        val_b = model_mean.get((loss_b, seed, fold))
        if val_b is None:
            continue
        d = val_a - val_b
        diffs.append(d)
        if HIGHER_BETTER[metric]:
            if d > 0:
                wins_a += 1
        else:
            if d < 0:
                wins_a += 1
    return diffs, wins_a


def paired_ci(
    rows: list[dict], split: str, loss_a: str, loss_b: str, metric: str,
    n_resamples: int, alpha: float, rng: np.random.Generator,
) -> dict:
    diffs, wins_a = paired_diffs_at_model_level(rows, split, loss_a, loss_b, metric)
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs) if n > 1 else 0.0
    ci_lo, ci_hi = bootstrap_paired_ci(diffs, n_resamples, alpha, rng)
    return {
        "n": n,
        "mean_diff_a_minus_b": mean_diff,
        "std_diff": std_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "win_rate_a": wins_a / n,
        "diffs": diffs,
    }


def _verdict(metric: str, loss_a: str, loss_b: str, res: dict) -> str:
    """One-line plain-English verdict from the CI / sign."""
    if res["n"] == 0:
        return "no data"
    lo, hi, md = res["ci_lo"], res["ci_hi"], res["mean_diff_a_minus_b"]
    higher = HIGHER_BETTER[metric]
    if lo > 0:
        winner = loss_a if higher else loss_b
    elif hi < 0:
        winner = loss_b if higher else loss_a
    else:
        return f"equivalent (95% CI of {loss_a}-{loss_b} includes 0)"
    mag = abs(md)
    unit = " HU" if metric in ("mae_hu", "rmse_hu") else ""
    if metric == "psnr_hu":
        unit = " dB"
    return f"{winner} better by {mag:.2f}{unit}  [95% CI {lo:+.2f}, {hi:+.2f}]"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _format_metric(mean: float, std: float, m: str) -> str:
    if not math.isfinite(mean):
        return "--"
    if m in ("mae_hu", "rmse_hu"):
        return f"{mean:.1f} +/- {std:.1f}"
    if m == "psnr_hu":
        return f"{mean:.2f} +/- {std:.2f}"
    if m == "dice_bone":
        return f"{mean:.3f} +/- {std:.3f}"
    return f"{mean:.4f} +/- {std:.4f}"


def _best_index(values: list[float], higher_better: bool) -> int | None:
    finite = [(i, v) for i, v in enumerate(values) if math.isfinite(v)]
    if not finite:
        return None
    return (max if higher_better else min)(finite, key=lambda x: x[1])[0]


def print_markdown_table(
    split: str,
    losses: list[str],
    summary: dict[str, dict[str, tuple[float, float, int]]],
    ci_lookup: dict[tuple[str, str], dict],
    baseline: str,
) -> None:
    """Markdown table with mean +/- std per loss. Asterisks mark cells whose
    paired CI vs the baseline excludes 0 (i.e. statistically distinguishable)."""
    print(f"\n## {split.upper()} split  (per-volume HU metrics + bone Dice)\n")

    header = "| Loss | " + " | ".join(
        f"{m} {'↑' if HIGHER_BETTER[m] else '↓'}" for m in METRIC_ORDER
    ) + " |"
    sep = "|------|" + "|".join(["----------"] * len(METRIC_ORDER)) + "|"
    print(header)
    print(sep)

    losses_sorted = sorted(losses)
    best_idx_per_metric: dict[str, int | None] = {}
    for m in METRIC_ORDER:
        vals = [summary[l][m][0] for l in losses_sorted]
        best_idx_per_metric[m] = _best_index(vals, HIGHER_BETTER[m])

    for i, loss in enumerate(losses_sorted):
        row_cells = [loss]
        for m in METRIC_ORDER:
            mean, std, _ = summary[loss][m]
            cell = _format_metric(mean, std, m)
            if best_idx_per_metric[m] == i:
                cell = f"**{cell}**"
            if loss != baseline:
                res = ci_lookup.get((loss, baseline)) or ci_lookup.get((baseline, loss))
                if res is not None:
                    ci_for_metric = res.get(m, {})
                    lo, hi = ci_for_metric.get("ci_lo"), ci_for_metric.get("ci_hi")
                    if lo is not None and hi is not None:
                        excludes_zero = (lo > 0) or (hi < 0)
                        cell = f"{cell}{'*' if excludes_zero else ''}"
            row_cells.append(cell)
        print("| " + " | ".join(row_cells) + " |")

    n_volumes = max(summary[l][METRIC_ORDER[0]][2] for l in losses_sorted)
    print(
        f"\nBold = best per column. "
        f"`*` marks cells whose paired bootstrap 95% CI vs baseline `{baseline}` "
        f"excludes 0 (i.e. statistically distinguishable from the baseline). "
        f"N volumes per loss = {n_volumes}."
    )


def print_paired_ci_block(
    split: str,
    losses: list[str],
    ci_lookup: dict[tuple[str, str], dict],
) -> None:
    print(f"\n--- {split.upper()} split: paired bootstrap 95% CIs ---")
    headers = ["Pair", "Metric", "Δ mean (a-b)", "95% CI",
               "Win rate (a)", "Verdict"]
    widths = [12, 10, 14, 22, 13, 60]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    pairs = [(losses[i], losses[j]) for i in range(len(losses)) for j in range(i + 1, len(losses))]
    for la, lb in pairs:
        per_metric = ci_lookup.get((la, lb), {})
        for m in METRIC_ORDER:
            res = per_metric.get(m)
            if not res or res["n"] == 0:
                continue
            md = res["mean_diff_a_minus_b"]
            ci = f"[{res['ci_lo']:+.2f}, {res['ci_hi']:+.2f}]"
            wr_count = int(round(res["win_rate_a"] * res["n"]))
            wr = f"{res['win_rate_a']*100:.0f}%  ({wr_count}/{res['n']})"
            verdict = _verdict(m, la, lb, res)
            cells = [
                f"{la} vs {lb}",
                m.replace("_hu", ""),
                f"{md:+.2f}",
                ci,
                wr,
                verdict,
            ]
            print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
    print()


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def write_summary_per_loss(
    summary: dict[str, dict[str, tuple[float, float, int]]],
    path: Path,
) -> None:
    fields = ["loss"]
    for m in METRIC_ORDER:
        fields += [f"{m}_mean", f"{m}_std", f"{m}_n"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for loss in sorted(summary):
            row: list = [loss]
            for m in METRIC_ORDER:
                mean, std, n = summary[loss][m]
                row += [f"{mean:.6f}", f"{std:.6f}", n]
            w.writerow(row)


def write_summary_per_model(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_paired_ci_csv(
    split: str,
    losses: list[str],
    ci_lookup: dict[tuple[str, str], dict],
    path: Path,
) -> None:
    fields = ["split", "loss_a", "loss_b", "metric", "n_pairs",
              "mean_diff_a_minus_b", "std_diff", "ci_lo_95", "ci_hi_95",
              "win_rate_a", "verdict"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        pairs = [(losses[i], losses[j]) for i in range(len(losses)) for j in range(i + 1, len(losses))]
        for la, lb in pairs:
            per_metric = ci_lookup.get((la, lb), {})
            for m in METRIC_ORDER:
                res = per_metric.get(m)
                if not res or res["n"] == 0:
                    continue
                w.writerow({
                    "split": split,
                    "loss_a": la,
                    "loss_b": lb,
                    "metric": m,
                    "n_pairs": res["n"],
                    "mean_diff_a_minus_b": f"{res['mean_diff_a_minus_b']:.6f}",
                    "std_diff": f"{res['std_diff']:.6f}",
                    "ci_lo_95": f"{res['ci_lo']:.6f}",
                    "ci_hi_95": f"{res['ci_hi']:.6f}",
                    "win_rate_a": f"{res['win_rate_a']:.4f}",
                    "verdict": _verdict(m, la, lb, res),
                })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    csv_path = args.per_volume_csv.resolve()
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")
    out_dir = (args.out_dir or csv_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No data rows in {csv_path}")

    losses = sorted({r["loss"] for r in rows})
    splits = sorted({r["split"] for r in rows})
    print(f"Loaded {len(rows)} per-volume rows from {csv_path}")
    print(f"Losses : {losses}")
    print(f"Splits : {splits}")
    print(f"Bootstrap: {args.bootstrap_n} resamples, seed={args.bootstrap_seed}, "
          f"alpha={args.alpha} (-> {int(100*(1-args.alpha))}% CI)")
    if args.baseline not in losses:
        print(f"[warn] baseline '{args.baseline}' not present; using '{losses[0]}'")
        args.baseline = losses[0]

    rng = np.random.default_rng(args.bootstrap_seed)
    pairs = [(losses[i], losses[j]) for i in range(len(losses)) for j in range(i + 1, len(losses))]

    # Run the bootstrap once per split, cache the CI lookups, then both the
    # CSV writers and the stdout block consume the same numbers.
    cached_ci: dict[str, dict[tuple[str, str], dict]] = {}
    for split in splits:
        summary = per_loss_summary(rows, split)
        per_model = per_model_summary(rows, split)

        ci_lookup: dict[tuple[str, str], dict] = {}
        for la, lb in pairs:
            ci_lookup[(la, lb)] = {}
            for m in METRIC_ORDER:
                ci_lookup[(la, lb)][m] = paired_ci(
                    rows, split, la, lb, m,
                    n_resamples=args.bootstrap_n,
                    alpha=args.alpha,
                    rng=rng,
                )
        cached_ci[split] = ci_lookup

        per_loss_csv = out_dir / f"summary_per_loss_{split}.csv"
        per_model_csv = out_dir / f"summary_per_model_{split}.csv"
        paired_ci_csv = out_dir / f"paired_ci_{split}.csv"

        write_summary_per_loss(summary, per_loss_csv)
        write_summary_per_model(per_model, per_model_csv)
        write_paired_ci_csv(split, losses, ci_lookup, paired_ci_csv)

        print_markdown_table(split, losses, summary, ci_lookup, baseline=args.baseline)
        print(f"\n  -> {per_loss_csv}")
        print(f"  -> {per_model_csv}")
        print(f"  -> {paired_ci_csv}")

    print("\n" + "=" * 72)
    print("PAIRED COMPARISON (paper-ready, bootstrap 95% CIs)")
    print("Pairing: one observation per (seed, fold) model "
          "[N=9 for a 3-fold x 3-seed sweep].")
    print("If a 95% CI excludes 0, the two losses differ on that metric.")
    print("=" * 72)
    for split in splits:
        print_paired_ci_block(split, losses, cached_ci[split])


if __name__ == "__main__":
    main()
