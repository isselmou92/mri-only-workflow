#!/usr/bin/env python
"""
Aggregate metrics across a loss-benchmark sweep produced by run_sweep.py.

Walks <sweep-root>/{loss}_seed{seed}/fold{N}/checkpoints/{latest,best}.pt and
collects the CV val metrics stored in each checkpoint payload into:
  - summary_per_run.csv     (one row per loss x fold x seed)
  - summary_per_loss.csv    (mean + std per loss across folds and seeds)

The external test cohort is *not* aggregated here - that is the job of
scripts/ensemble_test.py, which ensembles all CV-trained models for the
chosen loss and writes its own per-volume CSV + summary.

Also prints a paper-ready markdown table to stdout.

Examples
--------
    python scripts/aggregate_sweep.py --sweep-root output/loss_sweep_v1
    python scripts/aggregate_sweep.py --sweep-root output/loss_sweep_v1 --use best
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

import torch


def _finite_values(rows: list[dict], key: str) -> tuple[list[float], int]:
    """Return (finite values for ``key`` across ``rows``, count of non-finite skipped).

    Python 3.11's statistics.stdev raises AttributeError when given nan/inf
    (internal _sum returns float instead of Fraction). We strip them here and
    surface the dropped count so the paper-ready summary can flag it.
    """
    finite: list[float] = []
    dropped = 0
    for r in rows:
        if key not in r:
            continue
        v = r[key]
        if isinstance(v, float) and math.isfinite(v):
            finite.append(v)
        elif isinstance(v, (int,)):
            finite.append(float(v))
        else:
            dropped += 1
    return finite, dropped

RUN_DIR_RE = re.compile(r"^(?P<loss>[a-z0-9]+)_seed(?P<seed>\d+)$")
FOLD_DIR_RE = re.compile(r"^fold(?P<fold>\d+)$")

# Metrics highlighted in the paper-ready markdown table.
PAPER_METRICS_VAL = ["val_mae", "val_rmse", "val_psnr", "val_ssim"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-root", required=True, type=Path,
                    help="Sweep root produced by run_sweep.py")
    ap.add_argument("--use", default="latest", choices=["latest", "best"],
                    help="Which checkpoint to read in each fold dir")
    ap.add_argument("--out-csv", default=None, type=Path,
                    help="Override path for the per-run CSV (default: <sweep>/summary_per_run.csv)")
    return ap.parse_args()


def collect_rows(sweep_root: Path, which: str) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(sweep_root.iterdir()):
        if not run_dir.is_dir():
            continue
        m = RUN_DIR_RE.match(run_dir.name)
        if not m:
            continue
        loss = m.group("loss")
        seed = int(m.group("seed"))

        for fold_dir in sorted(run_dir.iterdir()):
            mf = FOLD_DIR_RE.match(fold_dir.name)
            if not mf:
                continue
            fold = int(mf.group("fold"))
            ckpt = fold_dir / "checkpoints" / f"{which}.pt"
            if not ckpt.exists():
                print(f"[miss] {ckpt}")
                continue
            try:
                # weights_only=False is required because our checkpoints store
                # optimizer state and metric dicts, not just tensors. The pickle
                # files we read here come from our own training runs, so the
                # security caveat in the FutureWarning does not apply.
                state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"[error] {ckpt}: {e}")
                continue

            row: dict = {
                "loss": loss,
                "fold": fold,
                "seed": seed,
                "epoch": int(state.get("epoch", -1)),
                "ckpt": str(ckpt.relative_to(sweep_root)),
            }
            for split_key in ("val", "test"):
                d = state.get(split_key, {}) or {}
                for k, v in d.items():
                    try:
                        row[f"{split_key}_{k}"] = float(v)
                    except (TypeError, ValueError):
                        continue
            rows.append(row)
    return rows


def write_per_run_csv(rows: list[dict], path: Path) -> list[str]:
    # Stable column ordering: identifiers first, then metric columns.
    head = ["loss", "fold", "seed", "epoch"]
    metric_keys = sorted({k for r in rows for k in r.keys()
                          if k.startswith("val_") or k.startswith("test_")})
    keys = head + metric_keys + ["ckpt"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return metric_keys


def write_per_loss_csv(rows: list[dict], metric_keys: list[str], path: Path) -> dict:
    by_loss: dict[str, list[dict]] = {}
    for r in rows:
        by_loss.setdefault(r["loss"], []).append(r)

    head = ["loss", "n_runs"]
    for k in metric_keys:
        # n_finite reports how many of n_runs contributed (i.e. excluded nan/inf).
        head += [f"{k}_mean", f"{k}_std", f"{k}_n_finite"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for loss in sorted(by_loss):
            runs = by_loss[loss]
            cells: list = [loss, len(runs)]
            for k in metric_keys:
                vals, _dropped = _finite_values(runs, k)
                if vals:
                    mean = statistics.mean(vals)
                    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                    cells += [f"{mean:.6f}", f"{std:.6f}", str(len(vals))]
                else:
                    cells += ["", "", "0"]
            w.writerow(cells)
    return by_loss


def print_paper_table(by_loss: dict, metric_keys: list[str]) -> None:
    available = [k for k in PAPER_METRICS_VAL if any(k in r for runs in by_loss.values() for r in runs)]
    if not available:
        print("\n(no val_* metrics found, skipping paper-ready table)")
        return
    print("\n## Paper-ready summary (held-out fold val metrics, mean +/- std)\n")
    header = "| Loss | N | " + " | ".join(available) + " |"
    sep = "|------|---|" + "|".join(["----------"] * len(available)) + "|"
    print(header)
    print(sep)

    total_dropped: dict[str, dict[str, int]] = {}  # loss -> {metric: dropped}
    for loss in sorted(by_loss):
        runs = by_loss[loss]
        cells = [loss, str(len(runs))]
        for k in available:
            vals, dropped = _finite_values(runs, k)
            if dropped:
                total_dropped.setdefault(loss, {})[k] = dropped
            if vals:
                m = statistics.mean(vals)
                s = statistics.stdev(vals) if len(vals) > 1 else 0.0
                if "mae" in k or "rmse" in k:
                    label = f"{m:.2f} +/- {s:.2f}"
                else:
                    label = f"{m:.4f} +/- {s:.4f}"
                if dropped:
                    # Mark cells where some runs were excluded (e.g. nan/inf PSNR).
                    label += f" (n={len(vals)})"
                cells.append(label)
            else:
                cells.append("--")
        print("| " + " | ".join(cells) + " |")
    print()

    if total_dropped:
        print("Non-finite values dropped (likely NaN/Inf in raw checkpoint metrics):")
        for loss, kv in total_dropped.items():
            for k, n in kv.items():
                print(f"  - {loss}.{k}: {n} run(s)")
        print()


def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()

    rows = collect_rows(sweep_root, which=args.use)
    if not rows:
        print(f"No runs found under {sweep_root}")
        return

    per_run_csv = args.out_csv or (sweep_root / "summary_per_run.csv")
    metric_keys = write_per_run_csv(rows, per_run_csv)
    print(f"Per-run CSV       : {per_run_csv}  ({len(rows)} rows)")

    per_loss_csv = sweep_root / "summary_per_loss.csv"
    by_loss = write_per_loss_csv(rows, metric_keys, per_loss_csv)
    print(f"Per-loss summary  : {per_loss_csv}")

    print_paper_table(by_loss, metric_keys)


if __name__ == "__main__":
    main()
