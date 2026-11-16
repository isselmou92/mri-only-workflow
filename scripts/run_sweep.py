#!/usr/bin/env python
"""
Parallel loss-function benchmark sweep launcher.

Generates one YAML per (loss, fold, seed) triple from a base config and runs
them as concurrent child processes pinned to specific GPUs via
CUDA_VISIBLE_DEVICES. A separate ThreadPoolExecutor per GPU enforces the
per-GPU concurrency cap exactly.

Tested target: 2x A6000 + 64-core CPU + 256 GB RAM (Windows or Linux), with
the 3D U-Net sCT model using ~9 GB VRAM/job at batch_size=2.

Default sweep: {l1, ssim} x {fold 0, 1, 2} x {seed 1337} = 6 runs.

Examples
--------
Dry-run (just print the plan):
    python scripts/run_sweep.py --dry-run

Real sweep, default 6 runs across 2 GPUs:
    python scripts/run_sweep.py --out output/loss_sweep_v1

Smoke test with 5 epochs to make sure plumbing works:
    python scripts/run_sweep.py --epochs 5 --per-gpu 1 --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Each preset isolates one loss term so the comparison is unambiguous.
LOSS_PRESETS: dict[str, dict[str, float]] = {
    "l1":   {"l1_weight": 1.0, "ssim_weight": 0.0},
    "ssim": {"l1_weight": 0.0, "ssim_weight": 1.0},
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-config", default="configs/default.yaml",
                    help="Base YAML to derive every per-run config from")
    ap.add_argument("--out", default="output/loss_sweep",
                    help="Sweep root: per-run configs, logs, and outputs go under here")

    ap.add_argument("--losses", nargs="+", choices=list(LOSS_PRESETS),
                    default=["l1", "ssim"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337],
                    help="One or more seeds. Default is a single seed; the "
                         "loss-comparison protocol does not require multiple.")

    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1],
                    help="Physical GPU indices to use")
    ap.add_argument("--per-gpu", type=int, default=4,
                    help="Concurrent runs per GPU. Default 4 targets ~36 GB on a "
                         "48 GB A6000 at ~9 GB peak/run, leaving ~12 GB headroom for "
                         "PyTorch caching + CUDA context spikes. Bump to 5 for "
                         "tighter packing or back to 3 if you see CUDA OOM.")
    ap.add_argument("--num-workers", type=int, default=2,
                    help="Per-run data.num_workers (kept low because many runs share the CPU)")
    ap.add_argument("--prefetch-factor", type=int, default=2)

    ap.add_argument("--epochs", type=int, default=None,
                    help="Optional override for train.epochs (e.g. for a quick smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan and write per-run YAMLs but do not launch anything")
    ap.add_argument("--skip-completed", action="store_true",
                    help="Skip a run when its latest.pt already shows epoch >= train.epochs")
    ap.add_argument("--heartbeat-sec", type=int, default=60,
                    help="How often to print the live-runs status snapshot")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Per-run YAML generation
# ---------------------------------------------------------------------------
def make_run_id(loss: str, fold: int, seed: int) -> str:
    return f"{loss}_fold{fold}_seed{seed}"


def make_run_yaml(
    base: dict,
    loss: str,
    fold: int,
    seed: int,
    num_workers: int,
    prefetch: int,
    epochs_override: int | None,
    sweep_root: Path,
    base_config_dir: Path,
) -> dict:
    cfg = deepcopy(base)

    cfg["seed"] = int(seed)
    cfg.setdefault("data", {})
    cfg["data"]["fold"] = int(fold)
    cfg["data"]["num_workers"] = int(num_workers)
    cfg["data"]["prefetch_factor"] = int(prefetch)

    # Make data dirs absolute so they don't depend on the new YAML's location.
    # (Skip POSIX-absolute strings on Windows: Path.is_absolute() returns False
    # for "/mnt/c/..." but we still want to keep them verbatim.)
    for key in ("mri_dir", "ct_dir", "test_mri_dir", "test_ct_dir"):
        if key not in cfg["data"]:
            continue
        val = str(cfg["data"][key])
        looks_posix_absolute = val.startswith("/") or val.startswith("~")
        if not looks_posix_absolute:
            p = Path(val)
            if not p.is_absolute():
                cfg["data"][key] = str((base_config_dir / p).resolve())

    cfg.setdefault("loss", {}).update(LOSS_PRESETS[loss])

    cfg.setdefault("train", {})
    if epochs_override is not None:
        cfg["train"]["epochs"] = int(epochs_override)

    # Per-run output paths. train.py auto-suffixes "/fold{N}" inside out_dir
    # and "_fold{N}" on experiment_name, so each (loss, seed) run gets one
    # base directory and the three folds land underneath.
    cfg.setdefault("logging", {})
    cfg["logging"]["out_dir"] = str(sweep_root / f"{loss}_seed{seed}")
    cfg["logging"]["log_dir"] = str(sweep_root / "tb")
    cfg["experiment_name"] = f"sct_{loss}_seed{seed}"

    return cfg


def already_completed(run_out_dir: Path, fold: int, target_epochs: int) -> bool:
    """Cheap resumability check: does a finished latest.pt already exist?"""
    ckpt = run_out_dir / f"fold{fold}" / "checkpoints" / "latest.pt"
    if not ckpt.exists():
        return False
    try:
        import torch  # local import: only needed when --skip-completed is used
        state = torch.load(str(ckpt), map_location="cpu")
        return int(state.get("epoch", 0)) >= int(target_epochs)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------
def run_one(
    run_id: str,
    yaml_path: Path,
    gpu_id: int,
    log_path: Path,
    active: dict,
) -> tuple[int, int, float]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [sys.executable, str(REPO_ROOT / "train.py"), "--config", str(yaml_path)]
    t0 = time.time()
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
        active[run_id] = (proc, gpu_id, t0)
        try:
            rc = proc.wait()
        finally:
            active.pop(run_id, None)
    return rc, gpu_id, time.time() - t0


def heartbeat_loop(active: dict, stop: threading.Event, every_sec: int) -> None:
    while not stop.wait(every_sec):
        snap = list(active.items())
        if not snap:
            continue
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] running {len(snap)}:")
        for run_id, (_proc, gpu, t0) in snap:
            elapsed = int(time.time() - t0)
            print(f"    GPU{gpu}  {run_id}  ({elapsed // 60}m{elapsed % 60:02d}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    sweep_root = Path(args.out)
    if not sweep_root.is_absolute():
        sweep_root = (REPO_ROOT / sweep_root).resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)
    cfg_dir = sweep_root / "configs"; cfg_dir.mkdir(exist_ok=True)
    log_dir = sweep_root / "logs"; log_dir.mkdir(exist_ok=True)

    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = (REPO_ROOT / base_path).resolve()
    with open(base_path) as f:
        base = yaml.safe_load(f) or {}
    base_config_dir = base_path.parent

    target_epochs = int(args.epochs if args.epochs is not None
                        else base.get("train", {}).get("epochs", 200))

    # ---------------- Build job list ----------------
    jobs: list[tuple[str, Path, Path, str]] = []  # (run_id, yaml_path, log_path, run_out_dir)
    skipped = 0
    for loss in args.losses:
        for fold in args.folds:
            for seed in args.seeds:
                run_id = make_run_id(loss, fold, seed)
                run_cfg = make_run_yaml(
                    base, loss, fold, seed,
                    args.num_workers, args.prefetch_factor, args.epochs,
                    sweep_root, base_config_dir,
                )
                yaml_path = cfg_dir / f"{run_id}.yaml"
                log_path = log_dir / f"{run_id}.log"
                with open(yaml_path, "w") as f:
                    yaml.safe_dump(run_cfg, f, sort_keys=False)

                run_out_dir = Path(run_cfg["logging"]["out_dir"])
                if args.skip_completed and already_completed(run_out_dir, fold, target_epochs):
                    print(f"[skip] {run_id} already at epoch >= {target_epochs}")
                    skipped += 1
                    continue
                jobs.append((run_id, yaml_path, log_path, str(run_out_dir)))

    if not jobs:
        print(f"No jobs to run (skipped {skipped}).")
        return

    # Round-robin GPU assignment for predictability.
    job_assignments = [
        (rid, ypath, args.gpus[i % len(args.gpus)], lpath)
        for i, (rid, ypath, lpath, _o) in enumerate(jobs)
    ]

    print(f"Sweep root      : {sweep_root}")
    print(f"Base config     : {base_path}")
    print(f"Losses          : {args.losses}")
    print(f"Folds           : {args.folds}")
    print(f"Seeds           : {args.seeds}")
    print(f"GPUs            : {args.gpus} (per_gpu={args.per_gpu}, "
          f"total concurrency = {len(args.gpus) * args.per_gpu})")
    print(f"Per-run workers : {args.num_workers}")
    print(f"Total jobs      : {len(job_assignments)} (skipped {skipped})")
    print()
    for run_id, yaml_path, gpu_id, _ in job_assignments:
        print(f"  GPU {gpu_id}  {run_id}  ->  {yaml_path.name}")

    if args.dry_run:
        print("\n[dry-run] No subprocesses launched.")
        return

    # ---------------- Launch ----------------
    active: dict[str, tuple[subprocess.Popen, int, float]] = {}
    stop = threading.Event()
    threading.Thread(target=heartbeat_loop, args=(active, stop, args.heartbeat_sec),
                     daemon=True).start()

    # One ThreadPoolExecutor per GPU. Each pool's max_workers is the per-GPU
    # cap, which is the simplest provably correct way to bound concurrency
    # per device. ThreadPoolExecutor.submit returns immediately and queues the
    # rest. Total in-flight = sum of pool sizes = len(gpus) * per_gpu.
    pools = {g: ThreadPoolExecutor(max_workers=args.per_gpu,
                                   thread_name_prefix=f"gpu{g}")
             for g in args.gpus}
    futs: dict = {}
    t_sweep_start = time.time()
    for run_id, yaml_path, gpu_id, log_path in job_assignments:
        fut = pools[gpu_id].submit(run_one, run_id, yaml_path, gpu_id, log_path, active)
        futs[fut] = (run_id, gpu_id)

    print(f"\nLaunched {len(futs)} runs at {datetime.now().strftime('%H:%M:%S')}.\n")

    failed: list[str] = []
    try:
        for fut in as_completed(futs):
            run_id, gpu = futs[fut]
            try:
                rc, gpu, took = fut.result()
                tag = "OK" if rc == 0 else f"FAIL rc={rc}"
                mins, secs = divmod(int(took), 60)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{tag}] {run_id} "
                      f"(GPU {gpu}, {mins}m{secs:02d}s)")
                if rc != 0:
                    failed.append(run_id)
            except Exception as e:  # pragma: no cover
                print(f"[ERROR] {run_id}: {e}")
                failed.append(run_id)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: terminating active runs...")
        for run_id, (proc, _g, _t) in list(active.items()):
            try:
                proc.terminate()
            except Exception:
                pass
        raise
    finally:
        stop.set()
        for p in pools.values():
            p.shutdown(wait=False)

    total = int(time.time() - t_sweep_start)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    print(f"\nSweep complete in {h}h{m:02d}m{s:02d}s. "
          f"failed={len(failed)}/{len(job_assignments)}")
    if failed:
        print("Failed runs (check their .log files in", log_dir, "):")
        for r in failed:
            print(f"  {r}")
    print(f"\nNext: python scripts/aggregate_sweep.py --sweep-root {sweep_root}")


if __name__ == "__main__":
    main()
