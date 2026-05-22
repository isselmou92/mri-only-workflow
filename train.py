# train.py
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader
# --- NEW AMP API ---
from torch.amp import autocast, GradScaler

from simple_sct.config import load_config, choose_device
from simple_sct.utils.logging import (
    get_logger,
    prepare_dirs,
    create_tb_writer,
    log_scalars,
    log_triplet_panels,
    log_xai_maps,
)
from simple_sct.data.volume_dataset import VolumePairs, unnormalize
from simple_sct.models.simple_unet3d import UNet3D
# >>> changed here: use your new ReconstructionLoss
from simple_sct.losses.basic import ReconstructionLoss
from simple_sct.metrics.metrics import compute_all
from simple_sct.xai.attribution import gradient_saliency, occlusion_sensitivity


def _to_hu(t: torch.Tensor, hu_min: float, hu_max: float, scale_target: str) -> torch.Tensor:
    """Map a tensor back to HU so metrics are interpretable.

    For ``scale_target='minmax'`` the tensor lives in [-1, 1]; we invert that.
    For ``scale_target='hu'`` it is already in HU and passes through.
    """
    if scale_target == "hu":
        return t
    return unnormalize(t, hu_min, hu_max)


# -----------------------------
# EMA (optional)
# -----------------------------
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * p.detach()
            self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].data)

    def restore(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.backup[name].data)
        self.backup = {}


# -----------------------------
# Train/Val/Test Split by Mouse ID and Timepoint
# -----------------------------
def _extract_mouse_id(stem: str) -> str:
    """
    Pull the mouse identifier from a NIfTI filename stem.

    Convention: the mouse ID is the leading underscore-separated token,
    e.g. 'MZEI00066_MR_HEAD_T1' -> 'MZEI00066'.
    """
    return stem.split("_", 1)[0].upper()


def _sample_filename_stems(ds: VolumePairs) -> list[str]:
    """Return one filename stem per dataset sample, e.g. ``MZEI00018_MR_T1``.

    Strips both the ``.nii.gz`` and ``.nii`` extensions by splitting on the
    first dot in the filename (``.split(".")[0]``), matching the rule used
    by ``_extract_mouse_id``.
    """
    return [Path(p).name.split(".")[0] for p in ds.files]


def _partition_into_folds(items: list, k: int) -> list[list]:
    """
    Deterministically split a list into k contiguous, near-equal-sized folds.
    Earlier folds receive the extra items when len(items) is not divisible by k.
    """
    n = len(items)
    base, rem = divmod(n, k)
    out: list[list] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        out.append(items[start:start + size])
        start += size
    return out


def split_kfold(
    ds: VolumePairs,
    fold: int,
    seed: int = 1337,
    k: int = 3,
) -> tuple[list[int], list[int], list[list[str]]]:
    """
    Pure k-fold cross-validation grouped by mouse ID (subject-level CV).

    Mouse IDs are extracted from the MRI filename stem (token before the first
    underscore), sorted alphabetically, and partitioned into ``k`` disjoint
    groups. Samples whose mouse belongs to ``fold_assignments[fold]`` form the
    held-out validation set; samples from all other mice form the training set.

    There is no additional train/test split: model selection (best checkpoint)
    is done on the held-out val fold, and that same fold provides the reported
    per-fold metric. Across the k runs, every mouse is held out exactly once,
    which is the textbook k-fold CV property. The ``seed`` only affects the
    deterministic shuffle of the training indices (for batch diversity); it
    has no effect on which mouse lands in which fold.

    Returns:
        (train_indices, val_indices, fold_assignments)
        where ``fold_assignments[i]`` is the sorted list of mouse IDs in fold i.
    """
    import random

    if not (0 <= fold < k):
        raise ValueError(f"fold must be in [0, {k}); got {fold}")

    sample_keys = _sample_filename_stems(ds)
    sample_mouse: list[str] = [_extract_mouse_id(s) for s in sample_keys]
    unique_mice: list[str] = sorted(set(sample_mouse))
    if len(unique_mice) < k:
        raise RuntimeError(
            f"Cannot do {k}-fold CV with only {len(unique_mice)} unique mouse IDs"
        )

    fold_assignments = _partition_into_folds(unique_mice, k)
    val_mice = set(fold_assignments[fold])

    val_indices: list[int] = []
    train_indices: list[int] = []
    for idx, mouse_id in enumerate(sample_mouse):
        (val_indices if mouse_id in val_mice else train_indices).append(idx)

    rng = random.Random(seed)
    rng.shuffle(train_indices)

    return train_indices, val_indices, fold_assignments


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train minimal, explainable 3D sCT model with TensorBoard + live XAI")
    ap.add_argument("--config", required=True, help="Path to YAML config (e.g., configs/default.yaml)")
    ap.add_argument("--device", default=None, help='Override device ("cuda","cpu","auto"); else use config preference')
    return ap.parse_args()


# -----------------------------
# Validation loop
# -----------------------------
@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    include_psnr: bool,
    include_ssim: bool,
    hu_min: float,
    hu_max: float,
    scale_target: str,
    *,
    writer: Optional[Any] = None,
    tb_tag_prefix: Optional[str] = None,
    tb_global_step: Optional[int] = None,
    tb_render_axes: Optional[Sequence[str]] = None,
    tb_slice_indices: Optional[Dict[str, Sequence[int]]] = None,
    tb_max_samples: int = 0,
) -> Dict[str, float]:
    """
    Run inference over a loader and aggregate metrics in HU.

    Predictions/targets are first un-normalised to HU so MAE/RMSE are directly
    interpretable. PSNR/SSIM use ``data_range = hu_max - hu_min``. Optionally
    also logs MRI/pred/CT triplet panels for the first ``tb_max_samples``
    samples under ``tb_tag_prefix/sample{i}`` (set ``tb_max_samples=0`` or
    leave ``writer`` as None to skip image logging).
    """
    model.eval()
    hu_range = float(hu_max) - float(hu_min)
    agg = {"mae": 0.0, "rmse": 0.0}
    if include_psnr:
        agg["psnr"] = 0.0
    if include_ssim:
        agg["ssim"] = 0.0

    log_imgs = (
        writer is not None
        and tb_max_samples > 0
        and tb_tag_prefix is not None
        and tb_global_step is not None
        and tb_render_axes is not None
        and tb_slice_indices is not None
    )

    n = 0
    logged = 0
    for batch in loader:
        x = batch["mri"].to(device)
        y = batch["ct"].to(device)
        with torch.no_grad():
            pred = model(x)
        pred_hu = _to_hu(pred.float(), hu_min, hu_max, scale_target)
        y_hu = _to_hu(y.float(), hu_min, hu_max, scale_target)
        m = compute_all(pred_hu, y_hu, include_psnr, include_ssim, data_range=hu_range)
        for k, v in m.items():
            agg[k] += float(v)
        n += 1

        if log_imgs and logged < tb_max_samples:
            # log_triplet_panels handles a single example at a time, so iterate
            # over the batch dim and stop once we've hit the budget.
            B = x.shape[0]
            for b in range(B):
                if logged >= tb_max_samples:
                    break
                log_triplet_panels(
                    writer=writer,
                    tag_prefix=f"{tb_tag_prefix}/sample{logged}",
                    mri=x[b:b + 1].detach().cpu(),
                    pred=pred[b:b + 1].detach().cpu(),
                    ct=y[b:b + 1].detach().cpu(),
                    axes=tb_render_axes,
                    slice_indices=tb_slice_indices,
                    global_step=tb_global_step,
                    normalize_input=True,
                    normalize_targets=True,
                )
                logged += 1

    for k in list(agg.keys()):
        agg[k] /= max(1, n)
    return agg


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Device selection (CUDA-first like your codebase)
    device = choose_device(args.device or cfg.train.device_preference)

    # Auto-isolate per-fold artifacts so concurrent folds never overwrite each other.
    fold = int(cfg.data.fold)
    cfg.experiment_name = f"{cfg.experiment_name}_fold{fold}"
    cfg.logging.out_dir = str(Path(cfg.logging.out_dir) / f"fold{fold}")

    logger = get_logger()
    out_dir, ckpt_dir, preds_dir, tb_dir = prepare_dirs(
        cfg.logging.out_dir, cfg.experiment_name, cfg.logging.tensorboard, cfg.logging.log_dir
    )
    writer = create_tb_writer(tb_dir)

    # Data - pure 3-fold cross-validation by mouse ID. The held-out val fold
    # is the only evaluation set: it drives both per-epoch monitoring and
    # final reporting (best.pt is selected on val_mae).
    #
    # Two VolumePairs instances are built: one with augmentation (train) and
    # one without (val). They scan the same directory with the same exclusion
    # list and the same cap, so their internal file lists are identical and
    # the indices returned by split_kfold address both consistently. The
    # assertion below makes that invariant explicit and crashes loudly if
    # someone ever changes one constructor path without the other.
    ds = VolumePairs(cfg)
    ds_eval = VolumePairs(cfg)
    ds_eval.augment = False
    assert ds.files == ds_eval.files, (
        "VolumePairs file lists diverged between train and val instances; "
        "k-fold indices would no longer be aligned."
    )

    train_indices, val_indices, fold_assignments = split_kfold(
        ds, fold=fold, seed=cfg.seed, k=3
    )

    train_ds = torch.utils.data.Subset(ds, train_indices)
    val_ds = torch.utils.data.Subset(ds_eval, val_indices)

    n_train, n_val = len(train_ds), len(val_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Model, loss, optimizer, scaler, EMA
    model = UNet3D(**cfg.model.__dict__).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # --- GradScaler with conservative initial scale ---
    # Default 2^16=65536 × loss~1330 ≈ 87M → fp16 gradient overflow.
    # Start at 2^10=1024 and let it grow automatically.
    scaler = GradScaler(device="cuda", enabled=bool(cfg.train.amp), init_scale=2**10)

    # Cosine-annealing LR schedule (decays to ~0 over all epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.train.epochs, eta_min=1e-6,
    )

    # SSIM data range follows the target scaling: HU targets keep their raw
    # range, minmax targets live in [-1, 1]. The reconstruction loss runs in
    # whatever space the target is in; metrics are always reported in HU below.
    ssim_dr = (float(cfg.data.hu_max) - float(cfg.data.hu_min)
               if cfg.data.scale_target == "hu" else 2.0)
    hu_range = float(cfg.data.hu_max) - float(cfg.data.hu_min)

    loss_fn = ReconstructionLoss(
        l1_weight=cfg.loss.l1_weight,
        ssim_weight=cfg.loss.ssim_weight,
        ssim_kernel=getattr(cfg.loss, "ssim_kernel", 7),
        ssim_sigma=getattr(cfg.loss, "ssim_sigma", 1.5),
        ssim_data_range=ssim_dr,
    )

    ema = EMA(model, decay=0.999) if cfg.train.ema else None

    # Training state. Best checkpoint is selected on val_mae (lowest wins).
    best_val = float("inf")
    global_step = 0

    logger.info(
        f"Start training | fold={fold}/3 | epochs={cfg.train.epochs} | "
        f"train={n_train} | val={n_val} | device={device} | amp={cfg.train.amp}"
    )
    excluded_mice = getattr(ds, "_excluded_mouse_ids", []) or []
    excluded_n = getattr(ds, "_excluded_sample_count", 0)
    if excluded_mice:
        logger.info(
            f"Excluded {excluded_n} sample(s) belonging to test mice: {excluded_mice}"
        )
    for i, mice in enumerate(fold_assignments):
        marker = " (val)" if i == fold else ""
        logger.info(f"  fold {i}{marker}: {mice}")
    _all_keys = _sample_filename_stems(ds)
    logger.info(f"Val samples: {[_all_keys[i] for i in val_indices]}")

    # --- XAI colormap (new): user wanted hot/cold
    xai_cmap = getattr(cfg.train, "xai_cmap", "hot")  # typical diverging map

    # Helper closures for TB logging
    def _maybe_log_triplets(step: int, x: torch.Tensor, pred: torch.Tensor, y: torch.Tensor):
        if writer is None or cfg.logging.tb_images_every_steps <= 0:
            return
        if step % cfg.logging.tb_images_every_steps != 0:
            return
        # Only log first example in batch to keep TB light
        x0 = x[:1].detach().cpu()
        p0 = pred[:1].detach().cpu()
        y0 = y[:1].detach().cpu()
        log_triplet_panels(
            writer=writer,
            tag_prefix="train/volumes",
            mri=x0,
            pred=p0,
            ct=y0,
            axes=cfg.logging.tb_render_axes,
            slice_indices=cfg.logging.tb_slice_indices.__dict__,
            global_step=step,
            normalize_input=True,
            normalize_targets=True,
        )

    def _maybe_log_xai(step: int, x: torch.Tensor, y: torch.Tensor):
        if writer is None or not cfg.train.xai_enabled or cfg.train.xai_log_every_steps <= 0:
            return
        if step % cfg.train.xai_log_every_steps != 0:
            return

        x0 = x[:1].to(device)  # (1,1,D,H,W)
        y0 = y[:1].to(device)

        # Saliency
        if "saliency" in cfg.train.xai_methods:
            sal = gradient_saliency(model, x0).detach().cpu()
            log_xai_maps(
                writer=writer,
                tag_prefix="train/xai/saliency",
                xai=sal,
                axes=cfg.train.xai_log_axes,
                slice_indices=cfg.train.xai_slice_indices.__dict__,
                global_step=step,
                normalize=cfg.train.xai_log_norm,
                cmap=xai_cmap,          # <<< hot/cold
            )

        # Occlusion (heavier)
        if "occlusion" in cfg.train.xai_methods:
            pd, ph, pw = cfg.train.xai_occlusion.patch
            sd, sh, sw = cfg.train.xai_occlusion.stride
            occ = occlusion_sensitivity(
                model=model,
                x=x0,
                y_true=y0,
                patch=(pd, ph, pw),
                stride=(sd, sh, sw),
                fill=None,
            ).detach().cpu()
            log_xai_maps(
                writer=writer,
                tag_prefix="train/xai/occlusion",
                xai=occ,
                axes=cfg.train.xai_log_axes,
                slice_indices=cfg.train.xai_slice_indices.__dict__,
                global_step=step,
                normalize=cfg.train.xai_log_norm,
                cmap=xai_cmap,          # <<< hot/cold
            )

    # ---------------- Train epochs ----------------
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_valid_steps = 0

        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_loader, start=1):
            x = batch["mri"].to(device, non_blocking=True)
            y = batch["ct"].to(device, non_blocking=True)

            # --- Forward in AMP; loss in fp32 ---
            with autocast(device_type="cuda", enabled=bool(cfg.train.amp)):
                pred = model(x)

            # Cast to fp32 *outside* autocast so loss is computed stably
            pred_f32 = pred.float()
            y_f32 = y.float()

            # ---- Guard: skip if model output contains NaN/Inf ----
            if not torch.isfinite(pred_f32).all():
                opt.zero_grad(set_to_none=True)
                global_step += 1
                continue

            loss = loss_fn(pred_f32, y_f32) / max(1, cfg.train.grad_accum_steps)

            # ---- Guard: a non-tensor loss has no autograd graph and can't be
            # backpropagated. Skip the batch and log a (rate-limited) warning
            # so we can still diagnose the underlying issue.
            if not isinstance(loss, torch.Tensor):
                if global_step % 50 == 0:
                    get_logger().warning(
                        f"loss_fn returned non-tensor "
                        f"({type(loss).__name__}={loss!r}); skipping batch. "
                        f"weights: l1={cfg.loss.l1_weight}, "
                        f"ssim={cfg.loss.ssim_weight}"
                    )
                opt.zero_grad(set_to_none=True)
                global_step += 1
                continue

            # ---- Guard: skip batches that produce inf/nan loss ----
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                global_step += 1
                continue

            scaler.scale(loss).backward()

            # Step with gradient accumulation
            if it % cfg.train.grad_accum_steps == 0:
                if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

            running += float(loss.item()) * max(1, cfg.train.grad_accum_steps)
            n_valid_steps += 1
            global_step += 1

            # Console + TB scalars
            if it % cfg.logging.log_interval == 0:
                avg_loss = running / max(1, n_valid_steps)
                msg = f"[Epoch {epoch}] step {it}/{len(train_loader)} | loss={avg_loss:.4f}"
                if writer is not None:
                    log_scalars(writer, {"loss": avg_loss}, global_step, prefix="train")
                # also log quick metrics on-the-fly for current batch (optional)
                if writer is not None:
                    with torch.no_grad():
                        pred_hu = _to_hu(
                            pred_f32, cfg.data.hu_min, cfg.data.hu_max, cfg.data.scale_target
                        )
                        y_hu = _to_hu(
                            y_f32, cfg.data.hu_min, cfg.data.hu_max, cfg.data.scale_target
                        )
                        m = compute_all(
                            pred_hu, y_hu,
                            include_psnr=cfg.metrics.compute_psnr,
                            include_ssim=cfg.metrics.compute_ssim,
                            data_range=hu_range,
                        )
                        log_scalars(writer, m, global_step, prefix="train/batch_metrics")
                        msg += f" | MAE={m['mae']:.2f} HU"
                get_logger().info(msg)

            # Images & XAI to TB
            _maybe_log_triplets(global_step, x, pred, y)
            _maybe_log_xai(global_step, x, y)

        # ---------------- Validation (held-out CV fold) ----------------
        # Evaluate EMA weights if enabled
        if ema is not None:
            ema.apply_shadow(model)
        # Decide once per epoch whether to log val predicted volumes to TB.
        _val_imgs_every = int(getattr(cfg.logging, "tb_log_val_images_every_epochs", 1))
        _log_val_imgs = (
            writer is not None
            and _val_imgs_every > 0
            and (epoch % _val_imgs_every == 0)
        )
        val_metrics = run_validation(
            model=model,
            loader=val_loader,
            device=device,
            include_psnr=cfg.metrics.compute_psnr,
            include_ssim=cfg.metrics.compute_ssim,
            hu_min=cfg.data.hu_min,
            hu_max=cfg.data.hu_max,
            scale_target=cfg.data.scale_target,
            writer=writer if _log_val_imgs else None,
            tb_tag_prefix="val/volumes" if _log_val_imgs else None,
            tb_global_step=epoch if _log_val_imgs else None,
            tb_render_axes=cfg.logging.tb_render_axes,
            tb_slice_indices=cfg.logging.tb_slice_indices.__dict__,
            tb_max_samples=int(getattr(cfg.logging, "tb_val_max_samples", 1)) if _log_val_imgs else 0,
        )
        if ema is not None:
            ema.restore(model)

        # TensorBoard: epoch metrics for val
        if writer is not None:
            log_scalars(writer, val_metrics, epoch, prefix="val")

        # Step the LR scheduler once per epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        took = time.time() - t0
        avg_epoch_loss = running / max(1, n_valid_steps)
        def _fmt_val(k: str, v: float) -> str:
            if k in ("mae", "rmse"):
                return f"val_{k}={v:.2f} HU"
            return f"val_{k}={v:.4f}"
        val_str = " | ".join(_fmt_val(k, v) for k, v in val_metrics.items())
        get_logger().info(
            f"[Epoch {epoch}] train_loss={avg_epoch_loss:.4f} | "
            + val_str
            + f" | lr={current_lr:.2e} | {took:.1f}s"
        )
        if writer is not None:
            log_scalars(writer, {"lr": current_lr}, epoch, prefix="train")

        # Save checkpoints: keep only the latest and the best (by val MAE).
        # Pure k-fold CV: the held-out val fold is the *only* evaluation set,
        # so best.pt is the result we report for this fold.
        ckpt_payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val": val_metrics,
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        latest_path = Path(ckpt_dir) / "latest.pt"
        torch.save(ckpt_payload, latest_path)

        # Save best (minimize val_mae)
        current = val_metrics.get("mae", None)
        if current is not None and current < best_val:
            best_val = current
            best_path = Path(ckpt_dir) / "best.pt"
            torch.save(ckpt_payload, best_path)
            get_logger().info(f"Saved new best checkpoint @ epoch {epoch} (val_mae={current:.4f})")

    # ---------------- Done ----------------
    # In pure k-fold CV, the held-out val fold is evaluated every epoch and
    # best.pt already holds the weights with the lowest val_mae for this fold.
    # No separate final-evaluation pass is needed.
    if best_val < float("inf"):
        get_logger().info(
            f"Training complete. Best val_mae for fold {fold}: {best_val:.4f} "
            f"(checkpoint: {Path(ckpt_dir) / 'best.pt'})"
        )
    else:
        get_logger().info("Training complete. No best checkpoint was saved (val_mae never finite?).")

    # Flush and close the TB writer so all events land on disk before the
    # process exits. Important when the sweep launcher runs many of these
    # concurrently and tears them down close together.
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
