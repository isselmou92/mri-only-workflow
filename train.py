# train.py
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

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
from simple_sct.data.volume_dataset import VolumePairs
from simple_sct.models.simple_unet3d import UNet3D
# >>> changed here: use your new ReconstructionLoss
from simple_sct.losses.basic import ReconstructionLoss
from simple_sct.metrics.metrics import compute_all
from simple_sct.xai.attribution import gradient_saliency, occlusion_sensitivity


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
def split_by_mouse_timepoint(ds: VolumePairs, seed: int = 1337) -> tuple[list[int], list[int], list[int]]:
    """
    Split dataset into train/val/test.
    
    Val set (held out): Mouse 66 T1,T2 + Mouse 30 T1,T2,T3 (5 samples)
    Train/Test: Remaining samples split 80/20
    
    Returns:
        (train_indices, val_indices, test_indices)
    """
    import random
    
    val_patterns = [
        ("66", ["T1", "T2"]),
        ("30", ["T1", "T2", "T3"]),
    ]
    
    val_indices = []
    remaining_indices = []
    
    for idx, pair in enumerate(ds.pairs):
        # Use the actual MRI filename for matching (not pair_keys which may be indices)
        filename = pair.mri.stem.upper()  # e.g., "MZEI00066_MR_HEAD_REGISTERED_T1"
        
        is_val = False
        
        # Check val patterns
        for mouse_id, timepoints in val_patterns:
            # Match mouse ID (handles both MZEI00066 and MZEI0066 formats)
            if f"MZEI000{mouse_id}" in filename or f"MZEI00{mouse_id}" in filename:
                for tp in timepoints:
                    # Match timepoint at end of filename (e.g., _T1, _T2)
                    if filename.endswith(f"_{tp}") or filename.endswith(tp):
                        is_val = True
                        break
            if is_val:
                break
        
        if is_val:
            val_indices.append(idx)
        else:
            remaining_indices.append(idx)
    
    # Split remaining indices 80/20 into train/test
    rng = random.Random(seed)
    rng.shuffle(remaining_indices)
    
    n_remaining = len(remaining_indices)
    n_train = int(0.8 * n_remaining)
    
    train_indices = remaining_indices[:n_train]
    test_indices = remaining_indices[n_train:]
    
    return train_indices, val_indices, test_indices


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
    data_range: float = 8000.0,
) -> Dict[str, float]:
    model.eval()
    agg = {"mae": 0.0, "rmse": 0.0}
    if include_psnr:
        agg["psnr"] = 0.0
    if include_ssim:
        agg["ssim"] = 0.0

    n = 0
    for batch in loader:
        x = batch["mri"].to(device)
        y = batch["ct"].to(device)
        pred = model(x)
        m = compute_all(pred, y, include_psnr, include_ssim, data_range=data_range)
        for k, v in m.items():
            agg[k] += float(v)
        n += 1

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

    logger = get_logger()
    out_dir, ckpt_dir, preds_dir, tb_dir = prepare_dirs(
        cfg.logging.out_dir, cfg.experiment_name, cfg.logging.tensorboard, cfg.logging.log_dir
    )
    writer = create_tb_writer(tb_dir)

    # Data - split by mouse ID and timepoint
    ds = VolumePairs(cfg)
    train_indices, val_indices, test_indices = split_by_mouse_timepoint(ds, seed=cfg.seed)
    
    train_ds = torch.utils.data.Subset(ds, train_indices)
    val_ds = torch.utils.data.Subset(ds, val_indices)
    test_ds = torch.utils.data.Subset(ds, test_indices)
    
    n_train, n_val, n_test = len(train_ds), len(val_ds), len(test_ds)

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
    test_loader = DataLoader(
        test_ds,
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

    # >>> build ReconstructionLoss fully from YAML (no hardcoding)
    # Compute the SSIM data range from the target scaling:
    #   - "hu"    targets stay in raw HU  → range = hu_max - hu_min
    #   - "minmax" targets are in [-1,1]  → range = 2.0
    if cfg.data.scale_target == "hu":
        ssim_dr = float(cfg.data.hu_max) - float(cfg.data.hu_min)
    else:
        ssim_dr = 2.0

    loss_fn = ReconstructionLoss(
        l1_weight=cfg.loss.l1_weight,
        ssim_weight=cfg.loss.ssim_weight,
        ssim_kernel=getattr(cfg.loss, "ssim_kernel", 7),
        ssim_sigma=getattr(cfg.loss, "ssim_sigma", 1.5),
        ssim_data_range=ssim_dr,
        nmi_weight=getattr(cfg.loss, "nmi_weight", 0.0),
        nmi_bins=getattr(cfg.loss, "nmi_bins", 64),
        nmi_value_min=getattr(cfg.loss, "nmi_value_min", -1.0),
        nmi_value_max=getattr(cfg.loss, "nmi_value_max", 1.0),
        nmi_sigma=getattr(cfg.loss, "nmi_sigma", None),
    )

    ema = EMA(model, decay=0.999) if cfg.train.ema else None

    # Training state
    best_key = cfg.logging.save_best_on
    best_val = float("inf")  # minimize
    global_step = 0

    logger.info(
        f"Start training | epochs={cfg.train.epochs} | train={n_train} | val={n_val} | test={n_test} | "
        f"device={device} | amp={cfg.train.amp}"
    )
    logger.info(f"Val samples: {[ds.pairs[i].mri.stem for i in val_indices]}")
    logger.info(f"Test samples: {[ds.pairs[i].mri.stem for i in test_indices]}")

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
                        m = compute_all(
                            pred_f32, y_f32,
                            include_psnr=cfg.metrics.compute_psnr,
                            include_ssim=cfg.metrics.compute_ssim,
                            data_range=ssim_dr,
                        )
                        log_scalars(writer, m, global_step, prefix="train/batch_metrics")
                        msg += f" | MAE={m['mae']:.2f} HU"
                get_logger().info(msg)

            # Images & XAI to TB
            _maybe_log_triplets(global_step, x, pred, y)
            _maybe_log_xai(global_step, x, y)

        # ---------------- Test ----------------
        # Evaluate EMA weights if enabled
        if ema is not None:
            ema.apply_shadow(model)
        test_metrics = run_validation(
            model=model,
            loader=test_loader,
            device=device,
            include_psnr=cfg.metrics.compute_psnr,
            include_ssim=cfg.metrics.compute_ssim,
            data_range=ssim_dr,
        )
        if ema is not None:
            ema.restore(model)

        # TensorBoard: epoch metrics for test
        if writer is not None:
            log_scalars(writer, test_metrics, epoch, prefix="test")

        # Step the LR scheduler once per epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        took = time.time() - t0
        avg_epoch_loss = running / max(1, n_valid_steps)
        test_str = " | ".join([f"test_{k}={v:.4f}" if k != "mae" else f"test_mae={v:.2f} HU" for k, v in test_metrics.items()])
        get_logger().info(
            f"[Epoch {epoch}] train_loss={avg_epoch_loss:.4f} | "
            + test_str
            + f" | lr={current_lr:.2e} | {took:.1f}s"
        )
        if writer is not None:
            log_scalars(writer, {"lr": current_lr}, epoch, prefix="train")

        # Save checkpoints
        ckpt_payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "test": test_metrics,
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if epoch % cfg.logging.checkpoint_every == 0:
            path = Path(ckpt_dir) / f"epoch_{epoch:04d}.pt"
            torch.save(ckpt_payload, path)

        # Save best (minimize test_mae)
        current = test_metrics.get("mae", None)
        if current is not None and current < best_val:
            best_val = current
            path = Path(ckpt_dir) / "best.pt"
            torch.save(ckpt_payload, path)
            get_logger().info(f"Saved new best checkpoint @ epoch {epoch} (test_mae={current:.4f})")

    # ---------------- Final Validation (held-out set) ----------------
    get_logger().info("Running final validation on held-out set...")
    if ema is not None:
        ema.apply_shadow(model)
    val_metrics = run_validation(
        model=model,
        loader=val_loader,
        device=device,
        include_psnr=cfg.metrics.compute_psnr,
        include_ssim=cfg.metrics.compute_ssim,
        data_range=ssim_dr,
    )
    if ema is not None:
        ema.restore(model)
    
    # Log final val metrics to TensorBoard
    if writer is not None:
        log_scalars(writer, val_metrics, cfg.train.epochs, prefix="val_final")
    
    val_str = " | ".join([f"val_{k}={v:.4f}" if k != "mae" else f"val_mae={v:.2f} HU" for k, v in val_metrics.items()])
    get_logger().info(f"[Final Validation] {val_str}")
    
    # Save final checkpoint with val metrics
    path = Path(ckpt_dir) / "final.pt"
    torch.save({"model": model.state_dict(), "epoch": cfg.train.epochs, "val": val_metrics, "test": test_metrics}, path)
    get_logger().info(f"Saved final checkpoint to {path}")
    
    get_logger().info("Training complete.")


if __name__ == "__main__":
    main()
