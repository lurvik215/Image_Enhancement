"""
train.py - trains MIRNet-v2 for low-light image enhancement.

    python train.py                                   # defaults from config.py
    python train.py --epochs 100 --batch_size 4 --patch_size 128
    python train.py --resume auto                     # continue from last.pth

Outputs go to  <results_root>/<run_name>/ :
    checkpoints/best.pth, last.pth, epoch_XXX.pth
    logs/train.log
    visualizations/val/epoch_XXX/*.png
    train_metrics.csv   (one row per epoch)
"""
from __future__ import annotations

import argparse
import copy
import time

import torch
from tqdm import tqdm

from config import LowLightConfig, add_common_args, cfg_from_args
from data_loader import build_dataloader
from losses import TotalLoss
from model_arch import build_model
from utils import (AverageMeter, append_csv_row, build_optimizer, build_scheduler, create_run_dirs,
                   evaluate_loop, get_device, load_checkpoint, save_checkpoint, seed_everything,
                   setup_logger)

LOSS_KEYS = ("total", "l1", "lpips", "fft")


def train(cfg: LowLightConfig) -> dict:
    seed_everything(cfg.seed)
    dirs = create_run_dirs(cfg)
    log = setup_logger("train", dirs["logs"] / "train.log")
    cfg.save(dirs["run"] / "config.json")

    device = get_device(cfg)
    use_amp = cfg.amp and device.type == "cuda"
    accum = max(1, cfg.accum_steps)
    per_iter = cfg.lr_schedule == "step"  # step schedule counts optimizer updates, cosine counts epochs
    if per_iter:
        log.info("LR schedule: x%.2f every %d iterations, starting at %g", cfg.lr_gamma, cfg.lr_step_iters, cfg.lr)
    else:
        log.info("LR schedule: warm-up + cosine, %g -> %g over %d epochs", cfg.lr, cfg.min_lr, cfg.epochs)
    log.info("Run: %s | dataset: %s | device: %s | AMP: %s", cfg.run_name, cfg.dataset, device, use_amp)
    log.info("Patch %d | batch %d x accumulation %d = effective batch %d",
             cfg.patch_size, cfg.batch_size, accum, cfg.batch_size * accum)

    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")
    log.info("Train pairs: %d | Val pairs: %d", len(train_loader.dataset), len(val_loader.dataset))

    model = build_model(cfg).to(device)
    log.info("Official MIRNet-v2 parameters: %.2f M", sum(p.numel() for p in model.parameters()) / 1e6)
    criterion = TotalLoss(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    metrics_csv = dirs["run"] / "train_metrics.csv"
    start_epoch, best_psnr = 1, 0.0

    resume = cfg.resume
    if resume == "auto":
        resume = str(dirs["checkpoints"] / "last.pth")
    if resume:
        ckpt = load_checkpoint(resume, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch, best_psnr = ckpt["epoch"] + 1, ckpt.get("best_psnr", 0.0)
        log.info("Resumed from %s (epoch %d, best PSNR %.2f)", resume, ckpt["epoch"], best_psnr)
        if cfg.resume_lr is not None:
            # Override the restored learning rate and start a fresh schedule from it. A cosine
            # schedule is stretched over the epochs that are still to run, without warm-up.
            for g in optimizer.param_groups:
                g["lr"] = g["initial_lr"] = cfg.resume_lr
            sched_cfg = copy.copy(cfg)
            sched_cfg.lr = cfg.resume_lr
            sched_cfg.epochs = max(1, cfg.epochs - ckpt["epoch"])
            sched_cfg.warmup_epochs = 0
            scheduler = build_scheduler(optimizer, sched_cfg)
            log.info("LR override: restarting the '%s' schedule from %g", cfg.lr_schedule, cfg.resume_lr)
        log.info("Learning rate in use after resume: %g", optimizer.param_groups[0]["lr"])
    elif metrics_csv.exists():
        metrics_csv.unlink()  # fresh run: start a clean metrics file

    if cfg.resume_lr is not None and not resume:
        log.warning("--resume_lr only applies when resuming (--resume); it is ignored here.")

    best_path = dirs["checkpoints"] / "best.pth"

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        meters = {k: AverageMeter() for k in LOSS_KEYS}
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        n_batches = len(train_loader)
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(bar):
            low = batch["low"].to(device, non_blocking=True)
            high = batch["high"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                pred = model(low)
            loss, parts = criterion(pred, high)  # computed in float32

            scaler.scale(loss / accum).backward()  # gradients are summed over `accum` batches
            if (i + 1) % accum == 0 or (i + 1) == n_batches:
                if cfg.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if per_iter:
                    scheduler.step()

            for k, v in parts.items():
                meters[k].update(v, low.shape[0])
            bar.set_postfix(loss=f"{meters['total'].avg:.4f}")

        lr = optimizer.param_groups[0]["lr"]
        if not per_iter:
            scheduler.step()

        train_time_s = time.time() - t0  # training part of the epoch only
        row = {"epoch": epoch, "lr": lr}
        row.update({f"train_{k}": round(meters[k].avg, 6) for k in LOSS_KEYS})
        row.update({"val_psnr": "", "val_ssim": ""})
        is_best = False
        val_time_s = 0.0

        if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
            t_val = time.time()
            summary, _ = evaluate_loop(
                model, val_loader, device, amp=use_amp, max_images=cfg.val_max_images,
                vis_dir=dirs["visualizations"] / "val" / f"epoch_{epoch:03d}", num_vis=cfg.num_vis,
                desc="validate",
            )
            val_time_s = time.time() - t_val
            row["val_psnr"], row["val_ssim"] = round(summary["psnr"], 4), round(summary["ssim"], 4)
            if summary["psnr"] > best_psnr:
                best_psnr, is_best = summary["psnr"], True
                save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_psnr, cfg)

        row["train_time_s"] = round(train_time_s, 1)
        row["val_time_s"] = round(val_time_s, 1)
        row["epoch_time_s"] = round(time.time() - t0, 1)
        row["is_best"] = int(is_best)
        append_csv_row(metrics_csv, row)

        save_checkpoint(dirs["checkpoints"] / "last.pth", model, optimizer, scheduler, scaler, epoch, best_psnr, cfg)
        if cfg.save_every and epoch % cfg.save_every == 0:
            save_checkpoint(dirs["checkpoints"] / f"epoch_{epoch:03d}.pth", model, None, None, None,
                            epoch, best_psnr, cfg)

        log.info("Epoch %3d | loss %.4f | val PSNR %s | val SSIM %s | best %.2f | train %.0fs + val %.0fs = %.0fs%s",
                 epoch, meters["total"].avg, row["val_psnr"] or "-", row["val_ssim"] or "-",
                 best_psnr, train_time_s, val_time_s, row["epoch_time_s"], "  *best*" if is_best else "")

    log.info("Training finished. Best val PSNR: %.2f dB -> %s", best_psnr, best_path)
    return {"best_psnr": best_psnr, "best_checkpoint": str(best_path), "run_dir": str(dirs["run"])}


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser(description="Train MIRNet-v2 for low-light enhancement"))
    train(cfg_from_args(parser.parse_args()))