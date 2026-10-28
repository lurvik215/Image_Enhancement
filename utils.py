"""
utils.py - helpers shared by train.py, evaluate.py and pipeline.py.

    * run folders / logging / CSV writing
    * tensor <-> numpy conversion
    * PSNR / SSIM and the evaluation loop
    * optimizer + LR scheduler construction
    * checkpoint save / load
"""
from __future__ import annotations

import csv
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm


# ------------------------------------------------------------------ setup helpers
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg) -> torch.device:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available - using CPU.")
        return torch.device("cpu")
    return torch.device(cfg.device)


def create_run_dirs(cfg) -> Dict[str, Path]:
    """Create <results_root>/<run_name>/{checkpoints,logs,predictions,visualizations}."""
    run = cfg.run_dir
    dirs = {
        "run": run,
        "checkpoints": run / "checkpoints",
        "logs": run / "logs",
        "predictions": run / "predictions",
        "visualizations": run / "visualizations",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


class AverageMeter:
    def __init__(self):
        self.sum, self.n = 0.0, 0

    def update(self, v: float, n: int = 1):
        self.sum += float(v) * n
        self.n += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.n, 1)


# --------------------------------------------------------------------- CSV helpers
def append_csv_row(path: Path, row: dict) -> None:
    """Append a row, writing the header first if the file is new."""
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------- tensor <-> numpy helpers
def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """CHW (or 1CHW) float tensor in [0,1]  ->  HWC uint8 array."""
    t = t.detach().float().clamp(0, 1).cpu()
    if t.dim() == 4:
        t = t[0]
    return (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 (or float in [0,1]) array  ->  CHW float tensor in [0,1]."""
    arr = img.astype(np.float32)
    if img.dtype == np.uint8:
        arr /= 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


# ------------------------------------------------------------------------- metrics
def calc_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR (dB) between two uint8 images."""
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else float(10 * math.log10(255.0 ** 2 / mse))


def calc_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM between two uint8 HWC images."""
    return float(structural_similarity(pred, gt, channel_axis=2, data_range=255))


def save_comparison(low: np.ndarray, pred: np.ndarray, gt: Optional[np.ndarray], path: Path) -> None:
    """Save a  low-light | enhanced | ground-truth  strip (no ground truth -> 2 panels)."""
    parts = [low, pred] + ([gt] if gt is not None else [])
    Image.fromarray(np.concatenate(parts, axis=1)).save(path)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate_loop(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    amp: bool = False,
    max_images: Optional[int] = None,
    save_pred_dir: Optional[Path] = None,
    vis_dir: Optional[Path] = None,
    num_vis: int = 0,
    lpips_model: Optional[torch.nn.Module] = None,
    desc: str = "eval",
    has_gt: bool = True,
):
    """Run the model over `loader` (batch size 1) and compute PSNR / SSIM (/ LPIPS).

    Returns (summary, rows): `summary` has the mean metrics, `rows` one dict per image.
    """
    model.eval()
    for d in (save_pred_dir, vis_dir):
        if d is not None:
            Path(d).mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for idx, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_images is not None and idx >= max_images:
            break
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)

        _sync(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
            pred = model(low)
        _sync(device)
        dt_ms = (time.perf_counter() - t0) * 1000 / low.shape[0]
        pred = pred.float().clamp(0, 1)

        for b in range(pred.shape[0]):
            name = batch["name"][b]
            p_np = tensor_to_numpy(pred[b])
            g_np = tensor_to_numpy(high[b]) if has_gt else None
            row = {"image": name}
            if has_gt:
                row["psnr"], row["ssim"] = calc_psnr(p_np, g_np), calc_ssim(p_np, g_np)
                if lpips_model is not None:
                    row["lpips"] = float(lpips_model(pred[b : b + 1] * 2 - 1, high[b : b + 1] * 2 - 1).mean())
            row["time_ms"] = dt_ms
            rows.append(row)

            if save_pred_dir is not None:
                Image.fromarray(p_np).save(Path(save_pred_dir) / f"{name}.png")
            if vis_dir is not None and len(rows) <= num_vis:
                save_comparison(tensor_to_numpy(low[b]), p_np, g_np, Path(vis_dir) / f"{name}.png")

    keys = [k for k in ("psnr", "ssim", "lpips", "time_ms") if rows and k in rows[0]]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary["num_images"] = len(rows)
    return summary, rows


# ------------------------------------------------- optimizer / scheduler builders
def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    """LR scheduler selected by cfg.lr_schedule.

    "step":   LR *= lr_gamma every lr_step_iters optimizer updates (stepped once per iteration).
    "cosine": linear warm-up for `warmup_epochs`, then cosine decay to `min_lr` (stepped per epoch).
    """
    if cfg.lr_schedule == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_iters, gamma=cfg.lr_gamma)
    if cfg.lr_schedule != "cosine":
        raise ValueError(f"Unknown lr_schedule '{cfg.lr_schedule}' (use 'step' or 'cosine')")
    floor = cfg.min_lr / cfg.lr

    def lr_lambda(epoch: int) -> float:
        if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------- checkpoints
def save_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None,
                    epoch: int = 0, best_psnr: float = 0.0, cfg=None) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "epoch": epoch,
        "best_psnr": best_psnr,
        "cfg": cfg.to_dict() if cfg is not None else None,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt