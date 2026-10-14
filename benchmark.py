"""
benchmark.py - finds out WHY training is slow: image loading or the GPU?

    python benchmark.py --dataset lolistreet --num_workers 4

It measures, in a few minutes and without training anything:
  1. the size of your training images and how long one sample takes to load,
  2. how fast the DataLoader delivers batches (no model involved),
  3. how fast the GPU runs a training step on random data for several batch sizes,
     together with the peak GPU memory.
"""
from __future__ import annotations

import argparse
import random
import time

import torch
from PIL import Image

from config import add_common_args, cfg_from_args
from data_loader import LowLightDataset, build_dataloader
from losses import TotalLoss
from model_arch import build_model
from utils import get_device


def inspect_data(cfg):
    ds = LowLightDataset(cfg, "train")
    print(f"\n[1] Training images: {len(ds)} pairs")
    if "onedrive" in str(cfg.dataset_root).lower():
        print("    WARNING: the dataset is inside OneDrive. OneDrive may download every file on first read.")
    for low, _ in ds.pairs[:3]:
        with Image.open(low) as im:
            print(f"    {low.name}: {im.size[0]}x{im.size[1]} px, {low.stat().st_size / 1e6:.2f} MB on disk")
    idx = random.Random(0).sample(range(len(ds)), min(8, len(ds)))
    t0 = time.perf_counter()
    for i in idx:
        ds[i]
    per_sample = (time.perf_counter() - t0) / len(idx)
    print(f"    loading + cropping ONE sample (1 process): {per_sample * 1000:.0f} ms")
    return per_sample


def time_loader(cfg, n_batches: int = 15):
    loader = build_dataloader(cfg, "train")
    it = iter(loader)
    next(it)  # the first batch includes worker start-up
    t0, n = time.perf_counter(), 0
    for _ in range(n_batches):
        try:
            next(it)
        except StopIteration:
            break
        n += 1
    per_batch = (time.perf_counter() - t0) / max(n, 1)
    print(f"\n[2] DataLoader alone (batch {cfg.batch_size}, {cfg.num_workers} workers): {per_batch:.2f} s per batch")
    print(f"    -> an epoch of {len(loader)} batches would need about {per_batch * len(loader) / 3600:.1f} h just for loading")
    return per_batch


def time_gpu(cfg, device, batch_size: int, amp: bool, iters: int = 8):
    model = build_model(cfg).to(device).train()
    crit = TotalLoss(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    ps = cfg.patch_size
    x = torch.rand(batch_size, 3, ps, ps, device=device)
    y = torch.rand(batch_size, 3, ps, ps, device=device)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
            pred = model(x)
        loss, _ = crit(pred, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    t = sum(times[2:]) / len(times[2:])  # skip warm-up iterations
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, crit, opt, x, y, pred, loss
    torch.cuda.empty_cache()
    return t, peak


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Find the training bottleneck"))
    cfg = cfg_from_args(parser.parse_args())
    device = get_device(cfg)
    amp = cfg.amp and device.type == "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)} | total memory {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB "
          f"| patch {cfg.patch_size} | AMP {amp}")

    inspect_data(cfg)
    per_batch = time_loader(cfg)

    print("\n[3] GPU training step on random data (no image loading):")
    iters_per_epoch = 30000 // 12
    results = {}
    for bs in (12, 8, 6, 4):
        try:
            t, peak = time_gpu(cfg, device, bs, amp)
            results[bs] = t
            print(f"    batch {bs:2d}: {t:5.2f} s/step = {t / bs:.3f} s/image | peak memory {peak:5.1f} GB")
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"    batch {bs:2d}: CUDA out of memory")
            torch.cuda.empty_cache()
    if 12 in results:
        print(f"\n    -> one epoch (2,500 steps of batch 12) would take about {results[12] * iters_per_epoch / 3600:.1f} h on the GPU alone")

    print("\nHOW TO READ THIS")
    print(" * If [2] is much slower than a GPU step in [3], image loading is the bottleneck.")
    print(" * If s/image is clearly worse at batch 12 than at batch 6 or 4, the GPU is probably running out of")
    print("   memory and spilling into system RAM. Use a smaller batch with --accum_steps.")


if __name__ == "__main__":
    main()