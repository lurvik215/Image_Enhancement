"""
baseline_identity.py - scores the UNENHANCED low-light images against their ground truth.

    python baseline_identity.py --dataset lolistreet --split val --run_name street_100ep
    python baseline_identity.py --dataset lolistreet --split val --eval_max_images 500   # quick check

This is the "do nothing" reference row of your results table: a trained model should clearly
beat it. It uses the same PSNR / SSIM functions as evaluate.py, so the numbers are comparable.
If the file names start with light_ / moderate_ / dense_, the results are also grouped by
that lighting level.
"""
from __future__ import annotations

import argparse
import collections

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import add_common_args, cfg_from_args
from data_loader import LowLightDataset
from utils import calc_psnr, calc_ssim, write_csv

LEVELS = ("light", "moderate", "dense")


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Score the unenhanced inputs"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    args = parser.parse_args()
    cfg = cfg_from_args(args)

    ds = LowLightDataset(cfg, args.split)
    if not ds.has_gt:
        raise SystemExit(f"The '{args.split}' split has no ground truth, so nothing can be scored.")
    pairs = ds.pairs[: cfg.eval_max_images] if cfg.eval_max_images else ds.pairs

    rows = []
    for low_p, high_p in tqdm(pairs, desc="scoring inputs"):
        low = np.asarray(Image.open(low_p).convert("RGB"))
        high_im = Image.open(high_p).convert("RGB")
        if high_im.size != (low.shape[1], low.shape[0]):
            high_im = high_im.resize((low.shape[1], low.shape[0]), Image.BICUBIC)
        high = np.asarray(high_im)
        rows.append({"image": low_p.stem, "psnr": calc_psnr(low, high), "ssim": calc_ssim(low, high)})

    def summarise(subset):
        return float(np.mean([r["psnr"] for r in subset])), float(np.mean([r["ssim"] for r in subset]))

    print(f"\nNo-enhancement baseline on '{args.split}' ({len(rows)} images)")
    psnr, ssim = summarise(rows)
    print(f"  all        : PSNR {psnr:.2f} dB | SSIM {ssim:.4f}")

    groups = collections.defaultdict(list)
    for r in rows:
        prefix = r["image"].split("_")[0].lower()
        if prefix in LEVELS:
            groups[prefix].append(r)
    for level in LEVELS:
        if groups[level]:
            p, s = summarise(groups[level])
            print(f"  {level:<10s} : PSNR {p:.2f} dB | SSIM {s:.4f}  ({len(groups[level])} images)")

    out_dir = cfg.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"no_enhancement_{args.split}.csv"
    write_csv(out, rows + [{"image": "MEAN", "psnr": round(psnr, 6), "ssim": round(ssim, 6)}])
    print(f"\nPer-image scores saved to {out}")


if __name__ == "__main__":
    main()