"""
pipeline.py - orchestrates the complete low-light enhancement project.

    python pipeline.py                      # check data -> train -> evaluate on Test
    python pipeline.py --stage train        # only training
    python pipeline.py --stage test         # only evaluation (needs a trained checkpoint)
    python pipeline.py --stage check        # only verify the dataset and create result folders

Any config.py value can be overridden, e.g.
    python pipeline.py --epochs 100 --batch_size 4 --run_name exp1
"""
from __future__ import annotations

import argparse

from config import add_common_args, cfg_from_args
from data_loader import dataset_summary
from evaluate import evaluate
from train import train
from utils import create_run_dirs, setup_logger


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Low-light enhancement pipeline (MIRNet-v2)"))
    parser.add_argument("--stage", choices=["all", "check", "train", "test"], default="all")
    args = parser.parse_args()
    cfg = cfg_from_args(args)

    dirs = create_run_dirs(cfg)  # results/<run_name>/{checkpoints,logs,predictions,visualizations}
    log = setup_logger("pipeline", dirs["logs"] / "pipeline.log")
    log.info("Task: %s | run: %s | stage: %s", cfg.task, cfg.run_name, args.stage)
    log.info("Results folder: %s", dirs["run"])

    if args.stage in ("all", "check", "train"):
        counts = dataset_summary(cfg)
        for split, info in counts.items():
            log.info("%-5s : %d images | ground truth: %s", split, info["n"], "yes" if info["has_gt"] else "NO (enhancement only)")

    if args.stage == "check":
        return

    if args.stage in ("all", "train"):
        result = train(cfg)
        log.info("Training done. Best val PSNR %.2f dB", result["best_psnr"])
        cfg.checkpoint = result["best_checkpoint"]

    if args.stage in ("all", "test"):
        summary = evaluate(cfg, split="test")
        if "psnr" not in summary:  # Test has no ground truth -> report metrics on Val instead
            log.info("Test set has no ground truth: enhanced images saved. Computing metrics on Val.")
            summary = evaluate(cfg, split="val")
        log.info("Pipeline finished. PSNR %.2f dB | SSIM %.4f", summary["psnr"], summary["ssim"])


if __name__ == "__main__":
    main()