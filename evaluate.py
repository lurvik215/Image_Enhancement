"""
evaluate.py - runs a trained MIRNet-v2 over the full test set and reports metrics.

    python evaluate.py                                  # uses checkpoints/best.pth
    python evaluate.py --checkpoint path/to/model.pth

Outputs go to  <results_root>/<run_name>/ :
    predictions/test/*.png          enhanced images
    visualizations/test/*.png       low | enhanced | ground-truth strips
    <split>_metrics.csv             per-image PSNR / SSIM / LPIPS / time (time only if no GT)
    <split>_summary.csv             mean and std of every metric
    logs/evaluate.log
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import LowLightConfig, add_common_args, cfg_from_args
from data_loader import build_dataloader
from model_arch import build_model
from utils import (create_run_dirs, evaluate_loop, get_device, load_checkpoint, setup_logger, write_csv)


def evaluate(cfg: LowLightConfig, split: str = "test") -> dict:
    dirs = create_run_dirs(cfg)
    log = setup_logger("evaluate", dirs["logs"] / "evaluate.log")
    device = get_device(cfg)
    use_amp = cfg.amp and device.type == "cuda"

    ckpt_path = Path(cfg.checkpoint) if cfg.checkpoint else dirs["checkpoints"] / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first or pass --checkpoint.")

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(ckpt_path, model, map_location=device)
    log.info("Loaded %s (epoch %s)", ckpt_path, ckpt.get("epoch"))

    loader = build_dataloader(cfg, split)
    has_gt = loader.dataset.has_gt
    log.info("Evaluating on %d %s images (ground truth: %s)", len(loader.dataset), split, has_gt)
    if not has_gt:
        log.warning("No ground truth for '%s': enhanced images are saved, but PSNR/SSIM/LPIPS "
                    "are skipped. Use --split val to get metrics.", split)

    # LPIPS is reported as an extra metric when the package/weights are available.
    lpips_model = None
    try:
        if not has_gt:
            raise RuntimeError('no ground truth')
        import lpips
        lpips_model = lpips.LPIPS(net=cfg.lpips_net, verbose=False).to(device).eval()
    except Exception as e:  # noqa: BLE001
        log.warning("LPIPS metric skipped (%s).", e)

    summary, rows = evaluate_loop(
        model, loader, device, amp=use_amp, max_images=cfg.eval_max_images,
        save_pred_dir=dirs["predictions"] / split,
        vis_dir=dirs["visualizations"] / split, num_vis=cfg.num_vis,
        lpips_model=lpips_model, desc=f"{split}", has_gt=has_gt,
    )

    # per-image CSV (+ a MEAN row at the end)
    mean_row = {"image": "MEAN", **{k: round(v, 6) for k, v in summary.items() if k != "num_images"}}
    write_csv(dirs["run"] / f"{split}_metrics.csv", rows + [mean_row])

    # summary CSV
    metric_keys = [k for k in ("psnr", "ssim", "lpips", "time_ms") if k in summary]
    summary_rows = [
        {"metric": k,
         "mean": round(float(np.mean([r[k] for r in rows])), 6),
         "std": round(float(np.std([r[k] for r in rows])), 6),
         "min": round(float(np.min([r[k] for r in rows])), 6),
         "max": round(float(np.max([r[k] for r in rows])), 6),
         "num_images": len(rows)}
        for k in metric_keys
    ]
    write_csv(dirs["run"] / f"{split}_summary.csv", summary_rows)

    log.info("%s RESULTS | " % split.upper() + " | ".join(f"{k}: {summary[k]:.4f}" for k in metric_keys))
    log.info("Saved metrics to %s", dirs["run"])
    return summary


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser(description="Evaluate MIRNet-v2"))
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="which split to run (Test has no ground truth -> no metrics)")
    args = parser.parse_args()
    evaluate(cfg_from_args(args), split=args.split)