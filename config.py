"""
config.py - configuration for the LOW-LIGHT IMAGE ENHANCEMENT task.

Every task in the pipeline gets its own config. This one is for MIRNet-v2 trained
on the LoLI-Street dataset. Every other module (data_loader, model_arch, losses,
utils, train, evaluate, pipeline) reads its settings from here, and any value can be
overridden from the command line (see add_common_args / cfg_from_args).

Two datasets are supported (select with `dataset`, or --dataset on the command line):

  * "lolistreet"  (default)  paired low/high images, plus a Test folder without ground truth
  * "lolv2"       LOL-v2 (Real_captured and/or Synthetic), see data_loader_lolv2.py

Expected LoLI-Street layout (same as the folder you showed):

    DATASETS/
    └── LoLI-Street Dataset/
        ├── Train/  ├── high/   (well-lit ground truth)
        │           └── low/    (low-light input)
        ├── Val/    ├── high/
        │           └── low/
        ├── Test/   ├── high/
        │           └── low/
        └── YOLO Annotations/   (not used by this task)

Expected LOL-v2 layout (folder names are matched case-insensitively):

    DATASETS/
    └── LOL-v2/
        ├── Real_captured/  ├── Train/  ├── Low/     Normal/
        │                   └── Test/   ├── Low/     Normal/
        └── Synthetic/      ├── Train/  ├── Low/     Normal/
                            └── Test/   ├── Low/     Normal/

LOL-v2 has no validation folder: a random `lolv2_val_fraction` of the training pairs is held
out for validation (fixed by `seed`); the official Test folder is used for the final test.

Results layout (created automatically):

    enhancement_results/<run_name>/
    ├── checkpoints/        best.pth, last.pth, epoch_XXX.pth
    ├── logs/               train.log, evaluate.log, pipeline.log
    ├── predictions/        enhanced images (PNG) produced at test time
    ├── visualizations/     low | enhanced | ground-truth comparison strips
    ├── config.json         the exact config used for the run
    ├── train_metrics.csv   per-epoch loss / PSNR / SSIM
    ├── test_metrics.csv    per-image PSNR / SSIM / LPIPS / time
    └── test_summary.csv    mean (and std) of the test metrics
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class LowLightConfig:
    # ------------------------------------------------------------------ task
    task: str = "lowlight_enhancement"
    dataset: str = "lolistreet"      # "lolistreet" | "lolv2"
    run_name: Optional[str] = None   # default: mirnetv2_<dataset>[_<lolv2_subset>]
    seed: int = 42
    device: str = "cuda"  # falls back to cpu automatically if CUDA is missing

    # ----------------------------------------------------------------- paths
    data_root: Optional[str] = None  # default depends on `dataset` (see dataset_root)
    results_root: str = str(PROJECT_ROOT / "enhancement_results")
    train_split: str = "Train"
    val_split: str = "Val"
    test_split: str = "Test"
    low_dir: str = "low"
    high_dir: str = "high"
    img_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # ------------------------------------------------------------------ LOL-v2
    lolv2_subset: str = "real"          # "real" | "synthetic" | "both"
    lolv2_val_fraction: float = 0.1     # share of Train held out for validation (0 -> use Test)

    # ----------------------------------------------------------- model (MIRNet-v2)
    inp_channels: int = 3
    out_channels: int = 3
    # Official MIRNet-v2 defaults. The number of resolution levels (3) and passes per block (2)
    # are fixed by the architecture. n_feat must be a multiple of 16 (e.g. 48, 64, 80).
    n_feat: int = 80          # base number of channels
    chan_factor: float = 1.5  # channel growth per resolution level
    n_rrg: int = 4            # recursive residual groups
    n_mrb: int = 2            # multi-scale residual blocks per RRG
    bias: bool = False

    # ------------------------------------------------------------------ data
    patch_size: int = 256     # random crop size used for training
    # Effective batch = batch_size * accum_steps = 12 (the LoLI-Street paper's batch size). A batch of
    # 12 in one go needs ~31 GB at 256px and overflowed a 20 GB GPU (16x slower per image), so it is
    # split into 4 x 3. Measured peak memory for batch 4: ~10.6 GB. Lower batch_size (and raise
    # accum_steps) if your GPU has less memory.
    batch_size: int = 4
    accum_steps: int = 3      # gradient accumulation
    num_workers: int = 4
    augment: bool = True      # flips + 90-degree rotations
    # Val/Test images are processed at full resolution (batch size 1).
    # Set to e.g. 1024 to shrink the longer side if full-size images don't fit in memory.
    eval_max_side: Optional[int] = None
    # Use only the first N validation images each epoch (None = all) to speed up training.
    val_max_images: Optional[int] = None
    # Quick-test helpers: use a random subset of N training images / only the first N
    # images when running evaluate.py (None = use everything).
    train_max_images: Optional[int] = None
    eval_max_images: Optional[int] = None

    # ---------------------------------------------------------- optimisation
    # Default: LR 2e-4 with 3 warm-up epochs, then cosine decay to min_lr (1e-6) at the LAST epoch,
    # so `epochs` must be fixed in advance (stopping early leaves the LR high).
    # The schedule published with LoLI-Street is still available:
    #     --lr 1e-4 --lr_schedule step --lr_step_iters 5000 --lr_gamma 0.8   (x0.8 every 5,000 iterations)
    epochs: int = 40
    lr: float = 2e-4
    lr_schedule: str = "cosine" # "cosine" (warm-up + cosine decay to min_lr) | "step" (paper)
    lr_step_iters: int = 5000   # step schedule: decay every N iterations (one iteration = one optimizer update)
    lr_gamma: float = 0.8       # step schedule: multiplicative decay factor
    min_lr: float = 1e-6        # cosine schedule only
    weight_decay: float = 1e-8
    betas: Tuple[float, float] = (0.9, 0.999)
    warmup_epochs: int = 3      # cosine schedule only
    grad_clip: Optional[float] = 1.0
    amp: bool = True          # mixed precision (CUDA only). Use --no_amp on Pascal GPUs (e.g. P5000), where it is slow

    # ------------------------------------------------------------------ loss
    # Total loss = sum(weight * term). A weight of 0 disables that term.
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"l1": 1.0, "lpips": 0.1, "fft": 0.05}
    )
    lpips_net: str = "alex"   # "alex" | "vgg" | "squeeze"

    # --------------------------------------------------------------- logging
    val_every: int = 1        # validate every N epochs
    save_every: int = 10      # keep a checkpoint every N epochs (best/last always kept)
    num_vis: int = 4          # comparison images saved per validation / test run
    resume: Optional[str] = None      # path to a checkpoint, or "auto" for last.pth
    # When resuming, the learning rate and its schedule are restored from the checkpoint, so
    # --lr / --lr_gamma / --lr_step_iters are ignored. Set resume_lr to override: the schedule
    # (cfg.lr_schedule, lr_step_iters, lr_gamma, min_lr) then restarts from this value at the resume epoch.
    resume_lr: Optional[float] = None
    checkpoint: Optional[str] = None  # checkpoint used by evaluate.py (default: best.pth)

    # ------------------------------------------------------------------ helpers
    def __post_init__(self):
        if self.run_name is None:
            self.run_name = self.default_run_name()

    def default_run_name(self) -> str:
        return f"mirnetv2_{self.dataset}" + (f"_{self.lolv2_subset}" if self.dataset == "lolv2" else "")

    @property
    def dataset_root(self) -> Path:
        if self.data_root:
            return Path(self.data_root)
        name = "LOL-v2" if self.dataset == "lolv2" else "LoLI-Street Dataset"
        return PROJECT_ROOT / "DATASETS" / name

    @property
    def run_dir(self) -> Path:
        return Path(self.results_root) / self.run_name

    def split_dir(self, split: str) -> Path:
        name = {"train": self.train_split, "val": self.val_split, "test": self.test_split}[split]
        return self.dataset_root / name

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def update(self, **kwargs) -> "LowLightConfig":
        for k, v in kwargs.items():
            if v is None:
                continue
            if not hasattr(self, k):
                raise AttributeError(f"Unknown config field: {k}")
            setattr(self, k, v)
        return self


# One entry per task. Add more tasks here if the project grows.
TASK_CONFIGS = {"lowlight_enhancement": LowLightConfig}


def get_config(task: str = "lowlight_enhancement") -> LowLightConfig:
    """Return a fresh config object for the given task."""
    if task not in TASK_CONFIGS:
        raise KeyError(f"Unknown task '{task}'. Available: {list(TASK_CONFIGS)}")
    return TASK_CONFIGS[task]()


# --------------------------------------------------------------------------- CLI
_OVERRIDABLE = [
    "dataset", "lolv2_subset", "lolv2_val_fraction", "accum_steps",
    "lr_schedule", "lr_step_iters", "lr_gamma",
    "run_name", "seed", "device", "data_root", "results_root", "n_feat", "n_rrg",
    "n_mrb", "patch_size", "batch_size", "num_workers", "eval_max_side",
    "val_max_images", "train_max_images", "eval_max_images", "epochs", "lr", "val_every", "save_every", "num_vis",
    "resume", "resume_lr", "checkpoint", "lpips_net",
]


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Command-line overrides shared by train.py, evaluate.py and pipeline.py."""
    p.add_argument("--task", default="lowlight_enhancement")
    p.add_argument("--dataset", choices=["lolistreet", "lolv2"], help="which dataset to use")
    p.add_argument("--lolv2_subset", choices=["real", "synthetic", "both"], help="LOL-v2 subset")
    p.add_argument("--lolv2_val_fraction", type=float, help="LOL-v2: share of Train held out for validation")
    p.add_argument("--accum_steps", type=int, help="gradient accumulation steps")
    p.add_argument("--run_name", type=str)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", type=str, help="cuda | cpu")
    p.add_argument("--data_root", type=str)
    p.add_argument("--results_root", type=str)
    p.add_argument("--n_feat", type=int)
    p.add_argument("--n_rrg", type=int)
    p.add_argument("--n_mrb", type=int)
    p.add_argument("--patch_size", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--num_workers", type=int)
    p.add_argument("--eval_max_side", type=int)
    p.add_argument("--val_max_images", type=int, help="validate on the first N val images each epoch")
    p.add_argument("--train_max_images", type=int, help="train on a random subset of N images (quick tests)")
    p.add_argument("--eval_max_images", type=int, help="evaluate.py: only the first N images")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--lr_schedule", choices=["step", "cosine"], help="step = x gamma every lr_step_iters iterations")
    p.add_argument("--lr_step_iters", type=int, help="step schedule: decay the LR every N iterations")
    p.add_argument("--lr_gamma", type=float, help="step schedule: decay factor")
    p.add_argument("--val_every", type=int)
    p.add_argument("--save_every", type=int)
    p.add_argument("--num_vis", type=int)
    p.add_argument("--resume", type=str, help="checkpoint path or 'auto'")
    p.add_argument("--resume_lr", type=float, help="when resuming: override the learning rate and restart the schedule from it")
    p.add_argument("--checkpoint", type=str, help="checkpoint to evaluate (default best.pth)")
    p.add_argument("--lpips_net", type=str)
    p.add_argument("--amp", action="store_true", help="enable mixed precision (CUDA)")
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    p.add_argument("--w_l1", type=float, help="weight of the L1 loss")
    p.add_argument("--w_lpips", type=float, help="weight of the LPIPS loss")
    p.add_argument("--w_fft", type=float, help="weight of the FFT L1 loss")
    return p


def cfg_from_args(args: argparse.Namespace) -> LowLightConfig:
    cfg = get_config(getattr(args, "task", "lowlight_enhancement"))
    cfg.update(**{k: getattr(args, k, None) for k in _OVERRIDABLE})
    if getattr(args, "amp", False):
        cfg.amp = True
    if getattr(args, "no_amp", False):
        cfg.amp = False
    if getattr(args, "run_name", None) is None:  # name the run after the (possibly overridden) dataset
        cfg.run_name = cfg.default_run_name()
    for key in ("l1", "lpips", "fft"):
        v = getattr(args, f"w_{key}", None)
        if v is not None:
            cfg.loss_weights[key] = v
    return cfg