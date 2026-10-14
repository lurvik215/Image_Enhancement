"""
data_loader.py - paired low-light / well-lit dataset loading.

The dataset is chosen by cfg.dataset:
    "lolistreet"  <data_root>/<Train|Val|Test>/{low,high}/*   (Test may have no ground truth)
    "lolv2"       see data_loader_lolv2.py (Real_captured / Synthetic, Low / Normal)

Pairs are matched by file name (stem), then by the name with a "low"/"normal"/... prefix
removed (LOL-v2 real: low00001.png <-> normal00001.png). If names still don't match but both
folders contain the same number of images, they are paired in sorted order (with a warning).

Train: random paired crops of cfg.patch_size + flips / 90-degree rotations.
Val/Test: full images, batch size 1 (optionally limited by cfg.eval_max_side).
"""
from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


def list_images(folder: Path, exts) -> List[Path]:
    exts = {e.lower() for e in exts}
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in exts)


_PREFIX = re.compile(r"^(low|normal|high|gt|input|target)[\s_\-]*", re.IGNORECASE)


def _key(stem: str) -> str:
    """File name without a leading low/normal/high/gt prefix: 'low00001' -> '00001'."""
    return _PREFIX.sub("", stem).lower()


def pair_files(low_dir: Path, high_dir: Path, exts) -> List[Tuple[Path, Path]]:
    if not Path(low_dir).is_dir() or not Path(high_dir).is_dir():
        raise FileNotFoundError(f"Expected folders:\n  {low_dir}\n  {high_dir}")
    low, high = list_images(low_dir, exts), list_images(high_dir, exts)
    high_by_stem = {p.stem: p for p in high}
    pairs = [(l, high_by_stem[l.stem]) for l in low if l.stem in high_by_stem]
    if len(pairs) == len(low):
        return pairs
    high_by_key = {_key(p.stem): p for p in high}
    pairs = [(l, high_by_key[_key(l.stem)]) for l in low if _key(l.stem) in high_by_key]
    if len(pairs) == len(low):
        return pairs
    if len(low) == len(high):
        log.warning("File names in %s and %s differ - pairing by sorted order.", low_dir, high_dir)
        return list(zip(low, high))
    log.warning("%d of %d low images have no matching high image; they are skipped.", len(low) - len(pairs), len(low))
    return pairs


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _lolistreet_pairs(cfg, split: str):
    """LoLI-Street: <root>/<Split>/{low,high}. has_gt=False -> the split only holds low-light
    images (e.g. the LoLI-Street Test folder): they can be enhanced and saved, but PSNR/SSIM
    cannot be computed. Allowed for "test" only."""
    root = cfg.split_dir(split)
    low_dir, high_dir = root / cfg.low_dir, root / cfg.high_dir
    if low_dir.is_dir() and high_dir.is_dir():
        return pair_files(low_dir, high_dir, cfg.img_exts), True
    if split != "test":
        raise FileNotFoundError(f"Expected folders:\n  {low_dir}\n  {high_dir}")
    src = low_dir if low_dir.is_dir() else root
    pairs = [(p, p) for p in list_images(src, cfg.img_exts)]
    log.warning("%s has no low/high folders - treating %d images as low-light only "
                "(no ground truth, metrics will be skipped).", root, len(pairs))
    return pairs, False


class LowLightDataset(Dataset):
    def __init__(self, cfg, split: str):
        assert split in ("train", "val", "test")
        self.split = split
        self.train = split == "train"
        self.patch_size = cfg.patch_size
        self.augment = cfg.augment and self.train
        self.max_side = cfg.eval_max_side
        if cfg.dataset == "lolv2":
            from data_loader_lolv2 import get_lolv2_pairs  # imported here to avoid a circular import
            self.pairs, self.has_gt = get_lolv2_pairs(cfg, split)
        elif cfg.dataset == "lolistreet":
            self.pairs, self.has_gt = _lolistreet_pairs(cfg, split)
        else:
            raise ValueError(f"Unknown dataset '{cfg.dataset}' (use 'lolistreet' or 'lolv2')")
        if not self.pairs:
            raise RuntimeError(f"No images found for dataset '{cfg.dataset}', split '{split}'")
        if self.train and cfg.train_max_images and cfg.train_max_images < len(self.pairs):
            self.pairs = random.Random(cfg.seed).sample(self.pairs, cfg.train_max_images)

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, idx: int):
        low_p, high_p = self.pairs[idx]
        low = Image.open(low_p).convert("RGB")
        high = Image.open(high_p).convert("RGB")
        if high.size != low.size:  # make sure the pair is aligned
            high = high.resize(low.size, Image.BICUBIC)
        return low, high, low_p.stem

    def __getitem__(self, idx: int):
        low, high, name = self._load(idx)

        if self.train:
            ps = self.patch_size
            if min(low.size) < ps:  # upscale tiny images so a crop fits
                s = ps / min(low.size)
                size = (max(ps, round(low.width * s)), max(ps, round(low.height * s)))
                low, high = low.resize(size, Image.BICUBIC), high.resize(size, Image.BICUBIC)
            x = random.randint(0, low.width - ps)
            y = random.randint(0, low.height - ps)
            box = (x, y, x + ps, y + ps)
            low, high = low.crop(box), high.crop(box)
        elif self.max_side and max(low.size) > self.max_side:
            s = self.max_side / max(low.size)
            size = (round(low.width * s), round(low.height * s))
            low, high = low.resize(size, Image.BICUBIC), high.resize(size, Image.BICUBIC)

        low, high = _to_tensor(low), _to_tensor(high)

        if self.augment:
            if random.random() < 0.5:
                low, high = low.flip(-1), high.flip(-1)
            if random.random() < 0.5:
                low, high = low.flip(-2), high.flip(-2)
            k = random.randint(0, 3)
            if k:
                low, high = torch.rot90(low, k, (1, 2)), torch.rot90(high, k, (1, 2))

        return {"low": low, "high": high, "name": name}


def build_dataloader(cfg, split: str) -> DataLoader:
    ds = LowLightDataset(cfg, split)
    train = split == "train"
    nw = cfg.num_workers
    return DataLoader(
        ds,
        batch_size=cfg.batch_size if train else 1,
        shuffle=train,
        drop_last=train and len(ds) >= cfg.batch_size,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=nw > 0,
    )


def dataset_summary(cfg) -> dict:
    """{split: {"n": number of images, "has_gt": bool}} (raises if train/val are missing)."""
    out = {}
    for split in ("train", "val", "test"):
        ds = LowLightDataset(cfg, split)
        out[split] = {"n": len(ds), "has_gt": ds.has_gt}
    return out