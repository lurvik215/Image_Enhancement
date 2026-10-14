"""
data_loader_lolv2.py - LOL-v2 dataset (Yang et al., "Sparse Gradient Regularized Deep
Retinex Network for Robust Low-Light Image Enhancement") for the enhancement pipeline.

Expected folders (names are matched case-insensitively, and the dataset may sit one folder
deeper, e.g. LOL-v2/LOL-v2/Real_captured):

    <data_root>/
    ├── Real_captured/  Train/{Low,Normal}   Test/{Low,Normal}
    └── Synthetic/      Train/{Low,Normal}   Test/{Low,Normal}

    real:       Low/low00001.png  <->  Normal/normal00001.png   (matched by the number)
    synthetic:  Low/<name>.png    <->  Normal/<name>.png        (identical names)

LOL-v2 has no validation split, so:
    train = Train pairs minus a random hold-out of `cfg.lolv2_val_fraction`
    val   = that hold-out (same random split every time: it is fixed by `cfg.seed`)
    test  = the official Test folder
Set cfg.lolv2_val_fraction = 0 to validate on the Test folder instead.

cfg.lolv2_subset selects "real", "synthetic" or "both" (pairs of both are combined).

This module only finds and pairs the files; loading, cropping and augmentation are done by
LowLightDataset in data_loader.py, so training / evaluation code is the same for every dataset.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

from data_loader import pair_files

log = logging.getLogger(__name__)

_SUBSET_NAMES = {"real": ("real_captured", "real"), "synthetic": ("synthetic", "syn")}
_LOW_NAMES = ("low", "input", "lq")
_HIGH_NAMES = ("normal", "high", "gt", "hq")


def _child_dir(parent: Path, names) -> Optional[Path]:
    """Sub-folder of `parent` whose lower-cased name is in `names` (or None)."""
    if not parent.is_dir():
        return None
    for child in sorted(parent.iterdir()):
        if child.is_dir() and child.name.lower() in names:
            return child
    return None


def _listing(path: Path) -> str:
    if not path.is_dir():
        return f"(folder does not exist: {path})"
    return ", ".join(sorted(p.name for p in path.iterdir())[:15]) or "(empty)"


def _find_subset_dir(root: Path, subset: str) -> Path:
    names = _SUBSET_NAMES[subset]
    found = _child_dir(root, names)
    if found is None and root.is_dir():  # one level deeper
        for child in sorted(root.iterdir()):
            if child.is_dir():
                found = _child_dir(child, names)
                if found is not None:
                    break
    if found is None:
        raise FileNotFoundError(
            f"LOL-v2 '{subset}' folder (one of {names}) not found in {root}.\n"
            f"Contents: {_listing(root)}\nUse --data_root to point to the LOL-v2 folder."
        )
    return found


def _subset_pairs(cfg, subset: str, folder: str) -> List[Tuple[Path, Path]]:
    """All (low, normal) pairs of one subset ('real'/'synthetic') and split folder ('train'/'test')."""
    subset_dir = _find_subset_dir(cfg.dataset_root, subset)
    split_dir = _child_dir(subset_dir, (folder,))
    if split_dir is None:
        raise FileNotFoundError(f"No '{folder}' folder in {subset_dir}. Contents: {_listing(subset_dir)}")
    low, high = _child_dir(split_dir, _LOW_NAMES), _child_dir(split_dir, _HIGH_NAMES)
    if low is None or high is None:
        raise FileNotFoundError(f"Expected Low/ and Normal/ folders in {split_dir}. Contents: {_listing(split_dir)}")
    return pair_files(low, high, cfg.img_exts)


def _all_pairs(cfg, folder: str) -> List[Tuple[Path, Path]]:
    subsets = ["real", "synthetic"] if cfg.lolv2_subset == "both" else [cfg.lolv2_subset]
    pairs: List[Tuple[Path, Path]] = []
    for s in subsets:
        pairs += _subset_pairs(cfg, s, folder)
    return pairs


def get_lolv2_pairs(cfg, split: str):
    """Return (pairs, has_gt) for split in {'train', 'val', 'test'}. LOL-v2 always has ground truth."""
    if split == "test":
        return _all_pairs(cfg, "test"), True

    frac = cfg.lolv2_val_fraction
    if split == "val" and frac <= 0:
        log.warning("lolv2_val_fraction=0: validating on the Test folder.")
        return _all_pairs(cfg, "test"), True

    pairs = sorted(_all_pairs(cfg, "train"), key=lambda p: str(p[0]))
    if frac <= 0:
        return pairs, True
    order = list(range(len(pairs)))
    random.Random(cfg.seed).shuffle(order)  # same shuffle for the train and the val call
    n_val = min(max(1, round(len(pairs) * frac)), len(pairs) - 1)
    val_idx = set(order[:n_val])
    if split == "val":
        return [p for i, p in enumerate(pairs) if i in val_idx], True
    return [p for i, p in enumerate(pairs) if i not in val_idx], True