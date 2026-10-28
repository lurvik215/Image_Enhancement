"""
losses.py - loss functions for low-light enhancement.

    L1PixelLoss   pixel-wise L1 reconstruction
    LPIPSLoss     learned perceptual image patch similarity (needs `pip install lpips`)
    FFTL1Loss     L1 between the 2-D FFTs of prediction and target
    TotalLoss     weighted sum of the above; weights come from cfg.loss_weights

All images are expected in [0, 1]. Losses are always computed in float32, even when
the model runs under mixed precision.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1PixelLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class FFTL1Loss(nn.Module):
    """L1 distance in the frequency domain (real + imaginary parts)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=pred.device.type, enabled=False):
            p = torch.fft.rfft2(pred.float(), dim=(-2, -1), norm="ortho")
            t = torch.fft.rfft2(target.float(), dim=(-2, -1), norm="ortho")
            return F.l1_loss(torch.view_as_real(p), torch.view_as_real(t))


class LPIPSLoss(nn.Module):
    """Perceptual loss. Inputs in [0,1] are mapped to [-1,1] as LPIPS expects.
    The first use downloads the pretrained backbone weights (internet needed once)."""

    def __init__(self, net: str = "alex"):
        super().__init__()
        try:
            import lpips
        except ImportError as e:  # pragma: no cover
            raise ImportError("LPIPS loss needs the `lpips` package: pip install lpips") from e
        self.net = lpips.LPIPS(net=net, verbose=False)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):  # keep the frozen network in eval mode
        return super().train(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=pred.device.type, enabled=False):
            return self.net(pred.float() * 2 - 1, target.float() * 2 - 1).mean()


class TotalLoss(nn.Module):
    """total = w_l1 * L1 + w_lpips * LPIPS + w_fft * FFT-L1   (weights from the task cfg)

    Terms with weight 0 are not built, so LPIPS is not required if you switch it off.
    Returns (total, {"total":..., "l1":..., "lpips":..., "fft":...}) for logging.
    """

    def __init__(self, cfg):
        super().__init__()
        self.weights: Dict[str, float] = dict(cfg.loss_weights)
        self.terms = nn.ModuleDict()
        if self.weights.get("l1", 0) > 0:
            self.terms["l1"] = L1PixelLoss()
        if self.weights.get("lpips", 0) > 0:
            self.terms["lpips"] = LPIPSLoss(cfg.lpips_net)
        if self.weights.get("fft", 0) > 0:
            self.terms["fft"] = FFTL1Loss()
        if len(self.terms) == 0:
            raise ValueError("All loss weights are 0 - nothing to optimise.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred, target = pred.float(), target.float()
        total = pred.new_zeros(())
        parts: Dict[str, float] = {}
        for name, fn in self.terms.items():
            value = fn(pred, target)
            total = total + self.weights[name] * value
            parts[name] = float(value.detach())
        parts["total"] = float(total.detach())
        return total, parts