"""Test-time augmentation for ResNet1D CV classifier.

Applies the same train-time augmentations (Gaussian noise + baseline drift)
at inference, averages softmax probs across n_aug passes. Scan-rate channels
(index 2 of each 3-channel block) are never augmented.
"""

from __future__ import annotations

import numpy as np
import torch


def tta_predict(
    model: torch.nn.Module,
    X: torch.Tensor,
    n_aug: int = 20,
    device: str | torch.device = "cuda",
    noise_sigma: float = 0.1,
    drift_max: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Run test-time augmentation inference.

    Args:
        model: Classifier returning logits of shape (N, num_classes).
        X: Input tensor of shape (N, 18, 250). 6 scan rates × 3 channels each
           (fwd current, rev current, scan rate). Must be pre-normalized.
        n_aug: Number of augmentation passes to average.
        device: Device to run on.
        noise_sigma: Gaussian noise std-dev applied to current channels.
        drift_max: Max amplitude of random linear ramp applied to current channels.

    Returns:
        (probs_mean, preds) where probs_mean has shape (N, num_classes) and
        preds has shape (N,) as int64 argmax.
    """
    model.eval()
    X = X.to(device)
    N, C, L = X.shape

    # Current-channel mask: channels 0,1,3,4,6,7,9,10,12,13,15,16 are currents;
    # channels 2,5,8,11,14,17 are scan-rate values (never augmented).
    is_current = torch.ones(C, dtype=torch.bool, device=device)
    is_current[2::3] = False
    is_current_col = is_current.view(1, C, 1)  # broadcast (1, C, 1)

    probs_sum = torch.zeros(N, dtype=torch.float32, device=device)
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for _ in range(n_aug):
            x_aug = X.clone()
            if noise_sigma > 0:
                noise = torch.randn_like(x_aug) * noise_sigma
                noise = noise * is_current_col
                x_aug = x_aug + noise
            if drift_max > 0:
                a = (torch.rand(N, C, 1, device=device) * 2 - 1) * drift_max
                b = (torch.rand(N, C, 1, device=device) * 2 - 1) * drift_max
                t = torch.linspace(0.0, 1.0, L, device=device).view(1, 1, L)
                ramp = a * (1 - t) + b * t
                ramp = ramp * is_current_col
                x_aug = x_aug + ramp
            logits = model(x_aug)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.detach().cpu().numpy())

    probs_mean = np.mean(np.stack(all_probs, axis=0), axis=0)
    preds = np.argmax(probs_mean, axis=-1)
    return probs_mean, preds
