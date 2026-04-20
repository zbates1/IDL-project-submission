"""CV dataset class for mechanism classification from cyclic voltammograms.

Loads simulated ``.npz`` data produced by :mod:`src.data.cv_simulator` and
provides PyTorch ``Dataset`` / ``DataLoader`` interfaces with optional
domain-randomisation augmentations (Gaussian noise, baseline drift).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


class CVDataset(Dataset):
    """PyTorch dataset for cyclic voltammogram mechanism classification.

    Each sample is a tensor of shape ``(C, L)`` where:

    - ``C = num_scan_rates * 3`` (3 channels per scan rate: forward current,
      reverse current, scan-rate value).
    - ``L = n_points`` (discretised potential axis, default 500).

    The 6 scan-rate blocks are **concatenated along the channel dimension**
    so the ResNet receives a ``(18, 500)`` input by default.

    Args:
        data_path: Path to an ``.npz`` file or directory containing one.
        split: ``"train"``, ``"val"``, or ``"test"``.
        train_frac: Fraction for training split.
        val_frac: Fraction for validation split.
        noise_sigma: Std-dev of Gaussian noise augmentation (relative to
            signal max).  ``0.0`` disables noise.
        baseline_drift: Max amplitude of random linear baseline drift.
        scale_jitter: Max fractional amplitude scaling per channel
            (e.g. 0.15 → multiply by uniform(0.85, 1.15)).
        time_shift_max: Max circular shift along the potential axis (int
            number of points).
        channel_dropout_prob: Probability of zeroing out an entire
            3-channel scan-rate block per sample.
        sr_shuffle_prob: Probability of randomly permuting the order of
            the 6 scan-rate blocks.
        seed: Seed for reproducible splitting.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        noise_sigma: float = 0.0,
        baseline_drift: float = 0.0,
        scale_jitter: float = 0.0,
        time_shift_max: int = 0,
        channel_dropout_prob: float = 0.0,
        sr_shuffle_prob: float = 0.0,
        seed: int = 42,
    ) -> None:
        data_path = Path(data_path)
        if data_path.is_dir():
            npz_files = list(data_path.glob("*.npz"))
            if not npz_files:
                raise FileNotFoundError(f"No .npz files found in {data_path}")
            data_path = npz_files[0]

        data = np.load(data_path)
        signals = data["signals"]
        labels = data["labels"]

        # Support both old format (N, n_sr, 3, L) and new format (N, n_sr, L)
        if signals.ndim == 4:
            # Old generated format: already split into (fwd, rev, v_n) channels
            n, n_sr, c, length = signals.shape
            self._signals = signals.reshape(n, n_sr * c, length).astype(np.float32)
        else:
            # New format: (N, n_sr, L) combined CV trace.
            # Reconstruct the paper's 3-channel layout per scan rate:
            #   channel 0: i_for  — forward (anodic) half-sweep  [0 : L//2]
            #   channel 1: i_rev  — reverse (cathodic) half-sweep [L//2 : L]
            #   channel 2: v_n    — scan rate value broadcast across the half-sweep
            # Output shape: (N, n_sr * 3, L // 2)  →  e.g. (N, 18, 250)
            n, n_sr, length = signals.shape
            mid = length // 2

            # Load per-sample scan rates if present (shape N, n_sr)
            sr_values = data["scan_rates"].astype(np.float32) if "scan_rates" in data else None

            out = np.empty((n, n_sr * 3, mid), dtype=np.float32)
            for sr_i in range(n_sr):
                out[:, sr_i * 3,     :] = signals[:, sr_i, :mid]    # i_for
                out[:, sr_i * 3 + 1, :] = signals[:, sr_i, mid:]    # i_rev
                if sr_values is not None:
                    # Broadcast scalar scan rate to (N, mid)
                    out[:, sr_i * 3 + 2, :] = sr_values[:, sr_i:sr_i + 1]
                else:
                    out[:, sr_i * 3 + 2, :] = 0.0
            self._signals = out  # (N, n_sr*3, mid)

        # Encode string labels to integers (supports both int and string label arrays)
        if labels.dtype.kind in ("U", "S", "O"):
            classes = sorted(set(labels.tolist()))
            label_to_idx = {lbl: i for i, lbl in enumerate(classes)}
            self._labels = np.array(
                [label_to_idx[lbl] for lbl in labels], dtype=np.int64
            )
            self.classes_ = classes
        else:
            self._labels = labels.astype(np.int64)
            self.classes_ = None

        # Deterministic stratified split
        rng = np.random.default_rng(seed)
        indices = np.arange(n)
        rng.shuffle(indices)

        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        if split == "train":
            self._indices = indices[:n_train]
        elif split == "val":
            self._indices = indices[n_train:n_train + n_val]
        elif split == "test":
            self._indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: '{split}'")

        # Per-channel normalization (computed on training split)
        # Current channels (i_fwd, i_rev) get z-score normalization.
        # Scan-rate channels (every 3rd) get min-max to [0, 1] since they
        # have near-zero variance within each channel but vary across channels.
        train_indices = indices[:n_train]
        train_data = self._signals[train_indices]
        C = self._signals.shape[1]

        self._channel_mean = np.zeros(C, dtype=np.float32)
        self._channel_std = np.ones(C, dtype=np.float32)

        for ch in range(C):
            if ch % 3 == 2:  # scan-rate channel
                ch_min = train_data[:, ch, :].min()
                ch_max = train_data[:, ch, :].max()
                rng = ch_max - ch_min if ch_max != ch_min else 1.0
                self._channel_mean[ch] = ch_min
                self._channel_std[ch] = rng  # maps to [0, 1]
            else:  # current channel
                self._channel_mean[ch] = train_data[:, ch, :].mean()
                self._channel_std[ch] = train_data[:, ch, :].std() + 1e-8

        # Apply normalization to ALL samples
        mean_bc = self._channel_mean[np.newaxis, :, np.newaxis]
        std_bc = self._channel_std[np.newaxis, :, np.newaxis]
        self._signals = (self._signals - mean_bc) / std_bc

        self.channel_mean = self._channel_mean  # (C,) — exposed for real data
        self.channel_std = self._channel_std     # (C,) — exposed for real data

        self.noise_sigma = noise_sigma
        self.baseline_drift = baseline_drift
        self.scale_jitter = scale_jitter
        self.time_shift_max = time_shift_max
        self.channel_dropout_prob = channel_dropout_prob
        self.sr_shuffle_prob = sr_shuffle_prob
        self.split = split
        self._rng = np.random.default_rng(seed + hash(split) % 2**31)

        logger.info(
            "CVDataset(%s): %d samples, shape=%s, noise=%.2f, drift=%.2f",
            split, len(self._indices), self._signals.shape[1:],
            noise_sigma, baseline_drift,
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self._indices[idx]
        x = self._signals[real_idx].copy()  # (C, L)
        y = int(self._labels[real_idx])

        # Augmentations (only during training)
        if self.split == "train":
            C, L = x.shape
            # Mask for current channels (skip scan-rate channels at indices 2,5,8,...)
            current_mask = np.array([ch % 3 != 2 for ch in range(C)])

            # 1. Scale jitter — per-channel amplitude scaling (current only)
            if self.scale_jitter > 0:
                scales = self._rng.uniform(
                    1.0 - self.scale_jitter, 1.0 + self.scale_jitter, (C, 1)
                ).astype(np.float32)
                scales[~current_mask, :] = 1.0
                x *= scales

            # 2. Gaussian noise (current channels only)
            if self.noise_sigma > 0:
                noise = self._rng.normal(0, self.noise_sigma, x.shape).astype(
                    np.float32
                )
                noise[~current_mask, :] = 0.0
                x += noise

            # 3. Baseline drift — random linear ramp (current channels only)
            if self.baseline_drift > 0:
                drift = np.linspace(
                    self._rng.uniform(-self.baseline_drift, self.baseline_drift),
                    self._rng.uniform(-self.baseline_drift, self.baseline_drift),
                    L,
                ).astype(np.float32)
                x[current_mask, :] += drift[np.newaxis, :]

            # 4. Time shift — circular shift along potential axis
            if self.time_shift_max > 0:
                shift = int(self._rng.integers(-self.time_shift_max, self.time_shift_max + 1))
                if shift != 0:
                    x = np.roll(x, shift, axis=-1)

            # 5. Channel dropout — zero out entire SR blocks (3 channels each)
            if self.channel_dropout_prob > 0:
                n_sr = C // 3
                for sr_i in range(n_sr):
                    if self._rng.random() < self.channel_dropout_prob:
                        x[sr_i * 3 : sr_i * 3 + 3, :] = 0.0

            # 6. Scan-rate shuffle — permute SR block order
            if self.sr_shuffle_prob > 0 and self._rng.random() < self.sr_shuffle_prob:
                n_sr = C // 3
                perm = self._rng.permutation(n_sr)
                x_shuffled = np.empty_like(x)
                for new_i, old_i in enumerate(perm):
                    x_shuffled[new_i * 3 : new_i * 3 + 3] = x[old_i * 3 : old_i * 3 + 3]
                x = x_shuffled

        return torch.from_numpy(x), y


def create_cv_dataloaders(
    config: Dict[str, Any],
    data_path: Optional[str | Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders from config.

    Args:
        config: Full project config dict.
        data_path: Override path to data. Falls back to
            ``config["data"]["root"]``.

    Returns:
        ``(train_loader, val_loader, test_loader)`` tuple.
    """
    data_cfg = config.get("data", {})
    path = data_path or data_cfg.get("root", "data/")
    batch_size = config.get("training", {}).get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 0)
    pin_memory = data_cfg.get("pin_memory", False)

    # Augmentation settings
    aug_cfg = data_cfg.get("augmentation", {})
    noise_sigma = aug_cfg.get("noise_sigma", 0.3)
    baseline_drift = aug_cfg.get("baseline_drift", 0.05)
    scale_jitter = aug_cfg.get("scale_jitter", 0.0)
    time_shift_max = aug_cfg.get("time_shift_max", 0)
    channel_dropout_prob = aug_cfg.get("channel_dropout_prob", 0.0)
    sr_shuffle_prob = aug_cfg.get("sr_shuffle_prob", 0.0)

    train_ds = CVDataset(
        path, split="train",
        noise_sigma=noise_sigma, baseline_drift=baseline_drift,
        scale_jitter=scale_jitter, time_shift_max=time_shift_max,
        channel_dropout_prob=channel_dropout_prob, sr_shuffle_prob=sr_shuffle_prob,
    )
    val_ds = CVDataset(path, split="val")
    test_ds = CVDataset(path, split="test")

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
