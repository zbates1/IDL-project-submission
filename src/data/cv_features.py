"""Hand-crafted electrochemical features for CV mechanism classification.

Extracts 59 features from raw ``(N, 6, 500)`` cyclic voltammogram signals
for traditional ML baselines (Random Forest, Logistic Regression, XGBoost).

Each 500-point CV trace per scan rate is split at the midpoint into a
forward (anodic) half-sweep ``[:250]`` and reverse (cathodic) half-sweep
``[250:]``.  Nine features are extracted per scan rate, plus five global
features computed across all six scan rates.

Usage::

    from src.data.cv_features import extract_features_batch, feature_names

    X_feat = extract_features_batch(signals)  # (N, 59)
    names = feature_names()                   # list of 59 str
"""

from __future__ import annotations

import numpy as np

# Default scan rates matching the simulated dataset.
SCAN_RATES = np.array([0.10, 0.16, 0.25, 0.40, 0.63, 1.00])
N_SR = len(SCAN_RATES)

PER_SR_NAMES = [
    "fwd_peak_current",
    "rev_peak_current",
    "fwd_peak_pos",
    "rev_peak_pos",
    "peak_separation",
    "peak_current_ratio",
    "half_peak_width",
    "fwd_area",
    "rev_area",
]

GLOBAL_NAMES = [
    "randles_sevcik_slope",
    "mean_peak_ratio",
    "std_peak_ratio",
    "mean_peak_separation",
    "std_peak_separation",
]


def feature_names() -> list[str]:
    """Return ordered list of 59 feature names."""
    names: list[str] = []
    for i in range(N_SR):
        for feat in PER_SR_NAMES:
            names.append(f"sr{i}_{feat}")
    names.extend(GLOBAL_NAMES)
    return names


def _half_peak_width(signal: np.ndarray, peak_idx: int, peak_val: float) -> float:
    """Width of peak at half its maximum height."""
    if peak_val <= 0:
        return 0.0
    half_max = peak_val / 2.0
    # Search left from peak
    left = peak_idx
    while left > 0 and signal[left] > half_max:
        left -= 1
    # Search right from peak
    right = peak_idx
    while right < len(signal) - 1 and signal[right] > half_max:
        right += 1
    return float(right - left)


def _extract_one(signal_6x500: np.ndarray, scan_rates: np.ndarray) -> np.ndarray:
    """Extract 59 features from a single (6, 500) signal.

    Args:
        signal_6x500: Raw CV signal array of shape ``(6, 500)``.
        scan_rates: Scan rate values of shape ``(6,)``.

    Returns:
        Feature vector of shape ``(59,)``.
    """
    n_sr, length = signal_6x500.shape
    mid = length // 2

    per_sr_feats = np.empty(n_sr * len(PER_SR_NAMES), dtype=np.float64)
    fwd_peaks = np.empty(n_sr, dtype=np.float64)
    ratios = np.empty(n_sr, dtype=np.float64)
    separations = np.empty(n_sr, dtype=np.float64)

    for i in range(n_sr):
        fwd = signal_6x500[i, :mid]
        rev = signal_6x500[i, mid:]

        fwd_peak_current = float(np.max(fwd))
        rev_peak_current = float(np.abs(np.min(rev)))
        fwd_peak_pos = int(np.argmax(fwd))
        rev_peak_pos = int(np.argmin(rev))
        peak_sep = abs(fwd_peak_pos - rev_peak_pos)

        # Clamp denominator to avoid division by zero
        denom = max(fwd_peak_current, 1e-12)
        ratio = rev_peak_current / denom

        width = _half_peak_width(fwd, fwd_peak_pos, fwd_peak_current)
        fwd_area = float(np.trapz(fwd))
        rev_area = float(np.trapz(rev))

        offset = i * len(PER_SR_NAMES)
        per_sr_feats[offset + 0] = fwd_peak_current
        per_sr_feats[offset + 1] = rev_peak_current
        per_sr_feats[offset + 2] = fwd_peak_pos
        per_sr_feats[offset + 3] = rev_peak_pos
        per_sr_feats[offset + 4] = peak_sep
        per_sr_feats[offset + 5] = ratio
        per_sr_feats[offset + 6] = width
        per_sr_feats[offset + 7] = fwd_area
        per_sr_feats[offset + 8] = rev_area

        fwd_peaks[i] = fwd_peak_current
        ratios[i] = ratio
        separations[i] = peak_sep

    # Global features
    sqrt_sr = np.sqrt(scan_rates[:n_sr])
    if np.std(sqrt_sr) > 1e-12:
        slope = np.polyfit(sqrt_sr, fwd_peaks, 1)[0]
    else:
        slope = 0.0

    global_feats = np.array([
        slope,
        float(np.mean(ratios)),
        float(np.std(ratios)),
        float(np.mean(separations)),
        float(np.std(separations)),
    ], dtype=np.float64)

    return np.concatenate([per_sr_feats, global_feats])


def extract_features_batch(
    signals: np.ndarray,
    scan_rates: np.ndarray | None = None,
) -> np.ndarray:
    """Extract features from a batch of CV signals.

    Args:
        signals: Raw CV signals of shape ``(N, 6, 500)``.
        scan_rates: Per-sample scan rate arrays of shape ``(N, 6)``.
            If ``None``, uses the default scan rates ``[0.10, ..., 1.00]``.

    Returns:
        Feature matrix of shape ``(N, 59)``.
    """
    n = signals.shape[0]
    n_features = N_SR * len(PER_SR_NAMES) + len(GLOBAL_NAMES)
    out = np.empty((n, n_features), dtype=np.float64)

    for i in range(n):
        sr = scan_rates[i] if scan_rates is not None else SCAN_RATES
        out[i] = _extract_one(signals[i], sr)

    return out
