"""Bandpower features for EEG epochs.

Extracts log power in the standard clinical bands from each epoch/channel using
Welch's method (via numpy FFT to avoid a scipy dependency). These are the classic
features that drive sleep staging and age estimation, so they make a fair,
model-agnostic probe input for comparing two EEG sources.
"""
from __future__ import annotations
import numpy as np

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (13.0, 16.0),   # sleep spindles
    "beta": (16.0, 30.0),
}


def _welch_psd(x: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Minimal Welch PSD along the last axis. Returns (freqs, psd)."""
    n = x.shape[-1]
    nperseg = min(nperseg, n)
    step = nperseg // 2 or 1
    win = np.hanning(nperseg)
    scale = 1.0 / (fs * (win ** 2).sum())
    segs = []
    start = 0
    while start + nperseg <= n:
        seg = x[..., start:start + nperseg] * win
        spec = np.fft.rfft(seg, axis=-1)
        segs.append((np.abs(spec) ** 2) * scale)
        start += step
    if not segs:
        seg = x * np.hanning(n)
        spec = np.fft.rfft(seg, axis=-1)
        psd = (np.abs(spec) ** 2) / (fs * (np.hanning(n) ** 2).sum())
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        return freqs, psd
    psd = np.mean(segs, axis=0)
    psd[..., 1:-1] *= 2.0
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, psd


def bandpower_features(epochs: np.ndarray, fs: float = 100.0, nperseg: int = 200) -> np.ndarray:
    """epochs: (N, C, T) -> features (N, C*len(BANDS)) of log bandpower.

    Works for (N, T) too (treated as single channel).
    """
    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim == 2:
        epochs = epochs[:, None, :]
    freqs, psd = _welch_psd(epochs, fs=fs, nperseg=nperseg)  # psd: (N, C, F)
    feats = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        bp = psd[..., mask].sum(axis=-1)  # (N, C)
        feats.append(np.log(bp + 1e-8))
    out = np.stack(feats, axis=-1)  # (N, C, B)
    return out.reshape(out.shape[0], -1)
