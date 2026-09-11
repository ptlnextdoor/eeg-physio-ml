"""Synthetic EEG generator with known structure, for pipeline validation.

Builds epochs where age drives alpha/delta band power and sleep stage drives a
distinct spectral signature. A correct harness must recover both above chance.
This lets the self-check assert correctness with no data download.
"""
from __future__ import annotations
import numpy as np

FS = 100.0
T = 3000  # 30 s epoch at 100 Hz
STAGES = ["W", "N1", "N2", "N3", "REM"]

# center frequency (Hz) that dominates each stage
_STAGE_FREQ = {"W": 18.0, "N1": 6.0, "N2": 13.5, "N3": 1.5, "REM": 7.0}


def _osc(freq, amp, n, rng):
    t = np.arange(n) / FS
    phase = rng.uniform(0, 2 * np.pi)
    return amp * np.sin(2 * np.pi * freq * t + phase)


def make_dataset(n=1200, seed=0, degrade=0.0):
    """Return (epochs (n,1,T), stage_labels (n,), ages (n,)).

    degrade in [0,1] injects noise and attenuates signal, simulating a lower
    quality EEG source (e.g. synthesized). Use degrade>0 to create a second
    'source' whose probe scores should be measurably worse.
    """
    rng = np.random.default_rng(seed)
    epochs = np.zeros((n, 1, T))
    stage_ids = rng.integers(0, len(STAGES), size=n)
    ages = rng.uniform(20, 80, size=n)
    for i in range(n):
        stage = STAGES[stage_ids[i]]
        sig = _osc(_STAGE_FREQ[stage], 1.0 * (1 - 0.85 * degrade), T, rng)
        # age drives alpha (down with age) and delta (up with age)
        alpha_amp = max(0.05, (80 - ages[i]) / 60.0)
        delta_amp = ages[i] / 80.0
        sig += _osc(10.0, alpha_amp * (1 - 0.7 * degrade), T, rng)
        sig += _osc(2.0, delta_amp * (1 - 0.7 * degrade), T, rng)
        noise = rng.standard_normal(T) * (0.3 + 1.5 * degrade)
        epochs[i, 0] = sig + noise
    return epochs, stage_ids, ages
