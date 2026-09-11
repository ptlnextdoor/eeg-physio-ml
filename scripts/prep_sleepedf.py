#!/usr/bin/env python3
"""Stub: convert Sleep-EDF Expanded (PhysioNet) into (epochs, stages, ages) .npz.

Fill this in against a downloaded copy. Kept as a stub on purpose so the repo
runs end-to-end on synthetic data (see selfcheck.py) without a multi-GB download.

Sleep-EDF: https://physionet.org/content/sleep-edfx/1.0.0/
Recommended reader: mne (pip install mne) to load .edf PSGs and .edf hypnograms.

Target output arrays:
  epochs : float32 (N, 1, 3000)   # 30 s at 100 Hz, single EEG channel (e.g. Fpz-Cz)
  stages : int     (N,)           # 0..4 mapping W,N1,N2,N3,REM
  ages   : float   (N,)           # subject age broadcast to each epoch
"""
from __future__ import annotations
import argparse, sys
import numpy as np

STAGE_MAP = {  # AASM/R&K -> our 5-class ids
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf-dir", required=True, help="dir with Sleep-EDF .edf files")
    ap.add_argument("--out", default="data/sleepedf.npz")
    args = ap.parse_args()

    print("TODO: implement with mne.io.read_raw_edf + read_annotations.", file=sys.stderr)
    print("See docstring for the exact target array shapes.", file=sys.stderr)
    # Example skeleton (uncomment once mne is installed and paths are set):
    # import mne, glob, os
    # epochs, stages, ages = [], [], []
    # for psg in glob.glob(os.path.join(args.edf_dir, "*PSG.edf")):
    #     raw = mne.io.read_raw_edf(psg, preload=True)
    #     raw.pick_channels(["EEG Fpz-Cz"]).resample(100)
    #     ... segment into 30s epochs, map hypnogram labels via STAGE_MAP ...
    # np.savez_compressed(args.out, epochs=..., stages=..., ages=...)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
