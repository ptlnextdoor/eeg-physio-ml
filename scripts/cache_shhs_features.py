#!/usr/bin/env python3
"""Cache SHHS bandpower features to a .npz so figures/runs are fast and the raw
EDFs never need to leave the machine. Point --edf-dir at a local NSRR copy.

Writes features (N,5 relative bandpower), ages (N,), to results/shhs_features.npz.
"""
from __future__ import annotations
import os, sys, glob, csv, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np


def load_ages(p):
    d = {}
    for row in csv.DictReader(open(p)):
        try:
            d[int(row["nsrrid"])] = float(row["nsrr_age"])
        except (KeyError, ValueError, TypeError):
            continue
    return d


def main():
    import mne
    from eeg_eval import bandpower_features
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf-dir", required=True)
    ap.add_argument("--harmonized", required=True)
    ap.add_argument("--out", default="results/shhs_features.npz")
    ap.add_argument("--minutes", type=float, default=180)
    args = ap.parse_args()

    ages = load_ages(args.harmonized)
    X, y = [], []
    for p in sorted(glob.glob(os.path.join(args.edf_dir, "*.edf"))):
        sid_digits = "".join(c for c in os.path.basename(p) if c.isdigit())
        if len(sid_digits) < 6:
            continue
        sid = int(sid_digits[-6:])
        if sid not in ages:
            continue
        try:
            r = mne.io.read_raw_edf(p, preload=False, verbose="ERROR")
            ch = next((c for c in ("EEG", "EEG(sec)") if c in r.ch_names), None)
            if ch is None:
                continue
            r.pick([ch]); r.crop(tmax=min(args.minutes * 60, r.times[-1]))
            r.load_data(verbose="ERROR"); r.resample(100.0, verbose="ERROR")
            x = r.get_data()[0]
            T = 3000; ne = len(x) // T
            if ne < 20:
                continue
            ep = x[: ne * T].reshape(ne, 1, T)
            bp = np.exp(bandpower_features(ep, fs=100.0))
            rel = bp / (bp.sum(1, keepdims=True) + 1e-12)
            feat = rel.mean(0)
            if not np.isfinite(feat).all():
                continue
            X.append(feat); y.append(ages[sid])
            print(f"ok {os.path.basename(p)} age={ages[sid]:.0f}")
        except Exception as e:
            print(f"skip {os.path.basename(p)}: {e}", file=sys.stderr)
    X = np.asarray(X); y = np.asarray(y)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, features=X, ages=y,
                        bands=np.array(["delta", "theta", "alpha", "sigma", "beta"]))
    print(f"\nsaved {args.out}: {X.shape[0]} recordings")


if __name__ == "__main__":
    main()
