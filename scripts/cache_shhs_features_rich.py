#!/usr/bin/env python3
"""Richer SHHS features: per-epoch bandpower summarized by DISTRIBUTION, not a
single whole-night mean. Also uses both EEG channels when available.

Per recording:
  - split full night into 30 s epochs (both EEG channels)
  - per epoch: absolute log bandpower (5 bands) + relative bandpower (5)
  - summarize each of those 10 numbers across the night by
    mean, std, 10th/50th/90th percentile  -> 10 x 5 = 50 features per channel
  - plus spectral-slope proxy (log delta / log beta) mean+std
That preserves sleep architecture (deep-sleep delta vs REM/wake) instead of
blurring the whole night into one average.

Output: results/shhs_features_rich.npz with features (N,F), ages (N,)
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


def summarize(M):
    """M: (n_epochs, k) -> (k*5,) mean,std,p10,p50,p90 per column."""
    return np.concatenate([M.mean(0), M.std(0),
                           np.percentile(M, 10, axis=0),
                           np.percentile(M, 50, axis=0),
                           np.percentile(M, 90, axis=0)])


def features_for_channel(x, fs=100.0):
    from eeg_eval import bandpower_features
    T = int(30 * fs); ne = len(x) // T
    if ne < 60:
        return None
    ep = x[: ne * T].reshape(ne, 1, T)
    logbp = bandpower_features(ep, fs=fs)              # (ne,5) log abs power
    bp = np.exp(logbp)
    rel = bp / (bp.sum(1, keepdims=True) + 1e-12)      # (ne,5)
    slope = (logbp[:, 0] - logbp[:, 4])[:, None]       # delta - beta in log space
    per_epoch = np.hstack([logbp, rel, slope])         # (ne, 11)
    return summarize(per_epoch)                        # (55,)


def main():
    import mne
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf-dir", required=True)
    ap.add_argument("--harmonized", required=True)
    ap.add_argument("--out", default="results/shhs_features_rich.npz")
    args = ap.parse_args()

    ages = load_ages(args.harmonized)
    X, y = [], []
    for p in sorted(glob.glob(os.path.join(args.edf_dir, "*.edf"))):
        digits = "".join(c for c in os.path.basename(p) if c.isdigit())
        if len(digits) < 6:
            continue
        sid = int(digits[-6:])
        if sid not in ages:
            continue
        try:
            r = mne.io.read_raw_edf(p, preload=False, verbose="ERROR")
            chs = [c for c in ("EEG", "EEG(sec)") if c in r.ch_names]
            if not chs:
                continue
            r.pick(chs); r.load_data(verbose="ERROR"); r.resample(100.0, verbose="ERROR")
            data = r.get_data()
            feats = []
            for i in range(len(chs)):
                f = features_for_channel(data[i])
                if f is None:
                    feats = None; break
                feats.append(f)
            if feats is None:
                continue
            # if only one channel, duplicate so feature length is constant
            if len(feats) == 1:
                feats.append(feats[0])
            fv = np.concatenate(feats[:2])
            if not np.isfinite(fv).all():
                continue
            X.append(fv); y.append(ages[sid])
            print(f"ok {os.path.basename(p)} age={ages[sid]:.0f} nfeat={len(fv)}")
        except Exception as e:
            print(f"skip {os.path.basename(p)}: {e}", file=sys.stderr)
    X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, features=X, ages=y)
    print(f"\nsaved {args.out}: {X.shape[0]} recordings x {X.shape[1]} features")


if __name__ == "__main__":
    main()
