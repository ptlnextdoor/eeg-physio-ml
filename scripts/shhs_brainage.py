#!/usr/bin/env python3
"""Real SHHS brain-age pilot: EEG bandpower -> ridge regression on age.

Reads SHHS1 EDFs (NSRR), extracts per-recording log bandpower from the EEG
channel, joins subject age from the harmonized dataset, and reports 5-fold
cross-validated MAE against a mean-age baseline.

This is the honest, runnable core behind the outreach claim. It does NOT ship
any data or token; point --edf-dir and --harmonized at a local NSRR copy.

Usage:
  python scripts/shhs_brainage.py --edf-dir ~/nsrr-work/edfs \
      --harmonized ~/nsrr-work/harmonized.csv --out results/shhs_brainage.json
"""
from __future__ import annotations
import os, sys, json, glob, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

EEG_CANDIDATES = ["EEG", "EEG(sec)", "EEG2", "EEG1", "C4-A1", "C3-A2"]


def load_edf_bandpower(path, fs_target=100.0, max_minutes=60):
    """Return a feature vector for one EDF, or None.

    Richer than a single mean: per epoch we compute absolute and relative
    log-bandpower, then summarize the night by the distribution (mean + 20/50/80
    percentiles). Averaging bandpower over the whole night alone is too crude for
    brain-age; the spread across epochs carries the sleep-architecture signal.
    """
    import mne
    from eeg_eval import bandpower_features, BANDS
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    ch = next((c for c in EEG_CANDIDATES if c in raw.ch_names), None)
    if ch is None:
        ch = next((c for c in raw.ch_names if "EEG" in c.upper()), None)
    if ch is None:
        return None
    raw.pick([ch])
    raw.crop(tmax=min(max_minutes * 60, raw.times[-1]))
    raw.load_data(verbose="ERROR")
    raw.resample(fs_target, verbose="ERROR")
    x = raw.get_data()[0]
    T = int(30 * fs_target)
    n_ep = len(x) // T
    if n_ep < 10:
        return None
    epochs = x[: n_ep * T].reshape(n_ep, 1, T)
    logbp = bandpower_features(epochs, fs=fs_target)      # (n_ep, n_bands) log power
    bp = np.exp(logbp)
    rel = bp / (bp.sum(axis=1, keepdims=True) + 1e-12)    # relative bandpower
    # summarize the night by distribution stats per band
    feats = []
    for M in (logbp, rel):
        feats.append(M.mean(0))
        feats.extend(np.percentile(M, [20, 50, 80], axis=0))
    return np.concatenate(feats)


def subject_id_from_edf(path):
    # shhs1-200001.edf -> 200001
    base = os.path.basename(path)
    digits = "".join(c for c in base if c.isdigit())
    return int(digits[-6:]) if len(digits) >= 6 else None


def load_ages(harmonized_csv):
    import csv
    ages = {}
    with open(harmonized_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                sid = int(row["nsrrid"])
                age = float(row["nsrr_age"])
                ages[sid] = age
            except (KeyError, ValueError, TypeError):
                continue
    return ages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf-dir", required=True)
    ap.add_argument("--harmonized", required=True)
    ap.add_argument("--out", default="results/shhs_brainage.json")
    ap.add_argument("--max-minutes", type=float, default=20)
    args = ap.parse_args()

    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import mean_absolute_error, r2_score

    ages = load_ages(args.harmonized)
    edfs = sorted(glob.glob(os.path.join(args.edf_dir, "*.edf")))
    X, y, ids = [], [], []
    for p in edfs:
        sid = subject_id_from_edf(p)
        if sid is None or sid not in ages:
            continue
        try:
            f = load_edf_bandpower(p, max_minutes=args.max_minutes)
        except Exception as e:
            print(f"skip {os.path.basename(p)}: {e}", file=sys.stderr)
            continue
        if f is None:
            continue
        X.append(f); y.append(ages[sid]); ids.append(sid)
        print(f"ok {os.path.basename(p)} age={ages[sid]:.0f}")
    X = np.asarray(X); y = np.asarray(y)
    # Guard: drop non-finite rows (artifact recordings) so a few bad EDFs can't
    # poison the fit. This is why the naive pipeline blew up before.
    finite = np.isfinite(X).all(axis=1)
    X, y = X[finite], y[finite]
    n = len(y)
    print(f"\nUsable recordings: {n} (dropped {int((~finite).sum())} non-finite)")
    if n < 8:
        print("Not enough recordings for CV; download more EDFs.", file=sys.stderr)
        return 2

    k = min(5, n)
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-1, 4, 20)),
    )
    pred = cross_val_predict(model, X, y, cv=KFold(k, shuffle=True, random_state=0))
    mae = mean_absolute_error(y, pred)
    baseline = mean_absolute_error(y, np.full_like(y, y.mean()))
    r2 = r2_score(y, pred)
    out = {
        "n_recordings": int(n), "cv_folds": k,
        "brainage_mae_years": round(float(mae), 3),
        "mean_baseline_mae_years": round(float(baseline), 3),
        "r2": round(float(r2), 3),
        "age_range": [float(y.min()), float(y.max())],
        "note": "SHHS1 pilot, EEG log-bandpower -> ridge. Bandpower baseline, not RnC yet.",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
