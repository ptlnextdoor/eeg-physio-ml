#!/usr/bin/env python3
"""Run a downstream probe on a real .npz dataset (epochs, stages, ages).

Usage:
  python scripts/run_eval.py --data data/sleepedf.npz --task sleep_staging
  python scripts/run_eval.py --data data/sleepedf.npz --task age_estimation

The .npz must contain: epochs (N,1,T), stages (N,), ages (N,).
Produces the probe score plus a saved metrics.json.
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.model_selection import train_test_split
from eeg_eval import (bandpower_features, sleep_staging_probe,
                      age_estimation_probe, FS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", choices=["sleep_staging", "age_estimation"], required=True)
    ap.add_argument("--fs", type=float, default=FS)
    ap.add_argument("--out", default="metrics.json")
    args = ap.parse_args()

    d = np.load(args.data)
    X = bandpower_features(d["epochs"], fs=args.fs)
    y = d["stages"] if args.task == "sleep_staging" else d["ages"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    probe = sleep_staging_probe if args.task == "sleep_staging" else age_estimation_probe
    res = probe(Xtr, ytr, Xte, yte)
    print(f"{res.task}: {res.metric_name} = {res.score:.4f}  extra={res.extra}")
    json.dump({"task": res.task, res.metric_name: res.score, **res.extra},
              open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
