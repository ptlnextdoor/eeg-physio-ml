#!/usr/bin/env python3
"""Self-check: prove the eval harness works on synthetic EEG with known structure.

Runs both probes on a clean source and a 'degraded' source (stand-in for
synthesized EEG). Asserts:
  1. sleep staging accuracy on clean data is well above chance (0.20 for 5 class)
  2. age MAE on clean data beats a mean-predictor baseline
  3. the degraded source scores measurably worse (the harness can DETECT a gap)

Exit code 0 on success. No downloads required.
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.model_selection import train_test_split
from eeg_eval import (make_dataset, bandpower_features,
                      sleep_staging_probe, age_estimation_probe,
                      compare_sources, FS)


def eval_source(degrade, seed=0):
    epochs, stages, ages = make_dataset(n=1500, seed=seed, degrade=degrade)
    X = bandpower_features(epochs, fs=FS)
    Xtr, Xte, s_tr, s_te, a_tr, a_te = train_test_split(
        X, stages, ages, test_size=0.3, random_state=seed)
    stage_res = sleep_staging_probe(Xtr, s_tr, Xte, s_te)
    age_res = age_estimation_probe(Xtr, a_tr, Xte, a_te)
    mean_baseline_mae = float(np.mean(np.abs(a_te - a_tr.mean())))
    return stage_res, age_res, mean_baseline_mae


def main():
    print("== eeg-physio-ml self-check ==\n")
    clean_stage, clean_age, base_mae = eval_source(degrade=0.0)
    deg_stage, deg_age, _ = eval_source(degrade=0.7)

    print("Sleep staging (5-class, chance=0.20):")
    print(compare_sources({"clean_eeg": clean_stage, "degraded_eeg": deg_stage}))
    print(f"\nAge estimation (MAE years, mean-baseline={base_mae:.2f}):")
    print(compare_sources({"clean_eeg": clean_age, "degraded_eeg": deg_age}))

    ok = True
    # 1. staging beats chance
    if clean_stage.score <= 0.35:
        print("\nFAIL: clean staging accuracy not above chance"); ok = False
    # 2. age beats mean baseline
    if clean_age.score >= base_mae:
        print("FAIL: clean age MAE not better than mean baseline"); ok = False
    # 3. harness detects source gap: degraded never strictly better on either
    #    task, and clearly worse on at least one.
    not_better = (deg_stage.score <= clean_stage.score + 1e-6) and \
                 (deg_age.score >= clean_age.score - 1e-6)
    clearly_worse = (deg_age.score > clean_age.score * 1.2) or \
                    (deg_stage.score < clean_stage.score - 0.02)
    if not (not_better and clearly_worse):
        print("FAIL: harness did not detect degraded source as worse"); ok = False

    if ok:
        print("\nPASS: harness recovers signal and detects source quality gap.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
