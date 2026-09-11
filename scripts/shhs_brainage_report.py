#!/usr/bin/env python3
"""SHHS brain-age from EEG relative bandpower: an honest baseline + figure.

Reads the cached feature .npz (see cache_shhs_features.py) and reports 5-fold
CV MAE vs a mean-age baseline, plus per-band Spearman correlation with age.

Finding (SHHS1, n=200): whole-night averaged single-channel relative bandpower
is only weakly age-informative. This is the point: it motivates the learned
representations (Rank-N-Contrast, the respiration->EEG model) that this repo is
built to evaluate. The figure makes that weakness visible rather than hiding it.
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import numpy as np


def main():
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.stats import spearmanr
    import figure_style as fs

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="npz from cache_shhs_features.py")
    ap.add_argument("--out-json", default="results/shhs_brainage.json")
    ap.add_argument("--out-fig", default="figures/shhs_brainage")
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, y = d["features"], d["ages"]
    bands = [str(b) for b in d["bands"]]
    n = len(y)

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30)))
    pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
    mae = mean_absolute_error(y, pred)
    baseline = mean_absolute_error(y, np.full_like(y, y.mean()))
    r2 = r2_score(y, pred)
    corrs = {b: float(spearmanr(X[:, j], y).statistic) for j, b in enumerate(bands)}

    out = {
        "dataset": "SHHS1 (NSRR)", "n_recordings": int(n), "cv_folds": 5,
        "brainage_mae_years": round(float(mae), 2),
        "mean_baseline_mae_years": round(float(baseline), 2),
        "beats_baseline": bool(mae < baseline),
        "r2": round(float(r2), 3),
        "per_band_spearman_r": {k: round(v, 3) for k, v in corrs.items()},
        "age_range": [float(y.min()), float(y.max())],
        "finding": ("Whole-night single-channel relative bandpower is only weakly "
                    "age-informative; motivates learned representations (RnC, "
                    "respiration->EEG) that this harness is built to evaluate."),
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)
    print(json.dumps(out, indent=2))

    # Figure: predicted vs true age (left) + per-band |correlation| (right)
    fs.use_paper_style()
    fig, axes = fs.new_figure(cols=2, panels=2)
    axA, axB = axes
    lo, hi = y.min() - 3, y.max() + 3
    axA.plot([lo, hi], [lo, hi], color=fs.PALETTE[7], lw=1, ls="--",
             label="perfect (y=x)")
    axA.scatter(y, pred, s=18, color=fs.PALETTE[0], alpha=0.6, edgecolor="none")
    axA.set_xlim(lo, hi)
    axA.set_xlabel("true age (years)")
    axA.set_ylabel("CV-predicted age (years)")
    axA.set_title(f"Bandpower brain-age: MAE {mae:.1f} yr vs {baseline:.1f} baseline")
    fs.opaque_legend(axA)

    order = np.argsort([abs(corrs[b]) for b in bands])[::-1]
    bn = [bands[i] for i in order]
    cv = [abs(corrs[bands[i]]) for i in order]
    axB.bar(bn, cv, color=fs.PALETTE[0])
    axB.set_ylabel("|Spearman r| with age")
    axB.set_xlabel("EEG band")
    axB.set_title("Even the best band is weakly age-informative")

    fig.suptitle(f"SHHS1 (n={n}): why crude bandpower is not enough for brain-age")
    fs.assert_no_clip(fig)
    os.makedirs(os.path.dirname(args.out_fig) or ".", exist_ok=True)
    fs.save(fig, args.out_fig)
    print("saved", args.out_fig + ".png/.pdf")


if __name__ == "__main__":
    main()
