#!/usr/bin/env python3
"""Brain-age on SHHS from rich per-epoch features. Compares three models and
makes the figure. Honest: reports whichever wins, plus the mean-age baseline.

  ridge   : linear, cross-validated alpha
  gbr     : gradient boosting (nonlinear)
  baseline: predict the mean age

Run: python scripts/shhs_brainage_rich.py --features feats_rich.npz
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import numpy as np


def main():
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, QuantileTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.stats import pearsonr
    import figure_style as fs

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out-json", default="results/shhs_brainage_rich.json")
    ap.add_argument("--out-fig", default="figures/shhs_brainage")
    args = ap.parse_args()

    d = np.load(args.features)
    X, y = d["features"].astype(np.float64), d["ages"].astype(np.float64)
    n, nf = X.shape
    cv = KFold(5, shuffle=True, random_state=0)
    baseline = float(np.abs(y - y.mean()).mean())

    models = {
        "ridge": make_pipeline(QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=0),
                               RidgeCV(alphas=np.logspace(0, 5, 40))),
        "gbr": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                         subsample=0.8, random_state=0),
    }
    preds, scores = {}, {}
    for name, m in models.items():
        p = cross_val_predict(m, X, y, cv=cv)
        preds[name] = p
        scores[name] = {"mae": float(np.abs(p - y).mean()),
                        "r2": float(r2_score(y, p)),
                        "pearson_r": float(pearsonr(p, y)[0])}
        print(f"{name:8s} MAE={scores[name]['mae']:.2f}  R2={scores[name]['r2']:.3f}  r={scores[name]['pearson_r']:.3f}")
    print(f"baseline MAE={baseline:.2f}")

    best = min(scores, key=lambda k: scores[k]["mae"])
    out = {"dataset": "SHHS1 (NSRR)", "n_recordings": int(n), "n_features": int(nf),
           "features": "per-epoch abs+rel bandpower + slope, night distribution stats, 2 EEG channels",
           "cv_folds": 5, "mean_baseline_mae": round(baseline, 2),
           "models": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in scores.items()},
           "best_model": best,
           "beats_baseline": bool(scores[best]["mae"] < baseline)}
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    # Figure: best model pred vs true (left), MAE by model vs baseline (right)
    fs.use_paper_style()
    fig, (axA, axB) = fs.new_figure(cols=2, panels=2)
    p = preds[best]; lo, hi = y.min() - 3, y.max() + 3
    axA.plot([lo, hi], [lo, hi], color=fs.PALETTE[7], lw=1, ls="--", label="perfect (y = x)")
    axA.scatter(y, p, s=16, color=fs.PALETTE[0], alpha=0.6, edgecolor="none")
    axA.set_xlim(lo, hi); axA.set_ylim(lo, hi)
    axA.set_xlabel("true age (years)"); axA.set_ylabel("predicted age (years, cross-validated)")
    axA.set_title(f"{best}: MAE {scores[best]['mae']:.1f} yr, r = {scores[best]['pearson_r']:.2f}")
    fs.opaque_legend(axA)

    names = ["baseline"] + list(scores)
    maes = [baseline] + [scores[k]["mae"] for k in scores]
    cols = [fs.PALETTE[7]] + [fs.PALETTE[0] if k != best else fs.PALETTE[3] for k in scores]
    axB.bar(names, maes, color=cols)
    axB.axhline(baseline, color=fs.PALETTE[7], ls="--", lw=1)
    axB.set_ylabel("brain-age MAE (years, lower is better)")
    axB.set_title("Per-epoch features vs guessing the mean")
    for i, v in enumerate(maes):
        axB.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"SHHS1 (n={n}): brain-age from {nf} sleep-EEG features")
    fs.assert_no_clip(fig)
    os.makedirs(os.path.dirname(args.out_fig) or ".", exist_ok=True)
    fs.save(fig, args.out_fig)
    print(json.dumps(out, indent=2)); print("saved", args.out_fig + ".png/.pdf")


if __name__ == "__main__":
    main()
