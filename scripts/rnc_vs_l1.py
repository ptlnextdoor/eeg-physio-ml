#!/usr/bin/env python3
"""Head-to-head: Rank-N-Contrast vs plain L1 for brain-age on SHHS features.

Uses Kaiwen Zha's RnCLoss verbatim (src/rnc/loss.py, from
github.com/kaiwenzha/Rank-N-Contrast). Everything else is held fixed so the
only difference is the training objective:

  L1 :  encoder + linear head trained end-to-end on |age - pred|
  RnC:  encoder pretrained with RnCLoss (two augmented views per sample), frozen,
        then the same linear head trained on top with L1

Same 2-layer MLP encoder, same 5-fold CV splits, same epochs, repeated over
several seeds. Reports mean +/- std MAE for both, plus the mean-age baseline.

Run:
  python scripts/rnc_vs_l1.py --features feats.npz
"""
from __future__ import annotations
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from rnc.loss import RnCLoss


def mlp(d_in, d_hid=64, d_out=32):
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Linear(d_hid, d_out))


def augment(x, noise=0.1):
    """Two views: small Gaussian jitter in standardized feature space."""
    return x + noise * torch.randn_like(x), x + noise * torch.randn_like(x)


def train_l1(Xtr, ytr, epochs, seed):
    torch.manual_seed(seed)
    enc, head = mlp(Xtr.shape[1]), nn.Linear(32, 1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    X, y = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (head(enc(X)).squeeze(1) - y).abs().mean()
        loss.backward(); opt.step()
    return enc, head


def train_rnc(Xtr, ytr, epochs, seed, temperature=2.0):
    torch.manual_seed(seed)
    enc = mlp(Xtr.shape[1])
    crit = RnCLoss(temperature=temperature, label_diff="l1", feature_sim="l2")
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    X, y = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        v1, v2 = augment(X)
        feats = torch.stack([enc(v1), enc(v2)], dim=1)   # [bs, 2, feat_dim]
        loss = crit(feats, y[:, None])
        loss.backward(); opt.step()
    # freeze encoder, fit linear head with L1
    for p in enc.parameters():
        p.requires_grad_(False)
    head = nn.Linear(32, 1)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    with torch.no_grad():
        Z = enc(X)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (head(Z).squeeze(1) - y).abs().mean()
        loss.backward(); opt.step()
    return enc, head


def cv_mae(X, y, trainer, epochs, seed):
    kf = KFold(5, shuffle=True, random_state=seed)
    pred = np.zeros_like(y)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        mu = y[tr].mean()
        enc, head = trainer(Xtr, y[tr] - mu, epochs, seed)
        with torch.no_grad():
            pred[te] = head(enc(torch.tensor(Xte, dtype=torch.float32))).squeeze(1).numpy() + mu
    return float(np.abs(pred - y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results/rnc_vs_l1.json")
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, y = d["features"].astype(np.float64), d["ages"].astype(np.float64)
    baseline = float(np.abs(y - y.mean()).mean())

    l1, rnc = [], []
    for s in range(args.seeds):
        l1.append(cv_mae(X, y, train_l1, args.epochs, s))
        rnc.append(cv_mae(X, y, train_rnc, args.epochs, s))
        print(f"seed {s}: L1 MAE={l1[-1]:.2f}  RnC MAE={rnc[-1]:.2f}")

    out = {
        "dataset": "SHHS1 (NSRR)", "n_recordings": int(len(y)),
        "features": "5 whole-night relative bandpowers",
        "encoder": "MLP 5->64->32, linear head", "epochs": args.epochs,
        "seeds": args.seeds, "cv_folds": 5,
        "mean_baseline_mae": round(baseline, 2),
        "l1_mae_mean": round(float(np.mean(l1)), 2), "l1_mae_std": round(float(np.std(l1)), 2),
        "rnc_mae_mean": round(float(np.mean(rnc)), 2), "rnc_mae_std": round(float(np.std(rnc)), 2),
        "rnc_beats_l1": bool(np.mean(rnc) < np.mean(l1)),
        "either_beats_baseline": bool(min(np.mean(rnc), np.mean(l1)) < baseline),
        "loss_source": "github.com/kaiwenzha/Rank-N-Contrast loss.py, unmodified",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
