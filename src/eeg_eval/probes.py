"""Downstream probes: train a light classifier/regressor on frozen features.

The point of a probe is to hold the evaluation fixed and vary only the EEG
*source*, so a difference in score reflects a difference in representation
quality, not in the head. This mirrors the linear-probe protocol common in
self-supervised representation-learning papers.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


@dataclass
class ProbeResult:
    task: str
    metric_name: str
    score: float
    extra: dict


def sleep_staging_probe(Xtr, ytr, Xte, yte) -> ProbeResult:
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    return ProbeResult("sleep_staging", "accuracy", float(acc),
                       {"macro_f1": float(f1)})


def age_estimation_probe(Xtr, ytr, Xte, yte) -> ProbeResult:
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    return ProbeResult("age_estimation", "MAE_years", float(mae), {})


def compare_sources(results_by_source: dict[str, ProbeResult]) -> str:
    """Pretty table comparing the same probe across EEG sources."""
    lines = ["source            metric        score"]
    lines.append("-" * 40)
    for src, r in results_by_source.items():
        lines.append(f"{src:<16}  {r.metric_name:<12}  {r.score:.4f}")
    return "\n".join(lines)
