# eeg-physio-ml

Reproducible evaluation harness for **self-supervised representation learning on
sleep/physiological EEG**, built to probe how well learned or synthesized EEG
representations transfer to downstream clinical tasks (sleep staging, age
estimation) on *public* data.

Motivated by the Katabi Lab line of work on physiological-signal ML
(e.g. *Physiology as Language: Translating Respiration to Sleep EEG*,
arXiv:2602.00526; *Rank-N-Contrast*, NeurIPS 2023). This repo is my own
independent reproduction/eval scaffold, not affiliated with the lab.

## What this does

Given EEG epochs (real, or synthesized by another model), it trains a light
linear/MLP probe on frozen features and reports downstream performance, so you
can measure the representation gap between two EEG sources on the *same* probe.

- `sleep staging` probe (5-class): W, N1, N2, N3, REM
- `age estimation` probe (regression, MAE in years)
- source-vs-source comparison table (real EEG vs synthesized EEG)

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/selfcheck.py          # runs on synthetic data, no download needed
```

The self-check builds a tiny synthetic dataset with a known age->band-power
relationship, runs both probes, and asserts the harness recovers signal above
chance. This proves the pipeline is correct before you plug in real data.

## Using real data (Sleep-EDF)

1. Download Sleep-EDF Expanded from PhysioNet.
2. Convert to `.npz` epochs with `scripts/prep_sleepedf.py` (stub; fill paths).
3. Run `python scripts/run_eval.py --data data/sleepedf.npz --task sleep_staging`.

## Repo layout

```
src/eeg_eval/       core: data, features, probes, metrics
scripts/            selfcheck, prep, run_eval
```

## Status

Working self-check on synthetic data. Sleep-EDF adapter is a stub to be filled
against a downloaded copy. Extending toward a real respiration->EEG comparison.

## Real-data result (SHHS1, NSRR, n=200)

Ran the pipeline on 200 real SHHS1 overnight recordings (single-channel EEG,
NSRR). Whole-night averaged relative bandpower + cross-validated ridge gives:

| metric | value |
|---|---|
| brain-age MAE | 9.53 yr |
| mean-age baseline MAE | 9.45 yr |
| R^2 | -0.01 |
| best single band (alpha) \|Spearman r\| with age | 0.11 |

Honest finding: crude bandpower is **not** age-informative on its own; the model
regularizes to the mean. That is the point of this harness. It is the baseline
that learned representations (Rank-N-Contrast, the respiration->EEG generative
model) must beat. See `figures/shhs_brainage.png`.

Reproduce (needs your own NSRR SHHS copy; no data or token is shipped):

```bash
python scripts/cache_shhs_features.py --edf-dir <edfs> --harmonized <harmonized.csv> --out feats.npz
python scripts/shhs_brainage_report.py --features feats.npz
```

## License

MIT
