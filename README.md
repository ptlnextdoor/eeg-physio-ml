# eeg-physio-ml

Can you guess someone's age from their sleeping brain waves?

This repo is a small, honest test of that question. It takes overnight EEG
(brain-wave) recordings, pulls out simple features, and checks how well those
features predict a person's age. It also gives you a clean way to compare two
different kinds of EEG on the same test.

I built it after reading the Katabi Lab's work at MIT on turning breathing
signals into EEG and on learning better representations for regression
(Rank-N-Contrast). This is my own independent project, not affiliated with
the lab.

## What I found (on real data)

I ran it on **200 real overnight recordings** from the Sleep Heart Health
Study (SHHS, via NSRR).

**First try was a flop, and it taught me something.** I summarized each
whole night as 5 numbers (how much slow vs fast brain activity, averaged over
8 hours). The model predicted "about 61" for everyone. It couldn't beat just
guessing the average age (9.53 vs 9.45 years error). Averaging a whole night
into one number throws away the part that carries age: how sleep is
*structured* through the night.

**Second try fixed it.** I split each night into 30-second chunks, measured
brain activity in each chunk, and kept the *spread* across the night (average,
variation, 10th/50th/90th percentiles) for both EEG channels. That's 110
numbers per night instead of 5.

| model | error in years | beats guessing? |
|---|---|---|
| Guess everyone is the average age | 9.45 | (baseline) |
| Linear model on 110 features | 8.69 | yes |
| Gradient-boosted trees on 110 features | **8.44** | yes, correlation r = 0.36 |

![result](figures/shhs_brainage.png)

Left: predicted vs true age. The dots now follow the diagonal instead of
sitting in a flat line. Right: the two real models beat the guess-the-mean bar.

This is still a modest result (an 8.4-year error on ages 40 to 89 is far from
clinical), but it's real signal from real data with only 200 people, and the
whole pipeline runs from raw EDF files.

## Does Rank-N-Contrast help? (head-to-head)

I tested the Katabi Lab's Rank-N-Contrast objective against plain L1, using
their `loss.py` unmodified, same small neural net, same folds, 3 seeds, on
the same 110 features.

| training objective | error in years (mean ± std over seeds) |
|---|---|
| Guess the average age | 9.45 |
| Gradient-boosted trees (from above) | 8.44 |
| Neural net, plain L1 | 10.29 ± 0.23 |
| Neural net, Rank-N-Contrast then L1 head | 10.10 ± 0.15 |

Two takeaways:
- **RnC beats plain L1 again**, on both the crude and the rich features, and
  is again more stable. The objective consistently does what the paper says.
- **But with only 200 people, a small neural net overfits and loses to trees.**
  Both nets land above the guess-the-mean bar. That is the honest limit of
  this dataset size, not of RnC. The natural next step is more subjects
  (SHHS has ~5,800) and per-epoch training rather than per-night summaries.

Reproduce: `python scripts/rnc_vs_l1.py --features feats_rich.npz`

## Try it in 30 seconds (no data needed)

```bash
pip install -r requirements.txt
python scripts/selfcheck.py
```

This builds fake brain-wave data where I *know* the answer, then checks that
the code recovers it. If this passes, the pipeline works.

## Run it on real data

You need your own copy of SHHS from NSRR (free with a signed data-use
agreement). Nothing from the dataset is included here.

```bash
python scripts/cache_shhs_features_rich.py --edf-dir <folder of .edf files> --harmonized <shhs-harmonized.csv> --out feats_rich.npz
python scripts/shhs_brainage_rich.py --features feats_rich.npz
```

The first command reads the recordings once and saves small feature files.
The second one trains the models, prints the numbers, and makes the figure.
(`cache_shhs_features.py` + `shhs_brainage_report.py` are the older 5-feature
version, kept so the flop above is reproducible too.)

## What's in here

```
src/eeg_eval/                    the core: features, models, fake data
scripts/selfcheck.py             quick test on fake data
scripts/cache_shhs_features_rich.py  real recordings -> 110 per-night features
scripts/shhs_brainage_rich.py        features -> numbers + figure
scripts/cache_shhs_features.py       older 5-feature version (the flop)
scripts/shhs_brainage_report.py      older report for the 5-feature version
scripts/rnc_vs_l1.py             Rank-N-Contrast vs L1 head-to-head
src/rnc/loss.py                  Kaiwen Zha's RnCLoss, copied unmodified
scripts/run_eval.py              run any saved dataset through the models
scripts/prep_sleepedf.py         loader for a second public dataset (not finished)
figures/                         the figure above
results/                         the numbers, as a small JSON file
```

## License

MIT
