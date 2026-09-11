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

| | error in years |
|---|---|
| Just guessing everyone is the average age | 9.45 |
| My model using brain-wave features | 9.53 |

**The model did no better than guessing the average.** Simple brain-wave
features (how much slow vs fast activity there is, averaged over the whole
night) carry almost no information about age. The strongest single feature
had a correlation with age of only 0.11.

That's not a failure, it's the point. Crude features aren't enough. This is the
baseline that smarter, learned features have to beat. Next up: testing whether
the Rank-N-Contrast method actually beats it.

![result](figures/shhs_brainage.png)

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
python scripts/cache_shhs_features.py --edf-dir <folder of .edf files> --harmonized <shhs-harmonized.csv> --out feats.npz
python scripts/shhs_brainage_report.py --features feats.npz
```

The first command reads the recordings once and saves small feature files.
The second one trains the model, prints the numbers, and makes the figure.

## What's in here

```
src/eeg_eval/                    the core: features, models, fake data
scripts/selfcheck.py             quick test on fake data
scripts/cache_shhs_features.py   real recordings -> saved features
scripts/shhs_brainage_report.py  features -> numbers + figure
scripts/run_eval.py              run any saved dataset through the models
scripts/prep_sleepedf.py         loader for a second public dataset (not finished)
figures/                         the figure above
results/                         the numbers, as a small JSON file
```

## License

MIT
