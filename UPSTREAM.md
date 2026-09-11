# Upstream: build on THEIR code

This repo is meant to extend the Katabi Lab's own published code, not rebuild
it blind. Target for the outreach fork:

- **kaiwenzha/Rank-N-Contrast** (NeurIPS 2023, MIT license, 139 stars)
  https://github.com/kaiwenzha/Rank-N-Contrast
  - `loss.py` -> `RnCLoss` is the reusable piece. Keep it unchanged.
  - `dataset.py` -> replace the AgeDB loader with a Sleep-EDF loader.
  - `main_rnc.py` / `main_linear.py` -> encoder then linear probe.
  - The pitch: "Applied your RnC loss to sleep-EEG brain-age regression."

- **YyzHarry/imbalanced-regression** (ICML 2021, famous)
  https://github.com/YyzHarry/imbalanced-regression
  - Has a healthcare DIR benchmark; useful baseline framing.

Relevant papers (read before emailing):
- Physiology as Language: Translating Respiration to Sleep EEG, arXiv:2602.00526
- Antidepressant use from one night of sleep, arXiv:2510.10364 (Mirzazadeh)
- Rank-N-Contrast, arXiv:2210.01189
- Delving into Deep Imbalanced Regression, arXiv:2102.09554

The self-check in this repo (scripts/selfcheck.py) is the standalone capability
proof; the RnC-on-EEG fork is the paper-specific extension you link in the email.
