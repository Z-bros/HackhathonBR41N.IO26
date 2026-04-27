# Silence Is Not Absence

**Replication and critical reassessment of a vibrotactile P300 BCI pipeline for patients with Unresponsive Wakefulness Syndrome.**

Built for [BR41N.IO 2026](https://www.br41n.io) by Group 45.

## What this is

This project replicates the published vibrotactile P300 BCI pipeline (Spataro et al. 2018; Guger et al. 2018) on a dataset of 8 EEG recordings from 2 UWS patients, then critically reassesses whether the dataset's pre-assigned high/low accuracy labels survive faithful reanalysis.

The data comes pre-labeled by the original researchers: each recording is tagged as either "high" (>95% accuracy in the original mindBEAGLE classifier) or "low" (<5%). We tested whether these labels would hold up when the published pipeline is re-implemented from scratch.

## Headline result

**Only 3 of 8 files cleared the 95% significance threshold (23%) in our re-implementation, scattered across both labels and patients.** One file labeled as a >95% accuracy run came back at 8.2% — below the 12.5% chance level — in our hands.

No spectral biomarker (pre-stimulus alpha, event-related power) and no alternative classifier (shrinkage LDA, xDAWN spatial filtering, Riemannian MDM) substantially exceeded the LDA baseline. The bottleneck is signal strength, not algorithm choice.

## Pipeline

The analysis is split across five MATLAB scripts, run in order:

| Script | Purpose | Key output |
|---|---|---|
| `uws_preprocessing.m` | Filter (0.1–30 Hz + 50 Hz notch), epoch (−100 to +600 ms), baseline-correct, reject ±100 µV. Extract three feature sets. | `processed_data.mat` |
| `uws_classification.m` | Single-trial LDA + group-of-8 LDA classification with leave-one-group-out CV. Permutation testing. | `classification_results.mat` |
| `uws_spectral.m` | Welch PSD, pre-stimulus band power per trial, event-related spectrograms. | `spectral_results.mat` |
| `uws_classifier_inspection.m` | LDA weights, Haufe-corrected activation patterns, scalp topographies. | `classifier_inspection.pdf` |
| `uws_classifier_comparison.m` | Comparison of four classifier variants on the same task. | `classifier_comparison.pdf` |

## Reproducing the analysis

### Requirements

- MATLAB R2020a or newer (for `exportgraphics`)
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox

### Data

The 8 `.mat` files (`P1_low1.mat`, `P1_low2.mat`, ..., `P2_high2.mat`) are provided by the BR41N.IO 2025 organizers and are not included in this repository. To request access, contact [BR41N.IO](https://www.br41n.io).

Each file contains:
- `y`: EEG data, samples × 8 channels (Fz, C3, Cz, C4, CP1, CPz, CP2, Pz)
- `trig`: trigger channel (−1 = distractor, 1 = non-target, 2 = target)
- `fs`: sampling rate (256 Hz)

Place the `.mat` files in the project root, then run the scripts in `scripts/` in the order listed above.

## Results

All output figures are in `results/`. Highlights:

- `group_classification_perfile.pdf` — per-file group-of-8 accuracy with chance and significance thresholds (the headline figure)
- `classifier_comparison.pdf` — four classifier variants compared on the same task
- `spectral_summary.pdf` — PSD, pre-stimulus alpha, and event-related spectrograms

## Limitations

A few important caveats on what this project does and doesn't show:

- **Two patients only.** Findings on signal-level fragility cannot be generalized to UWS as a population.
- **Replication does not equal critique.** The original Spataro and Guger papers report group-level findings (e.g., BCI accuracy correlates with 6-month CRS-R recovery, r = 0.89) that are not addressed here. Our work focuses on a narrower question: whether single-session run-level labels survive faithful reanalysis on this specific dataset.
- **Outcome data was not available.** A more complete reanalysis would correlate replication failure with clinical outcome.
- **Per-file analysis only.** We did not pool trials across files, since pooling introduces session-level confounds that can inflate accuracy artifactually.

## References

- Giacino JT, Fins JJ, Laureys S, Schiff ND. (2014). Disorders of consciousness after acquired brain injury: the state of the science. *Nat. Rev. Neurol.* 10:99–114.
- Spataro R, Heilinger A, Allison B, et al. (2018). Preserved somatosensory discrimination predicts consciousness recovery in unresponsive wakefulness syndrome. *Clin. Neurophysiol.* 129:1130–1136.
- Guger C, Spataro R, Pellas F, et al. (2018). Assessing command-following and communication with vibro-tactile P300 brain-computer interface tools in patients with unresponsive wakefulness syndrome. *Front. Neurosci.* 12:423.
- Wannez S, Heine L, Thonnard M, et al. (2017). The repetition of behavioral assessments in diagnosis of disorders of consciousness. *Ann. Neurol.* 81:883–889.

## Team

- Zidane Fatuna — [GitHub](https://github.com/Z-bros) | [LinkedIn](https://linkedin.com/in/zidanefatuna)
- Timoleon Fourfaro
- Johana Dominguez
- Neha Sharma

## License

Code: MIT License (see `LICENSE`).
Original publications and dataset: subject to their respective licenses.
