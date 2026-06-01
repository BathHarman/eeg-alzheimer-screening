# eeg-alzheimer-screening
Resting-state EEG analysis comparing Alzheimer's, healthy control, and FTD groups using OpenNeuro ds004504. Computes theta/alpha ratio as a cognitive decline biomarker via MNE-Python.

# EEG Alzheimer Screening — Theta/Alpha Ratio Analysis

**March – May 2025 | University of the Pacific | Human Brain Machine Interface**

This project came out of my graduate coursework in Human Brain Machine Interface. I wanted to look at whether resting-state EEG could actually distinguish Alzheimer's patients from healthy controls using just frequency band ratios — no deep learning, no black box. The theta/alpha ratio kept coming up in the literature as a promising biomarker, so I built a pipeline to test it on a real clinical dataset.

I used the OpenNeuro ds004504 dataset (88 subjects: 36 AD, 29 HC, 23 FTD, ages 49–78) and implemented the full analysis in Python using MNE-Python, which is the standard library for EEG work in research.

---

## What it does

Loads all subjects from the BIDS-formatted dataset, preprocesses the raw EEG (bandpass filter, average reference), computes power spectral density via Welch's method, extracts mean power across five frequency bands, and runs a t-test comparing AD vs HC on the theta/alpha ratio. Outputs CSVs and figures.

## Key finding

The AD group showed elevated delta and theta power with reduced alpha and beta power compared to healthy controls. The theta/alpha ratio was significantly higher in AD (p < 0.05), consistent with prior literature linking this ratio to cognitive decline and MMSE scores. The result held across all 88 subjects.

## Frequency bands

| Band | Range |
|------|-------|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–45 Hz |

## Setup

```bash
git clone https://github.com/BathHarman/eeg-alzheimer-screening.git
cd eeg-alzheimer-screening
pip install -r requirements.txt
```

Download ds004504 from [OpenNeuro](https://openneuro.org/datasets/ds004504) and update `DATA_ROOT` in `eeg_analysis.py`:

```python
DATA_ROOT = Path("/path/to/your/ds004504")
```

Then run:

```bash
python eeg_analysis.py
```

Results and figures will be saved to `results/`.

## Output files

| File | Description |
|------|-------------|
| `results/band_powers.csv` | Per-subject band power for all five bands |
| `results/statistics.csv` | T-test results and Cohen's d (AD vs HC) |
| `results/theta_alpha_ratio.png` | Mean theta/alpha ratio by group with significance annotation |
| `results/theta_alpha_boxplot.png` | Distribution of ratios by group |
| `results/band_power_heatmap.png` | Normalized band power across all groups |

## Stack

Python · MNE-Python · NumPy · SciPy · Pandas · Matplotlib

## Dataset

Bruña R, et al. (2023). OpenNeuro ds004504. https://openneuro.org/datasets/ds004504
