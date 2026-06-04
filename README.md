# TRACY — Fiber Photometry Analysis Suite

A desktop application for processing and analyzing fiber photometry recordings,
with optional behavior synchronization and a broad set of built-in analyses.
Built with Python/Tkinter; everything runs locally in a single GUI.

*Created by Jobe Ritchie — Kash Lab, University of North Carolina at Chapel Hill.*
Version 1.2.2.

## Getting Started
- Install [Git](https://git-scm.com/install/windows)
- Install [Python 3.9 or newer version](https://www.python.org/ftp/python/3.14.5/python-3.14.5-amd64.exe)
- Open command prompt in a new folder and type `git clone https://github.com/JobeRitchie/TRACY-Photometry-Processing-Suite.git`
- Double click Start_FP_GUI.bat
- Click project tab and create a new project
- Click processing tab and select either single files or a folder to process
- Click visualization tab, select a subject, select an option from the plot type drop down, and click visualize
- See Info tab for detailed information on all features

## Requirements
- Python 3.9+
- Git
- Dependencies listed in [requirements.txt](requirements.txt): pandas, numpy,
  scipy, matplotlib, openpyxl, statsmodels (`tkinter` ships with standard Python).

## Features

- **Project management** — organized projects with automatic save/load of
  parameters, processed data, groups, exclusions, and zone definitions.
- **Signal processing pipeline** — LED-state deinterleaving, photobleaching
  correction (biexponential), dF/F, isosbestic motion correction, and z-scoring.
  Single-subject or whole-folder batch processing with configurable file naming.
- **Behavior synchronization** — aligns photometry to tracking/timestamp data;
  derives velocity, distance traveled, zone occupancy, and zone entries.
- **Bout extraction & analysis** — onset-aligned traces per behavior, with
  summary statistics and histograms by subject or group.
- **FLMM time-course statistics** — Functional Linear Mixed Model (FUI) analysis
  of the whole peri-event window: a per-timepoint mixed model (random intercept
  per subject) yields an effect trace β(t) with pointwise and
  multiple-comparison-corrected *simultaneous* confidence bands, so you can see
  *when* an effect is significant. Reference points: signal ≠ 0, across bouts,
  between groups, and bout-order × group. (Bout Analysis tab → **Plot ▾ →
  Time-Course Statistics (FLMM-style)**.)
- **Coherence / connectivity** — spectral coherence between channels (Morlet
  wavelet or Welch): static, sliding, bout-epoch, and group comparisons.
- **Spike analysis** — MAD-based detection of calcium transients with rate,
  amplitude, and width metrics.
- **Decision probability** — explore-vs-retreat probability binned by z-score.
- **Signal integrity** — per-subject quality metrics (SNR, CV, isosbestic
  correlation, artifacts, photobleaching) with an overall quality score.
- **Visualization** — raw / dF/F / motion-corrected / z-scored traces, bout
  overlays, position heatmaps, and more, with content-aware figure sizing.


## Input Files

- **`{SubjectID}FPData0.csv`** — photometry data (frame, timestamps, LED state,
  1–4 channels). *Required.*
- **`{SubjectID}ComputerTS0.csv`** or **`{SubjectID}AnimalPosition0.csv`** —
  timestamps and optional X/Y position. *Optional* (enables behavior sync).
- **`boutframes.xlsx`** — one worksheet per subject; columns are behaviors,
  values are onset frame numbers. *Optional* (enables bout-aligned analysis).

File-name patterns and suffixes are configurable on the Processing tab, so
existing naming schemes do not need to be renamed.

## Optional: exact FLMM via R / fastFMM

The FLMM time-course feature ships with two engines:

- **Python FUI (default, no setup):** always available. The effect estimate
  β(t) matches R `lme4`/`fastFMM` to ~1e-14; the confidence bands use a GLS
  subject-clustered coefficient covariance and land within ~5–10% of `fastFMM`
  (the small-sample correction and spline covariance-smoothing differ).
- **R `fastFMM` (exact):** if R and the `fastFMM` package are installed, TRACY
  detects them automatically and offers an *"R fastFMM (exact)"* engine in the
  Time-Course Statistics dialog, producing bit-exact `fui()` results.

To enable the exact engine:

1. Install [R](https://cloud.r-project.org/) (4.x).
2. In R, install the package: `install.packages("fastFMM", dependencies = TRUE)`.
3. Restart TRACY. (TRACY finds `Rscript` on the PATH, under `R_HOME`, or in the
   standard `C:\Program Files\R\R-*\bin` location.)

A standalone demonstration that runs the Python engine on the bundled
`Sample Data/` and a reproducible comparison against R `fastFMM` live in
[Validation Tests/](Validation%20Tests/) and `flmm_validation/`.

## License

This project is distributed under the terms in [LICENCE.pdf](LICENCE.pdf).
Please review that file before use or redistribution.
