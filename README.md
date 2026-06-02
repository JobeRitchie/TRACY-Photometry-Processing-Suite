# TRACY — Fiber Photometry Analysis Suite

A desktop application for processing and analyzing fiber photometry recordings,
with optional behavior synchronization and a broad set of built-in analyses.
Built with Python/Tkinter; everything runs locally in a single GUI.

*Created by Jobe Ritchie — Kash Lab, University of North Carolina at Chapel Hill.*
Version 1.0.2.

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
  scipy, matplotlib, openpyxl (`tkinter` ships with standard Python).

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

## License

This project is distributed under the terms in [LICENCE.pdf](LICENCE.pdf).
Please review that file before use or redistribution.
