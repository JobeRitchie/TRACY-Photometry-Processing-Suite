# Validation Tests

Standalone demonstrations that TRACY's **FLMM time-course analysis** is a sound,
reproducible strategy — suitable for sharing with collaborators.

## `flmm_sample_data_gui_test.py` — GUI demo on the sample data

Loads `../Sample Data/FLMM_test_data_for_jobe.csv` (Loewinger et al. PV-cell
consumption dataset, 679 bouts × 181 timepoints, already in bout-trace format)
and runs TRACY's **pure-Python** FLMM time-course analysis in a small GUI.

**Why it's legitimate (no re-implementation):** the script imports the real
`FPAnalysisGUI` class and calls the *exact same methods* the TRACY app uses —
via a lightweight subclass (`TracyFLMMEngine`) that only skips the heavy Tk
setup. The analysis path is identical:

```
_fit_timecourse      per-timepoint random-intercept mixed model (FUI step 1)
_flmm_smooth         Savitzky–Golay smoothing of β(t), SE(t)
_flmm_python_qn      GLS subject-clustered coefficient covariance → joint multiplier
_flmm_assemble_bands pointwise + simultaneous bands & significance masks
_build_flmm_figure   the identical two-panel figure TRACY draws
_flmm_stats_lines    the identical text summary TRACY shows
```

**Run it** (use the same Python environment that runs TRACY — i.e. the one with
`numpy/scipy/statsmodels`, not a bare `.venv`):

```
python "Validation Tests/flmm_sample_data_gui_test.py"
```

Pick a **Reference** and, for group contrasts, a **Grouping variable**
(`session`, `sex`, `side`, …), then **Run analysis**. The per-timepoint mixed
models take ~20–30 s (run on a background thread; the window stays responsive).
Example: the *Signal ≠ 0* reference flags the post-onset consumption transient
as significant across a large contiguous window.

**Condition contrast (one level as the 0-point).** The *Condition contrast*
reference treats one level of the grouping variable as the baseline (0-point)
and compares another level — or **all others pooled** — against it, giving
β(t) = (comparison) − (reference). For the bundled data, set Grouping
variable = `session`, Reference level = `water`, Compare to = `‹all others›`
to get **water vs. others**. This mirrors the main app's *Behavior contrast*
reference, where the factor is the bout's behavior instead of a metadata column.

**Factor: all levels vs a reference (one model).** The *Factor* reference fits a
single model `Y ~ session + (1|mouse)` with a chosen reference level and draws a
panel per coefficient: the intercept (mean of the reference level) plus a
"reference vs level" panel for every other level. This is the analysis behind a
"significant difference based on session" result, and its confidence bands pool
the residual/random structure across **all** levels (unlike separate pairwise
fits). For the bundled data: Reference = *Factor*, Grouping variable = `session`,
Reference level = `water`. Validated against R `fastFMM` (`Y ~ session`): β(t)
matches lme4/fastFMM to ~1e-14 and joint multipliers land within ~5–11%. The main
app exposes the same thing as *Factor: ALL behaviors vs a reference*, and there an
exact R `fastFMM` engine is offered when R is installed.

> The `fps` / `samples-before-onset` fields only set the x-axis labels; they do
> not affect the statistics.

## Relationship to `../flmm_validation/`

`../flmm_validation/` contains the head-to-head numeric validation against the
real R package **`fastFMM` 1.0.1** (`RESULTS.md`): β(t) matches R `lme4`/
`fastFMM` to ~1e-14, and the simultaneous-band multiplier is within ~5–10% of
`fastFMM`. That folder has the data generator, the R script, and the comparison
harness for full reproducibility.
