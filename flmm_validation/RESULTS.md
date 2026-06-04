# TRACY FLMM-style vs R fastFMM 1.0.1 — validation results

Dataset: synthetic, 12 subjects (6/group), 8 bouts each, L=50 timepoints, AR(1)
residual correlation + per-subject random intercept. Model: `Y ~ group + (1|id)`.
R: fastFMM 1.0.1 / lme4, `fui(analytic=TRUE)`. Python: statsmodels MixedLM (REML).

## What matches exactly
- **β(t) point estimates (raw per-timepoint):** max|diff| = **1.2e-14** (identical to
  lme4). statsmodels MixedLM and lme4 give the same fixed-effect estimates.
- **Joint-multiplier machinery:** feeding fastFMM's own covariance into our
  simulate-max-|z| routine reproduces its qn to **0.008** (2.677 vs 2.685). Our qn
  procedure is correct.

## What differs (the bands)
| quantity | fastFMM | python (best) | gap |
|---|---|---|---|
| pointwise SE (median) | 0.711 | 0.675 | ~5% low |
| joint multiplier qn — GLS sandwich | 2.685 | 2.499 | ~7% low |
| joint multiplier qn — analytic G(s,t) | 2.685 | 2.95 | ~10% high |
| joint multiplier qn — raw-data proxy (old) | 2.685 | 3.03 | ~13% high |

### Why the SE/qn gap exists
1. **Small-sample cluster correction.** fastFMM's SE ≈ statsmodels SE × sqrt(G/(G-1))
   (G=12 clusters); observed ratio 0.952 ≈ sqrt((G-1)/G)=0.957. statsmodels omits it.
2. **Cross-location smoothing.** fastFMM smooths the variance components and the
   covariance surface (penalized bivariate splines / fbps) across timepoints; we use
   Savitzky–Golay on β/SE only. This adds ~5% location-wise variation in SE.

## Conclusion
- The **effect estimate β(t) is exact** vs fastFMM.
- The **GLS subject-clustered coefficient covariance** is the best pure-Python band
  (qn within ~7%, a proper coefficient covariance) and clearly beats the old raw-data
  proxy. But pure Python is **not bit-identical** to fastFMM without replicating its
  cluster correction + spline covariance smoothing.
- **Exact fastFMM** requires calling the R package. R 4.6 + fastFMM 1.0.1 are installed
  and work here via Rscript, so an optional R backend is feasible.
