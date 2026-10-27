"""The bi-exponential is fitted on block means, and must match a full fit.

Fitting every sample of a 40-minute session made the bleaching fit ~90% of
processing time. fit_biexponential now fits at most BIEXP_FIT_POINTS block
means with the analytic Jacobian and evaluates the curve on every sample; the
baseline it returns has to be indistinguishable from the all-samples fit.
"""

import numpy as np
import pytest

import fp_analysis_gui

G = fp_analysis_gui.FPAnalysisGUI


def _fitter(points):
    inst = G.__new__(G)          # fit_biexponential needs no GUI state
    inst.BIEXP_FIT_POINTS = points
    return inst


@pytest.mark.parametrize('seed', range(3))
def test_block_fit_matches_full_fit_on_long_session(seed):
    rng = np.random.default_rng(seed)
    n = 72000                    # 40 min at 30 Hz
    t = np.arange(n) / 30.0
    tn = t / t[-1]
    y = (900 * np.exp(-6 * tn) + 400 * np.exp(-0.4 * tn) + 12000
         + rng.normal(0, 40, n))
    fast = _fitter(G.BIEXP_FIT_POINTS).fit_biexponential(t, y)
    full = _fitter(10 ** 9).fit_biexponential(t, y)
    assert fast.shape == y.shape
    signal_sd = np.std(y - full)
    assert np.max(np.abs(fast - full)) < 0.01 * signal_sd


def test_short_signal_is_fitted_on_every_sample():
    t = np.arange(500) / 30.0
    y = 100 * np.exp(-t / 5) + 50 + np.random.default_rng(0).normal(0, 1, 500)
    a = _fitter(G.BIEXP_FIT_POINTS).fit_biexponential(t, y)
    b = _fitter(10 ** 9).fit_biexponential(t, y)
    assert np.array_equal(a, b)
