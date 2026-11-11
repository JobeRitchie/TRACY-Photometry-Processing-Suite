"""Tri-exponential bleach baseline is opt-in and chosen per channel by BIC."""

import numpy as np

import fp_analysis_gui

G = fp_analysis_gui.FPAnalysisGUI


def _app(on):
    inst = G.__new__(G)          # fitting needs no GUI state
    inst.params = {'detrend_auto_triexp': on}
    inst.log = []
    inst.log_message = inst.log.append
    return inst


def _session(fast_amp, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(43000) / 30.0              # 24 min at 30 Hz
    y = (8 * np.exp(-t / 45) + 6 * np.exp(-t / 500) + 25
         + fast_amp * np.exp(-t / 3) + rng.normal(0, 0.05, t.size))
    return t, y


def test_off_is_exactly_the_biexponential():
    t, y = _session(fast_amp=10)
    assert np.array_equal(_app(False).fit_bleach_baseline(t, y),
                          _app(False).fit_biexponential(t, y))


def test_fast_component_selects_triexponential():
    t, y = _session(fast_amp=10)
    app = _app(True)
    tri = app.fit_bleach_baseline(t, y)
    bi = app.fit_biexponential(t, y)
    assert 'tri-exponential' in app.log[-1]
    early = slice(0, 9000)
    assert np.std((y - tri)[early]) < 0.5 * np.std((y - bi)[early])


def test_plain_biexponential_keeps_biexponential():
    t, y = _session(fast_amp=0)
    app = _app(True)
    fitted = app.fit_bleach_baseline(t, y)
    assert 'bi-exponential' in app.log[-1]
    assert np.array_equal(fitted, app.fit_biexponential(t, y))
