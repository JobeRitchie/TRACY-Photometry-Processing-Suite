"""The photobleaching fit must always return, at any signal scale.

Two defects motivated these tests, both found on real NSF recordings and both
present in the single 303-sample excerpt embedded below:

1. The fit was run on unnormalised y while curve_fit's step sizes and
   convergence tolerances are absolute.  On raw detector counts (~1e4) the
   strict fit failed and the retry -- which had infinite upper bounds and
   maxfev=20000 -- searched indefinitely.  Processing froze on the first
   subject with no error message and no timeout.

2. p0 was never checked against the bounds.  A signal that RISES over the
   session gives a negative decay_amount, which violates the a >= 0 bound, so
   curve_fit raised ValueError before fitting anything and the whole channel
   silently fell through to a linear detrend.

Synthetic bleaching curves do NOT reproduce either failure -- they fit happily
at any scale -- so the regression fixture is real data: subject 1-51, channel
G0, LED state 2, from the pre-flattening rebuild, decimated 1/64.  Anything
smoother than a real recording lets the old code pass and pins nothing.
"""
import base64
import time
import zlib

import numpy as np
import pytest

import fp_analysis_gui as G


# Real photometry that hung the old fit: 303 samples, ~8000 counts, carrying the
# detector startup transient that makes the strict fit fail and hand over to the
# unbounded retry.  Stored inline so the test needs no data files.
_HANG_FIXTURE = (
    "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"
)


def _real_hanging_signal():
    """(time, signal) for the recording excerpt that used to hang the fit."""
    raw = zlib.decompress(base64.b64decode("".join(_HANG_FIXTURE)))
    both = np.frombuffer(raw, dtype=np.float32).astype(float)
    return both[:len(both) // 2].copy(), both[len(both) // 2:].copy()


def _app():
    """A shell exposing fit_biexponential without constructing any Tk widgets."""
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    return app


def _bleaching(n=6000, duration=900.0, scale=1.0, offset=1.0, noise=0.0, seed=0):
    """A synthetic recording: two decaying components on a baseline.

    `noise` is a standard deviation in unit space, added before scaling, so the
    signal-to-noise ratio stays fixed as `scale` changes -- otherwise a test
    sweeping the scale would be changing two things at once.
    """
    t = np.linspace(0.0, duration, n)
    y = 0.30 * np.exp(-t / 20.0) + 0.12 * np.exp(-t / 400.0) + offset
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, n)
    return t, y * scale


# --------------------------------------------------------------------------
# 1. Scale invariance -- the defect that froze the GUI
# --------------------------------------------------------------------------

def test_real_recording_that_used_to_hang_now_fits():
    """The exact signal that froze processing at subject 1/44.

    This is the test that pins the bug.  It is timed rather than merely
    asserted to return, because the failure mode was an unbounded search: the
    old code was still running after 20 s on these 303 samples, and after
    45 s on the full 19,366.
    """
    app = _app()
    t, y = _real_hanging_signal()

    started = time.time()
    fitted = app.fit_biexponential(t, y)
    elapsed = time.time() - started

    assert np.all(np.isfinite(fitted))
    assert fitted.min() > 0                    # usable as a dF/F denominator
    assert elapsed < 5.0, f'fit took {elapsed:.1f}s -- the search is unbounded again'


@pytest.mark.parametrize('scale', [1.0, 1e2, 1e4, 3e4, 1e-3])
def test_fit_is_invariant_to_signal_scale(scale):
    """Counts, volts or dF/F: the same recording must give the same dF/F.

    dF/F is a ratio, so multiplying the input by any constant must leave it
    unchanged.  Uses the real recording, because a clean synthetic curve fits
    at every scale even with the bug present and so proves nothing.
    """
    app = _app()
    t, y_unit = _real_hanging_signal()
    y_unit = y_unit / y_unit.mean()            # the units the pipeline expects

    base = app.fit_biexponential(t, y_unit)
    dff_unit = 100 * (y_unit - base) / base

    y = y_unit * scale
    fitted = app.fit_biexponential(t, y)
    dff = 100 * (y - fitted) / fitted

    assert np.all(np.isfinite(dff))
    # Same dF/F to far inside any biologically meaningful difference; the
    # traces themselves swing by whole percent.
    assert np.max(np.abs(dff - dff_unit)) < 1e-3


@pytest.mark.parametrize('scale', [1.0, 1e4, 3e4])
def test_fit_recovers_the_true_baseline_at_any_scale(scale):
    """The fitted curve tracks the real bleaching curve, not just any curve."""
    app = _app()
    t, y = _bleaching(scale=scale, noise=0.002, seed=3)

    fitted = app.fit_biexponential(t, y)
    truth = (0.30 * np.exp(-t / 20.0) + 0.12 * np.exp(-t / 400.0) + 1.0) * scale

    # Residual to truth is small next to the bleaching amplitude it must capture.
    assert np.max(np.abs(fitted - truth)) < 0.02 * 0.30 * scale


# --------------------------------------------------------------------------
# 2. Feasible initial guess -- the silent fall-through to linear detrend
# --------------------------------------------------------------------------

def test_rising_signal_fits_instead_of_raising():
    """A signal that brightens must not throw 'x0 is infeasible'.

    decay_amount goes negative, which used to put p0 outside the a >= 0 bound.
    curve_fit raised ValueError, and because calculate_dff catches everything
    the entire channel was linear-detrended without the user being told why.
    """
    app = _app()
    t = np.linspace(0.0, 900.0, 6000)
    y = 1.0 + 0.15 * (t / t[-1])          # rises steadily, never bleaches

    fitted = app.fit_biexponential(t, y)

    assert np.all(np.isfinite(fitted))
    # The model cannot rise (both amplitudes are held >= 0), so the honest
    # answer is a near-flat baseline -- and it must stay in the data's range
    # rather than running off to satisfy the fit.
    assert fitted.min() > 0
    assert y.min() - 0.05 <= fitted.min() and fitted.max() <= y.max() + 0.05


def test_rising_signal_at_raw_scale_also_fits():
    """Both defects at once -- a rising signal reported in raw counts."""
    app = _app()
    t = np.linspace(0.0, 900.0, 6000)
    y = 9400.0 + 1200.0 * (t / t[-1])

    fitted = app.fit_biexponential(t, y)
    assert np.all(np.isfinite(fitted))
    assert fitted.min() > 0


# --------------------------------------------------------------------------
# 3. Bounded work -- the guarantee, not just the observed speed
# --------------------------------------------------------------------------

def test_evaluation_caps_are_finite_and_leave_headroom():
    """Healthy real fits used ~350 evaluations; the worst measured was 7,871."""
    assert np.isfinite(G.FPAnalysisGUI.FIT_MAX_NFEV)
    assert np.isfinite(G.FPAnalysisGUI.FIT_RETRY_MAX_NFEV)
    assert np.isfinite(G.FPAnalysisGUI.FIT_MAX_WORK)
    assert G.FPAnalysisGUI.FIT_MAX_NFEV >= 10000
    assert G.FPAnalysisGUI.FIT_RETRY_MAX_NFEV >= 3500


def test_work_cap_bounds_cost_without_starving_real_recordings():
    """Cost is nfev * samples, so the budget has to account for length.

    Capping nfev alone still lets a long recording run for minutes.  The cap
    must leave room for the slowest fit actually observed (7,871 evaluations on
    a 19,390-sample channel) while keeping the worst case to a few seconds.
    """
    cap = G.FPAnalysisGUI._fit_nfev_cap
    ceiling = G.FPAnalysisGUI.FIT_MAX_NFEV

    # A typical NSF channel: room for the slowest real fit, with headroom.
    assert cap(19390, ceiling) > 7871 * 1.25

    # Longer recordings get proportionally fewer evaluations, so the product --
    # the thing that costs time -- stays bounded at roughly 5 s per channel.
    for n in (1_000, 20_000, 200_000, 5_000_000):
        assert cap(n, ceiling) * n <= max(G.FPAnalysisGUI.FIT_MAX_WORK, 2000 * n)
        assert cap(n, ceiling) >= 2000        # never starve a short recording
    assert cap(500, ceiling) == ceiling       # short signals keep the full budget


def test_retry_search_region_is_compact():
    """The retry's bounds must all be finite.

    An infinite upper bound is what let the trust-region solver wander without
    ever converging.  Read back from the source so the guarantee cannot be
    quietly reverted.
    """
    import inspect
    src = inspect.getsource(G.FPAnalysisGUI.fit_biexponential)
    relaxed = src.split('bounds_relaxed = (')[1].split(')')[0]
    assert 'np.inf' not in relaxed, 'retry bounds are unbounded again'
    assert 'inf' not in relaxed.replace('np.inf', '')


def test_pathological_signal_still_terminates():
    """Noise with no bleaching curve in it: the fit may fail, but it must stop."""
    app = _app()
    t = np.linspace(0.0, 900.0, 12000)
    y = 5000.0 + np.random.default_rng(11).normal(0.0, 2500.0, 12000)

    started = time.time()
    try:
        fitted = app.fit_biexponential(t, y)
        assert np.all(np.isfinite(fitted))
    except (ValueError, RuntimeError):
        pass          # an honest failure is fine; callers linear-detrend
    assert time.time() - started < 30.0


# --------------------------------------------------------------------------
# 4. The result is safe to divide by
# --------------------------------------------------------------------------

def test_non_finite_input_is_rejected_not_propagated():
    """NaNs must not silently become a NaN baseline for the whole channel."""
    app = _app()
    t, y = _bleaching()
    y[500] = np.nan

    with pytest.raises(ValueError):
        app.fit_biexponential(t, y)


def test_positive_signal_yields_a_positive_baseline():
    """The fitted curve is the dF/F denominator, so it must not reach zero."""
    app = _app()
    for scale in (1.0, 1e4):
        t, y = _bleaching(scale=scale, noise=0.003, seed=5)
        assert app.fit_biexponential(t, y).min() > 0


def test_calculate_dff_reports_why_it_fell_back():
    """A degraded channel has to be visible in the log, not inferred later."""
    app = _app()
    messages = []
    app.log_message = lambda m='', *a, **k: messages.append(str(m))

    n = 4000
    t = np.linspace(0.0, 900.0, n)
    fp = np.column_stack([t, t, np.full(n, np.nan)])   # forces the fallback
    app.calculate_dff(fp, t[0])

    warned = [m for m in messages if 'Biexponential fit failed' in m]
    assert warned, 'silent fallback -- the user cannot tell the fit was skipped'
    assert 'ValueError' in warned[0]                   # names the actual cause
