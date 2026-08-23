"""
Tests for the bout-analysis decay metrics: tau (mono-exponential fit) and
t½ (model-free half-decay time).

These exercise the real ``calculate_tau`` / ``calculate_t_half`` /
``_orient_transient`` from fp_analysis_gui.py, bound to a minimal stub that
supplies only ``self.params`` -- no GUI / tkinter mainloop involved.

Regressions guarded here (all were live bugs):
  1.  A proportional lead (len//10) for the baseline swallowed the rising edge
      in longer windows, inverting clean positive transients and placing the
      "peak" at frame 0.
  2.  Tau tracked the ANALYSIS WINDOW rather than the signal, because the fit
      ran to the end of the window and started from a window-scaled guess
      (len/3) that dropped the optimiser into a different local minimum for
      every window length.
  3.  Fits that did not describe the data at all (R² <= 0) still reported a
      number: pure noise returned taus of 19 s, 33 s, 0.0 s.
  4.  A decay slower than the window is long was reported as a short tau by
      extrapolating past the recorded data (true 5 s seen for 4 s -> 0.8 s).
  5.  NaNs were deleted rather than interpolated, compressing the time axis and
      corrupting every frames-to-seconds conversion.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

FPS = 30.0
_GUI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fp_analysis_gui.py')


def _load_gui_module():
    spec = importlib.util.spec_from_file_location('fp_analysis_gui_under_test', _GUI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)   # safe: the GUI only starts under __main__
    return module


_G = _load_gui_module().FPAnalysisGUI


class _Metrics:
    """Stub exposing the decay metrics; they touch nothing but self.params."""
    DECAY_STRICTNESS = _G.DECAY_STRICTNESS
    _decay_fill_nans = staticmethod(_G._decay_fill_nans)
    _decay_noise_sigma = staticmethod(_G._decay_noise_sigma)
    _decay_smooth = _G._decay_smooth
    _orient_transient = _G._orient_transient
    calculate_tau = _G.calculate_tau
    calculate_t_half = _G.calculate_t_half
    get_fps = _G.get_fps
    FPS_FALLBACK = _G.FPS_FALLBACK

    def __init__(self, fps=FPS, strictness='balanced'):
        self.params = {'decay_strictness': strictness}
        # The sampling rate is a property of the recording, not a parameter:
        # get_fps() reads it from the processed subject (see fp_analysis_gui).
        self.processed_data = {'stub': {'photometry_fps': fps}}
        self._mixed_fps_warned = False
        self.log_message = lambda *_args, **_kw: None


@pytest.fixture
def m():
    return _Metrics()


def make_transient(tau_s, n_frames, amp=5.0, noise=0.0, rise=15, pre=30,
                   baseline=0.0, seed=0, fps=FPS):
    """Onset-aligned window: flat baseline, fast rise, mono-exponential decay."""
    rng = np.random.default_rng(seed)
    sig = np.full(n_frames, float(baseline))
    sig[pre:pre + rise] = baseline + amp * (np.arange(rise) / rise)
    tail = np.arange(n_frames - pre - rise)
    sig[pre + rise:] = baseline + amp * np.exp(-tail / (tau_s * fps))
    if noise:
        sig = sig + rng.normal(0, noise, n_frames)
    return sig


# ---------------------------------------------------------------------------
# Recovery of a known time constant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tau_true', [0.5, 1.0, 2.0, 5.0])
def test_tau_recovers_known_noiseless_decay(m, tau_true):
    win = int(max(10.0, tau_true * 4) * FPS)
    tau = m.calculate_tau(make_transient(tau_true, win))
    assert tau is not None
    assert tau == pytest.approx(tau_true, rel=0.05)


def test_very_fast_decay_is_resolved_to_within_the_smoothing_width(m):
    """A 0.25 s decay is 7.5 frames at 30 fps -- the same order as the ~200 ms
    smoothing used for peak finding, so a slight overestimate is expected.  It
    must still land in the right ballpark rather than being refused or wild."""
    tau = m.calculate_tau(make_transient(0.25, 300, rise=5))
    assert tau is not None
    assert tau == pytest.approx(0.25, rel=0.20)


@pytest.mark.parametrize('tau_true', [0.5, 1.0, 2.0])
def test_tau_recovers_known_decay_with_noise(m, tau_true):
    tau = m.calculate_tau(make_transient(tau_true, 300, noise=0.3))
    assert tau is not None
    assert tau == pytest.approx(tau_true, rel=0.15)


@pytest.mark.parametrize('window_s', [5.0, 10.0, 30.0, 70.0])
def test_tau_is_independent_of_analysis_window_length(m, window_s):
    """Regression 2: the same transient must yield the same tau no matter how
    much empty window follows it."""
    tau = m.calculate_tau(make_transient(1.0, int(window_s * FPS)))
    assert tau is not None
    assert tau == pytest.approx(1.0, rel=0.05)


def test_t_half_matches_tau_times_ln2(m):
    for tau_true in (0.5, 1.0, 2.0):
        t_half = m.calculate_t_half(make_transient(tau_true, 900))
        assert t_half is not None
        assert t_half == pytest.approx(tau_true * np.log(2), abs=0.1)


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('window_s', [5.0, 10.0, 30.0, 70.0])
def test_positive_transient_is_not_inverted_in_long_windows(m, window_s):
    """Regression 1: a clean upward transient was flipped once the window grew,
    putting the peak at frame 0."""
    data = make_transient(1.0, int(window_s * FPS))
    sig, sm, baseline, peak_idx = m._orient_transient(data)
    assert peak_idx == pytest.approx(45, abs=10)      # true peak ~frame 45
    assert sig[peak_idx] > baseline
    assert np.allclose(sig, data)                     # not flipped


def test_negative_going_transient_is_flipped_and_measured(m):
    tau = m.calculate_tau(-make_transient(1.0, 300, noise=0.1))
    assert tau is not None
    assert tau == pytest.approx(1.0, rel=0.15)


def test_orientation_ignores_a_single_noise_spike(m):
    """A lone sample must not define the peak."""
    data = make_transient(1.0, 300, noise=0.05)
    data[250] += 20.0                                  # one-frame artefact
    sig, sm, baseline, peak_idx = m._orient_transient(data)
    assert peak_idx < 100                              # the real transient wins


# ---------------------------------------------------------------------------
# Refusals: report N/A rather than an unsupported number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('seed', range(8))
def test_pure_noise_yields_no_decay_estimate(m, seed):
    """Regression 3: noise used to return taus of 19 s, 33 s, 0.0 s."""
    noise = np.random.default_rng(seed).normal(0, 1, 300)
    assert m.calculate_tau(noise) is None
    assert m.calculate_t_half(noise) is None


@pytest.mark.parametrize('strictness', ['strict', 'balanced', 'permissive'])
def test_decay_slower_than_the_window_is_refused(m, strictness):
    """Regression 4: only ~18% of a 20 s decay is visible in a 5 s window --
    too little for any strictness level to identify a time constant."""
    m.params['decay_strictness'] = strictness
    assert m.calculate_tau(make_transient(20.0, 150, noise=0.1)) is None


# ---------------------------------------------------------------------------
# Decay strictness (precision vs coverage)
# ---------------------------------------------------------------------------

def test_strictness_levels_are_ordered_from_least_to_most_permissive():
    falls = [_G.DECAY_STRICTNESS[k][0] for k in ('strict', 'balanced', 'permissive')]
    caps = [_G.DECAY_STRICTNESS[k][1] for k in ('strict', 'balanced', 'permissive')]
    assert falls == sorted(falls, reverse=True)   # less fall required as it loosens
    assert caps == sorted(caps)                   # longer taus tolerated


def test_a_partly_observed_decay_is_refused_when_strict_and_kept_when_permissive():
    """The setting must actually change the yield: a decay cut off partway is
    exactly the case the levels disagree about."""
    # 4 s decay seen for ~3.6 s: ~59% of the amplitude is lost, which sits
    # between the balanced (50%) and strict (63%) thresholds.
    partial = make_transient(4.0, 150, noise=0.05)
    assert _Metrics(strictness='strict').calculate_tau(partial) is None
    assert _Metrics(strictness='permissive').calculate_tau(partial) is not None


def test_a_fully_observed_decay_is_reported_at_every_strictness():
    full = make_transient(1.0, 600, noise=0.1)
    for level in ('strict', 'balanced', 'permissive'):
        tau = _Metrics(strictness=level).calculate_tau(full)
        assert tau is not None, level
        assert tau == pytest.approx(1.0, rel=0.15), level


def test_unknown_strictness_falls_back_to_balanced():
    weird = _Metrics(strictness='nonsense')
    known = _Metrics(strictness='balanced')
    data = make_transient(1.0, 300, noise=0.1)
    assert weird.calculate_tau(data) == pytest.approx(known.calculate_tau(data))


def test_late_spontaneous_event_does_not_become_the_peak(m):
    """The peak search is limited to the first half of the window, so a bigger
    unrelated event later on cannot hijack the measurement."""
    data = make_transient(1.0, 600)
    data[400:430] += 12.0                       # much larger, late, unrelated
    sig, sm, baseline, peak_idx = m._orient_transient(data)
    assert peak_idx < 300                       # still the onset transient
    assert m.calculate_tau(data) == pytest.approx(1.0, rel=0.15)


def test_monotonic_rise_has_no_decay(m):
    assert m.calculate_tau(np.linspace(0, 5, 300)) is None
    assert m.calculate_t_half(np.linspace(0, 5, 300)) is None


def test_peak_at_the_very_end_is_refused(m):
    data = np.concatenate([np.zeros(295), np.linspace(0, 5, 5)])
    assert m.calculate_tau(data) is None
    assert m.calculate_t_half(data) is None


def test_windows_shorter_than_five_samples_are_refused(m):
    assert m._orient_transient(np.arange(4, dtype=float)) is None
    assert m.calculate_tau(np.arange(4, dtype=float)) is None


def test_flat_signal_is_refused(m):
    assert m.calculate_tau(np.zeros(300)) is None
    assert m.calculate_t_half(np.zeros(300)) is None


# ---------------------------------------------------------------------------
# Invariances
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('offset', [-5.0, 0.0, 5.0])
def test_tau_is_invariant_to_a_constant_offset(m, offset):
    """Baseline correction on/off must not change the measured decay rate."""
    tau = m.calculate_tau(make_transient(1.0, 300, noise=0.2, baseline=offset))
    assert tau is not None
    assert tau == pytest.approx(1.0, rel=0.15)


def test_tau_scales_with_the_sampling_rate(m):
    """A decay of the same shape sampled at 60 fps is the same duration in
    seconds -- tau must be reported in seconds, not frames."""
    fast = _Metrics(fps=60.0)
    tau_30 = m.calculate_tau(make_transient(1.0, 300, fps=30.0))
    tau_60 = fast.calculate_tau(make_transient(1.0, 600, rise=30, pre=60, fps=60.0))
    assert tau_30 == pytest.approx(1.0, rel=0.05)
    assert tau_60 == pytest.approx(1.0, rel=0.05)


def test_interior_nans_do_not_shift_the_time_axis(m):
    """Regression 5: NaNs were deleted, which compressed the trace and made the
    reported decay artificially fast."""
    clean = make_transient(1.0, 300)
    gapped = clean.copy()
    gapped[120:135] = np.nan                     # 0.5 s gap partway down the decay
    tau_clean = m.calculate_tau(clean)
    tau_gapped = m.calculate_tau(gapped)
    assert tau_gapped is not None
    assert tau_gapped == pytest.approx(tau_clean, rel=0.1)


def test_all_nan_window_is_refused(m):
    assert m.calculate_tau(np.full(300, np.nan)) is None
    assert m.calculate_t_half(np.full(300, np.nan)) is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_decay_fill_nans_trims_edges_and_interpolates_interior():
    data = np.array([np.nan, 1.0, np.nan, 3.0, 4.0, np.nan])
    out = _G._decay_fill_nans(data)
    assert np.allclose(out, [1.0, 2.0, 3.0, 4.0])


def test_noise_sigma_tracks_the_injected_noise():
    rng = np.random.default_rng(0)
    for sigma in (0.1, 0.5, 2.0):
        est = _G._decay_noise_sigma(rng.normal(0, sigma, 5000))
        assert est == pytest.approx(sigma, rel=0.1)


def test_noise_sigma_is_not_inflated_by_the_transient():
    """A big slow transient must not be mistaken for noise, or the amplitude
    gate would reject every real event."""
    quiet = _G._decay_noise_sigma(make_transient(1.0, 300, amp=0.0, noise=0.2))
    loud = _G._decay_noise_sigma(make_transient(1.0, 300, amp=10.0, noise=0.2))
    assert loud == pytest.approx(quiet, rel=0.5)
